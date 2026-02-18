"""Workflow engine for Sparky AI.

Controls macro-sequencing of research steps. Each step launches a Claude
session with a scoped prompt. The engine handles step ordering, completion
checks, retries, budget tracking, PAUSE/inject, and alerting.
"""

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from sparky.workflow.telemetry import SessionTelemetry, StreamParser, save_telemetry

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/home/akamath/sparky-ai")


def _upload_session_to_wandb(telemetry, log_path, step) -> None:
    """Upload session log and telemetry to wandb (background thread)."""
    import threading

    def _upload():
        try:
            import wandb
            run = wandb.init(
                project="sparky-ai",
                entity="datadex_ai",
                name=f"session_{telemetry.session_id}_{telemetry.step}",
                job_type="session",
                group="ceo_sessions",
                config={
                    "session_id": telemetry.session_id,
                    "step": telemetry.step,
                    "attempt": telemetry.attempt,
                    "exit_reason": telemetry.exit_reason,
                },
                tags=["session", telemetry.step, telemetry.exit_reason],
                reinit=True,
            )

            wandb.log({
                "duration_minutes": telemetry.duration_minutes,
                "cost_usd": getattr(telemetry, "cost_usd", 0),
                "estimated_cost_usd": telemetry.estimated_cost_usd,
                "tokens_input": telemetry.tokens_input,
                "tokens_output": telemetry.tokens_output,
                "tokens_cache_read": getattr(telemetry, "tokens_cache_read", 0),
                "tokens_cache_creation": getattr(telemetry, "tokens_cache_creation", 0),
                "tool_calls": telemetry.tool_calls,
                "num_turns": getattr(telemetry, "num_turns", 0),
            })

            if log_path and Path(log_path).exists():
                art = wandb.Artifact(
                    name=f"session-log-{telemetry.session_id}",
                    type="session_log",
                    description=f"step={telemetry.step} exit={telemetry.exit_reason}",
                )
                art.add_file(str(log_path))
                run.log_artifact(art)

            wandb.finish()
            logger.info(f"Uploaded session {telemetry.session_id} to wandb")

        except Exception as e:
            logger.warning(f"wandb session upload failed (non-fatal): {e}")

    t = threading.Thread(target=_upload, daemon=True)
    t.start()
STATE_DIR = PROJECT_ROOT / "workflows" / "state"
LOG_DIR = PROJECT_ROOT / "logs" / "ceo_sessions"
ALERT_SCRIPT = PROJECT_ROOT / "scripts" / "alert.sh"


def _always_false() -> bool:
    return False


@dataclass
class Step:
    """A single workflow step."""

    name: str
    prompt: str
    done_when: Callable[[], bool] = _always_false
    skip_if: Callable[[], bool] = _always_false
    max_duration_minutes: int = 120
    max_retries: int = 3
    tags: list[str] = field(default_factory=list)


@dataclass
class StepState:
    """Persistent state for a single step."""

    name: str
    status: str = "pending"  # pending | running | completed | skipped | failed
    attempts: int = 0
    last_attempt_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "attempts": self.attempts,
            "last_attempt_at": self.last_attempt_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StepState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class BudgetState:
    """Budget tracking across workflow execution."""

    max_hours: float = 24.0
    hours_used: float = 0.0
    estimated_cost_usd: float = 0.0
    runs_completed: int = 0
    warned_80_pct: bool = False

    def to_dict(self) -> dict:
        return {
            "max_hours": self.max_hours,
            "hours_used": self.hours_used,
            "estimated_cost_usd": self.estimated_cost_usd,
            "runs_completed": self.runs_completed,
            "warned_80_pct": self.warned_80_pct,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BudgetState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class WorkflowState:
    """Full persistent state for a workflow."""

    workflow_name: str
    current_step_index: int = 0
    steps: dict[str, StepState] = field(default_factory=dict)
    budget: BudgetState = field(default_factory=BudgetState)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "workflow_name": self.workflow_name,
            "current_step_index": self.current_step_index,
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
            "budget": self.budget.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowState":
        state = cls(
            workflow_name=d["workflow_name"],
            current_step_index=d.get("current_step_index", 0),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )
        state.steps = {
            k: StepState.from_dict(v) for k, v in d.get("steps", {}).items()
        }
        state.budget = BudgetState.from_dict(d.get("budget", {}))
        return state

    def save(self, state_dir: Path = STATE_DIR) -> None:
        """Atomically persist state to disk."""
        state_dir.mkdir(parents=True, exist_ok=True)
        filepath = state_dir / f"{self.workflow_name}.json"
        self.updated_at = datetime.now(timezone.utc).isoformat()

        # Atomic write via temp file + os.replace
        fd, tmp_path = tempfile.mkstemp(
            dir=str(state_dir), suffix=".tmp", prefix=".state_"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            os.replace(tmp_path, str(filepath))
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, workflow_name: str, state_dir: Path = STATE_DIR) -> Optional["WorkflowState"]:
        """Load state from disk, or return None if not found."""
        filepath = state_dir / f"{workflow_name}.json"
        if not filepath.exists():
            return None
        with open(filepath) as f:
            return cls.from_dict(json.load(f))


class Workflow:
    """Workflow engine that sequences Claude sessions through research steps."""

    def __init__(
        self,
        name: str,
        steps: list[Step],
        max_hours: float = 24.0,
        state_dir: Path = STATE_DIR,
    ):
        self.name = name
        self.steps = steps
        self.max_hours = max_hours
        self.state_dir = state_dir

    def _load_or_create_state(self) -> WorkflowState:
        """Load existing state or create fresh."""
        state = WorkflowState.load(self.name, self.state_dir)
        if state is None:
            state = WorkflowState(
                workflow_name=self.name,
                created_at=datetime.now(timezone.utc).isoformat(),
                budget=BudgetState(max_hours=self.max_hours),
            )
            for step in self.steps:
                state.steps[step.name] = StepState(name=step.name)
            state.save(self.state_dir)
        return state

    def _alert(self, severity: str, message: str) -> None:
        """Send alert via alert.sh."""
        try:
            subprocess.run(
                ["bash", str(ALERT_SCRIPT), severity, message],
                timeout=10,
                capture_output=True,
            )
        except Exception as e:
            logger.warning(f"Alert failed: {e}")

    def _clean_env(self) -> dict[str, str]:
        """Build clean environment for Claude subprocess."""
        env = os.environ.copy()
        for var in ("CLAUDECODE", "CLAUDE_CODE_SSE_PORT", "CLAUDE_CODE_ENTRYPOINT"):
            env.pop(var, None)
        return env

    def _build_prompt(self, step: Step, index: int, inject_text: str = "") -> str:
        """Build the full prompt for a Claude session."""
        total = len(self.steps)
        preamble = (
            f"Read CLAUDE.md. You are executing step {index + 1} of {total} "
            f"in workflow '{self.name}'. Step: '{step.name}'. "
            f"When done, exit cleanly. Do NOT advance to other steps. "
            f"Do NOT present option menus or ask for decisions — just work."
        )
        prompt = f"{preamble}\n\n{step.prompt}"
        if inject_text:
            prompt += f"\n\nAdditional guidance from AK: {inject_text}"
        return prompt

    def _launch_claude(
        self,
        prompt: str,
        max_duration_minutes: int,
        step: Step,
        attempt: int,
    ) -> SessionTelemetry:
        """Launch a Claude session and collect telemetry."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        session_id = timestamp
        log_filename = f"workflow_{self.name}_{timestamp}.log"

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / log_filename
        latest_link = LOG_DIR / "latest.log"

        # Create symlink
        try:
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(log_path)
        except OSError:
            pass

        parser = StreamParser(session_id=session_id, step=step.name, attempt=attempt)
        env = self._clean_env()
        timeout_seconds = max_duration_minutes * 60

        from sparky.tracking.experiment import set_current_session, clear_current_session
        set_current_session(session_id)

        with open(log_path, "w") as log_file:
            log_file.write(
                f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] "
                f"Workflow '{self.name}' step '{step.name}' attempt {attempt}\n"
            )
            log_file.flush()

            try:
                proc = subprocess.Popen(
                    [
                        "claude", "-p", prompt,
                        "--model", "sonnet",
                        "--verbose",
                        "--output-format", "stream-json",
                        "--dangerously-skip-permissions",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    cwd=str(PROJECT_ROOT),
                    text=True,
                    bufsize=1,
                )

                start_time = time.monotonic()
                for line in proc.stdout:
                    parser.feed(line, log_file)
                    if time.monotonic() - start_time > timeout_seconds:
                        proc.kill()
                        parser.telemetry.exit_reason = "timeout"
                        logger.warning(f"Session timed out after {max_duration_minutes}m")
                        break

                proc.wait(timeout=30)

                # Check stderr for rate limit signals
                stderr_output = ""
                if proc.stderr:
                    try:
                        stderr_output = proc.stderr.read()
                    except Exception:
                        pass

                if proc.returncode != 0 and parser.telemetry.exit_reason == "completed":
                    stderr_lower = stderr_output.lower()
                    if any(s in stderr_lower for s in ("rate limit", "too many requests", "out of extra usage")):
                        parser.telemetry.exit_reason = "rate_limit"
                    else:
                        parser.telemetry.exit_reason = "error"

                # Parser may have detected rate limit from stream-json
                if parser.telemetry.exit_reason == "rate_limit":
                    pass  # Already set by parser

            except subprocess.TimeoutExpired:
                proc.kill()
                parser.telemetry.exit_reason = "timeout"
            except FileNotFoundError:
                logger.error("'claude' command not found")
                parser.telemetry.exit_reason = "error"
            except Exception as e:
                logger.error(f"Error launching claude: {e}")
                parser.telemetry.exit_reason = "error"
            finally:
                clear_current_session()

            log_file.write(
                f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] "
                f"Session ended ({parser.telemetry.exit_reason})\n"
            )

        telemetry = parser.finalize()
        save_telemetry(telemetry)
        _upload_session_to_wandb(telemetry, log_path, step)
        return telemetry

    def run(self) -> int:
        """Execute the workflow. Returns exit code (0=done/paused, 1=retry via systemd)."""
        state = self._load_or_create_state()

        # PAUSE check
        pause_file = self.state_dir / "PAUSE"
        if pause_file.exists():
            logger.info("Paused by operator")
            self._alert("INFO", "Workflow paused by operator")
            return 0

        # Budget check
        if state.budget.hours_used >= state.budget.max_hours:
            best = self._get_best_result_summary()
            self._alert("CRITICAL", f"Budget exhausted ({state.budget.hours_used:.1f}h). {best}")
            return 0

        for i in range(state.current_step_index, len(self.steps)):
            step = self.steps[i]
            step_state = state.steps.get(step.name)
            if step_state is None:
                step_state = StepState(name=step.name)
                state.steps[step.name] = step_state

            # Check skip_if
            try:
                if step.skip_if():
                    step_state.status = "skipped"
                    state.current_step_index = i + 1
                    state.save(self.state_dir)
                    logger.info(f"Skipping step '{step.name}' (skip_if=True)")
                    continue
            except Exception as e:
                logger.warning(f"skip_if check failed for '{step.name}': {e}")

            # Check done_when
            try:
                if step.done_when():
                    step_state.status = "completed"
                    step_state.completed_at = datetime.now(timezone.utc).isoformat()
                    state.current_step_index = i + 1
                    state.save(self.state_dir)
                    best = self._get_best_result_summary()
                    self._alert(
                        "INFO",
                        f"Step '{step.name}' complete. {state.budget.runs_completed} runs. {best}",
                    )
                    logger.info(f"Step '{step.name}' already done")
                    continue
            except Exception as e:
                logger.warning(f"done_when check failed for '{step.name}': {e}")

            # Check retries exhausted
            if step_state.attempts >= step.max_retries:
                step_state.status = "failed"
                state.save(self.state_dir)
                self._alert(
                    "ERROR",
                    f"Step '{step.name}' failed after {step_state.attempts} attempts",
                )
                return 1

            # Read inject file
            inject_text = ""
            inject_file = self.state_dir / "inject.md"
            if inject_file.exists():
                try:
                    inject_text = inject_file.read_text().strip()
                    inject_file.unlink()
                    logger.info(f"Injected guidance: {inject_text[:100]}...")
                except Exception as e:
                    logger.warning(f"Failed to read inject file: {e}")

            # Build prompt and launch (with rate limit retry loop)
            prompt = self._build_prompt(step, i, inject_text)
            step_state.status = "running"
            step_state.attempts += 1
            step_state.last_attempt_at = datetime.now(timezone.utc).isoformat()
            state.save(self.state_dir)

            telemetry = self._launch_claude(
                prompt=prompt,
                max_duration_minutes=step.max_duration_minutes,
                step=step,
                attempt=step_state.attempts,
            )

            # Rate limit handling: retry with backoff, don't count against max_retries
            if telemetry.exit_reason == "rate_limit":
                step_state.attempts -= 1  # Don't count rate-limited session
                logger.info("Rate limited. Waiting for reset.")
                self._alert("INFO", f"Rate limited on step '{step.name}'. Waiting 5m.")

                # Update budget for wall time spent
                state.budget.hours_used += telemetry.duration_minutes / 60.0
                state.budget.estimated_cost_usd += telemetry.estimated_cost_usd
                state.save(self.state_dir)

                # Backoff: 5m first, then 10m intervals
                wait_minutes = [5, 10, 10, 10, 10, 10, 10, 10, 10, 10]
                for wait_idx, wait_m in enumerate(wait_minutes):
                    # Check budget before waiting
                    wait_hours = wait_m / 60.0
                    if state.budget.hours_used + wait_hours >= state.budget.max_hours:
                        self._alert("CRITICAL", f"Budget would be exhausted during rate limit wait.")
                        return 0

                    logger.info(f"Rate limit backoff: waiting {wait_m} minutes (attempt {wait_idx + 1})")
                    time.sleep(wait_m * 60)
                    state.budget.hours_used += wait_hours
                    state.save(self.state_dir)

                    # Retry
                    step_state.attempts += 1
                    step_state.last_attempt_at = datetime.now(timezone.utc).isoformat()
                    state.save(self.state_dir)

                    telemetry = self._launch_claude(
                        prompt=prompt,
                        max_duration_minutes=step.max_duration_minutes,
                        step=step,
                        attempt=step_state.attempts,
                    )

                    if telemetry.exit_reason != "rate_limit":
                        break  # Rate limit cleared
                    else:
                        step_state.attempts -= 1  # Don't count this one either
                        state.budget.hours_used += telemetry.duration_minutes / 60.0
                        state.budget.estimated_cost_usd += telemetry.estimated_cost_usd
                        state.save(self.state_dir)
                        logger.info("Still rate limited.")
                else:
                    # Exhausted all rate limit retries
                    self._alert("ERROR", f"Rate limit persisted through all backoff attempts on '{step.name}'")
                    step_state.status = "pending"
                    state.save(self.state_dir)
                    return 1

            # Update budget
            state.budget.hours_used += telemetry.duration_minutes / 60.0
            state.budget.estimated_cost_usd += telemetry.estimated_cost_usd
            state.budget.runs_completed += 1
            state.save(self.state_dir)

            # Budget 80% warning
            if (
                state.budget.hours_used / state.budget.max_hours >= 0.8
                and not state.budget.warned_80_pct
            ):
                state.budget.warned_80_pct = True
                self._alert(
                    "WARN",
                    f"Budget 80% used: {state.budget.hours_used:.1f}/{state.budget.max_hours:.1f}h",
                )
                state.save(self.state_dir)

            # Behavioral flag alerts
            for flag in telemetry.behavioral_flags:
                self._alert("WARN", f"{flag} in session {telemetry.session_id}")

            # Re-check done_when
            try:
                if step.done_when():
                    step_state.status = "completed"
                    step_state.completed_at = datetime.now(timezone.utc).isoformat()
                    state.current_step_index = i + 1
                    state.save(self.state_dir)
                    best = self._get_best_result_summary()
                    self._alert(
                        "INFO",
                        f"Step '{step.name}' complete. {state.budget.runs_completed} runs. {best}",
                    )
                    continue
                else:
                    # Not done — return 1 so systemd restarts
                    step_state.status = "pending"
                    state.save(self.state_dir)
                    return 1
            except Exception as e:
                logger.warning(f"done_when re-check failed for '{step.name}': {e}")
                step_state.status = "pending"
                state.save(self.state_dir)
                return 1

        # All steps done
        best = self._get_best_result_summary()
        self._alert(
            "INFO",
            f"All {len(self.steps)} steps done. {best}",
        )
        return 0

    def _get_best_result_summary(self) -> str:
        """Get a short summary of the best result from wandb."""
        try:
            from sparky.tracking.experiment import ExperimentTracker
            tracker = ExperimentTracker()
            best = tracker.get_best_run("sharpe")
            sharpe = best.get("metrics", {}).get("sharpe", "?")
            name = best.get("name", "?")
            return f"Best: Sharpe {sharpe} ({name})"
        except Exception:
            return "Best: N/A"
