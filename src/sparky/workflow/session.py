"""Session launcher and shared utilities for Sparky AI.

Extracted from the former workflow engine. Contains the session launcher,
idle-loop detection, alerting, and wandb upload used by both the
ResearchOrchestrator and tests.
"""

import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from sparky.workflow.telemetry import SessionTelemetry, StreamParser, save_telemetry

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Idle-loop detection: kill agent if it outputs many consecutive text blocks
# with no tool calls and repeated "done" phrases.
IDLE_LOOP_CONSECUTIVE_THRESHOLD = 5
IDLE_LOOP_PHRASES = re.compile(
    r"session is done|no further action|all complete|nothing more to do|"
    r"work is complete|task is complete|i.ve completed|have completed all|"
    r"no additional work|nothing left to do|finished all",
    re.IGNORECASE,
)


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
                group="research_sessions",
                config={
                    "session_id": telemetry.session_id,
                    "step": telemetry.step,
                    "attempt": telemetry.attempt,
                    "exit_reason": telemetry.exit_reason,
                },
                tags=["session", telemetry.step, telemetry.exit_reason],
                reinit=True,
            )

            wandb.log(
                {
                    "duration_minutes": telemetry.duration_minutes,
                    "cost_usd": getattr(telemetry, "cost_usd", 0),
                    "estimated_cost_usd": telemetry.estimated_cost_usd,
                    "tokens_input": telemetry.tokens_input,
                    "tokens_output": telemetry.tokens_output,
                    "tokens_cache_read": getattr(telemetry, "tokens_cache_read", 0),
                    "tokens_cache_creation": getattr(telemetry, "tokens_cache_creation", 0),
                    "tool_calls": telemetry.tool_calls,
                    "num_turns": getattr(telemetry, "num_turns", 0),
                }
            )

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
LOG_DIR = PROJECT_ROOT / "logs" / "research_sessions"
ALERT_SCRIPT = PROJECT_ROOT / "scripts" / "alert.sh"


def send_alert(severity: str, message: str) -> None:
    """Send alert via alert.sh."""
    try:
        subprocess.run(
            ["bash", str(ALERT_SCRIPT), severity, message],
            timeout=10,
            capture_output=True,
        )
    except Exception as e:
        logger.warning(f"Alert failed: {e}")


def clean_env() -> dict[str, str]:
    """Build clean environment for Claude subprocess."""
    env = os.environ.copy()
    for var in ("CLAUDECODE", "CLAUDE_CODE_SSE_PORT", "CLAUDE_CODE_ENTRYPOINT"):
        env.pop(var, None)
    return env


def launch_claude_session(
    prompt: str,
    max_duration_minutes: int,
    session_name: str,
    step_name: str = "session",
    attempt: int = 1,
    log_dir: Path = LOG_DIR,
    disallowed_tools: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
) -> SessionTelemetry:
    """Launch a Claude session and collect telemetry.

    Args:
        disallowed_tools: Tool patterns to block (e.g. ["Bash(git:*)", "Bash(gh:*)"]).
        extra_env: Additional environment variables to set for the subprocess.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session_id = timestamp
    log_filename = f"{session_name}_{timestamp}.log"

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_filename
    latest_link = log_dir / "latest.log"

    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(log_path)
    except OSError:
        pass

    parser = StreamParser(session_id=session_id, step=step_name, attempt=attempt)
    env = clean_env()
    if extra_env:
        env.update(extra_env)
    timeout_seconds = max_duration_minutes * 60

    from sparky.tracking.experiment import clear_current_session, set_current_session

    set_current_session(session_id)

    with open(log_path, "w") as log_file:
        log_file.write(
            f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] "
            f"Session '{session_name}' step '{step_name}' attempt {attempt}\n"
        )
        log_file.flush()

        try:
            cmd = [
                "claude",
                "-p",
                prompt,
                "--model",
                "sonnet",
                "--verbose",
                "--output-format",
                "stream-json",
                "--allowedTools",
                "Read,Write,Edit,Bash,Glob,Grep",
            ]
            if disallowed_tools:
                cmd.extend(["--disallowedTools", ",".join(disallowed_tools)])
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=str(PROJECT_ROOT),
                text=True,
                bufsize=1,
            )

            start_time = time.monotonic()
            idle_consecutive_text = 0
            idle_phrase_hits = 0
            for line in proc.stdout:
                parser.feed(line, log_file)

                stripped = line.strip()
                if stripped:
                    try:
                        msg = json.loads(stripped)
                        if msg.get("type") == "assistant":
                            content = msg.get("message", {}).get("content", [])
                            has_tool = False
                            has_idle_phrase = False
                            if isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict):
                                        if block.get("type") == "tool_use":
                                            has_tool = True
                                        elif block.get("type") == "text":
                                            text = block.get("text", "")
                                            if IDLE_LOOP_PHRASES.search(text):
                                                has_idle_phrase = True
                            elif isinstance(content, str):
                                if IDLE_LOOP_PHRASES.search(content):
                                    has_idle_phrase = True

                            if has_tool:
                                idle_consecutive_text = 0
                                idle_phrase_hits = 0
                            else:
                                idle_consecutive_text += 1
                                if has_idle_phrase:
                                    idle_phrase_hits += 1

                            if idle_consecutive_text >= IDLE_LOOP_CONSECUTIVE_THRESHOLD and idle_phrase_hits >= 3:
                                proc.kill()
                                parser.telemetry.exit_reason = "idle_loop_detected"
                                logger.warning(
                                    f"Idle loop detected: {idle_consecutive_text} "
                                    f"consecutive text blocks, {idle_phrase_hits} "
                                    f"idle phrases. Killing session."
                                )
                                break
                    except json.JSONDecodeError:
                        pass

                if time.monotonic() - start_time > timeout_seconds:
                    proc.kill()
                    parser.telemetry.exit_reason = "timeout"
                    logger.warning(f"Session timed out after {max_duration_minutes}m")
                    break

            proc.wait(timeout=30)

            stderr_output = ""
            if proc.stderr:
                try:
                    stderr_output = proc.stderr.read()
                except Exception:  # noqa: S110
                    pass

            if proc.returncode != 0 and parser.telemetry.exit_reason == "completed":
                stderr_lower = stderr_output.lower()
                if any(s in stderr_lower for s in ("rate limit", "too many requests", "out of extra usage")):
                    parser.telemetry.exit_reason = "rate_limit"
                else:
                    parser.telemetry.exit_reason = "error"

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
    return telemetry
