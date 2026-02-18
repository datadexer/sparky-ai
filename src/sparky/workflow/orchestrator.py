"""Research orchestrator for autonomous multi-session experiment loops.

Loads a YAML directive, launches sequential Claude sessions with inter-session
context built from wandb results, and enforces stopping criteria, crash loop
protection, diversity stall detection, and budget limits.
"""

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from sparky.oversight.holdout_guard import get_policy_hash
from sparky.workflow.engine import (
    LOG_DIR,
    PROJECT_ROOT,
    STATE_DIR,
    launch_claude_session,
    send_alert,
)

logger = logging.getLogger(__name__)

GATE_REQUEST_PATH = PROJECT_ROOT / "GATE_REQUEST.md"
GATE_RESPONSE_PATH = PROJECT_ROOT / "GATE_RESPONSE.md"

# Research agents must NOT use git, GitHub CLI, or CI tools.
# This is enforced at the tool level via --disallowedTools.
RESEARCH_DISALLOWED_TOOLS = [
    # Version control & CI
    "Bash(git:*)",
    "Bash(gh:*)",
    "Bash(ruff:*)",
    "Bash(pre-commit:*)",
    "Bash(pytest:*)",
    "Bash(black:*)",
    "Bash(flake8:*)",
    "Bash(mypy:*)",
    # Package management — no installing/removing packages
    "Bash(pip:*)",
    "Bash(pip3:*)",
    "Bash(uv:*)",
    "Bash(conda:*)",
    # Network — no arbitrary downloads
    "Bash(curl:*)",
    "Bash(wget:*)",
    # System administration
    "Bash(systemctl:*)",
    "Bash(chmod:*)",
    "Bash(chown:*)",
    # Destructive operations
    "Bash(rm -rf:*)",
    # Process management
    "Bash(kill:*)",
    "Bash(pkill:*)",
    "Bash(killall:*)",
    # OOS vault access — block common read commands
    "Bash(cat data/.oos_vault:*)",
    "Bash(cat ./data/.oos_vault:*)",
    "Bash(head data/.oos_vault:*)",
    "Bash(head ./data/.oos_vault:*)",
    "Bash(tail data/.oos_vault:*)",
    "Bash(tail ./data/.oos_vault:*)",
    "Bash(ls data/.oos_vault:*)",
    "Bash(ls ./data/.oos_vault:*)",
    "Bash(cp data/.oos_vault:*)",
    "Bash(cp ./data/.oos_vault:*)",
]


# ── Directive ─────────────────────────────────────────────────────────────


@dataclass
class StoppingCriteria:
    """When to declare success or stop."""

    success_min_sharpe: float = 1.0
    success_min_dsr: float = 0.95
    max_sessions: int = 20
    max_hours: float = 24.0
    max_cost_usd: float = 100.0
    digest_every: int = 5
    sessions_without_improvement: int = 5
    improvement_threshold: float = 0.05
    diversity_threshold: float = 0.80


@dataclass
class SessionLimits:
    """Per-session constraints."""

    max_session_minutes: int = 120
    max_cost_per_session: float = 10.0
    min_session_minutes: float = 5.0
    max_consecutive_crashes: int = 5


@dataclass
class StrategySpec:
    """A strategy family to explore."""

    family: str = ""
    description: str = ""
    priority: int = 1
    parameter_ranges: dict[str, list] = field(default_factory=dict)


@dataclass
class GateSpec:
    """A decision gate."""

    trigger: str = ""
    action: str = "pause_and_alert"


@dataclass
class ResearchDirective:
    """Parsed research directive from YAML."""

    name: str
    objective: str
    constraints: dict[str, Any] = field(default_factory=dict)
    strategy_space: list[StrategySpec] = field(default_factory=list)
    stopping_criteria: StoppingCriteria = field(default_factory=StoppingCriteria)
    session_limits: SessionLimits = field(default_factory=SessionLimits)
    wandb_tags: list[str] = field(default_factory=list)
    gates: list[GateSpec] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ResearchDirective":
        """Load directive from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Directive file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("Directive YAML must be a mapping")

        # Required fields
        for req in ("name", "objective"):
            if req not in data:
                raise ValueError(f"Directive missing required field: '{req}'")

        stopping = data.get("stopping_criteria", {})
        success = stopping.get("success", {})
        budget = stopping.get("budget", {})
        stall = stopping.get("stall", {})

        sc = StoppingCriteria(
            success_min_sharpe=success.get("min_sharpe", 1.0),
            success_min_dsr=success.get("min_dsr", 0.95),
            max_sessions=budget.get("max_sessions", 20),
            max_hours=budget.get("max_hours", 24.0),
            max_cost_usd=budget.get("max_cost_usd", 100.0),
            digest_every=budget.get("digest_every", 5),
            sessions_without_improvement=stall.get("sessions_without_improvement", 5),
            improvement_threshold=stall.get("improvement_threshold", 0.05),
            diversity_threshold=stall.get("diversity_threshold", 0.80),
        )

        sl_data = data.get("session_limits", {})
        sl = SessionLimits(
            max_session_minutes=sl_data.get("max_session_minutes", 120),
            max_cost_per_session=sl_data.get("max_cost_per_session", 10.0),
            min_session_minutes=sl_data.get("min_session_minutes", 5.0),
            max_consecutive_crashes=sl_data.get("max_consecutive_crashes", 5),
        )

        strategies = []
        for s in data.get("strategy_space", []):
            strategies.append(
                StrategySpec(
                    family=s.get("family", ""),
                    description=s.get("description", ""),
                    priority=s.get("priority", 1),
                    parameter_ranges=s.get("parameter_ranges", {}),
                )
            )

        gates = []
        for g in data.get("gates", []):
            gates.append(GateSpec(trigger=g.get("trigger", ""), action=g.get("action", "pause_and_alert")))

        return cls(
            name=data["name"],
            objective=data["objective"],
            constraints=data.get("constraints", {}),
            strategy_space=strategies,
            stopping_criteria=sc,
            session_limits=sl,
            wandb_tags=data.get("wandb_tags", []),
            gates=gates,
            exclusions=data.get("exclusions", []),
        )


# ── Session Record ────────────────────────────────────────────────────────


@dataclass
class SessionRecord:
    """Record of a single orchestrator session."""

    session_id: str
    session_number: int
    start_ts: str
    end_ts: str = ""
    duration_minutes: float = 0.0
    exit_code: int = 0
    best_sharpe: Optional[float] = None
    best_dsr: Optional[float] = None
    wandb_run_ids: list[str] = field(default_factory=list)
    wandb_run_configs: list[dict] = field(default_factory=list)
    estimated_cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "session_number": self.session_number,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "duration_minutes": self.duration_minutes,
            "exit_code": self.exit_code,
            "best_sharpe": self.best_sharpe,
            "best_dsr": self.best_dsr,
            "wandb_run_ids": self.wandb_run_ids,
            "wandb_run_configs": self.wandb_run_configs,
            "estimated_cost_usd": self.estimated_cost_usd,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Orchestrator State ────────────────────────────────────────────────────


@dataclass
class OrchestratorState:
    """Persistent state for the research orchestrator."""

    name: str
    status: str = "running"  # running | paused | done | gate_triggered
    session_count: int = 0
    best_result: dict = field(default_factory=dict)  # {sharpe, dsr, run_id, session_number}
    stall_counter: int = 0
    crash_counter: int = 0
    crash_backoff_seconds: int = 120
    total_cost_usd: float = 0.0
    total_hours: float = 0.0
    sessions: list[SessionRecord] = field(default_factory=list)
    gate_message: Optional[str] = None
    lockfile_pid: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "session_count": self.session_count,
            "best_result": self.best_result,
            "stall_counter": self.stall_counter,
            "crash_counter": self.crash_counter,
            "crash_backoff_seconds": self.crash_backoff_seconds,
            "total_cost_usd": self.total_cost_usd,
            "total_hours": self.total_hours,
            "sessions": [s.to_dict() for s in self.sessions],
            "gate_message": self.gate_message,
            "lockfile_pid": self.lockfile_pid,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OrchestratorState":
        sessions = [SessionRecord.from_dict(s) for s in d.get("sessions", [])]
        return cls(
            name=d["name"],
            status=d.get("status", "running"),
            session_count=d.get("session_count", 0),
            best_result=d.get("best_result", {}),
            stall_counter=d.get("stall_counter", 0),
            crash_counter=d.get("crash_counter", 0),
            crash_backoff_seconds=d.get("crash_backoff_seconds", 120),
            total_cost_usd=d.get("total_cost_usd", 0.0),
            total_hours=d.get("total_hours", 0.0),
            sessions=sessions,
            gate_message=d.get("gate_message"),
            lockfile_pid=d.get("lockfile_pid", 0),
        )

    def save(self, state_dir: Path = STATE_DIR) -> None:
        """Atomically persist state to disk."""
        state_dir.mkdir(parents=True, exist_ok=True)
        filepath = state_dir / f"orchestrator_{self.name}.json"

        fd, tmp_path = tempfile.mkstemp(dir=str(state_dir), suffix=".tmp", prefix=".orch_")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            os.replace(tmp_path, str(filepath))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, name: str, state_dir: Path = STATE_DIR) -> Optional["OrchestratorState"]:
        """Load state from disk, or return None if not found."""
        filepath = state_dir / f"orchestrator_{name}.json"
        if not filepath.exists():
            return None
        with open(filepath) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def reconstruct_from_wandb(cls, name: str, directive_tags: list[str]) -> Optional["OrchestratorState"]:
        """Reconstruct state from wandb runs when the state file is missing.

        Queries wandb for all runs matching the directive's tags, groups them
        by session_NNN tag, and rebuilds session records with best metrics.

        Returns None if no matching runs are found or wandb is unreachable.
        """
        try:
            from sparky.tracking.experiment import ExperimentTracker

            tracker = ExperimentTracker(experiment_name=name)
            runs = tracker._fetch_runs(filters={"tags": {"$all": directive_tags}} if directive_tags else {})
            if not runs:
                return None

            # Group runs by session tag
            sessions_map: dict[int, dict] = {}
            for run in runs:
                session_num = None
                for tag in run.tags:
                    if tag.startswith("session_") and tag[8:].isdigit():
                        session_num = int(tag[8:])
                        break
                if session_num is None:
                    continue

                if session_num not in sessions_map:
                    sessions_map[session_num] = {
                        "run_ids": [],
                        "configs": [],
                        "best_sharpe": None,
                        "best_dsr": None,
                        "created_at": None,
                    }

                entry = sessions_map[session_num]
                entry["run_ids"].append(run.id)
                entry["configs"].append(dict(run.config))

                s = run.summary.get("sharpe") or run.summary.get("best_sharpe")
                d = run.summary.get("dsr") or run.summary.get("best_dsr")
                if s is not None and (entry["best_sharpe"] is None or s > entry["best_sharpe"]):
                    entry["best_sharpe"] = s
                if d is not None and (entry["best_dsr"] is None or d > entry["best_dsr"]):
                    entry["best_dsr"] = d

                # Track earliest run creation time for session ordering
                created = getattr(run, "created_at", None)
                if created and (entry["created_at"] is None or created < entry["created_at"]):
                    entry["created_at"] = created

            if not sessions_map:
                return None

            # Build session records in order
            session_records = []
            for num in sorted(sessions_map):
                entry = sessions_map[num]
                record = SessionRecord(
                    session_id=f"reconstructed_{num:03d}",
                    session_number=num,
                    start_ts=entry["created_at"] or "",
                    end_ts="",
                    best_sharpe=entry["best_sharpe"],
                    best_dsr=entry["best_dsr"],
                    wandb_run_ids=entry["run_ids"],
                    wandb_run_configs=entry["configs"],
                )
                session_records.append(record)

            # Find overall best
            best_result: dict = {}
            for rec in session_records:
                s = rec.best_sharpe or 0
                curr = best_result.get("sharpe", 0) or 0
                if s > curr:
                    best_result = {
                        "sharpe": rec.best_sharpe,
                        "dsr": rec.best_dsr,
                        "session_number": rec.session_number,
                    }

            state = cls(
                name=name,
                status="done",
                session_count=max(sessions_map),
                best_result=best_result,
                sessions=session_records,
            )
            logger.info(
                f"Reconstructed state from wandb: {len(session_records)} sessions, "
                f"best Sharpe={best_result.get('sharpe', 'N/A')}"
            )
            return state
        except Exception as e:
            logger.warning(f"Failed to reconstruct state from wandb: {e}")
            return None


# ── Jaccard Diversity ─────────────────────────────────────────────────────


def jaccard_similarity(cfg_a: dict, cfg_b: dict) -> float:
    """Compute Jaccard similarity between two config dicts."""
    set_a = frozenset((k, str(v)) for k, v in cfg_a.items())
    set_b = frozenset((k, str(v)) for k, v in cfg_b.items())
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def mean_pairwise_jaccard(configs: list[dict]) -> float:
    """Mean pairwise Jaccard similarity across a list of configs."""
    if len(configs) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(configs)):
        for j in range(i + 1, len(configs)):
            total += jaccard_similarity(configs[i], configs[j])
            count += 1
    return total / count if count > 0 else 0.0


# ── Context Builder ───────────────────────────────────────────────────────


class ContextBuilder:
    """Builds inter-session context from wandb results.

    Pure Python — queries wandb via ExperimentTracker. No LLM calls.
    """

    def __init__(self, directive: ResearchDirective, state: OrchestratorState):
        self.directive = directive
        self.state = state

    def build(self) -> str:
        """Build context summary string (~500 tokens) for the next session prompt."""
        parts = []

        # Top results table
        top_results = self._get_top_results()
        has_low_cost = any(r.get("low_cost_warning") for r in top_results[:10]) if top_results else False
        if top_results:
            parts.append("## Previous Results (top by Sharpe)")
            if has_low_cost:
                parts.append(
                    "**WARNING**: Results marked with * used costs below 50 bps and are "
                    "NOT comparable to 50 bps results. Re-run these configs at 50 bps."
                )
            parts.append("| # | Family | Sharpe | DSR | Key Params |")
            parts.append("|---|--------|--------|-----|------------|")
            for i, r in enumerate(top_results[:10], 1):
                family = r.get("family", "?")
                sharpe = r.get("sharpe", "?")
                dsr = r.get("dsr", "?")
                params = r.get("params_summary", "")
                if isinstance(sharpe, float):
                    sharpe = f"{sharpe:.3f}"
                if isinstance(dsr, float):
                    dsr = f"{dsr:.3f}"
                warn = "*" if r.get("low_cost_warning") else ""
                parts.append(f"| {i} | {family} | {sharpe}{warn} | {dsr} | {params} |")
        else:
            parts.append("## Previous Results\nNo results yet — this is the first session.")

        # Stall / diversity status
        if self.state.stall_counter > 0:
            parts.append(
                f"\n**Stall warning**: {self.state.stall_counter} sessions without improvement. "
                f"Try substantially different configs."
            )

        # Parameter coverage hint (heuristic v1 — best-effort only)
        coverage = self._parameter_coverage()
        if coverage:
            parts.append(f"\n## Parameter Coverage\n{coverage}")

        # Last session summary
        if self.state.sessions:
            last = self.state.sessions[-1]
            sharpe_str = f"{last.best_sharpe:.3f}" if last.best_sharpe is not None else "N/A"
            parts.append(
                f"\n**Last session** (#{last.session_number}): "
                f"best Sharpe={sharpe_str}, duration={last.duration_minutes:.0f}m"
            )

        # Overall best
        if self.state.best_result:
            bs = self.state.best_result.get("sharpe", "?")
            bd = self.state.best_result.get("dsr", "?")
            if isinstance(bs, float):
                bs = f"{bs:.3f}"
            if isinstance(bd, float):
                bd = f"{bd:.3f}"
            parts.append(f"\n**Overall best**: Sharpe={bs}, DSR={bd}")

        # Cost audit note
        if has_low_cost:
            parts.append(
                "\n**COST AUDIT**: Some prior results used <50 bps costs (marked with *). "
                "These are NOT comparable to the standard 50 bps. Re-run promising configs "
                "at 50 bps before drawing conclusions."
            )

        return "\n".join(parts)

    def _get_top_results(self) -> list[dict]:
        """Fetch top results from wandb, filtered by directive tags."""
        try:
            from sparky.tracking.experiment import ExperimentTracker

            tracker = ExperimentTracker(experiment_name=self.directive.name)
            runs = tracker._fetch_runs(
                filters={"tags": {"$all": self.directive.wandb_tags}} if self.directive.wandb_tags else {}
            )
            results = []
            for run in runs:
                sharpe = run.summary.get("sharpe") or run.summary.get("best_sharpe")
                if sharpe is None:
                    continue
                cost_bps = run.config.get("transaction_costs_bps", run.config.get("costs_bps"))
                low_cost = cost_bps is None or float(cost_bps) < 30
                results.append(
                    {
                        "family": run.config.get("strategy_family", run.config.get("model_type", "?")),
                        "sharpe": sharpe,
                        "dsr": run.summary.get("dsr") or run.summary.get("best_dsr"),
                        "params_summary": self._summarize_params(dict(run.config)),
                        "low_cost_warning": low_cost,
                    }
                )
            results.sort(key=lambda x: x.get("sharpe", 0) or 0, reverse=True)
            return results
        except Exception as e:
            logger.warning(f"ContextBuilder: failed to fetch wandb results: {e}")
            return []

    def _summarize_params(self, config: dict) -> str:
        """One-line summary of key params."""
        skip = {"config_hash", "git_hash", "data_manifest_hash", "session_id", "strategy_family", "model_type"}
        items = [(k, v) for k, v in config.items() if k not in skip and not k.startswith("_")]
        if not items:
            return ""
        parts = []
        for k, v in items[:5]:
            if isinstance(v, float):
                parts.append(f"{k}={v:g}")
            else:
                parts.append(f"{k}={v}")
        return ", ".join(parts)

    def _parameter_coverage(self) -> str:
        """Heuristic coverage of parameter_ranges vs. configs seen in wandb."""
        # v1 limitation: best-effort hint only, not authoritative
        all_ranges = {}
        for strategy in self.directive.strategy_space:
            all_ranges.update(strategy.parameter_ranges)

        if not all_ranges:
            return ""

        # Collect all param values seen in session records
        seen: dict[str, set] = {k: set() for k in all_ranges}
        for session in self.state.sessions:
            for cfg in session.wandb_run_configs:
                for param, _values in all_ranges.items():
                    if param in cfg:
                        seen[param].add(str(cfg[param]))

        lines = []
        for param, expected_vals in all_ranges.items():
            expected_set = {str(v) for v in expected_vals}
            covered = seen.get(param, set()) & expected_set
            pct = len(covered) / len(expected_set) * 100 if expected_set else 0
            if pct < 100:
                missing = expected_set - covered
                lines.append(f"- {param}: {pct:.0f}% covered (missing: {', '.join(sorted(missing)[:5])})")
        return "\n".join(lines) if lines else "All parameter ranges covered."


# ── Research Orchestrator ─────────────────────────────────────────────────


class ResearchOrchestrator:
    """Autonomous research loop: directive → sessions → results → repeat."""

    def __init__(
        self,
        directive: ResearchDirective,
        state_dir: Path = STATE_DIR,
        log_dir: Path = LOG_DIR,
    ):
        self.directive = directive
        self.state_dir = state_dir
        self.log_dir = log_dir
        self._lockfile = state_dir / f"orchestrator_{directive.name}.lock"

    def _acquire_lock(self) -> None:
        """Acquire lockfile with stale PID detection."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if self._lockfile.exists():
            try:
                existing_pid = int(self._lockfile.read_text().strip())
                os.kill(existing_pid, 0)  # raises OSError if process gone
                raise RuntimeError(f"Orchestrator already running (PID {existing_pid})")
            except OSError:
                pass  # stale lock — safe to overwrite
            except ValueError:
                pass  # corrupt lock file
        self._lockfile.write_text(str(os.getpid()))

    def _release_lock(self) -> None:
        """Release lockfile."""
        self._lockfile.unlink(missing_ok=True)

    def _load_or_create_state(self) -> OrchestratorState:
        """Load existing state, reconstruct from wandb, or create fresh."""
        state = OrchestratorState.load(self.directive.name, self.state_dir)
        if state is not None:
            return state

        # State file missing — try to reconstruct from wandb
        state = OrchestratorState.reconstruct_from_wandb(self.directive.name, self.directive.wandb_tags)
        if state is not None:
            logger.info("State file missing — reconstructed from wandb runs")
            state.save(self.state_dir)
            return state

        return OrchestratorState(name=self.directive.name)

    def _check_gate_request(self, state: OrchestratorState) -> bool:
        """Check if the agent wrote GATE_REQUEST.md. Returns True if gate triggered."""
        if GATE_REQUEST_PATH.exists():
            try:
                msg = GATE_REQUEST_PATH.read_text().strip()
            except Exception:
                msg = "(unreadable)"
            state.status = "gate_triggered"
            state.gate_message = msg
            state.save(self.state_dir)
            send_alert("WARN", f"Gate request from agent: {msg[:200]}")
            logger.info(f"Gate triggered by agent: {msg[:200]}")
            return True
        return False

    def _evaluate_stopping(self, state: OrchestratorState) -> Optional[str]:
        """Check stopping criteria. Returns reason string or None to continue."""
        sc = self.directive.stopping_criteria

        # Success
        best_sharpe = state.best_result.get("sharpe", 0) or 0
        best_dsr = state.best_result.get("dsr", 0) or 0
        if best_sharpe >= sc.success_min_sharpe and best_dsr >= sc.success_min_dsr:
            return f"SUCCESS: Sharpe={best_sharpe:.3f} DSR={best_dsr:.3f}"

        # Session limit
        if state.session_count >= sc.max_sessions:
            return f"BUDGET: max sessions reached ({state.session_count}/{sc.max_sessions})"

        # Time budget
        if state.total_hours >= sc.max_hours:
            return f"BUDGET: max hours reached ({state.total_hours:.1f}/{sc.max_hours:.1f}h)"

        # Cost budget
        if state.total_cost_usd >= sc.max_cost_usd:
            return f"BUDGET: max cost reached (${state.total_cost_usd:.2f}/${sc.max_cost_usd:.2f})"

        # Stall
        if state.stall_counter >= sc.sessions_without_improvement:
            return f"STALL: {state.stall_counter} sessions without improvement"

        return None

    def _update_stall(self, state: OrchestratorState, session_record: SessionRecord) -> None:
        """Update stall counter based on Sharpe improvement and config diversity."""
        sc = self.directive.stopping_criteria
        sharpe_stall = False
        diversity_stall = False

        # Sharpe stall: did this session improve the best?
        prev_best = state.best_result.get("sharpe", 0) or 0
        curr_best = session_record.best_sharpe or 0
        if curr_best <= prev_best + sc.improvement_threshold:
            sharpe_stall = True

        # Diversity stall: Jaccard similarity of recent sessions
        if len(state.sessions) >= 3:
            recent_configs = []
            for s in state.sessions[-3:]:
                recent_configs.extend(s.wandb_run_configs)
            if len(recent_configs) >= 2:
                similarity = mean_pairwise_jaccard(recent_configs)
                if similarity > sc.diversity_threshold:
                    diversity_stall = True
                    logger.info(f"Diversity stall: mean Jaccard={similarity:.3f} > {sc.diversity_threshold}")

        if sharpe_stall or diversity_stall:
            state.stall_counter += 1
        else:
            state.stall_counter = 0

    def _update_best(self, state: OrchestratorState, session_record: SessionRecord) -> None:
        """Update best_result if this session beat the previous best."""
        curr_sharpe = session_record.best_sharpe or 0
        prev_sharpe = state.best_result.get("sharpe", 0) or 0
        if curr_sharpe > prev_sharpe:
            state.best_result = {
                "sharpe": session_record.best_sharpe,
                "dsr": session_record.best_dsr,
                "run_id": session_record.wandb_run_ids[0] if session_record.wandb_run_ids else None,
                "session_number": session_record.session_number,
            }

    def _query_session_results(self, session_tag: str) -> dict:
        """Query wandb for results from this session. Returns {best_sharpe, best_dsr, run_ids, configs}."""
        try:
            from sparky.tracking.experiment import ExperimentTracker

            tracker = ExperimentTracker(experiment_name=self.directive.name)
            runs = tracker._fetch_runs(filters={"tags": {"$all": [session_tag]}})

            best_sharpe = None
            best_dsr = None
            run_ids = []
            configs = []

            for run in runs:
                run_ids.append(run.id)
                configs.append(dict(run.config))
                s = run.summary.get("sharpe") or run.summary.get("best_sharpe")
                d = run.summary.get("dsr") or run.summary.get("best_dsr")
                if s is not None and (best_sharpe is None or s > best_sharpe):
                    best_sharpe = s
                if d is not None and (best_dsr is None or d > best_dsr):
                    best_dsr = d

            return {
                "best_sharpe": best_sharpe,
                "best_dsr": best_dsr,
                "run_ids": run_ids,
                "configs": configs,
            }
        except Exception as e:
            logger.warning(f"Failed to query session results: {e}")
            return {"best_sharpe": None, "best_dsr": None, "run_ids": [], "configs": []}

    def _build_session_prompt(self, session_number: int, context: str) -> str:
        """Build the prompt for a Claude session."""
        d = self.directive
        session_tag = f"session_{session_number:03d}"

        sl = self.directive.session_limits
        parts = [
            f"You are session {session_number} of research directive '{d.name}'.",
            "",
            "Read RESEARCH_AGENT.md for all API usage, rules, and examples. "
            "Do NOT read CLAUDE.md or explore source files — everything you need "
            "is in RESEARCH_AGENT.md and this prompt.",
            "",
            f"## Session Duration — you have up to {sl.max_session_minutes} minutes",
            "Do NOT exit after running one sweep. Treat this session like a workday:",
            "1. Design your first experiment batch, run it, analyze results.",
            "2. Based on what you learned, design the NEXT batch. Run that too.",
            "3. Keep iterating: run → analyze → design next → run → analyze → ...",
            "4. Only stop when you have genuinely exhausted productive ideas OR you've "
            "found a result that meets the success criteria.",
            "",
            "A good session runs 3-5 experiment rounds, each informed by the last. "
            "A bad session runs 1 sweep and exits. Log results to wandb after EACH round "
            "so progress is captured even if the session is interrupted.",
            "",
            f"## Objective\n{d.objective}",
            "",
        ]

        # Constraints
        if d.constraints:
            parts.append("## Constraints")
            for k, v in d.constraints.items():
                parts.append(f"- {k}: {v}")
            parts.append("")

        # Strategy space
        if d.strategy_space:
            parts.append("## Strategy Space")
            for s in d.strategy_space:
                parts.append(f"- **{s.family}** (priority {s.priority}): {s.description}")
                if s.parameter_ranges:
                    for param, vals in s.parameter_ranges.items():
                        parts.append(f"  - {param}: {vals}")
            parts.append("")

        # Exclusions
        if d.exclusions:
            parts.append("## Exclusions (DO NOT do these)")
            for ex in d.exclusions:
                parts.append(f"- {ex}")
            parts.append("")

        # Context from previous sessions
        if context:
            parts.append(context)
            parts.append("")

        # Wandb tag instructions
        all_tags = d.wandb_tags + [session_tag]
        parts.append(
            f"## Tagging\nTag all wandb runs in this session with: {all_tags}\n\n"
            "**IMPORTANT:** When logging to wandb, include `sharpe` and `dsr` as top-level "
            "summary keys (not `best_sharpe`/`best_dsr`). Example:\n"
            "```python\nwandb.log({'sharpe': best_sharpe, 'dsr': best_dsr, ...})\n```"
        )
        parts.append("")

        # Reminder: costs and git rules are in RESEARCH_AGENT.md

        # Stuck protocol
        parts.append(
            "## When to Exit\n"
            "Do NOT exit just because one sweep finished. Exit ONLY when:\n"
            "- You have genuinely exhausted all productive ideas (tried 3+ distinct approaches)\n"
            "- A result meets the success criteria and you've validated it\n"
            "- You hit a platform blocker (write `GATE_REQUEST.md` and exit)\n\n"
            "If your first sweep fails, that's INFORMATION — use it to design a better "
            "second sweep. Try different parameter ranges, different strategy families, "
            "or different feature combinations. Negative results narrow the search space."
        )

        # Hard exit protocol
        parts.append(
            "\n## CRITICAL: Clean Exit Protocol\n"
            "When you have finished ALL experiments and logged to wandb, EXIT IMMEDIATELY.\n"
            "Do NOT:\n"
            "- Look for additional work or tidy up code\n"
            "- Run git commands, lint, or CI\n"
            "- Edit CLAUDE.md, RESEARCH_AGENT.md, configs, src/, tests/, or docs/\n"
            "- Modify your own memory files or project instructions\n"
            "- Interact with GitHub PRs or issues\n\n"
            "Your session is sandboxed. Writes outside results/, scratch/, and "
            "scripts/*.py are BLOCKED. If you are done, just stop."
        )

        return "\n".join(parts)

    def run(self) -> int:
        """Execute the orchestrator loop. Returns 0 on clean exit, 1 on error."""
        try:
            self._acquire_lock()
        except RuntimeError as e:
            logger.error(str(e))
            return 1

        try:
            state = self._load_or_create_state()

            # Check existing gate request
            if self._check_gate_request(state):
                return 0

            # Pre-flight validation
            logger.info(
                f"Starting orchestrator '{self.directive.name}' "
                f"(session_count={state.session_count}, status={state.status})"
            )

            # Reset status if resuming from paused/gate_triggered (via respond command)
            if state.status in ("paused", "gate_triggered"):
                if GATE_RESPONSE_PATH.exists():
                    state.status = "running"
                    state.gate_message = None
                    GATE_RESPONSE_PATH.unlink(missing_ok=True)
                    # Also clean up gate request
                    GATE_REQUEST_PATH.unlink(missing_ok=True)
                    state.save(self.state_dir)
                else:
                    logger.info(f"Orchestrator is {state.status}. Use 'sparky orch respond' to resume.")
                    return 0

            state.status = "running"
            state.save(self.state_dir)

            # Record policy hash at startup for integrity checking
            initial_policy_hash = get_policy_hash()

            # Main loop
            while True:
                # Evaluate stopping criteria
                stop_reason = self._evaluate_stopping(state)
                if stop_reason:
                    is_success = stop_reason.startswith("SUCCESS")
                    state.status = "done"
                    state.save(self.state_dir)
                    severity = "INFO" if is_success else "WARN"
                    send_alert(severity, f"Orchestrator '{self.directive.name}' stopped: {stop_reason}")
                    logger.info(f"Stopping: {stop_reason}")
                    return 0

                # Check crash loop
                sl = self.directive.session_limits
                if state.crash_counter >= sl.max_consecutive_crashes:
                    state.status = "paused"
                    state.save(self.state_dir)
                    send_alert(
                        "CRITICAL",
                        f"{state.crash_counter} consecutive fast exits. Pausing orchestrator.",
                    )
                    return 0

                # Crash backoff
                if state.crash_counter > 0:
                    logger.info(f"Crash backoff: sleeping {state.crash_backoff_seconds}s")
                    time.sleep(state.crash_backoff_seconds)

                # Build context
                context_builder = ContextBuilder(self.directive, state)
                context = context_builder.build()

                # Build prompt
                session_number = state.session_count + 1
                prompt = self._build_session_prompt(session_number, context)

                # Launch session
                session_tag = f"session_{session_number:03d}"
                session_name = f"orch_{self.directive.name}"
                start_ts = datetime.now(timezone.utc).isoformat()

                logger.info(f"Launching session {session_number}")
                telemetry = launch_claude_session(
                    prompt=prompt,
                    max_duration_minutes=sl.max_session_minutes,
                    session_name=session_name,
                    step_name=session_tag,
                    attempt=session_number,
                    log_dir=self.log_dir,
                    disallowed_tools=RESEARCH_DISALLOWED_TOOLS,
                    extra_env={"SPARKY_RESEARCH_SANDBOX": "1"},
                )

                end_ts = datetime.now(timezone.utc).isoformat()

                # Check for crash (fast exit)
                if telemetry.duration_minutes < sl.min_session_minutes:
                    state.crash_counter += 1
                    state.crash_backoff_seconds = min(state.crash_backoff_seconds * 2, 1920)
                    logger.warning(
                        f"Fast exit ({telemetry.duration_minutes:.1f}m < {sl.min_session_minutes}m). "
                        f"Crash counter: {state.crash_counter}"
                    )
                else:
                    state.crash_counter = 0
                    state.crash_backoff_seconds = 120

                # Post-session integrity checks
                current_hash = get_policy_hash()
                if current_hash != initial_policy_hash:
                    state.status = "paused"
                    state.save(self.state_dir)
                    send_alert(
                        "CRITICAL",
                        "holdout_policy.yaml was modified during orchestrator run! "
                        "Pausing immediately. This requires human investigation.",
                    )
                    logger.error("Holdout policy tampered — halting orchestrator")
                    return 1

                # Query wandb for session results
                results = self._query_session_results(session_tag)

                # Build session record
                record = SessionRecord(
                    session_id=telemetry.session_id,
                    session_number=session_number,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    duration_minutes=telemetry.duration_minutes,
                    exit_code=0 if telemetry.exit_reason == "completed" else 1,
                    best_sharpe=results["best_sharpe"],
                    best_dsr=results["best_dsr"],
                    wandb_run_ids=results["run_ids"],
                    wandb_run_configs=results["configs"],
                    estimated_cost_usd=telemetry.estimated_cost_usd,
                )

                # Update state
                state.sessions.append(record)
                state.session_count = session_number
                state.total_cost_usd += telemetry.estimated_cost_usd
                state.total_hours += telemetry.duration_minutes / 60.0

                self._update_best(state, record)
                self._update_stall(state, record)

                # Periodic digest
                sc = self.directive.stopping_criteria
                if session_number % sc.digest_every == 0:
                    best_s = state.best_result.get("sharpe", "N/A")
                    if isinstance(best_s, float):
                        best_s = f"{best_s:.3f}"
                    send_alert(
                        "INFO",
                        f"Digest: {session_number} sessions, best Sharpe={best_s}, "
                        f"cost=${state.total_cost_usd:.2f}, stall={state.stall_counter}",
                    )

                # Check for agent gate request
                if self._check_gate_request(state):
                    return 0

                state.save(self.state_dir)

        except Exception as e:
            logger.error(f"Orchestrator error: {e}", exc_info=True)
            return 1
        finally:
            self._release_lock()
