"""Research program: multi-phase state machine with depth tracking.

A ResearchProgram is a YAML spec defining ordered phases (explore, investigate,
validate, portfolio, report) with coverage requirements, exit criteria, and
stall policies. The orchestrator executes it as a state machine, transitioning
between phases when coverage + exit criteria are met.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ── Safe access helpers ──────────────────────────────────────────────────


def _safe_get(d: Any, *keys, default=None):
    """Nested dict get that never raises."""
    try:
        for k in keys:
            d = d[k]
        return d
    except (KeyError, TypeError, IndexError, AttributeError):
        return default


def _safe_int(v, default=0) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def _safe_float(v, default=0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _safe_bool(v, default=False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return default


# ── Program YAML Parser ─────────────────────────────────────────────────


@dataclass
class PhaseConfig:
    name: str
    objective: str
    coverage_requirements: dict = field(default_factory=dict)
    exit_criteria: dict = field(default_factory=dict)
    min_sessions: int = 1
    max_sessions: int = 20
    stall_policy: dict = field(default_factory=dict)
    session_minutes: int = 360
    next_phase: str | None = None
    human_review: str = "none"
    agents: int = 1
    strategy_space: dict | None = None
    depth_protocol: dict | None = None
    investigation_battery: dict | None = None
    validation_battery: dict | None = None
    construction_protocol: dict | None = None
    conditional_branches: list[dict] = field(default_factory=list)
    deliverables: list[str] = field(default_factory=list)
    elimination_rules: str | None = None


@dataclass
class ResearchProgram:
    name: str
    version: int
    budget_cap_usd: float
    max_calendar_days: int
    success_definition: str
    phases: dict[str, PhaseConfig]
    phase_order: list[str]
    memory: dict = field(default_factory=dict)
    guardrails: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ResearchProgram":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Program file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError("Program YAML must be a mapping")

        prog = raw.get("program")
        if not isinstance(prog, dict):
            raise ValueError("Program YAML must have a 'program' key with a mapping")

        if "name" not in prog:
            raise ValueError("Program missing required field: 'name'")
        if "phases" not in prog or not isinstance(prog["phases"], dict):
            raise ValueError("Program missing required field: 'phases' (must be a mapping)")

        phase_order = list(prog["phases"].keys())
        phases: dict[str, PhaseConfig] = {}

        for i, (phase_name, phase_data) in enumerate(prog["phases"].items()):
            if not isinstance(phase_data, dict):
                phase_data = {}

            # Determine next_phase from explicit field or key order
            explicit_next = phase_data.get("next_phase")
            if explicit_next is not None:
                next_phase = explicit_next if explicit_next else None
            elif i < len(phase_order) - 1:
                next_phase = phase_order[i + 1]
            else:
                next_phase = None

            phases[phase_name] = PhaseConfig(
                name=phase_name,
                objective=phase_data.get("objective", ""),
                coverage_requirements=phase_data.get("coverage_requirements", {}),
                exit_criteria=phase_data.get("exit_criteria", {}),
                min_sessions=phase_data.get("min_sessions", 1),
                max_sessions=phase_data.get("max_sessions", 20),
                stall_policy=phase_data.get("stall_policy", {}),
                session_minutes=phase_data.get("session_minutes", 360),
                next_phase=next_phase,
                human_review=phase_data.get("human_review", "none"),
                agents=phase_data.get("agents", 1),
                strategy_space=phase_data.get("strategy_space"),
                depth_protocol=phase_data.get("depth_protocol"),
                investigation_battery=phase_data.get("investigation_battery"),
                validation_battery=phase_data.get("validation_battery"),
                construction_protocol=phase_data.get("construction_protocol"),
                conditional_branches=phase_data.get("conditional_branches", []),
                deliverables=phase_data.get("deliverables", []),
                elimination_rules=phase_data.get("elimination_rules"),
            )

        # Validate next_phase references
        for pc in phases.values():
            if pc.next_phase is not None and pc.next_phase not in phases:
                raise ValueError(f"Phase '{pc.name}' references unknown next_phase '{pc.next_phase}'")

        return cls(
            name=prog["name"],
            version=prog.get("version", 1),
            budget_cap_usd=prog.get("budget_cap_usd", 100.0),
            max_calendar_days=prog.get("max_calendar_days", 14),
            success_definition=prog.get("success_definition", ""),
            phases=phases,
            phase_order=phase_order,
            memory=prog.get("memory", {}),
            guardrails=prog.get("guardrails", {}),
        )

    def current_phase_config(self, phase_name: str) -> PhaseConfig:
        return self.phases[phase_name]

    def to_directive(self, phase_name: str | None = None):
        """Create a synthetic ResearchDirective from the current phase config."""
        from sparky.workflow.orchestrator import (
            ResearchDirective,
            SessionLimits,
            StoppingCriteria,
        )

        if phase_name is None:
            phase_name = self.phase_order[0]
        pc = self.phases[phase_name]

        return ResearchDirective(
            name=self.name,
            objective=pc.objective,
            constraints={},
            strategy_space=[],
            stopping_criteria=StoppingCriteria(
                stop_on_success=False,
                max_sessions=pc.max_sessions * len(self.phases),
                max_cost_usd=self.budget_cap_usd,
                max_hours=self.max_calendar_days * 24.0,
            ),
            session_limits=SessionLimits(
                max_session_minutes=pc.session_minutes,
            ),
            wandb_tags=[f"program_{self.name}"],
        )


# ── Phase State Machine ─────────────────────────────────────────────────


@dataclass
class PhaseState:
    current_phase: str
    phase_session_count: int = 0
    coverage_status: dict = field(default_factory=dict)
    phase_history: list[dict] = field(default_factory=list)
    pending_human_review: bool = False

    def to_dict(self) -> dict:
        return {
            "current_phase": self.current_phase,
            "phase_session_count": self.phase_session_count,
            "coverage_status": self.coverage_status,
            "phase_history": list(self.phase_history),
            "pending_human_review": self.pending_human_review,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PhaseState":
        if not isinstance(d, dict):
            raise ValueError("PhaseState.from_dict requires a dict")
        return cls(
            current_phase=d.get("current_phase", ""),
            phase_session_count=d.get("phase_session_count", 0),
            coverage_status=d.get("coverage_status", {}),
            phase_history=d.get("phase_history", []),
            pending_human_review=d.get("pending_human_review", False),
        )


def evaluate_phase_transition(
    program: ResearchProgram,
    phase_state: PhaseState,
    core_memory: dict,
) -> str | None:
    """Check if current phase should transition. Returns next phase name or None."""
    phase_cfg = program.phases.get(phase_state.current_phase)
    if phase_cfg is None:
        return None

    # Too early
    if phase_state.phase_session_count < phase_cfg.min_sessions:
        return None

    # Force transition at max_sessions
    if phase_state.phase_session_count >= phase_cfg.max_sessions:
        logger.info(
            f"Phase '{phase_state.current_phase}' hit max_sessions ({phase_cfg.max_sessions}), forcing transition"
        )
        return phase_cfg.next_phase

    # Check coverage requirements
    coverage = extract_coverage(core_memory, phase_state.current_phase)
    if phase_cfg.coverage_requirements:
        all_met, _ = evaluate_coverage(coverage, phase_cfg.coverage_requirements)
        if not all_met:
            return None

    # Check exit criteria
    if phase_cfg.exit_criteria:
        for key, required in phase_cfg.exit_criteria.items():
            if key == "coverage_requirements_met":
                continue  # already checked above
            actual = _safe_get(core_memory, key) or _safe_get(coverage, key)
            if actual is None:
                actual = _safe_get(core_memory, "exit_criteria_status", key)
            if required is True and not _safe_bool(actual):
                return None
            if isinstance(required, (int, float)):
                if _safe_float(actual) < required:
                    return None

    # Human review gate
    if phase_cfg.human_review == "required":
        phase_state.pending_human_review = True
        return None

    return phase_cfg.next_phase


def check_stall(
    program: ResearchProgram,
    phase_state: PhaseState,
    sessions_without_new_result: int,
) -> str | None:
    """Check phase-specific stall policy. Returns action string or None."""
    phase_cfg = program.phases.get(phase_state.current_phase)
    if phase_cfg is None or not phase_cfg.stall_policy:
        return None

    threshold_key = None
    for key in phase_cfg.stall_policy:
        if key.startswith("sessions_without"):
            threshold_key = key
            break

    if threshold_key is None:
        return None

    threshold = _safe_int(phase_cfg.stall_policy[threshold_key], default=999)
    if sessions_without_new_result >= threshold:
        action = phase_cfg.stall_policy.get("on_stall", "proceed_with_available")
        logger.info(
            f"Phase '{phase_state.current_phase}' stall detected: "
            f"{sessions_without_new_result} sessions without new result "
            f"(threshold: {threshold}). Action: {action}"
        )
        return action

    return None


# ── Coverage Tracking ────────────────────────────────────────────────────


def read_core_memory(path: str | Path) -> dict:
    """Read core_memory.json. Returns empty dict on any error."""
    try:
        p = Path(path)
        if not p.exists():
            return {}
        with open(p) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning(f"Core memory at {path} is not a dict, ignoring")
            return {}
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read core memory at {path}: {e}")
        return {}


def extract_coverage(core_memory: dict, phase_name: str) -> dict:
    """Extract coverage metrics from core memory for the given phase."""
    coverage = {}

    if phase_name == "explore":
        fam = _safe_get(core_memory, "coverage", "families_screened") or {}
        if isinstance(fam, dict):
            coverage["min_families_screened"] = len(fam)
            null_count = sum(1 for v in fam.values() if isinstance(v, dict) and _safe_get(v, "status") == "null_result")
            coverage["min_families_with_null_result"] = null_count
            deep_count = sum(1 for v in fam.values() if isinstance(v, dict) and _safe_int(_safe_get(v, "round")) >= 2)
            coverage["min_families_deep_explored"] = deep_count
        else:
            coverage["min_families_screened"] = 0
            coverage["min_families_with_null_result"] = 0
            coverage["min_families_deep_explored"] = 0

        coverage["min_total_configs"] = _safe_int(_safe_get(core_memory, "coverage", "total_configs_tested"))
        coverage["min_inv_vol_tested"] = _safe_int(_safe_get(core_memory, "coverage", "inv_vol_tested"))
        coverage["min_robustness_checks"] = _safe_int(_safe_get(core_memory, "coverage", "robustness_checks"))

        # Exit criteria metrics
        candidates = _safe_get(core_memory, "top_candidates") or []
        if isinstance(candidates, list):
            coverage["min_tier1_candidates"] = len(candidates)
            assets = {c.get("id", "").split("_")[0] for c in candidates if isinstance(c, dict) and c.get("id")}
            coverage["min_assets_represented"] = len(assets)
        else:
            coverage["min_tier1_candidates"] = 0
            coverage["min_assets_represented"] = 0

    elif phase_name == "investigate":
        candidates = _safe_get(core_memory, "top_candidates") or []
        if isinstance(candidates, list):
            total = len(candidates)
            investigated = sum(
                1 for c in candidates if isinstance(c, dict) and _safe_get(c, "investigation_complete") is True
            )
            coverage["all_tier1_candidates_investigated"] = investigated >= total and total > 0
            coverage["investigated_count"] = investigated
            coverage["total_candidates"] = total
        else:
            coverage["all_tier1_candidates_investigated"] = False

        coverage["cross_candidate_correlation_analyzed"] = _safe_bool(
            _safe_get(core_memory, "cross_candidate_correlation_analyzed")
        )
        coverage["edge_attribution_complete_for_top5"] = _safe_bool(
            _safe_get(core_memory, "edge_attribution_complete_for_top5")
        )

    elif phase_name == "validate":
        candidates = _safe_get(core_memory, "candidates") or {}
        if isinstance(candidates, dict):
            total = len(candidates)
            tested = sum(
                1
                for v in candidates.values()
                if isinstance(v, dict) and _safe_get(v, "status", default="").startswith("validated")
            )
            passing = sum(
                1 for v in candidates.values() if isinstance(v, dict) and _safe_get(v, "status") == "validated_pass"
            )
            coverage["all_candidates_tested"] = tested >= total and total > 0
            coverage["min_passing_candidates"] = passing
        else:
            coverage["all_candidates_tested"] = False
            coverage["min_passing_candidates"] = 0

    elif phase_name == "portfolio":
        portfolios = _safe_get(core_memory, "portfolios") or {}
        coverage["portfolios_tested"] = len(portfolios) if isinstance(portfolios, dict) else 0
        coverage["best_portfolio_sharpe"] = _safe_float(_safe_get(core_memory, "best_portfolio", "sharpe"))

    elif phase_name == "report":
        deliverables = _safe_get(core_memory, "deliverables_status") or {}
        if isinstance(deliverables, dict):
            total = len(deliverables)
            done = sum(1 for v in deliverables.values() if v is True or v == "done")
            coverage["deliverables_complete"] = done >= total and total > 0
        else:
            coverage["deliverables_complete"] = False

    return coverage


def evaluate_coverage(
    coverage: dict,
    requirements: dict,
) -> tuple[bool, dict]:
    """Check if coverage meets requirements. Returns (all_met, status_dict)."""
    status = {}
    all_met = True

    for req_key, req_val in requirements.items():
        current = coverage.get(req_key)
        if current is None:
            current = 0

        if isinstance(req_val, bool):
            current_bool = _safe_bool(current)
            met = current_bool == req_val
            status[req_key] = {"required": req_val, "current": current_bool, "met": met}
        elif isinstance(req_val, (int, float)):
            current_num = _safe_float(current) if isinstance(req_val, float) else _safe_int(current)
            met = current_num >= req_val
            status[req_key] = {"required": req_val, "current": current_num, "met": met}
        else:
            met = True
            status[req_key] = {"required": req_val, "current": current, "met": met}

        if not met:
            all_met = False

    return all_met, status


def format_coverage_gaps(status: dict) -> str:
    """Format unmet coverage requirements as markdown."""
    lines = []
    for key, info in status.items():
        if not isinstance(info, dict):
            continue
        if not info.get("met", True):
            req = info.get("required", "?")
            cur = info.get("current", "?")
            label = key.replace("_", " ").replace("min ", "")
            lines.append(f"- **{label}**: {cur} / {req}")

    if not lines:
        return "All coverage requirements met."
    return "### Coverage Gaps\n" + "\n".join(lines)
