"""Manager (Opus) session tracker for audit trail.

Records infrastructure decisions, sub-agent spawns, research launches,
and contract design rationale. Provides an append-only JSONL log for
post-hoc review of manager-level decisions.

Usage:
    from sparky.tracking.manager_log import ManagerLog

    log = ManagerLog()
    session = log.start_session("Contract 005 infrastructure", "manager/contract-005")
    log.log_code_agent(session, CodeAgentRecord(
        task="Build guardrails module",
        model="sonnet",
        files_created=["src/sparky/tracking/guardrails.py"],
    ))
    log.log_decision(session,
        decision="Use Protocol for interfaces",
        alternatives=["ABC", "Protocol", "duck typing"],
        rationale="Matches existing backtest/engine.py pattern",
    )
    log.end_session(session, summary="Infrastructure complete, PR created")
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR = Path("logs/manager_sessions")
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "session_log.jsonl"


@dataclass
class CodeAgentRecord:
    """Record of a sub-agent (Task tool) invocation."""

    task: str
    model: str = "sonnet"
    files_created: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    tests_passed: Optional[bool] = None
    duration_seconds: Optional[float] = None
    notes: str = ""


@dataclass
class ResearchLaunchRecord:
    """Record of a research agent (systemd/workflow) launch."""

    workflow: str
    contract: str
    branch: str
    service_name: str = "sparky-research"
    notes: str = ""


@dataclass
class ContractDesignRecord:
    """Record of contract design decisions."""

    contract_name: str
    objective: str
    steps: list[str] = field(default_factory=list)
    budget_hours: float = 0.0
    success_criteria: str = ""
    rationale: str = ""


@dataclass
class ManagerSession:
    """A single manager session with events."""

    session_id: str
    objective: str
    branch: str
    started_at: str = ""
    ended_at: str = ""
    code_agents: list[CodeAgentRecord] = field(default_factory=list)
    research_launches: list[ResearchLaunchRecord] = field(default_factory=list)
    contract_designs: list[ContractDesignRecord] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    infrastructure: list[dict] = field(default_factory=list)
    summary: str = ""


class ManagerLog:
    """Append-only JSONL logger for manager sessions.

    Each session is accumulated in memory and written to the JSONL file
    when end_session() is called. get_history() reads back completed sessions.
    """

    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file or DEFAULT_LOG_FILE
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def start_session(self, objective: str, branch: str) -> ManagerSession:
        """Start a new manager session.

        Args:
            objective: What this session aims to accomplish.
            branch: Git branch for this work.

        Returns:
            ManagerSession object to pass to subsequent log calls.
        """
        now = datetime.now(timezone.utc)
        session_id = now.strftime("%Y%m%d_%H%M%S_%f")
        session = ManagerSession(
            session_id=session_id,
            objective=objective,
            branch=branch,
            started_at=now.isoformat(),
        )
        logger.info(f"[MANAGER] Session {session_id} started: {objective}")
        return session

    def log_code_agent(self, session: ManagerSession, record: CodeAgentRecord) -> None:
        """Record a sub-agent (Task tool) invocation.

        Args:
            session: Active manager session.
            record: CodeAgentRecord with task details.
        """
        session.code_agents.append(record)
        logger.info(f"[MANAGER] Code agent: {record.task} (model={record.model})")

    def log_research_launch(self, session: ManagerSession, record: ResearchLaunchRecord) -> None:
        """Record a research agent launch (systemd/workflow start).

        Args:
            session: Active manager session.
            record: ResearchLaunchRecord with launch details.
        """
        session.research_launches.append(record)
        logger.info(f"[MANAGER] Research launch: {record.workflow} ({record.contract})")

    def log_contract_design(self, session: ManagerSession, record: ContractDesignRecord) -> None:
        """Record a contract design decision.

        Args:
            session: Active manager session.
            record: ContractDesignRecord with contract details.
        """
        session.contract_designs.append(record)
        logger.info(f"[MANAGER] Contract design: {record.contract_name}")

    def log_decision(
        self,
        session: ManagerSession,
        decision: str,
        alternatives: list[str],
        rationale: str,
    ) -> None:
        """Record a manager-level decision.

        Args:
            session: Active manager session.
            decision: What was decided.
            alternatives: Other options considered.
            rationale: Why this option was chosen.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "alternatives": alternatives,
            "rationale": rationale,
        }
        session.decisions.append(entry)
        logger.info(f"[MANAGER] Decision: {decision}")

    def log_infrastructure(
        self,
        session: ManagerSession,
        module: str,
        purpose: str,
        files: list[str],
        rationale: str,
    ) -> None:
        """Record an infrastructure change.

        Args:
            session: Active manager session.
            module: Module name (e.g., "guardrails", "interfaces").
            purpose: What this infrastructure enables.
            files: Files created or modified.
            rationale: Why this was built.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "module": module,
            "purpose": purpose,
            "files": files,
            "rationale": rationale,
        }
        session.infrastructure.append(entry)
        logger.info(f"[MANAGER] Infrastructure: {module} — {purpose}")

    def end_session(self, session: ManagerSession, summary: str) -> None:
        """Finalize session and append to JSONL log.

        Args:
            session: Active manager session.
            summary: Brief summary of what was accomplished.
        """
        session.ended_at = datetime.now(timezone.utc).isoformat()
        session.summary = summary

        # Serialize to JSONL
        entry = asdict(session)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        logger.info(f"[MANAGER] Session {session.session_id} ended: {summary}")

    def get_history(self, n_sessions: int = 10) -> list[ManagerSession]:
        """Read back completed sessions from JSONL log.

        Uses a deque to only keep the last n_sessions raw lines in memory,
        then deserializes only those lines. Returns newest first.

        Args:
            n_sessions: Maximum number of sessions to return (most recent first).

        Returns:
            List of ManagerSession objects, newest first.
        """
        from collections import deque

        if not self.log_file.exists():
            return []

        # Only keep the last n_sessions lines (tail-based)
        tail = deque(maxlen=n_sessions)
        with open(self.log_file) as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if stripped:
                    tail.append(stripped)

        # Deserialize only the kept lines
        sessions = []
        for raw in tail:
            try:
                data = json.loads(raw)
                session = ManagerSession(
                    session_id=data.get("session_id", ""),
                    objective=data.get("objective", ""),
                    branch=data.get("branch", ""),
                    started_at=data.get("started_at", ""),
                    ended_at=data.get("ended_at", ""),
                    summary=data.get("summary", ""),
                )
                # Reconstruct nested records
                session.code_agents = [CodeAgentRecord(**r) for r in data.get("code_agents", [])]
                session.research_launches = [ResearchLaunchRecord(**r) for r in data.get("research_launches", [])]
                session.contract_designs = [ContractDesignRecord(**r) for r in data.get("contract_designs", [])]
                session.decisions = data.get("decisions", [])
                session.infrastructure = data.get("infrastructure", [])
                sessions.append(session)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"[MANAGER] Failed to parse session entry: {e}")
                continue

        # Return most recent first
        sessions.reverse()
        return sessions
