"""Strategy pre-registration for multiple testing control.

Records strategy families, parameter ranges, and trial budgets BEFORE
running experiments. Tamper-evident via SHA-256 hash.
"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

__all__ = [
    "StrategyFamily",
    "PreRegistration",
    "save_pre_registration",
    "load_pre_registration",
    "check_trial_budget",
]


class StrategyFamily(BaseModel):
    name: str
    parameter_ranges: dict
    max_configs: int


class PreRegistration(BaseModel):
    project_id: str
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    strategy_families: list[StrategyFamily]
    kill_criteria: dict = Field(default_factory=dict)
    advance_criteria: dict = Field(default_factory=dict)
    max_total_trials: int
    holdout_date: str
    notes: str = ""


def _compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def save_pre_registration(reg: PreRegistration, path: str | Path) -> None:
    """Save pre-registration to YAML with SHA-256 hash appended."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = reg.model_dump(mode="json")
    # Convert datetime to ISO string for YAML
    if isinstance(data.get("registered_at"), datetime):
        data["registered_at"] = data["registered_at"].isoformat()
    content = yaml.dump(data, default_flow_style=False, sort_keys=False)
    sha = _compute_hash(content)
    path.write_text(content + f"# sha256: {sha}\n")


def load_pre_registration(path: str | Path) -> PreRegistration:
    """Load pre-registration from YAML and validate hash."""
    path = Path(path)
    raw = path.read_text()

    # Split hash line from content
    lines = raw.rstrip("\n").split("\n")
    hash_line = lines[-1] if lines[-1].startswith("# sha256:") else None
    if hash_line is None:
        raise ValueError(f"No SHA-256 hash found in {path}")

    stored_hash = hash_line.split("# sha256:")[1].strip()
    content = "\n".join(lines[:-1]) + "\n"
    computed = _compute_hash(content)

    if stored_hash != computed:
        raise ValueError(
            f"Hash mismatch in {path}: stored={stored_hash[:16]}... "
            f"computed={computed[:16]}... — file may have been tampered with"
        )

    data = yaml.safe_load(content)
    return PreRegistration(**data)


def check_trial_budget(
    reg: PreRegistration,
    current_trial_count: int,
    family: Optional[str] = None,
) -> bool:
    """Check if trial budget allows more experiments.

    Returns True if budget is available, False if exhausted.
    """
    if current_trial_count >= reg.max_total_trials:
        return False
    if family:
        for sf in reg.strategy_families:
            if sf.name == family:
                return current_trial_count < sf.max_configs
    return True
