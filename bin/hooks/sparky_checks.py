#!/usr/bin/env python3
"""Sparky-specific pre-commit checks.

Catches issues unique to the trading research system that generic
linters can't detect:
- Holdout date references in production code (src/) that might leak future data
- Guardrail bypass patterns
- Missing cost specifications in backtest code
- Use of --no-verify to bypass hooks
"""

import re
import subprocess
import sys
from pathlib import Path


def _load_holdout_patterns() -> list[str]:
    """Generate holdout date patterns dynamically from configs/holdout_policy.yaml.

    Uses regex parsing (no pyyaml dependency) to extract oos_start dates
    and generate patterns for the OOS year and beyond.
    """
    policy_path = Path(__file__).resolve().parent.parent.parent / "configs" / "holdout_policy.yaml"
    if not policy_path.exists():
        return ["2024-"]  # conservative fallback

    content = policy_path.read_text()
    dates = re.findall(r'oos_start:\s*"(\d{4})-(\d{2})-(\d{2})"', content)
    if not dates:
        return ["2024-"]

    earliest_year = min(int(d[0]) for d in dates)
    earliest_month = min(int(d[1]) for d in dates if int(d[0]) == earliest_year)

    patterns = set()
    # Exact start dates
    for year_str, month_str, day_str in dates:
        patterns.add(f"{year_str}-{month_str}-{day_str}")
    # All months from OOS start month onward in the start year
    for m in range(earliest_month, 13):
        patterns.add(f"{earliest_year}-{m:02d}")
    # Subsequent years
    for y in range(earliest_year + 1, earliest_year + 3):
        patterns.add(f"{y}-")

    return sorted(patterns)


# Holdout boundary patterns — dynamically generated from policy YAML
_HOLDOUT_PATTERNS = _load_holdout_patterns()

# Files/dirs exempt from holdout date checks
_HOLDOUT_EXEMPT = {
    "test",
    "guardrail",
    "holdout",
    "conftest",
    "configs/",
    "hooks/",
    ".md",
}

# Guardrail bypass patterns
_BYPASS_PATTERNS = [
    "guardrails = False",
    "guardrails=False",
    "skip_guardrails",
    "no_guardrails",
    "SKIP_CHECKS",
    "disable_guardrails",
]

# Vault access patterns — direct vault reads bypass holdout protection
_VAULT_PATTERNS = [
    ".oos_vault",
    "data/holdout",
    "split_holdout_data",
]

# Cost-related terms expected in backtest/strategy files
_COST_TERMS = {"costs_bps", "transaction_costs", "cost", "fee", "commission", "slippage"}


def _is_exempt_holdout(filepath: str) -> bool:
    """Check if a file is exempt from holdout date checks."""
    return any(exempt in filepath for exempt in _HOLDOUT_EXEMPT)


def _is_self(filepath: str) -> bool:
    """Check if this is the checker script itself."""
    return "sparky_checks" in filepath


def _is_exempt_vault(filepath: str) -> bool:
    """Check if a file is exempt from vault access checks.

    Only the loader and the split script are allowed to reference the vault.
    """
    exempt = {
        "loader.py",
        "split_holdout_data.py",
        "conftest.py",
        "orchestrator.py",
        "oos_evaluate.py",
        "build_holdout_resampled.py",
        "holdout_split.py",
        "scan_data_holdout.py",
        "split_p003_holdout.py",
    }
    return any(e in filepath for e in exempt) or _is_self(filepath) or any(e in filepath for e in _HOLDOUT_EXEMPT)


def check_file(filepath: str) -> list[str]:
    """Run Sparky-specific checks on a single Python file."""
    errors = []
    path = Path(filepath)

    if path.suffix != ".py":
        return errors

    # Don't check ourselves
    if _is_self(filepath):
        return errors

    try:
        content = path.read_text()
    except (OSError, UnicodeDecodeError):
        return errors

    lines = content.split("\n")

    # 1. Holdout date hardcoding — only check src/ production code
    if filepath.startswith("src/") and not _is_exempt_holdout(filepath):
        in_docstring = False
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Track triple-quote docstrings
            if '"""' in stripped or "'''" in stripped:
                quote = '"""' if '"""' in stripped else "'''"
                count = stripped.count(quote)
                if count == 1:
                    in_docstring = not in_docstring
            if in_docstring:
                continue
            if stripped.startswith("#"):
                continue
            if '= "' in line or "= '" in line or ">>>" in line:
                continue
            for pattern in _HOLDOUT_PATTERNS:
                if pattern in line and "config" not in line.lower():
                    errors.append(
                        f"{filepath}:{i}: Possible hardcoded holdout/future date '{pattern}'. Use data loader instead."
                    )

    # 2. --no-verify usage (all files)
    if "--no-verify" in content:
        for i, line in enumerate(lines, 1):
            if "--no-verify" in line and not line.strip().startswith("#"):
                errors.append(f"{filepath}:{i}: Contains '--no-verify' which bypasses pre-commit hooks.")

    # 3. Guardrail bypass patterns (all files)
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pattern in _BYPASS_PATTERNS:
            if pattern in line:
                errors.append(f"{filepath}:{i}: Possible guardrail bypass: '{pattern}'")

    # 4. Missing cost specs in backtest/strategy files
    fname_lower = filepath.lower()
    if ("backtest" in fname_lower or "strategy" in fname_lower) and "test" not in fname_lower:
        has_cost_ref = any(term in content.lower() for term in _COST_TERMS)
        if "def run" in content and not has_cost_ref:
            errors.append(
                f"{filepath}: Backtest/strategy file with 'def run' but no cost reference. "
                f"All backtests must account for transaction costs."
            )

    # 5. Direct vault access — only loader.py and split script may reference the vault
    if not _is_exempt_vault(filepath):
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pattern in _VAULT_PATTERNS:
                if pattern in line:
                    errors.append(
                        f"{filepath}:{i}: Direct OOS vault access ('{pattern}'). "
                        "Use sparky.data.loader.load(purpose='oos_evaluation') instead."
                    )

    return errors


def get_staged_files() -> list[str]:
    """Get list of staged Python files."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Not in a git repo or no staged files — scan src/ instead
        return [str(p) for p in Path("src").rglob("*.py") if p.exists()]

    files = [f for f in result.stdout.strip().split("\n") if f.endswith(".py") and f]
    return [f for f in files if Path(f).exists()]


def check_policy_immutability() -> list[str]:
    """BLOCK: Reject commits that modify holdout_policy.yaml."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    staged = result.stdout.strip().split("\n")
    errors = []
    for f in staged:
        if "holdout_policy.yaml" in f:
            errors.append(
                f"{f}: BLOCKED — holdout_policy.yaml is immutable. "
                "Changes require human approval (use --no-verify with AK sign-off)."
            )
    return errors


def main() -> int:
    """Check staged Python files (or all src/ files if not in git context)."""
    all_errors = []

    # Check policy immutability first
    all_errors.extend(check_policy_immutability())

    files = get_staged_files()
    if not files and not all_errors:
        print("Sparky pre-commit checks: no Python files to check")
        return 0

    for f in files:
        all_errors.extend(check_file(f))

    if all_errors:
        print("Sparky pre-commit checks FAILED:")
        for err in all_errors:
            print(f"  {err}")
        return 1

    print(f"Sparky pre-commit checks passed ({len(files)} files checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
