#!/usr/bin/env python3
"""Sparky-specific pre-commit checks.

Catches issues unique to the trading research system that generic
linters can't detect:
- Holdout date references in production code (src/) that might leak future data
- Guardrail bypass patterns
- Missing cost specifications in backtest code
- Use of --no-verify to bypass hooks
"""

import subprocess
import sys
from pathlib import Path


# Holdout boundary patterns — nobody should hardcode these in src/
_HOLDOUT_PATTERNS = [
    "2024-07-01",
    "2024-07-02",
    "2024-08",
    "2024-09",
    "2024-10",
    "2024-11",
    "2024-12",
    "2025-",
]

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

# Cost-related terms expected in backtest/strategy files
_COST_TERMS = {"costs_bps", "transaction_costs", "cost", "fee", "commission", "slippage"}


def _is_exempt_holdout(filepath: str) -> bool:
    """Check if a file is exempt from holdout date checks."""
    return any(exempt in filepath for exempt in _HOLDOUT_EXEMPT)


def _is_self(filepath: str) -> bool:
    """Check if this is the checker script itself."""
    return "sparky_checks" in filepath


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


def main() -> int:
    """Check staged Python files (or all src/ files if not in git context)."""
    files = get_staged_files()
    if not files:
        print("Sparky pre-commit checks: no Python files to check")
        return 0

    all_errors = []
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
