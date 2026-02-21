#!/usr/bin/env python3
"""Unified Code Review Agent for Sparky AI.

Runs as a pre-commit hook. Uses `claude` CLI to review staged files for both
quantitative research methodology correctness AND engineering soundness in a
single pass, replacing the separate research-validation and platform-validation hooks.

Exits 0 if no HIGH (Critical) severity issues found.
Exits 1 if HIGH severity issues found.
Gracefully skips if the `claude` CLI is not installed.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_MAX_CODE_CHARS = 40000
_MAX_CONTEXT_CHARS = 12000

_SYSTEM_PROMPT = """\
**Situation**
You are an expert code review agent operating within an autonomous agentic quantitative \
research system. The system processes market data, calculates trading signals, backtests \
portfolio strategies using historical data, and manages risk metrics and position sizing. \
Code quality, reliability, and correctness are critical as errors can lead to incorrect \
research conclusions, financial losses, or system failures.

**Task**
Conduct a comprehensive code review of the provided code with dual focus on quantitative \
research methodology and software engineering quality. The assistant should identify and \
document all issues across these critical categories:

**Quantitative Research & Statistical Methods:**
- Statistical methodology correctness and appropriateness
- Signal calculation accuracy and mathematical soundness
- Backtesting validity (look-ahead bias, survivorship bias, data snooping)
- Risk metric calculations and portfolio optimization logic
- Time-series analysis correctness (stationarity, autocorrelation handling)
- Position sizing logic and capital allocation algorithms
- Performance attribution and statistical significance testing
- Data integrity throughout the quantitative pipeline
- Calculation accuracy and numerical precision in financial computations

**Engineering & Security:**
- Logic errors and algorithmic correctness
- Data handling robustness and edge case management
- Error handling gaps and exception management
- Security vulnerabilities and data exposure risks
- Performance bottlenecks and computational efficiency
- Memory management and resource cleanup
- Race conditions and thread safety in concurrent operations
- Code maintainability, readability, and documentation quality
- Testing coverage and validation of edge cases
- Dependency management and version compatibility

**Objective**
Ensure the code meets production-grade standards for an autonomous quantitative research \
system where statistical rigor, data integrity, calculation accuracy, and system \
reliability are paramount. The review should prevent incorrect trading signals, flawed \
backtesting results, compromised risk calculations, system failures, and security \
vulnerabilities.

**Knowledge**
The assistant should evaluate code with particular attention to:

**Quantitative Research Context:**
- Market data processing workflows and data quality validation
- Trading signal generation methodologies and their statistical foundations
- Backtesting frameworks and common pitfalls (lookahead bias, overfitting, data leakage)
- Risk metrics calculation (VaR, Sharpe ratio, drawdown, beta, correlation matrices)
- Position sizing algorithms and portfolio rebalancing logic
- Financial data peculiarities (missing data, corporate actions, timezone handling, \
market hours, tick data)
- Statistical validity of research methods and hypothesis testing

**Engineering Standards:**
- Numerical computing best practices (floating-point precision, vectorization, \
numerical stability)
- Data integrity checks at each pipeline stage
- Comprehensive error handling for data anomalies and calculation failures
- Edge case handling (zero division, null values, empty datasets, extreme market conditions)
- Autonomous system requirements (error recovery, logging, monitoring, idempotency)
- Production system standards (scalability, reproducibility, auditability)
- Security best practices (input validation, credential management, data access controls)

When a code snippet is provided, analyze it in the context of quantitative research \
systems where both statistical rigor and computational accuracy are non-negotiable. \
Verify that calculations produce mathematically correct results and that the system \
handles real-world data anomalies gracefully.\
"""


def get_changed_files() -> list[dict]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
    )
    all_files = [f for f in result.stdout.strip().split("\n") if f.strip()]
    reviewable = [
        f for f in all_files if any(f.endswith(ext) for ext in (".py", ".yaml", ".yml", ".sh")) and Path(f).exists()
    ]

    changes = []
    for filepath in reviewable[:15]:
        diff = subprocess.run(
            ["git", "diff", "--cached", "--", filepath],
            capture_output=True,
            text=True,
        )
        content = Path(filepath).read_text(errors="replace")
        changes.append({"filepath": filepath, "diff": diff.stdout, "full_content": content})
    return changes


def get_codebase_context() -> dict[str, str]:
    context_files = [
        "src/sparky/backtest/costs.py",
        "src/sparky/tracking/guardrails.py",
        "src/sparky/data/loader.py",
        "configs/trading_rules.yaml",
        "scripts/infra/sweep_utils.py",
    ]
    context = {}
    for path in context_files:
        if Path(path).exists():
            context[path] = Path(path).read_text(errors="replace")[:3000]
    return context


def _budget_changes(changes: list[dict], max_chars: int) -> str:
    if not changes:
        return ""
    per_file = max(500, max_chars // len(changes))
    diff_budget = int(per_file * 0.7)
    content_budget = per_file - diff_budget

    text = ""
    for c in changes:
        text += f"\n### File: {c['filepath']}\n"
        text += f"#### Diff:\n```\n{c['diff'][:diff_budget]}\n```\n"
        if content_budget > 100:
            text += f"#### Full content (truncated):\n```\n{c['full_content'][:content_budget]}\n```\n"
    return text


def should_skip(changes: list[dict]) -> tuple[bool, str]:
    infra_paths = {"src/", "scripts/", "tests/", "configs/", ".github/", "bin/"}
    infra_extensions = {".py", ".yaml", ".yml", ".sh"}
    for c in changes:
        path = c["filepath"]
        if any(path.startswith(p) for p in infra_paths):
            if any(path.endswith(ext) for ext in infra_extensions):
                return False, ""
    return True, "No infrastructure files changed"


def run_validation(changes: list[dict], context: dict[str, str]) -> dict:
    rubric = (Path(__file__).parent / "rubric.md").read_text()
    changes_text = _budget_changes(changes, _MAX_CODE_CHARS)

    context_text = ""
    ctx_per_file = max(500, _MAX_CONTEXT_CHARS // max(len(context), 1))
    for path, content in context.items():
        context_text += f"\n### {path} (for reference):\n```python\n{content[:ctx_per_file]}\n```\n"

    user_message = (
        "Review the following staged code changes. Be specific and technical — "
        "reference exact rubric sections for every issue. Only flag concrete violations "
        "that are visible in the code. Verify function signatures before flagging "
        "keyword argument issues.\n\n"
        "CRITICAL: Read Section 0 (Shared Utility Auto-Pass Rule) FIRST. Scripts "
        "importing from `sweep_utils` or other shared utility modules do NOT need to "
        "implement signal shifting, cost deduction, guardrails, or n_trials at the "
        "call site — these are handled inside the utility functions. Do NOT flag them.\n\n"
        f"## REVIEW RUBRIC\n{rubric}\n\n"
        f"## CODEBASE REFERENCE (APIs and shared utilities — for context)\n{context_text}\n\n"
        f"## CODE CHANGES TO REVIEW\n{changes_text}\n\n"
        "## OUTPUT FORMAT\n"
        "Respond with ONLY a valid JSON object (no markdown fences, no preamble):\n"
        "{\n"
        '  "summary": "one-sentence overall verdict",\n'
        '  "issues": [\n'
        "    {\n"
        '      "severity": "HIGH|MEDIUM|LOW",\n'
        '      "file": "path/to/file.py",\n'
        '      "line": "approx line or function name",\n'
        '      "section": "rubric section number",\n'
        '      "description": "what is wrong and why it matters",\n'
        '      "fix": "concrete suggested fix"\n'
        "    }\n"
        "  ],\n"
        '  "positive_observations": ["well-implemented aspects worth noting"],\n'
        '  "recommendations": ["actionable improvements beyond the flagged issues"],\n'
        '  "passed": true\n'
        "}\n\n"
        "Severity mapping:\n"
        "  HIGH   = Critical Issues (incorrect trading signals, data leakage, system failure, security breach)\n"
        "  MEDIUM = Major Issues (statistical validity concerns, reliability risks, maintainability)\n"
        "  LOW    = Minor Issues (style, documentation, optimization opportunities)\n\n"
        "Set passed=false if ANY HIGH severity issue exists. "
        "Do not escalate MEDIUM to HIGH to be cautious — false positives block commits "
        "and erode trust in the review system."
    )

    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    proc = subprocess.run(
        ["claude", "-p", "--model", "sonnet"],
        input=user_message,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI failed (rc={proc.returncode}): {proc.stderr[:500]}")

    text = proc.stdout.strip()
    # Strip markdown fences if the model wraps the JSON anyway
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    decoder = json.JSONDecoder()
    result, _ = decoder.raw_decode(text)
    return result


def main() -> None:
    if not shutil.which("claude"):
        print("claude CLI not found — skipping code review")
        sys.exit(0)

    changes = get_changed_files()
    if not changes:
        print("No reviewable files staged — skipping code review")
        sys.exit(0)

    skip, reason = should_skip(changes)
    if skip:
        print(f"Skipping code review: {reason}")
        sys.exit(0)

    context = get_codebase_context()
    print(f"Code Review Agent reviewing {len(changes)} file(s)...", flush=True)

    try:
        result = run_validation(changes, context)
    except Exception as e:  # noqa: BLE001
        print(f"Code review failed with error: {e}")
        print("Skipping code review due to error (non-blocking)")
        sys.exit(0)

    print(f"\n{'=' * 60}")
    print(f"Summary: {result.get('summary', 'N/A')}")

    issues = result.get("issues", [])
    print(f"Issues: {len(issues)}")
    print(f"Passed: {result.get('passed', True)}")

    if issues:
        print()
        for issue in issues:
            icon = {"HIGH": "[CRITICAL]", "MEDIUM": "[MAJOR]", "LOW": "[MINOR]"}.get(issue.get("severity", "?"), "[?]")
            section = issue.get("section", "")
            section_tag = f" [{section}]" if section else ""
            print(f"  {icon}{section_tag} {issue.get('file', '?')}: {issue.get('description', '')}")
            if issue.get("fix"):
                print(f"    Fix: {issue['fix']}")

    observations = result.get("positive_observations", [])
    if observations:
        print("\nPositive observations:")
        for obs in observations:
            print(f"  [+] {obs}")

    recommendations = result.get("recommendations", [])
    if recommendations:
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  [>] {rec}")

    if not result.get("passed", True):
        critical_count = len([i for i in issues if i.get("severity") == "HIGH"])
        print(f"\nBLOCKED: {critical_count} critical issue(s) must be resolved before committing")
        sys.exit(1)

    print("\nCode review passed ✓")


if __name__ == "__main__":
    main()
