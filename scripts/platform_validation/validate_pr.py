#!/usr/bin/env python3
"""Platform Validation Agent for Sparky AI.

Runs as a GitHub Action on every PR. Uses Claude Sonnet to review
changed files against the engineering QA rubric.

Complements research-validation (which checks statistical methodology).
This agent focuses on infrastructure correctness: backtesting plumbing,
data access patterns, wandb parameter flow, cost model usage, guardrails,
and testing coverage.

Exits 0 if no HIGH severity issues found.
Exits 1 if HIGH severity issues found OR if validation was inconclusive
        (rate limit, API error, non-JSON response). A PR must not go
        green when validation was skipped due to API failures.
Posts findings as PR comment via GitHub API.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import anthropic

# Retry delays (seconds) on 429 rate-limit errors.
# Three total attempts: immediate, +30s, +60s.
_RATE_LIMIT_RETRY_DELAYS = [30, 60]


def get_changed_files():
    """Get diff of changed files in this PR.

    Includes Python files, YAML configs, and shell scripts — unlike
    research-validation which only reviews Python. Platform engineering
    issues often live in config files and CI workflows.
    """
    result = subprocess.run(
        ["git", "diff", "origin/main...HEAD", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
    )
    all_files = [f for f in result.stdout.strip().split("\n") if f.strip()]

    # Review Python, YAML, and shell files
    reviewable = [
        f for f in all_files if any(f.endswith(ext) for ext in (".py", ".yaml", ".yml", ".sh")) and Path(f).exists()
    ]

    changes = []
    for filepath in reviewable[:18]:  # limit to 18 files
        diff = subprocess.run(
            ["git", "diff", "origin/main...HEAD", "--", filepath],
            capture_output=True,
            text=True,
        )
        content = Path(filepath).read_text(errors="replace")
        changes.append(
            {
                "filepath": filepath,
                "diff": diff.stdout[:5000],
                "full_content": content[:8000],
            }
        )
    return changes


def get_codebase_context():
    """Gather lightweight codebase context to help the agent understand the system."""
    context_files = [
        "src/sparky/backtest/costs.py",
        "src/sparky/tracking/guardrails.py",
        "src/sparky/data/loader.py",
        "configs/trading_rules.yaml",
    ]
    context = {}
    for path in context_files:
        if Path(path).exists():
            context[path] = Path(path).read_text(errors="replace")[:3000]
    return context


def _create_message_with_retry(client, **kwargs):
    """Call client.messages.create with exponential backoff on rate limits.

    Raises anthropic.RateLimitError if all retries are exhausted.
    """
    for attempt, delay in enumerate([0] + _RATE_LIMIT_RETRY_DELAYS):
        if delay:
            print(
                f"Rate limited (429) — retrying in {delay}s "
                f"(attempt {attempt + 1}/{len(_RATE_LIMIT_RETRY_DELAYS) + 1})...",
                flush=True,
            )
            time.sleep(delay)
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError:
            if attempt == len(_RATE_LIMIT_RETRY_DELAYS):
                raise  # All retries exhausted — propagate to caller


def run_validation(changes, context):
    """Send changes to Claude Sonnet for platform engineering review."""
    client = anthropic.Anthropic()

    rubric_path = Path(__file__).parent / "rubric.md"
    rubric = rubric_path.read_text()

    # Format changed files
    changes_text = ""
    for c in changes:
        changes_text += f"\n### File: {c['filepath']}\n"
        changes_text += f"#### Diff:\n```\n{c['diff']}\n```\n"
        changes_text += f"#### Full content (truncated):\n```\n{c['full_content'][:3000]}\n```\n\n"

    # Format codebase context
    context_text = ""
    for path, content in context.items():
        context_text += f"\n### {path} (for reference):\n```python\n{content}\n```\n"

    prompt = (
        "You are an engineering QA reviewer for Sparky AI, an autonomous crypto "
        "trading research platform. Your job is to review code changes for "
        "infrastructure correctness and engineering soundness — NOT statistical "
        "methodology (a separate agent handles that).\n\n"
        "Review the following code changes against the engineering rubric below. "
        "For each issue found:\n"
        "1. State the severity: HIGH, MEDIUM, or LOW\n"
        "2. State the file and approximate line\n"
        "3. Explain the concrete engineering issue\n"
        "4. Suggest the fix\n\n"
        "Focus areas: backtesting plumbing (signal timing, cost application order), "
        "data access patterns (loader usage, holdout enforcement), wandb parameter "
        "flow (correct key names, n_trials), cost model API usage, guardrails "
        "invocation, and testing coverage for new infrastructure.\n\n"
        "Do NOT flag: code style, import ordering, statistical methodology, "
        "performance without correctness impact, or test helper simplifications.\n\n"
        "IMPORTANT: Be specific. Reference the exact rubric section. Only flag "
        "concrete violations you can see in the code — not hypothetical risks.\n\n"
        f"## ENGINEERING RUBRIC\n{rubric}\n\n"
        f"## CODEBASE REFERENCE (current state of key files)\n{context_text}\n\n"
        f"## CODE CHANGES TO REVIEW\n{changes_text}\n\n"
        "## OUTPUT FORMAT\n"
        "Respond with a JSON object:\n"
        "{\n"
        '  "summary": "One-line summary of findings",\n'
        '  "issues": [\n'
        "    {\n"
        '      "severity": "HIGH|MEDIUM|LOW",\n'
        '      "file": "path/to/file.py",\n'
        '      "line": "approximate line or range",\n'
        '      "rubric_section": "e.g., 3.1 Required Logging Keys",\n'
        '      "description": "What\'s wrong and why it matters",\n'
        '      "fix": "Concrete fix"\n'
        "    }\n"
        "  ],\n"
        '  "passed": true\n'
        "}\n"
        "passed must be false if ANY HIGH severity issues exist."
    )

    response = _create_message_with_retry(
        client,
        model="claude-sonnet-4-5-20250929",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        # Non-JSON response is inconclusive — do NOT treat as pass
        raise ValueError(f"Validation agent returned non-JSON response (len={len(text)}): {text[:300]}") from exc

    return result


def post_pr_comment(result):
    """Post validation results as a PR comment via gh CLI."""
    pr_number = os.environ.get("PR_NUMBER", "")
    if not pr_number:
        print("No PR_NUMBER env var — skipping comment")
        return

    # Build comment header
    if result.get("inconclusive"):
        status = "## :warning: Platform Validation: INCONCLUSIVE"
    elif result["passed"]:
        status = "## :white_check_mark: Platform Validation: PASSED"
    else:
        status = "## :x: Platform Validation: FAILED (HIGH severity engineering issues found)"

    comment = f"{status}\n\n{result['summary']}\n\n"

    if result.get("issues"):
        comment += "| Severity | File | Section | Issue |\n"
        comment += "|----------|------|---------|-------|\n"
        for issue in result["issues"]:
            sev_icon = {
                "HIGH": ":red_circle:",
                "MEDIUM": ":yellow_circle:",
                "LOW": ":blue_circle:",
            }.get(issue["severity"], "")
            comment += (
                f"| {sev_icon} {issue['severity']} | `{issue['file']}` | "
                f"{issue.get('rubric_section', '-')} | {issue['description']} |\n"
            )
        comment += "\n"

        high_issues = [i for i in result["issues"] if i["severity"] == "HIGH"]
        if high_issues:
            comment += "### HIGH Severity Issues (must fix before merge)\n\n"
            for issue in high_issues:
                comment += f"**{issue['file']}** (line ~{issue.get('line', '?')})\n"
                comment += f"> {issue['description']}\n\n"
                comment += f"Fix: {issue.get('fix', 'See description')}\n\n"

    if result.get("inconclusive"):
        comment += (
            "\n> **This PR cannot be merged until validation re-runs successfully.**\n> Re-trigger CI to retry.\n"
        )

    comment += "\n---\n*Automated review by Sparky Platform Validation Agent (Sonnet)*"

    subprocess.run(["gh", "pr", "comment", pr_number, "--body", comment], check=False)


def should_skip(changes):
    """Return (skip, reason) — skip if only docs/non-infra files changed."""
    infra_extensions = {".py", ".yaml", ".yml", ".sh"}
    infra_paths = {"src/", "scripts/", "tests/", "configs/", ".github/"}

    for c in changes:
        path = c["filepath"]
        if any(path.startswith(p) for p in infra_paths):
            if any(path.endswith(ext) for ext in infra_extensions):
                return False, ""

    return True, "No infrastructure files changed"


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — skipping platform validation")
        sys.exit(0)

    changes = get_changed_files()

    if not changes:
        print("No reviewable files changed — skipping platform validation")
        sys.exit(0)

    skip, reason = should_skip(changes)
    if skip:
        print(f"Skipping platform validation: {reason}")
        sys.exit(0)

    context = get_codebase_context()

    print(f"Platform Validation Agent reviewing {len(changes)} files...")

    try:
        result = run_validation(changes, context)
    except anthropic.RateLimitError as e:
        # Rate limit exhausted after all retries — BLOCK the PR, do not pass
        print(f"RATE LIMIT: All retries exhausted: {e}", flush=True)
        result = {
            "summary": "INCONCLUSIVE: Anthropic API rate limit (429) — validation did not execute. Re-trigger CI to retry.",
            "issues": [],
            "passed": False,
            "inconclusive": True,
        }
        post_pr_comment(result)
        Path("platform-validation-report.json").write_text(json.dumps(result, indent=2))
        print("\nBLOCKED: rate limit — validation inconclusive")
        sys.exit(1)
    # All other errors propagate as unhandled exceptions (clear traceback in CI logs)

    print(f"\n{'=' * 60}")
    print(f"Summary: {result['summary']}")
    print(f"Issues found: {len(result.get('issues', []))}")
    print(f"Passed: {result['passed']}")

    if result.get("issues"):
        for issue in result["issues"]:
            icon = {"HIGH": "[HIGH]", "MEDIUM": "[MEDIUM]", "LOW": "[LOW]"}.get(issue["severity"], "[?]")
            print(f"  {icon} {issue['file']}: {issue['description']}")

    post_pr_comment(result)

    Path("platform-validation-report.json").write_text(json.dumps(result, indent=2))

    if not result["passed"]:
        high_count = len([i for i in result["issues"] if i["severity"] == "HIGH"])
        print(f"\nBLOCKED: {high_count} HIGH severity engineering issues")
        sys.exit(1)

    print("\nPlatform validation passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
