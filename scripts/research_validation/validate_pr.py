#!/usr/bin/env python3
"""Research Validation Agent for Sparky AI.

Runs as a GitHub Action on every PR. Uses Claude Sonnet to review
changed files against the quantitative finance rubric.

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
    """Get diff of changed Python files in this PR."""
    result = subprocess.run(
        ["git", "diff", "origin/main...HEAD", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
    )
    py_files = [f for f in result.stdout.strip().split("\n") if f.endswith(".py") and f.strip()]

    changes = []
    for filepath in py_files[:15]:  # limit to 15 files
        if not Path(filepath).exists():
            continue
        diff = subprocess.run(
            ["git", "diff", "origin/main...HEAD", "--", filepath],
            capture_output=True,
            text=True,
        )
        content = Path(filepath).read_text()
        changes.append(
            {
                "filepath": filepath,
                "diff": diff.stdout[:5000],  # truncate large diffs
                "full_content": content[:10000],  # truncate large files
            }
        )
    return changes


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


def run_validation(changes):
    """Send changes to Claude Sonnet for research validation review."""
    client = anthropic.Anthropic()

    rubric_path = Path(__file__).parent / "rubric.md"
    rubric = rubric_path.read_text()

    # Format the changes for review
    changes_text = ""
    for c in changes:
        changes_text += f"\n### File: {c['filepath']}\n"
        changes_text += f"#### Diff:\n```\n{c['diff']}\n```\n"
        changes_text += f"#### Full content (truncated):\n```python\n{c['full_content'][:3000]}\n```\n\n"

    prompt = (
        "You are a quantitative finance code reviewer for an autonomous "
        "crypto trading research system called Sparky AI.\n\n"
        "Review the following code changes against the rubric below. For each issue found:\n"
        "1. State the severity: HIGH, MEDIUM, or LOW\n"
        "2. State the file and approximate line\n"
        "3. Explain the issue clearly\n"
        "4. Suggest the fix\n\n"
        "Focus on quantitative finance correctness, statistical methodology, and "
        "backtesting validity. Do NOT flag general code style issues — another tool "
        "handles that.\n\n"
        "If you find no issues, say 'No research methodology issues found.'\n\n"
        "IMPORTANT: Be specific and technical. Reference the exact rubric section that "
        "applies. Do not flag speculative concerns — only flag concrete violations you "
        "can identify in the code.\n\n"
        f"## RUBRIC\n{rubric}\n\n"
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
        '      "rubric_section": "e.g., 1.2 Cross-validation",\n'
        '      "description": "What\'s wrong",\n'
        '      "fix": "How to fix it"\n'
        "    }\n"
        "  ],\n"
        '  "passed": true/false\n'
        "}\n"
        "passed should be false if ANY HIGH severity issues exist."
    )

    response = _create_message_with_retry(
        client,
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )

    # Parse response — handle both clean JSON and markdown-wrapped JSON
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
        status = "## :warning: Research Validation: INCONCLUSIVE"
    elif result["passed"]:
        status = "## :white_check_mark: Research Validation: PASSED"
    else:
        status = "## :x: Research Validation: FAILED (HIGH severity issues found)"

    comment = f"{status}\n\n{result['summary']}\n\n"

    if result.get("issues"):
        comment += "| Severity | File | Section | Issue |\n"
        comment += "|----------|------|---------|-------|\n"
        for issue in result["issues"]:
            sev_icon = {"HIGH": ":red_circle:", "MEDIUM": ":yellow_circle:", "LOW": ":blue_circle:"}.get(
                issue["severity"], ""
            )
            comment += (
                f"| {sev_icon} {issue['severity']} | `{issue['file']}` | "
                f"{issue.get('rubric_section', '-')} | {issue['description']} |\n"
            )
        comment += "\n"

        # Detail HIGH issues
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

    comment += "\n---\n*Automated review by Sparky Research Validation Agent (Sonnet)*"

    # Post via gh CLI
    subprocess.run(["gh", "pr", "comment", pr_number, "--body", comment], check=False)


def main():
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — skipping research validation")
        sys.exit(0)

    changes = get_changed_files()

    if not changes:
        print("No Python files changed — skipping research validation")
        sys.exit(0)

    # Skip if only test files changed
    non_test = [c for c in changes if "test" not in c["filepath"]]
    if not non_test:
        print("Only test files changed — skipping research validation")
        sys.exit(0)

    print(f"Research Validation Agent reviewing {len(changes)} files...")

    try:
        result = run_validation(changes)
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
        Path("research-validation-report.json").write_text(json.dumps(result, indent=2))
        print("\nBLOCKED: rate limit — validation inconclusive")
        sys.exit(1)
    # All other errors propagate as unhandled exceptions (clear traceback in CI logs)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Summary: {result['summary']}")
    print(f"Issues found: {len(result.get('issues', []))}")
    print(f"Passed: {result['passed']}")

    if result.get("issues"):
        for issue in result["issues"]:
            icon = {"HIGH": "[HIGH]", "MEDIUM": "[MEDIUM]", "LOW": "[LOW]"}.get(issue["severity"], "[?]")
            print(f"  {icon} {issue['file']}: {issue['description']}")

    # Post PR comment
    post_pr_comment(result)

    # Save results
    Path("research-validation-report.json").write_text(json.dumps(result, indent=2))

    # Exit code: 1 if HIGH severity issues found
    if not result["passed"]:
        high_count = len([i for i in result["issues"] if i["severity"] == "HIGH"])
        print(f"\nBLOCKED: {high_count} HIGH severity issues")
        sys.exit(1)

    print("\nResearch validation passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
