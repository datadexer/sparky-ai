#!/usr/bin/env python3
"""Research Validation Agent for Sparky AI.

Runs as a GitHub Action on every PR. Uses Claude Sonnet to review
changed files against the quantitative finance rubric.

Exits 0 if no HIGH severity issues found.
Exits 1 if HIGH severity issues found or rate-limited after retries.
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
# Four total attempts: immediate, +30s, +30s, +30s = 90s of retrying.
_RATE_LIMIT_RETRY_DELAYS = [30, 30, 30]

# Free tier: 10K input tokens/min ≈ 40K chars. Target well under to avoid 429s.
# Budget: ~5K tokens for rubric+prompt, ~3K tokens for code context.
_MAX_CODE_CHARS = 12000  # ~3K tokens for all changed files combined
_MAX_CONTEXT_CHARS = 3000  # reference files budget


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
                "diff": diff.stdout,
                "full_content": content,
            }
        )
    return changes


def _budget_changes(changes, max_chars):
    """Distribute character budget across changed files, prioritizing diffs."""
    if not changes:
        return ""
    n = len(changes)
    per_file = max(500, max_chars // n)
    # Give 70% to diff, 30% to content
    diff_budget = int(per_file * 0.7)
    content_budget = per_file - diff_budget

    text = ""
    for c in changes:
        text += f"\n### File: {c['filepath']}\n"
        diff = c["diff"][:diff_budget]
        text += f"#### Diff:\n```\n{diff}\n```\n"
        if content_budget > 100:
            content = c["full_content"][:content_budget]
            text += f"#### Full content (truncated):\n```python\n{content}\n```\n"
        text += "\n"
    return text


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


def get_codebase_context():
    """Gather lightweight codebase context for shared utility functions."""
    context_files = [
        "scripts/infra/sweep_utils.py",
    ]
    context = {}
    for path in context_files:
        if Path(path).exists():
            context[path] = Path(path).read_text(errors="replace")[:3000]
    return context


def run_validation(changes):
    """Send changes to Claude Sonnet for research validation review."""
    client = anthropic.Anthropic()

    rubric_path = Path(__file__).parent / "rubric.md"
    rubric = rubric_path.read_text()

    # Budget-aware formatting
    changes_text = _budget_changes(changes, _MAX_CODE_CHARS)

    # Gather context for shared utility functions
    context = get_codebase_context()
    context_text = ""
    ctx_per_file = max(500, _MAX_CONTEXT_CHARS // max(len(context), 1))
    for path, content in context.items():
        context_text += f"\n### {path} (shared utility — for reference):\n```python\n{content[:ctx_per_file]}\n```\n"

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
        "CRITICAL: Before flagging a function call's keyword argument as wrong, verify "
        "the function's actual signature as documented in the rubric. Different functions "
        "use different parameter names for different purposes (e.g., n_trials vs n_trades "
        "are parameters of different functions). See Section 6.1.\n\n"
        "CRITICAL: Read Section 0 of the rubric (Shared Utility Function Delegation) "
        "FIRST. Scripts that import from utility modules like sweep_utils.py and call "
        "evaluate() do NOT need to implement DSR, n_trials, signal shifting, or cost "
        "deduction at the call site — these are handled inside the utility functions. "
        "Check the CODEBASE REFERENCE section to see the utility implementations.\n\n"
        f"## RUBRIC\n{rubric}\n\n"
        f"## CODEBASE REFERENCE (shared utilities)\n{context_text}\n\n"
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

    # Use assistant prefill to force JSON output
    response = _create_message_with_retry(
        client,
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "{"},
        ],
    )

    # Reconstruct JSON (we prefilled "{", model continues from there)
    text = "{" + response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        # Use raw_decode to stop at end of first JSON object — model may
        # append prose explanation after the closing brace.
        decoder = json.JSONDecoder()
        result, _ = decoder.raw_decode(text)
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
        # Rate limit exhausted after retries (~90s) — block the PR.
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
