#!/usr/bin/env python3
"""Research Validation Agent for Sparky AI.

Runs as a pre-commit hook. Uses `claude` CLI to review staged files
against the quantitative finance rubric.

Exits 0 if no HIGH severity issues found.
Exits 1 if HIGH severity issues found.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_MAX_CODE_CHARS = 40000
_MAX_CONTEXT_CHARS = 8000


def get_changed_files():
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
    )
    py_files = [f for f in result.stdout.strip().split("\n") if f.endswith(".py") and f.strip()]

    changes = []
    for filepath in py_files[:15]:
        if not Path(filepath).exists():
            continue
        diff = subprocess.run(
            ["git", "diff", "--cached", "--", filepath],
            capture_output=True,
            text=True,
        )
        content = Path(filepath).read_text(errors="replace")
        changes.append({"filepath": filepath, "diff": diff.stdout, "full_content": content})
    return changes


def _budget_changes(changes, max_chars):
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
            text += f"#### Full content (truncated):\n```python\n{c['full_content'][:content_budget]}\n```\n"
    return text


def get_codebase_context():
    context_files = ["scripts/infra/sweep_utils.py"]
    context = {}
    for path in context_files:
        if Path(path).exists():
            context[path] = Path(path).read_text(errors="replace")[:8000]
    return context


def run_validation(changes):
    rubric = (Path(__file__).parent / "rubric.md").read_text()
    changes_text = _budget_changes(changes, _MAX_CODE_CHARS)

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
        "backtesting validity. Do NOT flag general code style issues.\n\n"
        "If you find no issues, say 'No research methodology issues found.'\n\n"
        "IMPORTANT: Be specific and technical. Reference the exact rubric section. "
        "Only flag concrete violations you can identify in the code.\n\n"
        "CRITICAL: Before flagging a function call's keyword argument as wrong, verify "
        "the function's actual signature as documented in the rubric. See Section 6.1.\n\n"
        "CRITICAL: Read Section 0 of the rubric (Shared Utility Function Delegation) "
        "FIRST. Scripts that import from utility modules like sweep_utils.py and call "
        "evaluate() do NOT need to implement DSR, n_trials, signal shifting, or cost "
        "deduction at the call site — these are handled inside the utility functions.\n\n"
        f"## RUBRIC\n{rubric}\n\n"
        f"## CODEBASE REFERENCE (shared utilities)\n{context_text}\n\n"
        f"## CODE CHANGES TO REVIEW\n{changes_text}\n\n"
        "## OUTPUT FORMAT\n"
        "Respond with ONLY a JSON object (no markdown fences, no explanation):\n"
        '{"summary": "...", "issues": [{"severity": "HIGH|MEDIUM|LOW", '
        '"file": "...", "line": "...", "rubric_section": "...", '
        '"description": "...", "fix": "..."}], "passed": true/false}\n'
        "passed should be false if ANY HIGH severity issues exist."
    )

    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    proc = subprocess.run(
        ["claude", "-p", "--model", "sonnet"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI failed (rc={proc.returncode}): {proc.stderr[:500]}")

    text = proc.stdout.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    decoder = json.JSONDecoder()
    result, _ = decoder.raw_decode(text)
    return result


def main():
    if not shutil.which("claude"):
        print("claude CLI not found — skipping research validation")
        sys.exit(0)

    changes = get_changed_files()
    if not changes:
        print("No Python files staged — skipping research validation")
        sys.exit(0)

    non_test = [c for c in changes if "test" not in c["filepath"]]
    if not non_test:
        print("Only test files staged — skipping research validation")
        sys.exit(0)

    print(f"Research Validation Agent reviewing {len(changes)} files...", flush=True)
    result = run_validation(changes)

    print(f"\n{'=' * 60}")
    print(f"Summary: {result['summary']}")
    print(f"Issues found: {len(result.get('issues', []))}")
    print(f"Passed: {result['passed']}")

    for issue in result.get("issues", []):
        icon = {"HIGH": "[HIGH]", "MEDIUM": "[MEDIUM]", "LOW": "[LOW]"}.get(issue["severity"], "[?]")
        print(f"  {icon} {issue['file']}: {issue['description']}")

    if not result["passed"]:
        high_count = len([i for i in result["issues"] if i["severity"] == "HIGH"])
        print(f"\nBLOCKED: {high_count} HIGH severity issues")
        sys.exit(1)

    print("\nResearch validation passed")


if __name__ == "__main__":
    main()
