#!/usr/bin/env python3
"""Platform Validation Agent for Sparky AI.

Runs as a pre-commit hook. Uses `claude` CLI to review staged files
against the engineering QA rubric.

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
_MAX_CONTEXT_CHARS = 12000


def get_changed_files():
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


def get_codebase_context():
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
            text += f"#### Full content (truncated):\n```\n{c['full_content'][:content_budget]}\n```\n"
    return text


def should_skip(changes):
    infra_paths = {"src/", "scripts/", "tests/", "configs/", ".github/"}
    infra_extensions = {".py", ".yaml", ".yml", ".sh"}
    for c in changes:
        path = c["filepath"]
        if any(path.startswith(p) for p in infra_paths):
            if any(path.endswith(ext) for ext in infra_extensions):
                return False, ""
    return True, "No infrastructure files changed"


def run_validation(changes, context):
    rubric = (Path(__file__).parent / "rubric.md").read_text()
    changes_text = _budget_changes(changes, _MAX_CODE_CHARS)

    context_text = ""
    ctx_per_file = max(500, _MAX_CONTEXT_CHARS // max(len(context), 1))
    for path, content in context.items():
        context_text += f"\n### {path} (for reference):\n```python\n{content[:ctx_per_file]}\n```\n"

    prompt = (
        "You are an engineering QA reviewer for Sparky AI, an autonomous crypto "
        "trading research platform. Review code changes for infrastructure "
        "correctness and engineering soundness — NOT statistical methodology.\n\n"
        "For each issue found:\n"
        "1. State the severity: HIGH, MEDIUM, or LOW\n"
        "2. State the file and approximate line\n"
        "3. Explain the concrete engineering issue\n"
        "4. Suggest the fix\n\n"
        "Focus: backtesting plumbing, data access patterns, wandb parameter flow, "
        "cost model API usage, guardrails invocation, testing coverage.\n\n"
        "Do NOT flag: code style, import ordering, statistical methodology, "
        "performance without correctness impact.\n\n"
        "IMPORTANT: Be specific. Reference the exact rubric section. Only flag "
        "concrete violations visible in the code.\n\n"
        "CRITICAL: Before flagging keyword arguments, verify the actual function "
        "signature. The rubric's 'What NOT to Flag' section has authoritative signatures.\n\n"
        "CRITICAL: Read Section 0 (Shared Utility Function Delegation) FIRST. "
        "Scripts importing from sweep_utils.py do NOT need to implement signal "
        "shifting, cost deduction, guardrails, or n_trials at the call site.\n\n"
        f"## ENGINEERING RUBRIC\n{rubric}\n\n"
        f"## CODEBASE REFERENCE\n{context_text}\n\n"
        f"## CODE CHANGES TO REVIEW\n{changes_text}\n\n"
        "## OUTPUT FORMAT\n"
        "Respond with ONLY a JSON object (no markdown fences, no explanation):\n"
        '{"summary": "...", "issues": [{"severity": "HIGH|MEDIUM|LOW", '
        '"file": "...", "line": "...", "rubric_section": "...", '
        '"description": "...", "fix": "..."}], "passed": true/false}\n'
        "passed must be false if ANY HIGH severity issues exist."
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
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    decoder = json.JSONDecoder()
    result, _ = decoder.raw_decode(text)
    return result


def main():
    if not shutil.which("claude"):
        print("claude CLI not found — skipping platform validation")
        sys.exit(0)

    changes = get_changed_files()
    if not changes:
        print("No reviewable files staged — skipping platform validation")
        sys.exit(0)

    skip, reason = should_skip(changes)
    if skip:
        print(f"Skipping platform validation: {reason}")
        sys.exit(0)

    context = get_codebase_context()
    print(f"Platform Validation Agent reviewing {len(changes)} files...", flush=True)
    result = run_validation(changes, context)

    print(f"\n{'=' * 60}")
    print(f"Summary: {result['summary']}")
    print(f"Issues found: {len(result.get('issues', []))}")
    print(f"Passed: {result['passed']}")

    for issue in result.get("issues", []):
        icon = {"HIGH": "[HIGH]", "MEDIUM": "[MEDIUM]", "LOW": "[LOW]"}.get(issue["severity"], "[?]")
        print(f"  {icon} {issue['file']}: {issue['description']}")

    if not result["passed"]:
        high_count = len([i for i in result["issues"] if i["severity"] == "HIGH"])
        print(f"\nBLOCKED: {high_count} HIGH severity engineering issues")
        sys.exit(1)

    print("\nPlatform validation passed")


if __name__ == "__main__":
    main()
