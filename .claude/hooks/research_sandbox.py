#!/usr/bin/env python3
"""PreToolUse hook: filesystem sandbox for research agents.

Only enforced when SPARKY_RESEARCH_SANDBOX=1 is set in the environment.
Oversight and interactive sessions skip all checks (exit 0 immediately).

Exit codes:
  0 = allow the tool call
  2 = block the tool call (with reason on stderr)
"""

import json
import os
import re
import sys
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────

# Allowlisted path prefixes for Write/Edit (relative to project root).
# Default-deny: anything not listed here is blocked.
ALLOWED_WRITE_PREFIXES = [
    "results/",
    "scratch/",
    "GATE_REQUEST.md",
]

# scripts/*.py is allowed, but NOT subdirectories or specific protected files
ALLOWED_SCRIPT_PATTERN = re.compile(r"^scripts/[^/]+\.py$")

PROTECTED_SCRIPTS = {
    "scripts/sparky",
    "scripts/alert.sh",
    "scripts/ceo_runner.sh",
}
# Note: scripts/infra/ files are automatically blocked because the sandbox
# only allows scripts/[^/]+\.py$ (no subdirectories).

# Absolute temp dirs always allowed
ALLOWED_ABSOLUTE_PREFIXES = [
    "/tmp/",  # noqa: S108
    "/var/tmp/",  # noqa: S108
]

# Blocked path components (anywhere in the path)
BLOCKED_COMPONENTS = {
    ".oos_vault",
    "oos_vault",
}

# Bash commands referencing OOS vault
OOS_VAULT_PATTERN = re.compile(r"\.?oos_vault", re.IGNORECASE)


# ── Path Resolution ──────────────────────────────────────────────────────


def _get_project_root() -> Path:
    """Get project root from CLAUDE_PROJECT_DIR or by walking up from this file."""
    env_root = os.environ.get("CLAUDE_PROJECT_DIR")
    if env_root:
        return Path(env_root).resolve()
    # Fallback: this file is at .claude/hooks/research_sandbox.py
    return Path(__file__).resolve().parent.parent.parent


def _resolve_to_relative(file_path: str, project_root: Path) -> str | None:
    """Resolve a file path to a project-relative path. Returns None if outside project."""
    try:
        resolved = Path(file_path).resolve()
        return str(resolved.relative_to(project_root))
    except ValueError:
        # Path is outside project root — check absolute allowlist
        return None


def is_path_allowed(file_path: str, project_root: Path) -> tuple[bool, str]:
    """Check if a file path is allowed for writing.

    Returns (allowed, reason).
    """
    # Block path traversal attempts
    if ".." in file_path:
        return False, f"Path traversal blocked: {file_path}"

    # Check for blocked components
    for component in BLOCKED_COMPONENTS:
        if component in file_path:
            return False, f"OOS vault access blocked: {file_path}"

    # Check absolute paths to temp dirs
    for prefix in ALLOWED_ABSOLUTE_PREFIXES:
        if file_path.startswith(prefix):
            return True, "temp directory"

    # Resolve to project-relative path
    rel_path = _resolve_to_relative(file_path, project_root)
    if rel_path is None:
        # Outside project — check if it's in an allowed absolute prefix
        try:
            resolved = str(Path(file_path).resolve())
            for prefix in ALLOWED_ABSOLUTE_PREFIXES:
                if resolved.startswith(prefix):
                    return True, "temp directory (resolved)"
        except Exception:  # noqa: S110
            pass
        return False, f"Path outside project root: {file_path}"

    # Re-check blocked components on resolved path
    for component in BLOCKED_COMPONENTS:
        if component in rel_path:
            return False, f"OOS vault access blocked: {rel_path}"

    # Check against allowlisted prefixes
    for prefix in ALLOWED_WRITE_PREFIXES:
        if prefix.endswith("/"):
            if rel_path.startswith(prefix) or rel_path == prefix.rstrip("/"):
                return True, f"allowed prefix: {prefix}"
        else:
            if rel_path == prefix:
                return True, f"allowed file: {prefix}"

    # Check scripts/*.py pattern (not subdirectories, not protected)
    if ALLOWED_SCRIPT_PATTERN.match(rel_path):
        if rel_path not in PROTECTED_SCRIPTS:
            return True, "allowed script"
        return False, f"Protected script: {rel_path}"

    return False, f"Path not in allowlist: {rel_path}"


def is_bash_command_allowed(command: str, project_root: Path) -> tuple[bool, str]:
    """Check if a bash command is allowed.

    We block:
    1. Commands referencing .oos_vault
    2. Write operations (redirects, cp/mv/tee, sed -i) targeting protected paths

    We allow:
    - .venv/bin/python execution (research needs it)
    - Read-only commands
    """
    # Block any reference to OOS vault
    if OOS_VAULT_PATTERN.search(command):
        return False, "OOS vault reference in bash command"

    # Check for write operations targeting protected paths
    # Extract potential target paths from redirect operators
    redirect_matches = re.findall(r">>?\s*(?:\"([^\"]*)\"|'([^']*)'|(\S+))", command)
    for match_groups in redirect_matches:
        target = next(g for g in match_groups if g)
        allowed, reason = is_path_allowed(target, project_root)
        if not allowed:
            return False, f"Redirect to protected path: {target} ({reason})"

    # Check cp/mv/tee targets (last argument is usually the destination)
    for cmd_pattern in (r"\bcp\b", r"\bmv\b", r"\btee\b"):
        if re.search(cmd_pattern, command):
            # Extract arguments after the command
            parts = command.split()
            for i, part in enumerate(parts):
                if re.match(cmd_pattern.strip(r"\b"), part):
                    # Check remaining args that look like paths (skip flags)
                    targets = [p for p in parts[i + 1 :] if not p.startswith("-")]
                    for target in targets:
                        target = target.strip("\"'")
                        allowed, reason = is_path_allowed(target, project_root)
                        if not allowed:
                            return False, f"{part} to protected path: {target} ({reason})"
                    break

    # Check sed -i / perl -i
    if re.search(r"\bsed\s+.*-i", command) or re.search(r"\bperl\s+.*-i", command):
        # Extract file arguments
        parts = command.split()
        for part in parts:
            part = part.strip("\"'")
            if "/" in part or part.endswith(".py") or part.endswith(".md") or part.endswith(".yaml"):
                allowed, reason = is_path_allowed(part, project_root)
                if not allowed:
                    return False, f"In-place edit of protected path: {part} ({reason})"

    return True, "allowed"


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> int:
    """Read tool call from stdin, validate, return exit code."""
    # Only enforce in sandbox mode
    if os.environ.get("SPARKY_RESEARCH_SANDBOX") != "1":
        return 0

    project_root = _get_project_root()

    try:
        raw = sys.stdin.read()
        event = json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        # Fail closed in sandbox mode
        print(f"SANDBOX: Failed to parse hook input: {e}", file=sys.stderr)
        return 2

    tool_name = event.get("tool_name", "")
    tool_input = event.get("tool_input", {})

    if tool_name in ("Write", "Edit"):
        file_path = tool_input.get("file_path", "")
        if not file_path:
            print("SANDBOX: No file_path in Write/Edit tool call", file=sys.stderr)
            return 2

        allowed, reason = is_path_allowed(file_path, project_root)
        if not allowed:
            print(f"SANDBOX BLOCKED: {reason}", file=sys.stderr)
            return 2

    elif tool_name == "Bash":
        command = tool_input.get("command", "")
        if not command:
            return 0

        allowed, reason = is_bash_command_allowed(command, project_root)
        if not allowed:
            print(f"SANDBOX BLOCKED: {reason}", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
