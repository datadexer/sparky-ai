"""Tests for the research sandbox hook path validation."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Import the hook module directly for unit testing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".claude" / "hooks"))
import research_sandbox


# ── Helpers ───────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Write/Edit Path Validation ───────────────────────────────────────────


class TestPathValidation:
    """Test is_path_allowed for Write/Edit tool calls."""

    def test_results_dir_allowed(self):
        allowed, _ = research_sandbox.is_path_allowed("results/sweep_001.json", PROJECT_ROOT)
        assert allowed

    def test_results_subdir_allowed(self):
        allowed, _ = research_sandbox.is_path_allowed("results/regime_donchian/session_003.json", PROJECT_ROOT)
        assert allowed

    def test_scratch_dir_allowed(self):
        allowed, _ = research_sandbox.is_path_allowed("scratch/temp_analysis.py", PROJECT_ROOT)
        assert allowed

    def test_state_dir_allowed(self):
        allowed, _ = research_sandbox.is_path_allowed("state/core_memory.json", PROJECT_ROOT)
        assert allowed

    def test_gate_request_allowed(self):
        allowed, _ = research_sandbox.is_path_allowed("GATE_REQUEST.md", PROJECT_ROOT)
        assert allowed

    def test_scripts_py_allowed(self):
        allowed, _ = research_sandbox.is_path_allowed("scripts/my_sweep.py", PROJECT_ROOT)
        assert allowed

    def test_scripts_subdir_blocked(self):
        """scripts/research_validation/rubric.md should be blocked."""
        allowed, _ = research_sandbox.is_path_allowed("scripts/research_validation/rubric.md", PROJECT_ROOT)
        assert not allowed

    def test_protected_script_sparky_blocked(self):
        allowed, _ = research_sandbox.is_path_allowed("scripts/sparky", PROJECT_ROOT)
        assert not allowed

    def test_protected_script_alert_blocked(self):
        allowed, _ = research_sandbox.is_path_allowed("scripts/alert.sh", PROJECT_ROOT)
        assert not allowed

    def test_protected_script_ceo_runner_blocked(self):
        allowed, _ = research_sandbox.is_path_allowed("scripts/ceo_runner.sh", PROJECT_ROOT)
        assert not allowed

    def test_infra_sweep_utils_blocked(self):
        """bin/infra/ is blocked (protected platform utilities)."""
        allowed, _ = research_sandbox.is_path_allowed("bin/infra/sweep_utils.py", PROJECT_ROOT)
        assert not allowed

    def test_infra_sweep_two_stage_blocked(self):
        allowed, _ = research_sandbox.is_path_allowed("bin/infra/sweep_two_stage.py", PROJECT_ROOT)
        assert not allowed

    def test_src_blocked(self):
        allowed, _ = research_sandbox.is_path_allowed("src/sparky/backtest/engine.py", PROJECT_ROOT)
        assert not allowed

    def test_claude_dir_blocked(self):
        allowed, _ = research_sandbox.is_path_allowed(".claude/settings.local.json", PROJECT_ROOT)
        assert not allowed

    def test_claude_md_blocked(self):
        allowed, _ = research_sandbox.is_path_allowed("CLAUDE.md", PROJECT_ROOT)
        assert not allowed

    def test_research_agent_md_blocked(self):
        allowed, _ = research_sandbox.is_path_allowed("RESEARCH_AGENT.md", PROJECT_ROOT)
        assert not allowed

    def test_configs_blocked(self):
        allowed, _ = research_sandbox.is_path_allowed("configs/holdout_policy.yaml", PROJECT_ROOT)
        assert not allowed

    def test_tests_blocked(self):
        allowed, _ = research_sandbox.is_path_allowed("tests/test_engine.py", PROJECT_ROOT)
        assert not allowed

    def test_workflows_blocked(self):
        allowed, _ = research_sandbox.is_path_allowed("workflows/contract_004.py", PROJECT_ROOT)
        assert not allowed

    def test_data_blocked(self):
        allowed, _ = research_sandbox.is_path_allowed("data/processed/features.parquet", PROJECT_ROOT)
        assert not allowed

    def test_docs_blocked(self):
        allowed, _ = research_sandbox.is_path_allowed("docs/FULL_GUIDELINES.md", PROJECT_ROOT)
        assert not allowed

    def test_pyproject_blocked(self):
        allowed, _ = research_sandbox.is_path_allowed("pyproject.toml", PROJECT_ROOT)
        assert not allowed

    def test_roadmap_blocked(self):
        allowed, _ = research_sandbox.is_path_allowed("roadmap/02_RESEARCH_LOG.md", PROJECT_ROOT)
        assert not allowed

    def test_directives_blocked(self):
        allowed, _ = research_sandbox.is_path_allowed("directives/regime_donchian_v3.yaml", PROJECT_ROOT)
        assert not allowed

    def test_path_traversal_blocked(self):
        allowed, _ = research_sandbox.is_path_allowed("results/../../src/sparky/engine.py", PROJECT_ROOT)
        assert not allowed

    def test_oos_vault_blocked(self):
        allowed, _ = research_sandbox.is_path_allowed("data/.oos_vault/btc_2024.parquet", PROJECT_ROOT)
        assert not allowed

    def test_oos_vault_no_dot_blocked(self):
        allowed, _ = research_sandbox.is_path_allowed("data/oos_vault/btc_2024.parquet", PROJECT_ROOT)
        assert not allowed

    def test_tmp_allowed(self):
        allowed, _ = research_sandbox.is_path_allowed("/tmp/my_temp_file.csv", PROJECT_ROOT)  # noqa: S108
        assert allowed

    def test_absolute_results_allowed(self):
        abs_path = str(PROJECT_ROOT / "results" / "sweep.json")
        allowed, _ = research_sandbox.is_path_allowed(abs_path, PROJECT_ROOT)
        assert allowed

    def test_absolute_src_blocked(self):
        abs_path = str(PROJECT_ROOT / "src" / "sparky" / "data" / "loader.py")
        allowed, _ = research_sandbox.is_path_allowed(abs_path, PROJECT_ROOT)
        assert not allowed

    def test_rubric_blocked(self):
        allowed, _ = research_sandbox.is_path_allowed("scripts/research_validation/rubric.md", PROJECT_ROOT)
        assert not allowed

    def test_memory_md_blocked(self):
        allowed, _ = research_sandbox.is_path_allowed(".claude/projects/memory/MEMORY.md", PROJECT_ROOT)
        assert not allowed


# ── Bash Command Validation ──────────────────────────────────────────────


class TestBashValidation:
    """Test is_bash_command_allowed for Bash tool calls."""

    def test_python_execution_allowed(self):
        allowed, _ = research_sandbox.is_bash_command_allowed(".venv/bin/python scripts/my_sweep.py", PROJECT_ROOT)
        assert allowed

    def test_redirect_to_results_allowed(self):
        allowed, _ = research_sandbox.is_bash_command_allowed('echo "hello" > results/output.txt', PROJECT_ROOT)
        assert allowed

    def test_redirect_to_src_blocked(self):
        allowed, _ = research_sandbox.is_bash_command_allowed('echo "hack" > src/sparky/data/loader.py', PROJECT_ROOT)
        assert not allowed

    def test_oos_vault_cat_blocked(self):
        allowed, _ = research_sandbox.is_bash_command_allowed("cat data/.oos_vault/btc_2024.parquet", PROJECT_ROOT)
        assert not allowed

    def test_oos_vault_reference_blocked(self):
        allowed, _ = research_sandbox.is_bash_command_allowed("ls -la data/.oos_vault/", PROJECT_ROOT)
        assert not allowed

    def test_cp_to_src_blocked(self):
        allowed, _ = research_sandbox.is_bash_command_allowed("cp scratch/evil.py src/sparky/evil.py", PROJECT_ROOT)
        assert not allowed

    def test_cp_to_results_allowed(self):
        allowed, _ = research_sandbox.is_bash_command_allowed("cp /tmp/data.csv results/data.csv", PROJECT_ROOT)
        assert allowed

    def test_mv_to_configs_blocked(self):
        allowed, _ = research_sandbox.is_bash_command_allowed(
            "mv scratch/new.yaml configs/holdout_policy.yaml", PROJECT_ROOT
        )
        assert not allowed

    def test_sed_i_on_src_blocked(self):
        allowed, _ = research_sandbox.is_bash_command_allowed(
            "sed -i 's/old/new/' src/sparky/data/loader.py", PROJECT_ROOT
        )
        assert not allowed

    def test_tee_to_claude_md_blocked(self):
        allowed, _ = research_sandbox.is_bash_command_allowed('echo "new content" | tee CLAUDE.md', PROJECT_ROOT)
        assert not allowed

    def test_readonly_commands_allowed(self):
        """Read-only commands should be allowed."""
        allowed, _ = research_sandbox.is_bash_command_allowed("cat results/output.json", PROJECT_ROOT)
        assert allowed

    def test_python_with_args_allowed(self):
        allowed, _ = research_sandbox.is_bash_command_allowed(
            ".venv/bin/python -c 'import sparky; print(sparky.__version__)'",
            PROJECT_ROOT,
        )
        assert allowed


# ── Environment Gating ───────────────────────────────────────────────────


class TestEnvironmentGating:
    """Test that the hook only enforces in sandbox mode."""

    def test_no_env_var_allows_everything(self):
        """Without SPARKY_RESEARCH_SANDBOX=1, the hook exits 0 for everything."""
        hook_path = PROJECT_ROOT / ".claude" / "hooks" / "research_sandbox.py"
        if not hook_path.exists():
            pytest.skip("Hook script not found")

        # Simulate a Write to a protected path (src/)
        event = json.dumps(
            {
                "tool_name": "Write",
                "tool_input": {"file_path": "src/sparky/evil.py", "content": "evil"},
            }
        )

        env = os.environ.copy()
        env.pop("SPARKY_RESEARCH_SANDBOX", None)

        result = subprocess.run(
            [sys.executable, str(hook_path)],
            input=event,
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        assert result.returncode == 0, f"Expected 0 (allow), got {result.returncode}: {result.stderr}"

    def test_sandbox_env_blocks_protected(self):
        """With SPARKY_RESEARCH_SANDBOX=1, writing to src/ is blocked."""
        hook_path = PROJECT_ROOT / ".claude" / "hooks" / "research_sandbox.py"
        if not hook_path.exists():
            pytest.skip("Hook script not found")

        event = json.dumps(
            {
                "tool_name": "Write",
                "tool_input": {"file_path": "src/sparky/evil.py", "content": "evil"},
            }
        )

        env = os.environ.copy()
        env["SPARKY_RESEARCH_SANDBOX"] = "1"
        env["CLAUDE_PROJECT_DIR"] = str(PROJECT_ROOT)

        result = subprocess.run(
            [sys.executable, str(hook_path)],
            input=event,
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        assert result.returncode == 2, f"Expected 2 (block), got {result.returncode}: {result.stderr}"
        assert "SANDBOX BLOCKED" in result.stderr

    def test_sandbox_env_allows_results(self):
        """With SPARKY_RESEARCH_SANDBOX=1, writing to results/ is allowed."""
        hook_path = PROJECT_ROOT / ".claude" / "hooks" / "research_sandbox.py"
        if not hook_path.exists():
            pytest.skip("Hook script not found")

        event = json.dumps(
            {
                "tool_name": "Write",
                "tool_input": {
                    "file_path": str(PROJECT_ROOT / "results" / "test.json"),
                    "content": "{}",
                },
            }
        )

        env = os.environ.copy()
        env["SPARKY_RESEARCH_SANDBOX"] = "1"
        env["CLAUDE_PROJECT_DIR"] = str(PROJECT_ROOT)

        result = subprocess.run(
            [sys.executable, str(hook_path)],
            input=event,
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        assert result.returncode == 0, f"Expected 0 (allow), got {result.returncode}: {result.stderr}"


# ── Idle Loop Detection (engine.py) ──────────────────────────────────────


class TestIdleLoopConstants:
    """Test the idle loop detection patterns from engine.py."""

    def test_idle_phrases_match(self):
        from sparky.workflow.session import IDLE_LOOP_PHRASES

        assert IDLE_LOOP_PHRASES.search("The session is done.")
        assert IDLE_LOOP_PHRASES.search("No further action needed.")
        assert IDLE_LOOP_PHRASES.search("All complete.")
        assert IDLE_LOOP_PHRASES.search("I've completed all the work.")
        assert IDLE_LOOP_PHRASES.search("There is nothing more to do.")
        assert IDLE_LOOP_PHRASES.search("The work is complete now.")

    def test_idle_phrases_no_false_positive(self):
        from sparky.workflow.session import IDLE_LOOP_PHRASES

        assert not IDLE_LOOP_PHRASES.search("Running experiment sweep...")
        assert not IDLE_LOOP_PHRASES.search("Training XGBoost model")
        assert not IDLE_LOOP_PHRASES.search("Best Sharpe: 1.23")


# ── Disallowed Tools Expansion ───────────────────────────────────────────


class TestDisallowedTools:
    """Test that RESEARCH_DISALLOWED_TOOLS includes all required patterns."""

    def test_git_blocked(self):
        from sparky.workflow.orchestrator import RESEARCH_DISALLOWED_TOOLS

        assert "Bash(git:*)" in RESEARCH_DISALLOWED_TOOLS

    def test_pip_blocked(self):
        from sparky.workflow.orchestrator import RESEARCH_DISALLOWED_TOOLS

        assert "Bash(pip:*)" in RESEARCH_DISALLOWED_TOOLS
        assert "Bash(pip3:*)" in RESEARCH_DISALLOWED_TOOLS

    def test_curl_wget_blocked(self):
        from sparky.workflow.orchestrator import RESEARCH_DISALLOWED_TOOLS

        assert "Bash(curl:*)" in RESEARCH_DISALLOWED_TOOLS
        assert "Bash(wget:*)" in RESEARCH_DISALLOWED_TOOLS

    def test_systemctl_blocked(self):
        from sparky.workflow.orchestrator import RESEARCH_DISALLOWED_TOOLS

        assert "Bash(systemctl:*)" in RESEARCH_DISALLOWED_TOOLS

    def test_kill_blocked(self):
        from sparky.workflow.orchestrator import RESEARCH_DISALLOWED_TOOLS

        assert "Bash(kill:*)" in RESEARCH_DISALLOWED_TOOLS
        assert "Bash(pkill:*)" in RESEARCH_DISALLOWED_TOOLS

    def test_oos_vault_patterns(self):
        from sparky.workflow.orchestrator import RESEARCH_DISALLOWED_TOOLS

        assert "Bash(cat data/.oos_vault:*)" in RESEARCH_DISALLOWED_TOOLS


# ── launch_claude_session extra_env ──────────────────────────────────────


class TestExtraEnv:
    """Test that launch_claude_session accepts extra_env parameter."""

    def test_signature_accepts_extra_env(self):
        """Verify the function signature includes extra_env."""
        import inspect

        from sparky.workflow.session import launch_claude_session

        sig = inspect.signature(launch_claude_session)
        assert "extra_env" in sig.parameters
        assert sig.parameters["extra_env"].default is None

    def test_signature_accepts_disallowed_tools(self):
        """Verify the function signature includes disallowed_tools."""
        import inspect

        from sparky.workflow.session import launch_claude_session

        sig = inspect.signature(launch_claude_session)
        assert "disallowed_tools" in sig.parameters
        assert sig.parameters["disallowed_tools"].default is None


class TestSettingsHooks:
    """Verify .claude/settings.json registers sandbox hooks."""

    def test_settings_json_has_hooks(self):
        settings_path = PROJECT_ROOT / ".claude" / "settings.json"
        assert settings_path.exists(), ".claude/settings.json must exist"
        data = json.load(open(settings_path))
        hooks = data.get("hooks", {}).get("PreToolUse", [])
        matchers = {h["matcher"] for h in hooks}
        assert {"Write", "Edit", "Bash"} <= matchers
