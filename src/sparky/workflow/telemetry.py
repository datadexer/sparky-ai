"""Stream-JSON parser and session telemetry for Claude subprocess output.

Parses stream-json lines from `claude --output-format stream-json`,
extracts human-readable text for log files, collects token/tool counts,
and detects behavioral flags.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Sonnet pricing (per token) — update as needed
INPUT_RATE_USD = 3.0 / 1_000_000  # $3 per 1M input tokens
OUTPUT_RATE_USD = 15.0 / 1_000_000  # $15 per 1M output tokens
CACHE_WRITE_RATE_USD = 3.75 / 1_000_000  # $3.75 per 1M cache creation tokens
CACHE_READ_RATE_USD = 0.30 / 1_000_000  # $0.30 per 1M cache read tokens


@dataclass
class SessionTelemetry:
    """Telemetry collected from a single Claude session."""

    session_id: str
    step: str
    attempt: int
    started_at: str
    ended_at: str = ""
    duration_minutes: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cache_read: int = 0
    tokens_cache_creation: int = 0
    num_turns: int = 0
    cost_usd: float = 0.0  # Claude CLI's own cost calculation (ground truth)
    tool_calls: int = 0
    estimated_cost_usd: float = 0.0
    behavioral_flags: list[str] = field(default_factory=list)
    exit_reason: str = "completed"
    done_when_result: bool = False

    def compute_cost(self) -> float:
        """Compute estimated cost from token counts, including cache tokens."""
        self.estimated_cost_usd = (
            self.tokens_input * INPUT_RATE_USD
            + self.tokens_cache_creation * CACHE_WRITE_RATE_USD
            + self.tokens_cache_read * CACHE_READ_RATE_USD
            + self.tokens_output * OUTPUT_RATE_USD
        )
        return self.estimated_cost_usd

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return {
            "session_id": self.session_id,
            "step": self.step,
            "attempt": self.attempt,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_minutes": self.duration_minutes,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "tokens_cache_read": self.tokens_cache_read,
            "tokens_cache_creation": self.tokens_cache_creation,
            "num_turns": self.num_turns,
            "cost_usd": self.cost_usd,
            "tool_calls": self.tool_calls,
            "estimated_cost_usd": self.estimated_cost_usd,
            "behavioral_flags": self.behavioral_flags,
            "exit_reason": self.exit_reason,
            "done_when_result": self.done_when_result,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionTelemetry":
        """Deserialize from dict."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class StreamParser:
    """Parses stream-json output from Claude, writes readable logs, collects telemetry.

    Usage:
        parser = StreamParser(session_id="20260217_031500", step="sweep", attempt=1)
        for line in process.stdout:
            parser.feed(line, log_file=f)
        telemetry = parser.finalize()
    """

    def __init__(self, session_id: str, step: str, attempt: int):
        self.telemetry = SessionTelemetry(
            session_id=session_id,
            step=step,
            attempt=attempt,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._start_time = time.monotonic()
        self._text_tokens_estimate = 0
        self._total_output_blocks = 0

    def feed(self, line: str, log_file=None) -> None:
        """Process a single stream-json line.

        Args:
            line: Raw line from Claude stdout.
            log_file: Optional file object to write human-readable output.
        """
        line = line.strip()
        if not line:
            return

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            if log_file:
                log_file.write(line + "\n")
                log_file.flush()
            return

        msg_type = msg.get("type")

        if msg_type == "assistant":
            content = msg.get("message", {}).get("content", [])
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    self._total_output_blocks += 1
                    if block_type == "text":
                        text = block.get("text", "")
                        if log_file:
                            log_file.write(text + "\n")
                            log_file.flush()
                        self._text_tokens_estimate += len(text.split())
                        self._check_behavioral_flags(text)
                    elif block_type == "tool_use":
                        self.telemetry.tool_calls += 1
                        tool_name = block.get("name", "unknown")
                        if log_file:
                            log_file.write(f"[tool: {tool_name}]\n")
                            log_file.flush()
            elif isinstance(content, str):
                if log_file:
                    log_file.write(content + "\n")
                    log_file.flush()
                self._check_behavioral_flags(content)

        elif msg_type == "result":
            result_text = msg.get("result", "")
            if result_text and log_file:
                log_file.write(result_text + "\n")
                log_file.flush()

            # Check for rate limit in result text
            if result_text:
                result_lower = result_text.lower()
                if any(s in result_lower for s in ("rate limit", "too many requests", "out of extra usage")):
                    self.telemetry.exit_reason = "rate_limit"

            # Extract usage if present
            usage = msg.get("usage", {})
            if usage:
                self.telemetry.tokens_input = usage.get("input_tokens", 0)
                self.telemetry.tokens_output = usage.get("output_tokens", 0)
                self.telemetry.tokens_cache_read = usage.get("cache_read_input_tokens", 0)
                self.telemetry.tokens_cache_creation = usage.get("cache_creation_input_tokens", 0)

            # Capture CLI's own cost and turn count (ground truth)
            cost = msg.get("total_cost_usd", 0)
            if cost:
                self.telemetry.cost_usd = cost
            turns = msg.get("num_turns", 0)
            if turns:
                self.telemetry.num_turns = turns

        elif msg_type == "error":
            error_text = str(msg.get("error", ""))
            error_lower = error_text.lower()
            if any(s in error_lower for s in ("rate limit", "too many requests", "out of extra usage")):
                self.telemetry.exit_reason = "rate_limit"
            if log_file:
                log_file.write(f"[ERROR] {error_text}\n")
                log_file.flush()

    def _check_behavioral_flags(self, text: str) -> None:
        """Scan text for behavioral patterns."""
        flags = self.telemetry.behavioral_flags
        text_lower = text.lower()

        if re.search(r"option\s+[ab]", text_lower) and "option_menu_detected" not in flags:
            flags.append("option_menu_detected")

        if "escalat" in text_lower and "escalation_detected" not in flags:
            flags.append("escalation_detected")

        if ("not applicable" in text_lower or "skip" in text_lower) and "step_skip_attempt" not in flags:
            flags.append("step_skip_attempt")

    def finalize(self) -> SessionTelemetry:
        """Finalize telemetry after session ends."""
        elapsed = time.monotonic() - self._start_time
        self.telemetry.ended_at = datetime.now(timezone.utc).isoformat()
        self.telemetry.duration_minutes = elapsed / 60.0

        # Idle session check: 0 tool calls after 5+ minutes
        if self.telemetry.tool_calls == 0 and elapsed >= 300:
            if "idle_session" not in self.telemetry.behavioral_flags:
                self.telemetry.behavioral_flags.append("idle_session")

        # Narration heavy: >50% of output blocks are text
        if self._total_output_blocks > 0:
            text_ratio = self._text_tokens_estimate / max(self._text_tokens_estimate + self.telemetry.tool_calls, 1)
            if text_ratio > 0.5 and self.telemetry.tool_calls > 0:
                if "narration_heavy" not in self.telemetry.behavioral_flags:
                    self.telemetry.behavioral_flags.append("narration_heavy")

        self.telemetry.compute_cost()
        return self.telemetry


def save_telemetry(telemetry: SessionTelemetry, base_path: str = "logs/telemetry") -> Path:
    """Save telemetry to JSON file.

    Args:
        telemetry: SessionTelemetry to save.
        base_path: Directory for telemetry files.

    Returns:
        Path to the saved file.
    """
    path = Path(base_path)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / f"{telemetry.session_id}.json"
    with open(filepath, "w") as f:
        json.dump(telemetry.to_dict(), f, indent=2)
    return filepath
