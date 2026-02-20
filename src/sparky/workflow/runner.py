"""CLI entry point for running research directives via the orchestrator.

Usage:
    python -m sparky.workflow.runner <phase.yaml>
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m sparky.workflow.runner <phase.yaml>")
        sys.exit(2)

    filepath = sys.argv[1]

    from sparky.workflow.orchestrator import ResearchDirective, ResearchOrchestrator

    logger.info(f"Loading directive from {filepath}")
    try:
        directive = ResearchDirective.from_yaml(filepath)
    except Exception as e:
        logger.error(f"Failed to load directive: {e}")
        sys.exit(2)

    logger.info(f"Running orchestrator '{directive.name}'")
    orch = ResearchOrchestrator(directive)
    exit_code = orch.run()
    logger.info(f"Orchestrator exited with code {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
