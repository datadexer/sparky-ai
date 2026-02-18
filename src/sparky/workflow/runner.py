"""CLI entry point for running workflows and research directives.

Usage:
    python -m sparky.workflow.runner <workflow_file.py>    # workflow mode
    python -m sparky.workflow.runner <directive.yaml>      # orchestrator mode

Workflow files must define a `build_workflow() -> Workflow` function.
YAML files are loaded as ResearchDirective and run via ResearchOrchestrator.
"""

import importlib.util
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_workflow(filepath: str):
    """Dynamically load a workflow from a Python file.

    The file must define `build_workflow() -> Workflow`.
    """
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Workflow file not found: {path}")

    spec = importlib.util.spec_from_file_location("workflow_module", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "build_workflow"):
        raise AttributeError(f"Workflow file {path} must define a build_workflow() function")

    return module.build_workflow()


def _is_yaml(path: str) -> bool:
    """Check if the path ends with .yaml or .yml."""
    return path.endswith((".yaml", ".yml"))


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m sparky.workflow.runner <workflow.py | directive.yaml>")
        sys.exit(2)

    filepath = sys.argv[1]

    if _is_yaml(filepath):
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
    else:
        logger.info(f"Loading workflow from {filepath}")
        try:
            workflow = load_workflow(filepath)
        except Exception as e:
            logger.error(f"Failed to load workflow: {e}")
            sys.exit(2)

        logger.info(f"Running workflow '{workflow.name}' ({len(workflow.steps)} steps)")
        exit_code = workflow.run()
        logger.info(f"Workflow exited with code {exit_code}")
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
