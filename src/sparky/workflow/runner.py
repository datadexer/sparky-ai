"""CLI entry point for running workflows.

Usage:
    python -m sparky.workflow.runner <workflow_file.py>

The workflow file must define a `build_workflow() -> Workflow` function.
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
        raise AttributeError(
            f"Workflow file {path} must define a build_workflow() function"
        )

    return module.build_workflow()


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m sparky.workflow.runner <workflow_file.py>")
        sys.exit(2)

    workflow_path = sys.argv[1]
    logger.info(f"Loading workflow from {workflow_path}")

    try:
        workflow = load_workflow(workflow_path)
    except Exception as e:
        logger.error(f"Failed to load workflow: {e}")
        sys.exit(2)

    logger.info(f"Running workflow '{workflow.name}' ({len(workflow.steps)} steps)")
    exit_code = workflow.run()
    logger.info(f"Workflow exited with code {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
