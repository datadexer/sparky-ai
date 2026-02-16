# Multi-Agent Coordination System

A Python API for coordinating multiple AI agents working on the Sparky AI project.

## Quick Start

```python
from pathlib import Path
from coordination import CoordinationAPI, AgentRole, TaskPriority, MessagePriority

# Initialize API (do this once at project root)
api = CoordinationAPI(Path.cwd())

# Register CEO agent
api.register_agent("ceo", AgentRole.CEO)

# CEO startup checklist (run at start of every session)
summary = api.ceo_startup_checklist("ceo")
print(f"Unread messages: {summary['unread_messages']}")
print(f"Active tasks: {summary['active_tasks']}")
```

## Features

- **Task Management**: Create, assign, and track tasks to prevent duplicate work
- **Agent Messaging**: Send messages between agents with priority levels
- **Agent Registry**: Track active agents and enforce constraints (e.g., only one CEO)
- **Markdown Export**: Export tasks and inbox to human-readable markdown files
- **Persistence**: All data saved to JSON files and persists across sessions

## Installation

No installation needed - just import from the `coordination` package.

## Usage Examples

### CEO Agent Workflow

```python
from coordination import CoordinationAPI, AgentRole

api = CoordinationAPI(Path("/home/akamath/sparky-ai"))

# At session start
api.register_agent("ceo", AgentRole.CEO)
summary = api.ceo_startup_checklist("ceo")

# Check for messages
for msg in summary['messages']:
    print(f"From {msg.from_agent}: {msg.subject}")
    print(msg.body)
    api.mark_message_read(msg.message_id)

# Check for tasks
for task in summary['tasks']:
    print(f"Task: {task.description}")

# Start a task
api.start_task("data-expansion")

# ... do work ...

# Complete task
api.complete_task("data-expansion")

# Update activity periodically
api.update_agent_activity("ceo", "Currently working on data expansion")
```

### Sub-Agent Workflow

```python
from coordination import CoordinationAPI, AgentRole, MessagePriority

api = CoordinationAPI(Path("/home/akamath/sparky-ai"))

# Register sub-agent
api.register_agent("validation-001", AgentRole.VALIDATION)

# Get assigned task
tasks = api.get_my_tasks("validation-001")
task = tasks[0]

# Start task
api.start_task(task.task_id)

# ... do work ...

# Send report and terminate
api.subagent_report_and_terminate(
    agent_id="validation-001",
    report_subject="Audit Complete",
    report_body="""
    Completed audit of PHASE_3_VALIDATION_SUMMARY.md

    Found 10 issues:
    - 3 CRITICAL
    - 4 HIGH
    - 3 MEDIUM

    See attached report for details.
    """
)
```

### Creating Tasks

```python
# Create high-priority task
api.create_task(
    task_id="data-expansion-hourly",
    description="Fetch hourly BTC data from 2019-2025",
    assigned_to="ceo",
    priority=TaskPriority.HIGH
)

# Check for duplicate work before starting
duplicates = api.check_duplicate_work("hourly BTC data")
if duplicates:
    print(f"Warning: Similar work already in progress: {duplicates[0].task_id}")
```

### Sending Messages

```python
# Send high-priority message to CEO
api.send_message(
    from_agent="validation-001",
    to_agent="ceo",
    subject="CRITICAL: Data leakage detected in new features",
    body="Shuffled-label test failed with 87% accuracy. New features leak future data.",
    priority=MessagePriority.CRITICAL
)
```

### Export to Markdown

```python
# Export tasks for human reading
api.export_task_markdown(Path("coordination/TASK_ASSIGNMENTS.md"))

# Export CEO inbox
api.export_inbox_markdown("ceo", Path("coordination/CEO_INBOX.md"))
```

## File Structure

```
sparky-ai/
├── coordination/
│   ├── __init__.py
│   ├── coordination_api.py (main API)
│   ├── task_manager.py
│   ├── inbox_manager.py
│   ├── agent_registry.py
│   ├── README.md (this file)
│   └── data/
│       ├── tasks.json (task database)
│       ├── inbox.json (message database)
│       └── agents.json (agent registry)
├── tests/
│   └── test_coordination_api.py (comprehensive tests)
└── agents/
    ├── CEO_AGENT.md (agent documentation)
    └── VALIDATION_AGENT.md
```

## API Reference

### CoordinationAPI

Main interface for all coordination operations.

#### Agent Lifecycle
- `register_agent(agent_id, role, metadata=None)` - Register new agent
- `update_agent_activity(agent_id, current_task=None)` - Update activity timestamp
- `terminate_agent(agent_id)` - Mark agent as terminated
- `get_ceo()` - Get active CEO agent info

#### Task Management
- `create_task(task_id, description, assigned_to, priority, dependencies)` - Create task
- `start_task(task_id)` - Mark task as IN_PROGRESS
- `complete_task(task_id)` - Mark task as COMPLETED
- `get_my_tasks(agent_id)` - Get agent's active tasks
- `check_duplicate_work(description_pattern)` - Check for similar work

#### Messaging
- `send_message(from_agent, to_agent, subject, body, priority)` - Send message
- `get_unread_messages(agent_id)` - Get unread messages (sorted by priority)
- `mark_message_read(message_id)` - Mark message as read
- `mark_all_messages_read(agent_id)` - Mark all messages read

#### Export
- `export_task_markdown(output_file)` - Export tasks to markdown
- `export_inbox_markdown(agent_id, output_file)` - Export inbox to markdown

#### High-Level Workflows
- `ceo_startup_checklist(ceo_id)` - Execute CEO startup checklist
- `subagent_report_and_terminate(agent_id, subject, body)` - Report and terminate

## Enums

### AgentRole
- `CEO` - Primary orchestrator (only one allowed)
- `VALIDATION` - Validation sub-agent
- `DATA_ENGINEER` - Data engineering sub-agent
- `RESEARCH` - Research sub-agent

### TaskStatus
- `QUEUED` - Task created but not started
- `IN_PROGRESS` - Task being worked on
- `COMPLETED` - Task finished
- `BLOCKED` - Task blocked by dependency
- `CANCELLED` - Task cancelled

### TaskPriority / MessagePriority
- `CRITICAL` - Urgent, blocks other work
- `HIGH` - Important, should do soon
- `MEDIUM` - Normal priority
- `LOW` - Nice to have

## Testing

Run tests with pytest:

```bash
cd /home/akamath/sparky-ai
pytest tests/test_coordination_api.py -v
```

All tests should pass. Tests cover:
- Agent lifecycle and registration
- Task creation and lifecycle
- Messaging and priority sorting
- Duplicate work detection
- Data persistence
- Edge cases and error handling

## Design Principles

1. **Single Source of Truth**: All coordination data in JSON files
2. **Type Safety**: Uses dataclasses and enums for structure
3. **Persistence**: Data survives across agent sessions
4. **Human Readable**: Export to markdown for oversight
5. **Testable**: Comprehensive test coverage
6. **Simple API**: One main interface (`CoordinationAPI`)

## Integration with Existing System

The Python API replaces the markdown-only coordination files:

**Old (markdown-only)**:
- `roadmap/90_CEO_INBOX.md` - Manual editing
- `roadmap/91_TASK_ASSIGNMENTS.md` - Manual editing

**New (Python API)**:
- `coordination/data/inbox.json` - Programmatic access
- `coordination/data/tasks.json` - Programmatic access
- Export to markdown for human reading

Agents should use the Python API for all operations, then export to markdown for human oversight.
