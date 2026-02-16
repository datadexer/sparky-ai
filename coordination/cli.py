#!/usr/bin/env python3
"""CLI interface for agent coordination.

This is what Claude Code agents use via Bash to interact with the coordination system.

Usage:
    python3 coordination/cli.py startup <agent_id>           # Run startup checklist
    python3 coordination/cli.py inbox <agent_id>             # Check inbox
    python3 coordination/cli.py inbox-read <agent_id>        # Mark all messages read
    python3 coordination/cli.py tasks <agent_id>             # Check my tasks
    python3 coordination/cli.py task-start <task_id>         # Start a task
    python3 coordination/cli.py task-done <task_id>          # Complete a task
    python3 coordination/cli.py send <from> <to> <subject> <body> [priority]  # Send message
    python3 coordination/cli.py register <agent_id> <role>   # Register agent
    python3 coordination/cli.py status                       # Show system status
    python3 coordination/cli.py resources                    # Show resource usage
    python3 coordination/cli.py check-spawn [agent_type]     # Check if can spawn agent
    python3 coordination/cli.py check-duplicates <pattern>   # Check for duplicate work
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from coordination import (
    CoordinationAPI,
    AgentRole,
    TaskPriority,
    MessagePriority,
)


def get_api() -> CoordinationAPI:
    """Get coordination API instance."""
    return CoordinationAPI(PROJECT_ROOT)


def cmd_startup(agent_id: str):
    """Run startup checklist for an agent."""
    api = get_api()
    summary = api.ceo_startup_checklist(agent_id)

    print("=" * 70)
    print(f"STARTUP CHECKLIST — {agent_id.upper()}")
    print("=" * 70)

    # Messages
    print(f"\nUNREAD MESSAGES: {summary['unread_messages']}")
    if summary["messages"]:
        print("-" * 70)
        for msg in summary["messages"]:
            priority_marker = {"critical": "!!!", "high": "!!", "medium": "!", "low": ""}
            marker = priority_marker.get(msg.priority.value, "")
            print(f"\n{marker} From: {msg.from_agent}")
            print(f"  Subject: {msg.subject}")
            print(f"  Priority: {msg.priority.value.upper()}")
            print(f"  Time: {msg.timestamp[:19]}")
            print(f"  ---")
            for line in msg.body.strip().split("\n"):
                print(f"  {line}")
        print()

    # Tasks
    print(f"\nACTIVE TASKS: {summary['active_tasks']}")
    if summary["tasks"]:
        print("-" * 70)
        for task in summary["tasks"]:
            status = "IN PROGRESS" if task.status.value == "in_progress" else "QUEUED"
            print(f"  [{task.priority.value.upper()}] {task.task_id}")
            print(f"    {task.description}")
            print(f"    Status: {status}")
        print()

    print("=" * 70)


def cmd_inbox(agent_id: str):
    """Check inbox for an agent."""
    api = get_api()
    messages = api.get_unread_messages(agent_id)

    if not messages:
        print(f"No unread messages for {agent_id}")
        return

    print(f"INBOX — {agent_id.upper()} ({len(messages)} unread)")
    print("-" * 70)
    for msg in messages:
        print(f"\nFrom: {msg.from_agent}")
        print(f"Subject: {msg.subject}")
        print(f"Priority: {msg.priority.value.upper()}")
        print(f"Time: {msg.timestamp[:19]}")
        print(f"ID: {msg.message_id}")
        print("---")
        print(msg.body.strip())
        print()


def cmd_inbox_read(agent_id: str):
    """Mark all messages as read."""
    api = get_api()
    api.mark_all_messages_read(agent_id)
    print(f"All messages marked as read for {agent_id}")


def cmd_tasks(agent_id: str):
    """List active tasks for an agent."""
    api = get_api()
    tasks = api.get_my_tasks(agent_id)

    if not tasks:
        print(f"No active tasks for {agent_id}")
        return

    print(f"TASKS — {agent_id.upper()} ({len(tasks)} active)")
    print("-" * 70)
    for task in tasks:
        status = "IN PROGRESS" if task.status.value == "in_progress" else "QUEUED"
        print(f"  [{task.priority.value.upper()}] {task.task_id}")
        print(f"    {task.description}")
        print(f"    Status: {status}")
        if task.dependencies:
            print(f"    Depends on: {', '.join(task.dependencies)}")
        print()


def cmd_task_start(task_id: str):
    """Start a task."""
    api = get_api()
    api.start_task(task_id)
    print(f"Task {task_id} started")


def cmd_task_done(task_id: str):
    """Complete a task."""
    api = get_api()
    api.complete_task(task_id)
    print(f"Task {task_id} completed")


def cmd_task_create(task_id: str, description: str, assigned_to: str, priority: str = "medium"):
    """Create a new task."""
    api = get_api()
    p = TaskPriority(priority)
    task = api.create_task(task_id, description, assigned_to, p)
    print(f"Task created: {task.task_id} -> {assigned_to} [{priority.upper()}]")


def cmd_send(from_agent: str, to_agent: str, subject: str, body: str, priority: str = "medium"):
    """Send a message."""
    api = get_api()
    p = MessagePriority(priority)
    msg = api.send_message(from_agent, to_agent, subject, body, p)
    print(f"Message sent: {msg.message_id}")


def cmd_register(agent_id: str, role: str):
    """Register a new agent."""
    api = get_api()
    r = AgentRole(role)
    try:
        agent = api.register_agent(agent_id, r)
        print(f"Agent registered: {agent.agent_id} (role={role})")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_status():
    """Show full system status."""
    api = get_api()

    print("=" * 70)
    print("COORDINATION SYSTEM STATUS")
    print("=" * 70)

    # Active agents
    agents = api.agent_registry.get_active_agents()
    print(f"\nACTIVE AGENTS: {len(agents)}")
    for agent in agents:
        task_info = f" — working on: {agent.current_task}" if agent.current_task else ""
        print(f"  {agent.agent_id} ({agent.role.value}){task_info}")

    # Active tasks
    all_tasks = api.task_manager.get_active_tasks()
    print(f"\nACTIVE TASKS: {len(all_tasks)}")
    for task in all_tasks:
        print(f"  [{task.priority.value.upper()}] {task.task_id} -> {task.assigned_to}: {task.description}")

    # Recent completed
    completed = api.task_manager.get_completed_tasks()[:3]
    if completed:
        print(f"\nRECENTLY COMPLETED:")
        for task in completed:
            print(f"  {task.task_id} ({task.completed_at[:10]})")

    print()


def cmd_export():
    """Export coordination data to readable markdown."""
    api = get_api()
    coord_dir = PROJECT_ROOT / "coordination"
    api.export_task_markdown(coord_dir / "TASK_ASSIGNMENTS.md")
    api.export_inbox_markdown("ceo", coord_dir / "CEO_INBOX.md")
    print("Exported to coordination/TASK_ASSIGNMENTS.md and coordination/CEO_INBOX.md")


def cmd_check_duplicates(pattern: str):
    """Check for duplicate work."""
    api = get_api()
    dupes = api.check_duplicate_work(pattern)
    if dupes:
        print(f"WARNING: Found {len(dupes)} potentially duplicate tasks:")
        for d in dupes:
            print(f"  - {d.task_id}: {d.description} (assigned to {d.assigned_to})")
    else:
        print("No duplicates found")


def cmd_resources():
    """Show system resource status."""
    api = get_api()
    status = api.get_resource_status()

    if not status:
        print("Resource manager not available")
        return

    print("=" * 70)
    print("SYSTEM RESOURCES")
    print("=" * 70)

    print(f"\nCPU Usage: {status.cpu_percent:.1f}%")
    print(f"Memory Usage: {status.memory_percent:.1f}%")
    print(f"Disk Free: {status.disk_free_gb:.1f} GB")

    print(f"\nActive Agents: {status.active_agents}")
    if status.agents_by_type:
        for agent_type, count in status.agents_by_type.items():
            print(f"  - {agent_type}: {count}")

    if status.circuit_breaker_open:
        print("\n⚠️  CIRCUIT BREAKER OPEN - No new agents allowed")

    if status.under_pressure:
        print("\n⚠️  SYSTEM UNDER PRESSURE - Reduced concurrency in effect")

    if status.warnings:
        print("\nWARNINGS:")
        for warning in status.warnings:
            print(f"  ⚠️  {warning}")

    print()


def cmd_check_spawn(agent_type: str = "general"):
    """Check if resources allow spawning a new agent."""
    api = get_api()
    can_spawn, reason = api.check_can_spawn_agent(agent_type)

    if can_spawn:
        print(f"✓ Can spawn {agent_type} agent: {reason}")
    else:
        print(f"✗ Cannot spawn {agent_type} agent: {reason}")
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    try:
        if cmd == "startup" and len(sys.argv) >= 3:
            cmd_startup(sys.argv[2])
        elif cmd == "inbox" and len(sys.argv) >= 3:
            cmd_inbox(sys.argv[2])
        elif cmd == "inbox-read" and len(sys.argv) >= 3:
            cmd_inbox_read(sys.argv[2])
        elif cmd == "tasks" and len(sys.argv) >= 3:
            cmd_tasks(sys.argv[2])
        elif cmd == "task-start" and len(sys.argv) >= 3:
            cmd_task_start(sys.argv[2])
        elif cmd == "task-done" and len(sys.argv) >= 3:
            cmd_task_done(sys.argv[2])
        elif cmd == "task-create" and len(sys.argv) >= 5:
            priority = sys.argv[5] if len(sys.argv) > 5 else "medium"
            cmd_task_create(sys.argv[2], sys.argv[3], sys.argv[4], priority)
        elif cmd == "send" and len(sys.argv) >= 6:
            priority = sys.argv[6] if len(sys.argv) > 6 else "medium"
            cmd_send(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], priority)
        elif cmd == "register" and len(sys.argv) >= 4:
            cmd_register(sys.argv[2], sys.argv[3])
        elif cmd == "status":
            cmd_status()
        elif cmd == "export":
            cmd_export()
        elif cmd == "check-duplicates" and len(sys.argv) >= 3:
            cmd_check_duplicates(sys.argv[2])
        elif cmd == "resources":
            cmd_resources()
        elif cmd == "check-spawn":
            agent_type = sys.argv[2] if len(sys.argv) >= 3 else "general"
            cmd_check_spawn(agent_type)
        else:
            print(f"Unknown command or missing args: {cmd}")
            print(__doc__)
            sys.exit(1)
    except (KeyError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
