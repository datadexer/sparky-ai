# Multi-Agent System Quick Start Guide

**For**: New users of the Sparky AI multi-agent coordination system
**Time**: 5-minute read
**Goal**: Get started immediately

---

## What is this system?

A coordination framework that allows multiple AI agents to work together on the Sparky AI ML trading project without conflicts, duplicate work, or lost context.

**Key Innovation**: File-based communication + clear ownership + full audit trail

---

## The 3-Minute Quick Start

### If you are the CEO Agent

**Every session, do this**:

```bash
# 1. Read your startup checklist
cat /home/akamath/sparky-ai/agents/CEO_STARTUP_CHECKLIST.md

# 2. Check your inbox (messages from sub-agents)
cat /home/akamath/sparky-ai/roadmap/CEO_INBOX.md

# 3. Check task assignments (prevent duplicate work)
cat /home/akamath/sparky-ai/roadmap/TASK_ASSIGNMENTS.md

# 4. Check validation directive (any blocking requirements?)
cat /home/akamath/sparky-ai/roadmap/VALIDATION_DIRECTIVE.md

# 5. NOW you can start working
```

**That's it!** These 4 files keep you coordinated.

### If you are a Sub-Agent

**You were spawned to do a specific task**:

```bash
# 1. Read your task assignment
cat /home/akamath/sparky-ai/roadmap/TASK_ASSIGNMENTS.md
# Find your agent ID, read what you need to do

# 2. Update status to IN PROGRESS
# Edit TASK_ASSIGNMENTS.md

# 3. Do the work (as specified in your assignment)
# Stay in scope, don't deviate

# 4. Write report to CEO
# Edit CEO_INBOX.md, add your findings

# 5. Update status to COMPLETED
# Edit TASK_ASSIGNMENTS.md

# 6. Terminate (you're done!)
```

---

## The 5 Core Files

| File | Purpose | Who Reads | Who Writes |
|------|---------|-----------|------------|
| `agents/CEO_STARTUP_CHECKLIST.md` | Session startup steps | CEO | System |
| `roadmap/CEO_INBOX.md` | Messages to CEO | CEO | Sub-agents, humans |
| `roadmap/TASK_ASSIGNMENTS.md` | Who's doing what | All agents | All agents |
| `roadmap/VALIDATION_DIRECTIVE.md` | Blocking requirements | CEO | Humans |
| `logs/agent_activity/*.jsonl` | Audit trail | All agents | All agents |

**Memory aid**:
- **Inbox** = incoming messages
- **Assignments** = current work
- **Directive** = must-dos
- **Logs** = history

---

## Common Commands

### For CEO Agent

```bash
# Session start (do this FIRST every time)
cat /home/akamath/sparky-ai/agents/CEO_STARTUP_CHECKLIST.md

# Check inbox
cat /home/akamath/sparky-ai/roadmap/CEO_INBOX.md

# Check tasks
cat /home/akamath/sparky-ai/roadmap/TASK_ASSIGNMENTS.md

# Check validations
cat /home/akamath/sparky-ai/roadmap/VALIDATION_DIRECTIVE.md

# Check recent activity
tail -20 /home/akamath/sparky-ai/logs/agent_activity/ceo_$(date +%Y-%m-%d).jsonl

# Read research log
cat /home/akamath/sparky-ai/roadmap/RESEARCH_LOG.md
```

### For Sub-Agents

```bash
# Read your assignment
cat /home/akamath/sparky-ai/roadmap/TASK_ASSIGNMENTS.md

# Read your agent definition (understand your role)
cat /home/akamath/sparky-ai/agents/VALIDATION_AGENT.md
# or
cat /home/akamath/sparky-ai/agents/DATA_ENGINEER_AGENT.md
```

---

## Decision Tree: What should I do?

```
START
  │
  ├─ Are you CEO?
  │   │
  │   YES → Read CEO_STARTUP_CHECKLIST.md
  │         Follow all 7 steps
  │         │
  │         ├─ Any CRITICAL inbox messages?
  │         │   │
  │         │   YES → Address those FIRST
  │         │   NO → Continue
  │         │
  │         ├─ Is VALIDATION_DIRECTIVE.md BLOCKING?
  │         │   │
  │         │   YES → Run validations BEFORE other work
  │         │   NO → Continue
  │         │
  │         ├─ Any tasks IN PROGRESS by you?
  │         │   │
  │         │   YES → Resume that work
  │         │   NO → Pull next from queue
  │         │
  │         └─ START WORK
  │               ├─ Update TASK_ASSIGNMENTS.md (IN PROGRESS)
  │               ├─ Log task_started
  │               ├─ Do the work
  │               ├─ Log task_completed
  │               └─ Update TASK_ASSIGNMENTS.md (COMPLETED)
  │
  └─ Are you a sub-agent?
      │
      YES → Read your agent definition file
            │
            ├─ Read TASK_ASSIGNMENTS.md
            ├─ Find your task (by agent ID)
            ├─ Update status to IN PROGRESS
            ├─ Execute work (stay in scope)
            ├─ Write report to CEO_INBOX.md
            ├─ Update status to COMPLETED
            └─ TERMINATE
```

---

## Examples

### Example 1: CEO Starting a New Session

```
Session: Monday morning, new week of work

1. Read CEO_STARTUP_CHECKLIST.md
   → See I need to check inbox, tasks, validations

2. Read CEO_INBOX.md
   → 1 unread message from validation-sub-agent-001
   → Report says: "Found 5 issues, recommend DEBUG"
   → Priority: HIGH
   → OK, noted

3. Read TASK_ASSIGNMENTS.md
   → Currently Active: None
   → Queued: "Address validation findings" (HIGH priority)
   → Completed: validation-sub-agent-001 completed audit

4. Read VALIDATION_DIRECTIVE.md
   → Status: BLOCKING
   → Must run holdout test before proceeding
   → OK, this is CRITICAL

5. Review activity log
   → Last session: Completed Phase 3 experiments
   → Results suspicious (Sharpe 0.999)
   → Spawned validation agent

6. Create session plan
   Priority 1: Run holdout test (BLOCKING)
   Priority 2: Address validation findings
   Priority 3: If validations pass, continue to multi-seed

7. Update TASK_ASSIGNMENTS.md
   Move "Run holdout test" to Currently Active
   Status: IN PROGRESS

8. Log task_started

9. NOW start work (execute holdout test)
```

### Example 2: CEO Spawning a Sub-Agent

```
Situation: Phase 3 results look suspicious, need independent audit

1. Decide to spawn validation agent

2. Create task in TASK_ASSIGNMENTS.md:

   ### Task: Audit Phase 3 Results

   **Priority**: HIGH
   **Assigned To**: validation-sub-agent-001
   **Created**: 2026-02-16 10:00 UTC
   **Deadline**: 2026-02-16 12:00 UTC

   **Objective**:
   Audit PHASE_3_VALIDATION_SUMMARY.md for errors

   **Deliverables**:
   1. Audit report in CEO_INBOX.md
   2. Issues list (prioritized)
   3. Recommendation: PROCEED or DEBUG

   **Pass Criteria**:
   - All results verified OR issues documented

   **Resources**:
   - /home/akamath/sparky-ai/roadmap/PHASE_3_VALIDATION_SUMMARY.md

3. Spawn agent (tell them their ID: validation-sub-agent-001)

4. Agent does work...

5. Later, check CEO_INBOX.md
   → New message from validation-sub-agent-001
   → Read report
   → Found 5 issues
   → Recommendation: DEBUG

6. Act on findings
   → Create tasks to address issues
```

### Example 3: Validation Agent Lifecycle

```
Agent ID: validation-sub-agent-001

1. SPAWN (by CEO)

2. Read TASK_ASSIGNMENTS.md
   → Find task assigned to validation-sub-agent-001
   → Objective: Audit Phase 3 results
   → Deliverable: Report to CEO_INBOX.md

3. Update status to IN PROGRESS

4. Execute work:
   - Read PHASE_3_VALIDATION_SUMMARY.md
   - Review code in scripts/
   - Check for leakage, overfitting, bugs
   - Found 5 issues:
     * 3 CRITICAL (unrealistic Sharpe)
     * 2 HIGH (missing validations)

5. Write report to CEO_INBOX.md:

   ### [2026-02-16 11:30] From: validation-sub-agent-001
   **Subject**: Phase 3 Audit Complete
   **Priority**: HIGH

   **Summary**: Found 5 issues requiring attention

   **Critical Issues**:
   1. Sharpe 0.999 is unrealistic
   2. 30d horizon previously failed leakage test
   3. Removing features improved results (backwards)

   **Recommendation**: DEBUG before proceeding

   **Signed**: validation-sub-agent-001

6. Update TASK_ASSIGNMENTS.md to COMPLETED

7. TERMINATE (job done)
```

---

## Cheat Sheet

### CEO Agent Must-Dos

```
✅ Read CEO_STARTUP_CHECKLIST.md (EVERY session)
✅ Check CEO_INBOX.md (for messages)
✅ Check TASK_ASSIGNMENTS.md (prevent duplicates)
✅ Check VALIDATION_DIRECTIVE.md (blocking requirements)
✅ Update task status BEFORE and AFTER work
✅ Log all activities
✅ Never skip BLOCKING validations
```

### Sub-Agent Must-Dos

```
✅ Read TASK_ASSIGNMENTS.md (find your task)
✅ Update status to IN PROGRESS
✅ Stay in scope (don't deviate)
✅ Write report to CEO_INBOX.md
✅ Update status to COMPLETED
✅ Terminate after delivery
```

### Red Flags

```
🚨 CEO starts work without reading inbox → STOP, read inbox
🚨 Task already IN PROGRESS by another agent → STOP, pick different task
🚨 VALIDATION_DIRECTIVE.md status = BLOCKING → STOP, run validations
🚨 Sub-agent deviates from assigned scope → STOP, stay in scope
🚨 Work not logged in activity logs → STOP, log everything
```

---

## FAQ

**Q: How do I know if I'm CEO or a sub-agent?**

A: If you're reading this to start a session, you're probably CEO. Sub-agents are explicitly spawned by CEO with a specific agent ID (e.g., "validation-sub-agent-001").

**Q: What if I forget to check the inbox?**

A: Stop immediately, read it now. Missing critical messages can waste hours of work.

**Q: What if a task is already IN PROGRESS?**

A: Don't duplicate it. Pick a different task from the queue, or investigate if the task is stale (>24h).

**Q: What if VALIDATION_DIRECTIVE.md says BLOCKING?**

A: Stop everything else. Run the required validations FIRST. Never skip blocking gates.

**Q: Where do I log my activities?**

A: `/home/akamath/sparky-ai/logs/agent_activity/{agent_type}_YYYY-MM-DD.jsonl`

**Q: How do I communicate with other agents?**

A: You don't (directly). All communication goes through files:
- Sub-agents → CEO: Write to CEO_INBOX.md
- CEO → Sub-agents: Write to TASK_ASSIGNMENTS.md

**Q: What if I'm stuck or confused?**

A: Write a message to CEO_INBOX.md with your question. Mark it HIGH or CRITICAL priority.

---

## Next Steps

Now that you understand the basics:

1. **If you're CEO**: Read `/home/akamath/sparky-ai/agents/CEO_STARTUP_CHECKLIST.md` and follow it

2. **If you're a sub-agent**: Read your agent definition:
   - Validation: `/home/akamath/sparky-ai/agents/VALIDATION_AGENT.md`
   - Data Engineer: `/home/akamath/sparky-ai/agents/DATA_ENGINEER_AGENT.md`

3. **For deeper understanding**: Read `/home/akamath/sparky-ai/agents/COORDINATION_PROTOCOL.md`

4. **For system overview**: Read `/home/akamath/sparky-ai/agents/README.md`

---

## Remember

**The golden rules**:

1. **CEO**: Read startup checklist EVERY session
2. **All agents**: Check TASK_ASSIGNMENTS.md BEFORE starting work
3. **All agents**: Log EVERYTHING
4. **CEO**: Never skip BLOCKING validations
5. **Sub-agents**: Stay in scope, report clearly

**Follow these rules and coordination is effortless.**

---

Good luck! The system is designed to be intuitive and prevent common mistakes. Trust the process.
