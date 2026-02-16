# CEO Agent Startup Checklist

**Purpose**: Mandatory checklist for CEO agent at the start of EVERY session
**Location**: `/home/akamath/sparky-ai/agents/CEO_STARTUP_CHECKLIST.md`
**Version**: 1.0

---

## Why This Checklist Exists

Starting a session without proper context leads to:
- **Duplicate work** (doing what was already done)
- **Missed messages** (ignoring sub-agent reports)
- **Skipped validations** (proceeding when blocked)
- **Lost context** (not knowing what happened last time)

This checklist ensures you start every session with full awareness.

---

## Mandatory Startup Steps

### ☐ Step 1: Read CEO Inbox

**File**: `/home/akamath/sparky-ai/roadmap/CEO_INBOX.md`

**Purpose**: Check for messages from sub-agents, humans, or system

**Actions**:
```bash
# Read the entire inbox
cat /home/akamath/sparky-ai/roadmap/CEO_INBOX.md

# Or use Read tool
Read(file_path="/home/akamath/sparky-ai/roadmap/CEO_INBOX.md")
```

**What to look for**:
- [ ] Any UNREAD messages in the inbox
- [ ] CRITICAL or HIGH priority messages (address first)
- [ ] Reports from sub-agents (validation, data, research)
- [ ] Directives from humans
- [ ] System notifications

**Process each message**:
1. Read message completely
2. Note action required
3. Prioritize by: CRITICAL > HIGH > MEDIUM > LOW
4. Plan response
5. Mark as "Read" after addressing (or add to todo)

**Red Flags**:
- 🚨 CRITICAL priority message → Address immediately before other work
- 🚨 BLOCKING status in validation directive → Stop, validate first
- 🚨 Multiple unread HIGH priority messages → Catch up before proceeding

---

### ☐ Step 2: Read Task Assignments

**File**: `/home/akamath/sparky-ai/roadmap/TASK_ASSIGNMENTS.md`

**Purpose**: Understand current task status and queue

**Actions**:
```bash
# Read task assignments
cat /home/akamath/sparky-ai/roadmap/TASK_ASSIGNMENTS.md

# Or use Read tool
Read(file_path="/home/akamath/sparky-ai/roadmap/TASK_ASSIGNMENTS.md")
```

**What to check**:
- [ ] **Currently Active** section: Is anyone already working on what you planned?
- [ ] **Queued Tasks** section: What's next in priority order?
- [ ] **Completed Tasks** section: What was done recently?

**Prevent duplicates**:
```python
# Before starting a task, ask yourself:
# 1. Is this already in "Currently Active"? → DON'T duplicate
# 2. Is this in "Completed Tasks" (last 24h)? → Probably already done
# 3. Is this in "Queued Tasks"? → Pull from queue
```

**Red Flags**:
- 🚨 Task you planned to do is already IN PROGRESS → Conflict
- 🚨 No tasks in queue and no clear next step → Need to plan
- 🚨 Task stuck IN PROGRESS for >24h → Stale, may need intervention

---

### ☐ Step 3: Check Validation Directive

**File**: `/home/akamath/sparky-ai/roadmap/VALIDATION_DIRECTIVE.md`

**Purpose**: Check for BLOCKING validation requirements

**Actions**:
```bash
# Read validation directive
cat /home/akamath/sparky-ai/roadmap/VALIDATION_DIRECTIVE.md

# Or use Read tool
Read(file_path="/home/akamath/sparky-ai/roadmap/VALIDATION_DIRECTIVE.md")
```

**What to check**:
- [ ] **Status** field: BLOCKING or PASS?
- [ ] **Priority** field: CRITICAL or HIGH?
- [ ] **Required Validations**: What must be done?

**If status is BLOCKING**:
1. 🛑 **STOP** planned work
2. Read the directive completely
3. Understand what validations are required
4. Execute validations FIRST (before proceeding)
5. Only after validations pass → Resume normal work

**If status is PASS**:
- ✅ Proceed normally
- Keep validation directive in mind for context

**Red Flags**:
- 🚨 Status = BLOCKING → Must address before ANY other work
- 🚨 Validations failed previously → Understand why, fix root cause
- 🚨 Multiple validation failures → Deeper issue, may need pivot

---

### ☐ Step 4: Review Recent Activity Log

**File**: `/home/akamath/sparky-ai/logs/agent_activity/ceo_YYYY-MM-DD.jsonl`

**Purpose**: Understand what happened in the last session

**Actions**:
```bash
# Get today's date
TODAY=$(date +%Y-%m-%d)

# Read last 20 entries from today's log
tail -20 /home/akamath/sparky-ai/logs/agent_activity/ceo_${TODAY}.jsonl

# If today's log doesn't exist, check yesterday
YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)
tail -20 /home/akamath/sparky-ai/logs/agent_activity/ceo_${YESTERDAY}.jsonl
```

**What to look for**:
- [ ] Last task that was worked on
- [ ] Whether last task completed or is still in progress
- [ ] Any experiments run (check results)
- [ ] Any decisions made (understand rationale)
- [ ] Any errors or issues encountered

**Continuity check**:
- If last task was NOT completed → Resume it
- If last task completed → Check queue for next task
- If experiments run → Review results in RESEARCH_LOG.md
- If errors occurred → Resolve before proceeding

**Red Flags**:
- 🚨 Last log entry is "task_started" but no "task_completed" → Incomplete work
- 🚨 Error logs → Resolve issues before continuing
- 🚨 No log entries in 24h → Context might be lost

---

### ☐ Step 5: Read Research Log (Context)

**File**: `/home/akamath/sparky-ai/roadmap/RESEARCH_LOG.md`

**Purpose**: Understand experimental findings and decisions

**Actions**:
```bash
# Read the research log (focus on recent entries)
# Or use Read tool with offset/limit if very long
Read(file_path="/home/akamath/sparky-ai/roadmap/RESEARCH_LOG.md")
```

**What to look for**:
- [ ] Recent experimental results
- [ ] Key findings and insights
- [ ] Open questions or next steps
- [ ] Strategic decisions made

**Use this for**:
- Understanding what's been tried
- Avoiding repeating experiments
- Building on previous findings
- Staying aligned with strategic goals

---

### ☐ Step 6: Update Your Session Plan

Based on Steps 1-5, create a clear plan for this session:

**Template**:
```markdown
## Session Plan: [Date/Time]

**Context**:
- Last session: [What was done]
- Inbox status: [X unread messages, Y critical]
- Active tasks: [None / Task X by Agent Y]
- Blocking issues: [None / Validation directive active]

**Priorities** (in order):
1. [CRITICAL: Address inbox message about...]
2. [HIGH: Complete validation X]
3. [MEDIUM: Continue work on Y]

**This Session Goals**:
- [ ] Goal 1 (estimated time: 30m)
- [ ] Goal 2 (estimated time: 1h)
- [ ] Goal 3 (estimated time: 30m)

**Not Doing This Session**:
- [Lower priority items for later]
```

---

### ☐ Step 7: Before Starting Work

After completing Steps 1-6 and having a clear plan:

**Actions**:
1. **Update TASK_ASSIGNMENTS.md**:
   - Move your task from "Queued" to "Currently Active"
   - Set status to 🔄 IN PROGRESS
   - Record start timestamp
   - Set ETA

2. **Log task_started**:
   ```python
   log_agent_activity(
       agent_id="ceo",
       session_id="[descriptive-session-id]",
       action_type="task_started",
       phase="phase_X",
       task="[task_identifier]",
       description="[Brief description of what you're starting]"
   )
   ```

3. **Now begin work**

---

## Quick Reference Card

Print this for fast reference:

```
╔══════════════════════════════════════════════════════════╗
║         CEO AGENT STARTUP CHECKLIST v1.0                 ║
╠══════════════════════════════════════════════════════════╣
║ At START of EVERY session:                               ║
║                                                           ║
║ ☐ 1. Read CEO_INBOX.md                                   ║
║      → Check for CRITICAL/HIGH priority messages         ║
║      → Address blocking issues FIRST                     ║
║                                                           ║
║ ☐ 2. Read TASK_ASSIGNMENTS.md                            ║
║      → Check for duplicate work                          ║
║      → Review task queue                                 ║
║                                                           ║
║ ☐ 3. Check VALIDATION_DIRECTIVE.md                       ║
║      → If BLOCKING → Stop and validate                   ║
║      → Never skip validation gates                       ║
║                                                           ║
║ ☐ 4. Review recent activity log                          ║
║      → Understand last session's work                    ║
║      → Check for incomplete tasks                        ║
║                                                           ║
║ ☐ 5. Read RESEARCH_LOG.md                                ║
║      → Get experimental context                          ║
║      → Avoid duplicate experiments                       ║
║                                                           ║
║ ☐ 6. Create session plan                                 ║
║      → Prioritize work                                   ║
║      → Set goals                                         ║
║                                                           ║
║ ☐ 7. Update TASK_ASSIGNMENTS.md + log task_started       ║
║      → Mark work IN PROGRESS                             ║
║      → Begin working                                     ║
╚══════════════════════════════════════════════════════════╝
```

---

## Common Scenarios

### Scenario 1: Clean Start

**Situation**: New day, no messages, clear queue

**Checklist Result**:
- ✅ Inbox: No unread messages
- ✅ Tasks: No active conflicts
- ✅ Validation: PASS status
- ✅ Log: Previous work completed
- ✅ Research: Context clear

**Action**: Pull next task from queue, proceed normally

---

### Scenario 2: Validation Blocking

**Situation**: VALIDATION_DIRECTIVE.md has BLOCKING status

**Checklist Result**:
- ⚠️ Inbox: May have message about validation
- ⚠️ Tasks: Validation tasks in queue
- 🛑 Validation: BLOCKING status
- ⚠️ Log: Previous phase completed
- ⚠️ Research: Results need validation

**Action**:
1. 🛑 STOP planned work
2. Read validation directive completely
3. Execute required validations (in order)
4. Report results
5. Only if validations PASS → Proceed

---

### Scenario 3: Sub-Agent Reports

**Situation**: Validation agent completed audit

**Checklist Result**:
- 📬 Inbox: Unread report from validation-sub-agent-001
- ✅ Tasks: Validation task marked completed
- ✅ Validation: May need attention based on report
- ✅ Log: Previous work completed
- ✅ Research: Context clear

**Action**:
1. Read validation report in inbox
2. Review findings (critical, high, medium, low)
3. Evaluate recommendation (PROCEED vs DEBUG)
4. If DEBUG → Create tasks to address issues
5. If PROCEED → Continue with next phase
6. Mark message as read, archive

---

### Scenario 4: Incomplete Work

**Situation**: Last session didn't complete task

**Checklist Result**:
- ✅ Inbox: No urgent messages
- ⚠️ Tasks: Task still IN PROGRESS from yesterday
- ✅ Validation: PASS status
- ⚠️ Log: Last entry is "task_started", no "task_completed"
- ✅ Research: Context clear

**Action**:
1. Review what was attempted
2. Check for any errors or blockers
3. Resume incomplete work
4. Complete the task
5. Log task_completed
6. Update TASK_ASSIGNMENTS.md

---

### Scenario 5: Conflicting Tasks

**Situation**: Task you planned is already active

**Checklist Result**:
- ✅ Inbox: No urgent messages
- 🚨 Tasks: Your planned task is IN PROGRESS by another agent!
- ✅ Validation: PASS status
- ✅ Log: Previous work completed
- ✅ Research: Context clear

**Action**:
1. Check who owns the task
2. Check when they started (timestamp)
3. If stale (>24h) → May be abandoned, investigate
4. If recent → Pick different task from queue
5. Update your session plan

---

## Troubleshooting

### "I forgot to check the inbox"

**Problem**: Started work without reading CEO_INBOX.md

**Fix**:
1. Stop current work
2. Read inbox NOW
3. If critical messages → Pivot to address them
4. If no critical messages → Resume work
5. Mark messages as read
6. Don't forget next time

### "Task is already done"

**Problem**: Started working on task that's in "Completed Tasks"

**Fix**:
1. Stop work immediately
2. Check TASK_ASSIGNMENTS.md completed tasks
3. Verify if truly done (check files, commits)
4. If done → Pick different task
5. If not actually done → Resume
6. Update your understanding

### "Validation directive is blocking"

**Problem**: Missed BLOCKING status, proceeded anyway

**Fix**:
1. Stop current work immediately
2. Read VALIDATION_DIRECTIVE.md completely
3. Understand what's required
4. Execute validations (may invalidate recent work)
5. If validations fail → Recent work may be wasted
6. Learn: Always check directive first

### "No clear next task"

**Problem**: Completed checklist but unclear what to do

**Fix**:
1. Review roadmap phases (`roadmap/phases/*.md`)
2. Check strategic goals
3. Review recent experimental results
4. Identify knowledge gaps or next logical step
5. Create task in TASK_ASSIGNMENTS.md (move to queue)
6. If still unclear → Write to CEO_INBOX.md as question to human

---

## Anti-Patterns

### ❌ Starting work without reading inbox

**Why bad**: Miss critical messages, ignore sub-agent reports

**Do instead**: Always check inbox first (Step 1)

### ❌ Skipping validation directive

**Why bad**: Proceed on invalid results, waste time

**Do instead**: Check directive every session (Step 3)

### ❌ Not updating TASK_ASSIGNMENTS.md

**Why bad**: Other agents don't know what you're doing, duplicate work

**Do instead**: Update before starting work (Step 7)

### ❌ Rushing through checklist

**Why bad**: Miss important context, make mistakes

**Do instead**: Take 5-10 minutes to thoroughly complete checklist

### ❌ Only doing checklist once

**Why bad**: This is NOT one-time, must do EVERY session

**Do instead**: Make it a habit, start every session the same way

---

## Success Metrics

You're using the checklist well if:

- ✅ You NEVER duplicate work
- ✅ You NEVER miss critical inbox messages
- ✅ You NEVER skip validation gates
- ✅ You ALWAYS know what happened in last session
- ✅ You ALWAYS update task status before working
- ✅ You feel confident and oriented at session start

---

## Appendix: File Paths Quick Reference

```
Coordination Files:
  /home/akamath/sparky-ai/roadmap/CEO_INBOX.md
  /home/akamath/sparky-ai/roadmap/TASK_ASSIGNMENTS.md
  /home/akamath/sparky-ai/roadmap/VALIDATION_DIRECTIVE.md

Agent Definitions:
  /home/akamath/sparky-ai/agents/CEO_AGENT.md
  /home/akamath/sparky-ai/agents/COORDINATION_PROTOCOL.md
  /home/akamath/sparky-ai/agents/CEO_STARTUP_CHECKLIST.md  ← You are here

Logs and Context:
  /home/akamath/sparky-ai/logs/agent_activity/ceo_YYYY-MM-DD.jsonl
  /home/akamath/sparky-ai/roadmap/RESEARCH_LOG.md
  /home/akamath/sparky-ai/roadmap/DECISIONS.md

Roadmap:
  /home/akamath/sparky-ai/roadmap/phases/phase_*.md
  /home/akamath/sparky-ai/CLAUDE.md
```

---

## Version History

- v1.0 (2026-02-16): Initial CEO startup checklist
