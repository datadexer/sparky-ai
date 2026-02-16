# CEO Agent Inbox

**Purpose**: Central communication hub for messages to the CEO Agent
**Location**: `/home/akamath/sparky-ai/roadmap/CEO_INBOX.md`

---

## 🔔 CRITICAL: Read this file at START of EVERY session

The CEO agent MUST check this inbox at the beginning of each session to:
- Receive reports from sub-agents
- Get alerts about blocking issues
- Review recommendations for next steps
- Stay informed of system state

---

## Unread Messages

Messages that require CEO attention and action.

### [2026-02-16 04:00] From: SYSTEM
**Subject**: Multi-Agent Coordination System Initialized
**Priority**: HIGH
**Type**: System Notification

**Message**:
The multi-agent coordination system has been created and is now active. Key files:

- `/home/akamath/sparky-ai/agents/CEO_AGENT.md` - Your role and responsibilities
- `/home/akamath/sparky-ai/agents/COORDINATION_PROTOCOL.md` - How agents coordinate
- `/home/akamath/sparky-ai/roadmap/TASK_ASSIGNMENTS.md` - Task tracking
- `/home/akamath/sparky-ai/roadmap/CEO_INBOX.md` - This inbox
- `/home/akamath/sparky-ai/agents/CEO_STARTUP_CHECKLIST.md` - Session startup steps

**Action Required**:
1. Read CEO_STARTUP_CHECKLIST.md before starting any work
2. Follow coordination protocol when spawning sub-agents
3. Update TASK_ASSIGNMENTS.md for all work

**Signed**: System
**Read**: [ ] Unread | [ ] Read

---

### [2026-02-16 01:40] From: HUMAN (via VALIDATION_DIRECTIVE.md)
**Subject**: BLOCKING Validation Required Before Phase 4
**Priority**: CRITICAL
**Type**: Directive

**Message**:
Phase 3 results showing Sharpe 0.999 on 30d horizon are suspicious and require validation BEFORE proceeding to multi-seed validation.

**Concerns**:
1. Sharpe 0.999 is unrealistic for crypto with transaction costs
2. 30d horizon previously FAILED leakage test, now shows best performance
3. Removing returns_1d IMPROVED results (contradictory)
4. Technical-only beats all features (contradicts project hypothesis)

**Required Actions** (in order):
1. **PRIORITY 1**: Run holdout test (2025-10-01 to 2025-12-31)
   - If holdout Sharpe ~= 0.999 → Maybe real
   - If holdout Sharpe << 0.999 → Overfitting confirmed
2. **PRIORITY 2**: Re-run leakage detector on technical-only, 30d
3. **PRIORITY 3**: Run sanity checks (baseline, trades, features)

**Status**: 🛑 BLOCKING - Do NOT proceed to Phase 4 until complete

**See Also**: `/home/akamath/sparky-ai/roadmap/VALIDATION_DIRECTIVE.md`

**Action Required**: Execute validations and report results

**Signed**: Human
**Read**: [ ] Unread | [ ] Read

---

## Read Messages

Messages that have been addressed. Archive for reference.

### [Example] [2026-02-15 12:00] From: validation-sub-agent-000
**Subject**: Example Audit Report
**Priority**: HIGH
**Type**: Audit Report

**Message**:
This is an example of what a validation agent report looks like. Real reports will appear in "Unread Messages" section above.

**Action Taken**: Example acknowledged
**Archived**: 2026-02-16 04:00 UTC

---

## Message Template

When sub-agents or humans send messages, use this format:

```markdown
### [YYYY-MM-DD HH:MM] From: [agent-id or HUMAN or SYSTEM]
**Subject**: [Brief subject line]
**Priority**: CRITICAL | HIGH | MEDIUM | LOW
**Type**: Report | Alert | Directive | Question | Recommendation

**Message**:
[Detailed message content]

**Action Required**:
[Specific actions the CEO should take]

**See Also**: [Related files or references]

**Signed**: [agent-id or name]
**Read**: [ ] Unread | [ ] Read
```

---

## How to Use This Inbox

### For CEO Agent

**At Session Start**:
1. Read all unread messages (top to bottom)
2. Prioritize by: CRITICAL > HIGH > MEDIUM > LOW
3. For each message:
   - Read content thoroughly
   - Note action required
   - Plan response
4. After reading, mark as read (check the box)
5. Move addressed messages to "Read Messages" section

**During Session**:
- Check inbox periodically for new messages
- Sub-agents may add messages while you work

**Before Session End**:
- Ensure all unread messages are addressed or have plan
- Don't leave CRITICAL or HIGH priority unread

### For Sub-Agents

**Sending a Report**:
1. Use the message template above
2. Add to "Unread Messages" section (at the top)
3. Set appropriate priority
4. Be specific about action required
5. Sign with your agent ID

**Priority Guidelines**:
- CRITICAL: Blocks all progress, immediate action required
- HIGH: Important, should address within current session
- MEDIUM: Should address soon, but not urgent
- LOW: FYI, address when convenient

### For Humans

**Sending Directives**:
1. Use message template
2. Be explicit about what's required
3. Set clear deadlines if time-sensitive
4. Reference supporting documents

---

## Message Types

### Report
Sub-agent completed work and is reporting results.
- Audit reports from validation agents
- Data quality reports from data engineers
- Research findings from research agents

### Alert
Something requires immediate attention.
- Validation failures
- Critical errors
- System issues

### Directive
Human or system instruction to CEO.
- Strategic direction changes
- New requirements
- Blocking conditions

### Question
Needs CEO decision or clarification.
- Ambiguous requirements
- Trade-off decisions
- Resource allocation

### Recommendation
Suggestion from sub-agent or human.
- Process improvements
- Next steps
- Alternative approaches

---

## Inbox Maintenance

### Archiving Messages

When a message is fully addressed:
1. Mark as "Read"
2. Add "Action Taken" note
3. Move to "Read Messages" section
4. Add "Archived" timestamp

### Cleaning Old Messages

Periodically (weekly):
- Archive read messages older than 7 days
- Keep critical decision records longer
- Move very old messages to separate archive file

### Inbox Zero

Goal: No unread CRITICAL or HIGH priority messages at end of session
- All blocking issues addressed
- Clear plan for non-blocking issues
- Next session knows what to prioritize

---

## Example Workflow

```
Session Start:
1. Open CEO_INBOX.md
2. See 2 unread messages: 1 CRITICAL, 1 HIGH
3. CRITICAL: Validation directive - need to run holdout test
4. HIGH: System notification - new coordination system
5. Plan: Address CRITICAL first, then HIGH

During Session:
6. Run holdout test (addressing CRITICAL message)
7. Mark CRITICAL as read, move to archived
8. Read system notification (HIGH message)
9. Mark HIGH as read, move to archived

Session End:
10. Inbox zero ✅
11. All messages addressed
12. Ready for next session
```

---

## Notes

- This inbox is the **primary way** for agents to communicate with CEO
- CEO should check inbox **at start of every session** (non-negotiable)
- Sub-agents should write clear, actionable reports
- Humans can use this to send directives or alerts
- Keep inbox clean (move read messages to archive)

---

## Version History

- v1.0 (2026-02-16): Initial CEO inbox created with example messages
