#!/bin/bash
# Enable GitHub branch protection on main.
# Run once by AK: bash scripts/setup_branch_protection.sh
# Requires: gh CLI authenticated (gh auth login)

set -euo pipefail

REPO="datadexer/sparky-ai"

echo "Setting branch protection on main for $REPO..."

gh api "repos/$REPO/branches/main/protection" \
  --method PUT \
  --input - <<'EOF'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["quality-gate", "research-validation"]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 0
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF

echo ""
echo "Branch protection enabled on main:"
echo "  - CI (quality-gate) must pass before merge"
echo "  - Research validation (Sonnet agent) must pass before merge"
echo "  - Branch must be up-to-date with main"
echo "  - No force pushes to main"
echo "  - No direct pushes (PRs required)"
echo "  - No human review required (agents can self-merge if CI passes)"
echo ""
echo "Agents can still create PRs and merge them IF CI passes."
echo "They CANNOT push directly to main or bypass CI."
