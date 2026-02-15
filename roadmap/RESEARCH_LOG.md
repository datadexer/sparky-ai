# RESEARCH LOG — Sparky AI

Running log of all findings, experiments, and insights.
Newest entries at the top.

---

## Academic Literature Review (Pre-bootstrap)

### Multi-Agent LLM Trading Frameworks
The field of LLM-based trading is rapidly maturing. Key papers and findings:

**TradingAgents (arXiv:2412.20138, Xiao et al. 2024)**
- Multi-agent framework mimicking trading firm structure
- Bull vs Bear researcher DEBATE mechanism produces balanced market assessment
- Relevance: debate mechanism interesting for Phase 5 hypothesis generation

**QuantAgent (arXiv:2402.03755, Wang et al. 2024)**
- Inner-loop/outer-loop architecture for autonomous alpha factor mining
- Relevance: Phase 5 research loop adopts this inner/outer pattern

**AlphaAgent (Tang et al. 2025)**
- Multi-agent system with hard-coded constraints to enforce originality
- Relevance: confirms our approach of constraints/guardrails over raw model power

**AI-Trader Benchmark (arXiv:2512.10971, Fan et al. 2025)**
- CRITICAL: "General intelligence does not automatically translate to effective trading capability"
- Relevance: validates domain-specific quantitative models over LLM-as-trader

**Alpha Arena Live Competition (nof1.ai, Oct-Nov 2025)**
- Only 2 of 6 frontier LLMs beat buy-and-hold BTC
- Risk management differentiates winners from losers
- MAJOR VALIDATION of our approach

**QuantaAlpha (arXiv:2602.07085, 2026)**
- Evolutionary alpha mining with trajectory-level self-evolution
- Relevance: future direction for Phase 5

### Key Takeaway
1. Domain specialization > general intelligence for trading
2. Risk management is the differentiator
3. Structured feedback loops enable self-improvement
4. Constraints and guardrails prevent overfitting
5. Simple models that work > complex models that might work
6. LLMs are better at research/analysis than direct trading

---

## Context from Previous Research (v1)

Key findings to validate or build upon:
- On-chain features improved directional accuracy from 48% to 55% (p<0.001)
- Hash Ribbon showed 0.81 Chatterjee correlation with BTC direction
- 30-day prediction horizon outperformed 7-day
- Portfolio-level edge more stable than individual asset picks
- XGBoost was competitive with deep learning on tabular features

Critical bugs found in v1 (do NOT repeat):
- Sign inversion bug: models predicted opposite direction, masked by -predictions hack
- RSI off by 28 points from Wilder's textbook definition
- Momentum strategy Sharpe of 0.76 was never independently reproduced

These findings inform our hypotheses but must be independently validated.
