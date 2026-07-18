# Stock Analysis & Portfolio Management — Multi-Agent AI System

A multi-agent AI system that turns a plain-language stock question into a single, data-grounded investment recommendation. Built with LangGraph, it routes queries to specialized agents covering fundamentals, valuation, technicals, and news, then synthesizes everything into one decisive, justified verdict — instead of a pile of disconnected numbers.

## Why this exists

Investing decisions demand more than one person can track at once. Financial statements, valuation ratios, price action, and news all live in different places and formats. Reading a balance sheet, judging valuation, and spotting a technical trend each require distinct expertise — and even after gathering everything, turning it into one clear buy, hold, or avoid decision is still manual and slow. Track a handful of stocks and it stops scaling entirely.

This project automates that pipeline: fetch the data, run it through the right specialists, and return one clear, reasoned answer.

## What it does

- **Understands the query** — classifies intent and extracts the company/ticker from natural language
- **Analyzes automatically** — runs only the specialist agents a given query actually needs
- **Concludes decisively** — synthesizes every agent's findings into a single justified recommendation
- **Manages your portfolio** — add, update, remove, and review holdings, entirely through chat

## Architecture

```
User query (Streamlit chat)
        │
        ▼
  Query router          — classifies intent, extracts ticker
        │
        ▼
Analysis orchestrator    — LLM selects the relevant agents
        │
        ▼
┌───────────────────────────────────────────┐
│         Specialist agents (parallel)        │
│  Income stmt   Balance sheet   Cash flow    │
│  Valuation     Technical       News          │
└───────────────────────────────────────────┘
        │
        ▼
   Final conclusion      — one synthesized recommendation
```

Each specialist agent is its own LangGraph sub-graph following the same execution model: independent data-fetch nodes run in parallel, feed into derived-metric nodes (ratios, growth rates, indicators), and fan back into a single LLM synthesis node. That pattern repeats identically across all six agents.

A separate **portfolio manager** agent, backed by SQLite, handles holdings through the same chat interface — add, update, remove, list — pausing to ask when required details (ticker, quantity, price, date) are missing, instead of guessing.

## Specialist agents

| Agent | Covers |
|---|---|
| Income statement | Revenue, net income, EPS, margins, growth over recent periods |
| Balance sheet | Assets, liabilities, equity, current/quick ratio, return on assets |
| Cash flow | Operating cash flow, free cash flow, FCF margin, CapEx trend |
| Valuation | P/E, PEG, EV/EBITDA, price-to-book, dividend and earnings yield |
| Technical | Price history, SMA, RSI, MACD, trend and momentum signals |
| News | Recent headlines, filtered by an LLM for relevance to the query |

## Orchestration logic

Every query is triaged before any analysis runs:

1. **Classify** — label the query as stock-related, general, or reject
2. **Judge necessity** — a stricter critic model decides if it needs real analysis or just a direct answer
3. **Resolve ticker** — map the company name to its exact Yahoo Finance ticker (NSE/BSE suffix included)
4. **Pick agents** — the orchestrator LLM selects only the agents relevant to that specific question

This keeps the system cheap and fast — a pure valuation question, for example, never triggers the technical or news agents.

## Tech stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — stateful, graph-based orchestration; explicit state typing, parallel fan-out/fan-in, conditional routing
- **LangChain + OpenAI** — GPT-4o-mini for extraction and per-agent analysis, GPT-4 as a stricter critic for classification decisions
- **[yfinance](https://github.com/ranaroussi/yfinance)** — market data: financial statements, price history, valuation metrics, news
- **Streamlit** — the entire user-facing chat interface
- **SQLite** — lightweight, persistent storage for portfolio holdings
- **Python** — glue for every agent and tool

## Engineering details

- **Null-safe financial math** — every ratio/growth helper tolerates missing rows, unequal list lengths, and zero denominators without crashing
- **No fabricated data** — prompts explicitly instruct agents to omit a metric they can't determine rather than invent a number; missing fields are treated as "not disclosed," never defaulted to zero
- **Two-model strategy** — GPT-4o-mini for high-volume extraction/analysis, GPT-4 reserved for the higher-stakes classification/critic step
- **Modular agent registry** — new analysis agents plug in through a single registry dictionary, not a rewrite of the orchestrator

## Example

**Query:** `analyse tcs stock`

The orchestrator selects `income_statement_agent`, `balance_sheet_analysis_agent`, `cashFlowAnalysisAgent`, `technical_agent`, and `newsAnalysisAgent` (portfolio manager skipped — nothing about holdings in the query). Each agent fetches and analyzes its slice of TCS.NS independently, and the final synthesis returns:

> **Recommendation: Hold** — consistent revenue growth (+4.58%) and a strong balance sheet, offset by net income lagging revenue growth and a bearish technical trend bias.

## Getting started

```bash
git clone https://github.com/<your-username>/stock-analysis-multi-agent.git
cd stock-analysis-multi-agent
pip install -r requirements.txt   # or: uv sync
```

Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your-key-here
```

Run the app:
```bash
streamlit run interface.py
```

## Limitations & what's next

This is a capstone project, not a finished product — a few known gaps:

- Portfolio holdings are stored as unstructured text; no structured P&L math yet
- Agents in the selection layer run sequentially, not in parallel
- No retry/error handling around external API calls yet
- Valuation agent is built but not yet wired into the orchestrator's registry

Planned next: live broker integration for real trade execution, deeper technical analysis with backtesting, real-time price alerts and portfolio risk dashboards, multi-currency/mutual-fund/ETF support, and persistent long-term memory across sessions (replacing the current in-memory checkpointer).

## License

This project is licensed under the [MIT License](LICENSE) — you're free to use, modify, and distribute this code, provided the original copyright notice is retained.

## Disclaimer

This project is for educational purposes as part of a capstone submission. It does not constitute financial advice. Recommendations are generated by language models reasoning over publicly available data and may be incomplete, delayed, or incorrect.
