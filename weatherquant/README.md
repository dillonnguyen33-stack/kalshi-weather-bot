# weatherquant

Probabilistic forecasting system pricing Kalshi daily-high-temperature markets.
Successor to the v3 Discord-alert bot. See `CLAUDE.md` for the locked stack and
`.planning/` for phase plans.

## Dev setup

```bash
uv sync               # resolve env (Python 3.12, pinned deps)
uv run pytest -q      # full suite (ledger integration needs DATABASE_URL)
uv run pytest -m "not integration" -q   # fast subset, no DB
```

`DATABASE_URL` must use the `postgresql+psycopg://` scheme (see `.env.example`).
