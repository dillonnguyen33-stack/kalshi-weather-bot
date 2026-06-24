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

### Test database isolation (DATA-LOSS GUARD)

The ledger integration tests do `metadata.drop_all`/`create_all` to rebuild the schema
each session. They run against an **isolated** database, NEVER your dev `DATABASE_URL`:

- Set `TEST_DATABASE_URL` to a dedicated test DB, **or** leave it unset and the suite
  derives a `*_test` database name from `DATABASE_URL` (e.g. `.../weatherquant` →
  `.../weatherquant_test`). The fixture auto-creates that DB if it does not exist.
- A hard guard in `tests/conftest.py` **refuses to run** (raises) if the resolved test URL
  is empty or equals `DATABASE_URL`, so `drop_all` can never touch dev data.

This guard exists because a `pytest` run on 2026-06-24 wiped the real dev ledger (the
`pg_engine` fixture was bound to the dev DB). See
`.planning/debug/resolved/test-suite-wipes-dev-ledger.md`. If you ever need to recreate the
test DB manually: `docker exec weatherquant-pg createdb -U postgres weatherquant_test`.
