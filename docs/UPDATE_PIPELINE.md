# Update Pipeline

Repo now has one orchestration entrypoint for the operating schedule:

```powershell
.\.venv\Scripts\python.exe -X utf8 data2\update_pipeline.py daily
```

Use `daily` after market close or on a near-real-time cadence. It refreshes:

- equity OHLCV under `data/raw/market/equity/<SYMBOL>/ohlcv.csv`
- index OHLCV under `data/raw/market/index/<INDEX>/ohlcv.csv`
- news preprocessing into `data/processed/chunks.jsonl`
- RAG vector stores under `data/vector_store/`
- materialized analytics/insights
- data quality audit

For a lighter daily run without rebuilding embeddings:

```powershell
.\.venv\Scripts\python.exe -X utf8 data2\update_pipeline.py daily --skip-rag-index
```

Run the quarterly job after financial statements are available:

```powershell
.\.venv\Scripts\python.exe -X utf8 data2\update_pipeline.py quarterly --end-year 2026
```

Quarterly refreshes historical fundamentals, cleans CSVs, materializes features,
runs audit, trains the growth models, and updates model comparison metrics.

To run both daily and quarterly work in one command:

```powershell
.\.venv\Scripts\python.exe -X utf8 data2\update_pipeline.py quarterly --include-daily --end-year 2026
```

Useful options:

- `--symbols ACB,VCB,FPT` limits the run to selected symbols.
- `--indexes VNINDEX,VN30,HNX30` controls index refresh.
- `--overlap-days 10` controls how many previous market days are refetched and merged.
- `--pause 0.15` controls API pacing.
- `--target target_profit_growth_1q` changes the training target.
