# Crawl Data Pipeline

Thu muc nay chi phu trach crawl, chuan hoa, lam sach va tao feature.

File chay chinh:

```text
pipeline_crawl_data/update_pipeline.py
```

Thu muc du lieu chuan:

```text
data/
```

Khong dung `data2/data/`.

## Chay tu dau

Nen chay tu thu muc goc project:

```powershell
cd E:\financial_rag_forecast
```

Crawl lai tu dau cho mot tap ma co phieu, tao feature, nhung chua train model:

```powershell
python pipeline_crawl_data/update_pipeline.py quarterly --symbols ACB,VCB,FPT,HPG,MWG --include-daily --crawl-news-rss --skip-rag-index --skip-train
```

Crawl + tao feature + train regression model:

```powershell
python pipeline_crawl_data/update_pipeline.py quarterly --symbols ACB,VCB,FPT,HPG,MWG --include-daily --crawl-news-rss --skip-rag-index
```

Neu da co BCTC va chi cap nhat market/news:

```powershell
python pipeline_crawl_data/update_pipeline.py daily --crawl-news-rss --skip-rag-index
```

## Cau truc

```text
pipeline_crawl_data/
  update_pipeline.py                  # File chay chinh
  pipeline_common.py                  # Xu ly duong dan va danh sach ma

  crawl_historical_fundamental.py     # Crawl BCTC quy lich su
  crawl_news_rss.py                   # Crawl tin tuc RSS
  vnstock_ml_tree_crawler.py          # Crawl reference/market/fundamental nhanh

  normalize_news_external.py          # Chuan hoa tin tuc, map ticker
  clean_csv_quality.py                # Xoa duplicate/cot rong
  materialize_ml_outputs.py           # Tao data/features/growth_features.csv
  audit_ml_tree.py                    # Kiem tra chat luong du lieu
```

## Dau ra

```text
data/raw/fundamental/<SYMBOL>/
  income_statement.csv
  balance_sheet.csv
  cash_flow.csv
  ratio.csv

data/raw/market/equity/<SYMBOL>/ohlcv.csv
data/raw/news_external/news_merged.csv
data/processed/chunks.jsonl
data/features/growth_features.csv
outputs/data_quality/ml_tree_audit.csv
```

## Chay rieng tung buoc

Crawl BCTC:

```powershell
python pipeline_crawl_data/crawl_historical_fundamental.py --symbols ACB,VCB,FPT --start-year 2015
```

Crawl tin tuc:

```powershell
python pipeline_crawl_data/crawl_news_rss.py --symbols ACB,VCB,FPT
python pipeline_crawl_data/normalize_news_external.py
```

Tao feature:

```powershell
python pipeline_crawl_data/materialize_ml_outputs.py
```

Kiem tra du lieu:

```powershell
python pipeline_crawl_data/audit_ml_tree.py
```
