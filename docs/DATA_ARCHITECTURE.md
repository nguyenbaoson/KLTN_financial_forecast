# Cau truc data ML hien tai

Muc tieu: tree gon, de doc, van dung duoc cho ML/RAG va khong mang cam giac "enterprise data lake" qua nang.

## Tree chuan

```text
data/
  catalog/
    symbols.csv
    data_inventory.csv
    ml_tree_inventory.csv

  raw/
    reference/
      equity/
        list.csv
        list_by_exchange.csv
        list_by_industry.csv
      company/
        FPT/
          overview.csv
          officers.csv
          shareholders.csv
          subsidiaries.csv

    market/
      equity/
        FPT/
          ohlcv.csv
          intraday.csv
        VCB/
          ohlcv.csv
      index/
        VNINDEX/
          ohlcv.csv
      futures/
        VN30F1M/
          ohlcv.csv

    fundamental/
      FPT/
        income_statement.csv
        balance_sheet.csv
        cash_flow.csv
        ratio.csv
      VCB/
        income_statement.csv
        balance_sheet.csv
        cash_flow.csv
        ratio.csv

    macro/
      macro_data.csv

    analytics/

    insights/

    news_external/
      cafef_raw.csv
      cafef_clean.csv
      vietstock_raw.csv
      news_merged.csv

  processed/
    financial_quarterly.csv
    market_quarterly.csv
    macro_annual.csv
    news_merged.csv

  features/
    growth_panel_quarterly.csv

  vector_store/
    faiss/
    chroma/
```

## Vai tro tung lop

`catalog/`: metadata va inventory de biet data den tu dau.

`raw/`: du lieu goc theo nhom nghiep vu. Khong sua truc tiep file raw khi clean data.

`processed/`: du lieu da chuan hoa schema, drop duplicate, xu ly missing nhe.

`features/`: bang feature panel dung cho model.

`vector_store/`: FAISS/Chroma cho RAG.

## Trang thai hien tai

Tree legacy da duoc xoa. Hien tai chi giu cac nhom lam viec chinh:

```text
data/
  catalog/
  raw/
  processed/
  features/
  vector_store/
```

Trong `data/raw/` chi giu cac nhom chuan:

```text
analytics/
fundamental/
insights/
macro/
market/
news_external/
reference/
```

## Ly do dung tree nay

1. Duong dan ngan.
2. De giai thich trong luan van.
3. Gan voi nhom du lieu thay vi gan voi source.
4. Source nam trong cot `source`, `source_system`, hoac inventory.
5. Pipeline ML doc truc tiep tu `data/raw/fundamental`, `data/raw/market`, `data/raw/macro`, `data/raw/news_external`.

## Train bang tree moi

```powershell
.\.venv\Scripts\python.exe -m train.train_growth_models --data-layout ml_tree --data-root data --target target_profit_growth_1q
```

Vi default hien da la `ml_tree`, co the chay ngan gon:

```powershell
.\.venv\Scripts\python.exe -m train.train_growth_models --target target_profit_growth_1q
```

## Audit tree hien tai

```powershell
.\.venv\Scripts\python.exe data2\audit_ml_tree.py --data-root data --output outputs/data_quality/ml_tree_audit.csv
```

## Script data con lai

```text
data2/
  vnstock_ml_tree_crawler.py  # crawler moi theo tree hien tai
  audit_ml_tree.py       # kiem tra chat luong data hien tai
```
