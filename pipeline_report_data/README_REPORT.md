# Report Data Pipeline

Thu muc nay chi phu trach ve hinh, so sanh ket qua va danh gia mo hinh.
No khong crawl du lieu.

File chay chinh:

```text
pipeline_report_data/report_pipeline.py
```

## Chay nhanh

Chay tat ca nhom report:

```powershell
python pipeline_report_data/report_pipeline.py all
```

Chay rieng nhom classification:

```powershell
python pipeline_report_data/report_pipeline.py classification --model-output-dir outputs/adaptive_compare_models_1q/s80_q10_p10
```

Chay rieng nhom adaptive target:

```powershell
python pipeline_report_data/report_pipeline.py adaptive --adaptive-input-dir outputs/adaptive_compare_models_1q/s80_q10_p10
```

In lenh se chay nhung khong thuc thi:

```powershell
python pipeline_report_data/report_pipeline.py all --dry-run
```

## Cau truc

```text
pipeline_report_data/
  report_pipeline.py                  # File chay chinh
  compare_adaptive_configs_models.py  # So sanh hai cau hinh adaptive
  evaluate_adaptive_by_quarter_sector.py
                                      # Danh gia theo quy/nganh

  plot_*.py                           # Ve tung loai hinh rieng
```
