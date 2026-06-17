# Ap dung hai paper vao he thong

## 1. Patel et al. (2015): Two-stage SVR fusion

Paper de xuat dung SVR o stage 1 de du bao cac bien trang thai tuong lai, sau do dua cac bien tuong lai do vao model stage 2. Trong repo, huong nay nam o:

```text
train/train_growth_two_stage_svr_fusion.py
```

Script hien da duoc bo sung:

- `--stage1-target-set paper_technical`: gan nhat voi paper, stage 1 chi du bao nhom bien market/technical.
- `--stage1-target-set state_all`: mo rong cho bai toan hien tai, stage 1 co the du bao ca news/event, transformer va bank indicators.
- `--stage2-input current_plus_future`: dung ca feature hien tai va feature tuong lai do SVR du bao.
- `--stage2-input future_only`: dung dung tinh than paper hon, stage 2 chi nhan feature tuong lai du bao.
- `single_stage_model_results.csv`: baseline single-stage.
- `stage1_svr_results.csv`: chat luong du bao cac feature tuong lai cua SVR.
- `stage1_svr_tuning.csv`: ket qua chon tham so stage-1 SVR neu bat `--tune-stage1`.
- `fusion_improvement.csv`: so sanh single-stage voi two-stage fusion.
- `--stage1-oof`: tao du bao stage-1 tren train bang expanding-window de giam overfit.
- `--min-stage1-validation-r2 0`: chi giu cac future feature ma SVR du bao tot hon baseline tren validation.
- `--stage2-selection-metric`: chon model stage-2 theo metric validation ro rang.
- `--threshold-metric balanced_accuracy|f1|recall`: metric dung de tune decision threshold.

Lenh khuyen nghi cho tap 24 ma trong luan van:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m train.train_growth_two_stage_svr_fusion `
  --dataset-path data/features/growth_features.csv `
  --target target_profit_up_4q `
  --tickers ACB,BCM,BID,CII,DGW,DIG,FPT,GAS,HCM,HPG,KBC,KDH,MBB,MWG,REE,SHB,SSI,STB,TCB,VCB,VHM,VIC,VNM,VRE `
  --feature-set all `
  --stage1-target-set market_technical `
  --stage2-input current_plus_future `
  --max-stage1-targets 12 `
  --tune-stage1 `
  --stage1-c-values 1,10,100 `
  --stage1-gamma-values scale,0.01,0.1 `
  --stage1-epsilon-values 0.01,0.1,0.5 `
  --stage1-oof `
  --min-stage1-validation-r2 0 `
  --stage2-selection-metric val_F1 `
  --tune-threshold `
  --threshold-metric f1 `
  --output-dir outputs/thesis_patel_svr_fusion_24
```

Lenh mo rong voi news/event theo bai toan cua ban:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m train.train_growth_two_stage_svr_fusion `
  --dataset-path data/features/growth_features.csv `
  --target target_profit_up_4q `
  --feature-set all `
  --stage1-target-set state_all `
  --stage2-input current_plus_future `
  --max-stage1-targets 45 `
  --tune-threshold `
  --output-dir outputs/thesis_patel_svr_fusion_all_symbols
```

## 2. FS-Graph RAG: Graph-based retrieval cho bao cao tai chinh

Paper FS-Graph RAG de xuat bieu dien bao cao tai chinh bang knowledge graph de truy xuat dung section/subsection/metric thay vi chi vector search tren text chunk. He thong cua ban da co du lieu tai chinh quy o dang structured panel, nen co the ap dung truoc bang graph retriever nhe:

```text
src/rag/financial_graph_retriever.py
```

Module nay xem moi dong `symbol-year-quarter` trong `data/features/growth_features.csv` la mot report node:

```text
Company(SYMBOL) -> Report(yearQquarter) -> Metric(column)
```

Chatbot trong `src/rag/chatbot.py` da duoc cap nhat de lay them `[Graph X]` context. Khi nguoi dung hoi ve doanh thu, loi nhuan, ROE, P/E, tang truong loi nhuan, quy cu the, he thong lay so lieu truc tiep tu feature panel va ket hop voi `[Nguon X]` tin tuc neu co.

Vi du cau hoi:

```text
Loi nhuan va doanh thu cua ACB quy 1 2026 thay doi nhu the nao?
```

Ket qua tra loi se dua tren structured context tu `growth_features.csv`, giam phu thuoc vao vector search tin tuc va giam rui ro hallucination khi hoi so lieu.
