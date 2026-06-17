# Tự động hóa KLTN bằng Docker

Tài liệu này gom các bước thực nghiệm và biên dịch PDF vào một chương trình chính:

```bash
python tools/run_kltn_pipeline.py all
```

## Chạy bằng Docker Compose

Từ thư mục gốc dự án:

```bash
docker compose build
docker compose run --rm kltn
```

Lệnh trên sẽ:

1. Tạo bộ đặc trưng cơ sở cho các kỳ hạn dự báo.
2. Huấn luyện lại riêng các mô hình cho kỳ hạn 2, 3, 4 và 8 quý.
3. Xuất bảng tổng hợp tại `outputs/adaptive_compare_models_multiq/s80_q10_p10/`.
4. Cập nhật bảng đa kỳ hạn trong `latex_kltn_nbs/main.tex`.
5. Biên dịch `latex_kltn_nbs/main.pdf` bằng XeLaTeX.

## Chạy từng bước

```bash
python tools/run_kltn_pipeline.py base
python tools/run_kltn_pipeline.py multiq
python tools/run_kltn_pipeline.py metrics
python tools/run_kltn_pipeline.py update-latex
python tools/run_kltn_pipeline.py pdf
```

Nếu muốn chạy lại từ đầu:

```bash
python tools/run_kltn_pipeline.py all --force-base --force-train
```

## Ghi chú

Docker image không copy toàn bộ dữ liệu vào trong image. Thư mục dự án được mount vào `/workspace`, nên các file `data/`, `outputs/` và `latex_kltn_nbs/main.pdf` vẫn nằm trực tiếp trong workspace hiện tại.
