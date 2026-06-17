# KLTN Financial Forecast

Dự án khóa luận tốt nghiệp về dự báo tăng trưởng lợi nhuận doanh nghiệp niêm yết tại Việt Nam bằng dữ liệu tài chính, dữ liệu thị trường, tin tức và mô hình học máy. Repo bao gồm mã nguồn thu thập dữ liệu, tạo đặc trưng, huấn luyện mô hình, đánh giá thực nghiệm, chatbot RAG minh họa và bộ LaTeX để biên dịch báo cáo KLTN.

## Nội dung chính

- Xây dựng pipeline dữ liệu doanh nghiệp-quý từ báo cáo tài chính, dữ liệu thị trường, vĩ mô và tin tức.
- Tạo nhãn dự báo tăng trưởng lợi nhuận thích nghi theo ngành.
- Huấn luyện và so sánh các mô hình dự báo tăng trưởng lợi nhuận.
- Bổ sung thực nghiệm đa kỳ hạn dự báo 2, 3, 4 và 8 quý.
- Cung cấp chatbot RAG để truy vấn thông tin tài chính và kết quả dự báo.
- Tự động hóa quá trình chạy thực nghiệm và biên dịch `latex_kltn_nbs/main.pdf`.

## Cấu trúc thư mục

```text
app/
  streamlit_app.py                    # Giao diện demo Streamlit

feature_engineering/
  growth_feature_engineering.py       # Tạo đặc trưng tăng trưởng lợi nhuận

pipeline_crawl_data/
  update_pipeline.py                  # Crawl, làm sạch, tạo dữ liệu đầu vào

pipeline_report_data/
  report_pipeline.py                  # Tạo hình, bảng và báo cáo đánh giá

src/
  preprocessing/                      # Tiền xử lý văn bản/tin tức
  rag/                                # RAG, chatbot, vector store, retriever

tools/
  run_kltn_pipeline.py                # Entrypoint tự động hóa KLTN

train/
  train_*.py                          # Huấn luyện và so sánh mô hình

latex_kltn_nbs/
  main.tex                            # Nguồn LaTeX báo cáo
  main.pdf                            # Bản PDF đã biên dịch
  figures/                            # Hình minh họa trong báo cáo

docs/
  AUTOMATION_DOCKER.md                # Hướng dẫn chạy tự động/Docker
  DATA_ARCHITECTURE.md                # Mô tả cấu trúc dữ liệu
  UPDATE_PIPELINE.md                  # Ghi chú cập nhật pipeline
```

## Chuẩn bị môi trường

Yêu cầu khuyến nghị:

- Python 3.10 trở lên.
- Docker Desktop nếu muốn chạy bằng Docker.
- XeLaTeX nếu muốn biên dịch PDF trực tiếp trên máy.

Cài đặt môi trường Python:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu dùng API cho chatbot hoặc mô hình ngôn ngữ, tạo file `.env` ở thư mục gốc:

```text
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

File `.env` đã được đưa vào `.gitignore` và không được commit lên GitHub.

## Dữ liệu

Thư mục `data/`, `outputs/`, `backups/` và các file sinh tự động không được commit lên repo. Khi clone dự án mới, cần tự chuẩn bị dữ liệu hoặc chạy pipeline thu thập dữ liệu.

Chạy pipeline cập nhật dữ liệu theo quý:

```powershell
python pipeline_crawl_data/update_pipeline.py quarterly --symbols ACB,VCB,FPT,HPG,MWG --include-daily --crawl-news-rss --skip-rag-index
```

Chạy cập nhật dữ liệu thị trường/tin tức hằng ngày:

```powershell
python pipeline_crawl_data/update_pipeline.py daily --crawl-news-rss --skip-rag-index
```

## Chạy thực nghiệm KLTN

Chạy toàn bộ pipeline thực nghiệm và biên dịch PDF:

```powershell
python tools/run_kltn_pipeline.py all
```

Chạy lại từ đầu, bỏ qua kết quả cũ:

```powershell
python tools/run_kltn_pipeline.py all --force-base --force-train
```

Chạy từng bước riêng:

```powershell
python tools/run_kltn_pipeline.py base
python tools/run_kltn_pipeline.py multiq
python tools/run_kltn_pipeline.py metrics
python tools/run_kltn_pipeline.py update-latex
python tools/run_kltn_pipeline.py pdf
```

Kết quả chính:

- Bảng/metric thực nghiệm: `outputs/adaptive_compare_models_multiq/s80_q10_p10/`
- Báo cáo PDF: `latex_kltn_nbs/main.pdf`

## Chạy bằng Docker

Từ thư mục gốc dự án:

```powershell
docker compose build
docker compose run --rm kltn
```

Docker sẽ mount workspace hiện tại vào container, chạy `tools/run_kltn_pipeline.py all`, cập nhật bảng thực nghiệm trong LaTeX và biên dịch lại `latex_kltn_nbs/main.pdf`.

Xem thêm: `docs/AUTOMATION_DOCKER.md`.

## Chạy giao diện demo

```powershell
streamlit run app/streamlit_app.py
```

Giao diện dùng để minh họa chatbot/RAG và một số kết quả dự báo. Một số chức năng có thể cần dữ liệu, vector store hoặc API key tương ứng.

## Biên dịch báo cáo LaTeX

Nếu chỉ muốn biên dịch PDF:

```powershell
python tools/run_kltn_pipeline.py pdf
```

Hoặc chạy trực tiếp trong thư mục LaTeX nếu máy đã có XeLaTeX:

```powershell
cd latex_kltn_nbs
xelatex main.tex
xelatex main.tex
```

## Ghi chú khi đưa lên GitHub

Repo chỉ lưu mã nguồn, tài liệu, hình minh họa và bản PDF khóa luận. Các nhóm file sau được loại khỏi Git:

- `.env`, `.venv/`, `.vscode/`
- `data/`, `outputs/`, `backups/`
- file phụ khi biên dịch LaTeX như `.aux`, `.log`, `.toc`
- báo cáo Turnitin và thư mục nháp/tài liệu tham khảo cá nhân

## Tác giả

Khóa luận tốt nghiệp: dự báo tăng trưởng lợi nhuận doanh nghiệp niêm yết Việt Nam bằng học máy và RAG.
