# Lab DAY 19 — Flat RAG vs Graph RAG

So sánh 2 pipeline RAG trên cùng bộ dữ liệu về các công ty AI (OpenAI, DeepSeek, Anthropic, Meta, Google DeepMind, XAI).

## Cấu trúc dự án

```
Lab_DAY_19/
├── data/                        # 7 file .md về các công ty AI
├── flat_rag/
│   ├── ingest.py                # Load → chunk → embed → ChromaDB
│   └── chain.py                 # RetrievalQA với GPT-4o-mini
├── graph_rag/
│   ├── ingest.py                # Extract entities/relations → Neo4j
│   └── chain.py                 # GraphCypherQAChain với GPT-4o-mini
├── benchmark/
│   ├── testset.json             # Bộ câu hỏi benchmark (có ground_truth)
│   ├── eval_flat_rag.py         # Chạy RAGAS cho Flat RAG
│   ├── eval_graph_rag.py        # Chạy RAGAS cho Graph RAG
│   └── compare_results.py       # So sánh kết quả + vẽ chart
├── notebooks/
│   └── flat_rag_demo.ipynb
├── docker-compose.yml           # Neo4j
├── .env.example
└── requirements.txt
```

---

## Cài đặt

### 1. Tạo và kích hoạt virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Cấu hình environment

```powershell
Copy-Item .env.example .env
```

Mở `.env` và điền `OPENAI_API_KEY`. Các biến Neo4j giữ nguyên nếu dùng docker-compose.

### 3. Khởi động Neo4j (Docker)

```powershell
docker-compose up -d
```

Neo4j Browser: http://localhost:7474 (login: `neo4j` / `password123`)

---

## Chạy Flat RAG

### Bước 1 — Ingest vào ChromaDB

```powershell
python -m flat_rag.ingest
```

### Bước 2 — Test nhanh 1 câu hỏi

```powershell
python -m flat_rag.chain
```

### Bước 3 — Chạy benchmark

```powershell
python -m benchmark.eval_flat_rag
```

Kết quả lưu tại: `benchmark/flat_rag_results.csv`

---

## Chạy Graph RAG

### Bước 1 — Ingest vào Neo4j

```powershell
python -m graph_rag.ingest
```

> Quá trình này gọi GPT-4o-mini để extract entities/relationships từ từng chunk. Sẽ mất vài phút.

### Bước 2 — Test nhanh 1 câu hỏi

```powershell
python -m graph_rag.chain
```

### Bước 3 — Chạy benchmark

```powershell
python -m benchmark.eval_graph_rag
```

Kết quả lưu tại: `benchmark/graph_rag_results.csv`

---

## So sánh kết quả

```powershell
python -m benchmark.compare_results
```

In bảng so sánh mean scores và lưu chart tại: `benchmark/comparison_chart.png`

---

## Format testset

File `benchmark/testset.json` — mỗi test case có format:

```json
{
  "id": "q01",
  "question": "...",
  "ground_truth": "...",
  "category": "single-hop | multi-hop | relationship"
}
```

| Field | Mô tả |
|---|---|
| `id` | ID duy nhất, dùng để merge kết quả 2 pipeline |
| `question` | Câu hỏi benchmark |
| `ground_truth` | Đáp án chuẩn (dùng cho RAGAS `context_recall`) |
| `category` | Phân loại câu hỏi để phân tích kết quả |

---

## Metrics (RAGAS)

| Metric | Ý nghĩa |
|---|---|
| `faithfulness` | Answer có dựa trên context không (tránh hallucination) |
| `answer_relevancy` | Answer có trả lời đúng câu hỏi không |
| `context_precision` | Context retrieved có liên quan không |
| `context_recall` | Context có đủ thông tin để trả lời không |
