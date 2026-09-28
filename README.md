# 📄 Enterprise PDF RAG Agent

A Retrieval-Augmented Generation (RAG) app that lets you upload a PDF and ask questions about it. Answers are generated **strictly from the document's content**, and the exact source chunks used are shown for every answer, so responses are transparent and verifiable.

Built with Streamlit, LangChain, ChromaDB, HuggingFace embeddings, and the Groq API.

---

## ✨ Features

- **PDF upload and parsing** – works with reports, slide decks exported to PDF, and other text-based documents
- **Hybrid retrieval** – combines semantic vector search (ChromaDB, MMR) with keyword search (BM25)
- **Query expansion** – the LLM rewrites your question into several variations, so it finds answers even when your wording differs from the document's
- **Boilerplate cleaning** – removes repeating headers and footers that pollute search results
- **Grounded answers** – the model is instructed not to hallucinate and to say so when the answer isn't in the document
- **Source transparency** – an expandable panel shows the exact chunks and page numbers behind each answer
- **Cached indexing** – each document is embedded once, not on every interaction

---

## 🏗️ Architecture

```
PDF Upload
   │
   ▼
PyPDFLoader ──► Clean boilerplate ──► RecursiveCharacterTextSplitter
                                              │
                                              ▼
                              HuggingFace Embeddings (bge-base-en-v1.5)
                                              │
                                              ▼
                                   ChromaDB (vector store)
                                              │
User Question ──► Query Expansion (LLM) ──► Hybrid Retriever (Vector + BM25)
                                              │
                                              ▼
                           Top chunks + Strict system prompt
                                              │
                                              ▼
                              Groq LLM (openai/gpt-oss-120b)
                                              │
                                              ▼
                                Answer + Source chunks (Streamlit)
```

---

## 🧰 Tech Stack

| Component        | Technology                                   |
| ---------------- | -------------------------------------------- |
| UI               | Streamlit                                    |
| Orchestration    | LangChain                                    |
| Document loader  | PyPDFLoader                                  |
| Text splitting   | RecursiveCharacterTextSplitter (800 / 150)   |
| Embeddings       | HuggingFace `BAAI/bge-base-en-v1.5` (local)  |
| Vector database  | ChromaDB (in-memory)                         |
| Keyword search   | BM25 (`rank_bm25`)                           |
| LLM              | Groq API – `openai/gpt-oss-120b`             |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and activate a virtual environment

```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 💡 Usage

1. Upload a PDF using the file uploader.
2. Wait for the "Document ready" message (the first run downloads the embedding model, ~400 MB).
3. Type a question, for example: *"Who are the team members?"* or *"What technologies were used?"*
4. Read the answer, and open **View Source Chunks** to see exactly where it came from.

---

## ⚙️ Configuration

You can tune the pipeline in `app.py`:

| Setting | Location | Purpose |
| --- | --- | --- |
| `chunk_size`, `chunk_overlap` | `build_index()` | Chunk granularity |
| `k`, `fetch_k` | vector retriever | How many chunks are retrieved |
| `weights=[0.6, 0.4]` | `EnsembleRetriever` | Balance between semantic and keyword search |
| `BOILERPLATE` regex | top of file | Pattern for headers/footers to strip |
| `model_name` | `get_llm()` | Any chat model available on Groq |

---

## 🛠️ Troubleshooting

- **"I cannot answer this based on the provided document"** – open **View Source Chunks** to check what was retrieved. If the relevant text isn't there, adjust the boilerplate regex or increase `k`.
- **Empty or garbled text** – the PDF may be image-based. Use OCR: `pip install rapidocr-onnxruntime` and load with `PyPDFLoader(path, extract_images=True)`.
- **`GROQ_API_KEY` not found** – confirm `.env` is in the project root and contains the key.

---

## 🔮 Future Improvements

- Multi-document upload and cross-document Q&A
- Persistent vector store on disk
- Conversational memory for follow-up questions
- Cross-encoder reranking for higher precision
- OCR fallback for scanned PDFs

---

## 📜 License

This project is licensed under the MIT License. Add a `LICENSE` file if you'd like to make this official.
