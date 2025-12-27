# 📄 RAG-Powered PDF Chatbot

An end-to-end intelligent chatbot leveraging **Retrieval-Augmented Generation (RAG)** to answer questions from PDF documents.
Built using **LangChain**, **Pinecone**, **HuggingFace embeddings**, and **cTransformers** for local LLM inference.

---

## ⚙️ Architecture Overview

### 🧠 Retrieval-Augmented Generation (RAG)

The system follows a **two-phase pipeline**:

### 1️⃣ Indexing Pipeline

* **Load** PDFs using `PyPDF`
* **Split** text with `RecursiveCharacterTextSplitter`
* **Embed** chunks using `HuggingFaceEmbeddings`
* **Store** vectors in **Pinecone**

### 2️⃣ Retrieval + Generation

* **Retrieve** relevant chunks from Pinecone
* **Generate** answers using a **local LLM** via `ctransformers`

---

## 📦 Tech Stack

| Component     | Purpose                  |
| ------------- | ------------------------ |
| LangChain     | RAG orchestration        |
| Pinecone      | Vector database          |
| HuggingFace   | Text embeddings          |
| cTransformers | Local GGML LLM inference |
| PyPDF         | PDF parsing              |
| python-dotenv | Environment variables    |

---

## 📁 Project Structure (IMPORTANT)

```
PDF-ChatBot/
│── .env                 # Main environment file (used by BOTH modes)
│── requirements.txt
│
│── project/             # FULL Frontend + Backend (Flask App)
│   ├── app.py
│   ├── templates/
│   ├── static/
│   └── services/
│
│── quick_run/            # QUICK testing & notebook-based usage
│   ├── trails.ipynb
│   └── helpers/
│
└── README.md
```

---

## 🔐 Environment Variables

Update **ONLY the root `.env` file**:

```env
PINECONE_API_KEY=your_key_here
PINECONE_INDEX=your_index_name
```

✅ Used by **both `project/` and `quick_run/`**

---

## 🚀 How to Run

### 🧪 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# OR
.venv\Scripts\activate      # Windows
```

### 📦 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run FULL Application (Frontend + Backend)

```bash
cd project
python app.py
```

✔ Uses Flask
✔ Frontend + Backend
✔ Production-style usage

---

## ⚡ Quick Run (Notebook / Fast Testing)

```bash
cd quick_run
jupyter notebook trails.ipynb
```

✔ Fast experimentation
✔ Index PDFs
✔ Test queries quickly

---

## ⚠️ Important Notes

* ✅ update `.env`**
* `project/` = full application
* `quick_run/` = rapid testing & debugging

---
