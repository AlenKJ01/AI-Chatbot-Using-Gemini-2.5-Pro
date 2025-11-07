# GeminiDocQA
## 🧠 AI Chatbot Using Gemini 2.5 Pro

An intelligent, document-aware chatbot web application built with **FastAPI**, **Gemini 2.5 Pro**, **Hugging Face Transformers**, and **Sentence Transformers**.  
The chatbot can **summarize uploaded documents** and **answer user queries** based on document context — all via a clean HTML + CSS frontend, with no JavaScript.

---

## 🚀 Overview

This project demonstrates how to integrate **Google’s Gemini 2.5 Pro LLM** with a local vector-based retrieval pipeline.  
Users can upload **PDF**, **DOCX**, or **TXT** files, have them summarized automatically, and then **converse intelligently** with the document content.

It combines:
- NLP summarization (Hugging Face BART)
- Semantic embeddings (Sentence Transformers)
- Vector similarity search (pure NumPy cosine similarity)
- Conversational reasoning (Gemini 2.5 Pro)
- A minimal, responsive **FastAPI** web UI (HTML + CSS only)

---

## 🧩 Features

| Feature | Description |
|----------|-------------|
| 📄 **Document Upload** | Upload PDF, DOCX, or TXT files via FastAPI. |
| ✂️ **Text Extraction** | Extracts text with `pdfplumber` or `python-docx`. |
| 🧠 **Summarization** | Uses `facebook/bart-large-cnn` from Hugging Face to produce concise summaries. |
| 🧭 **Embeddings + Vector Store** | Splits text into chunks, embeds them using `all-MiniLM-L6-v2`, and stores vectors as `.npy` arrays (no FAISS dependency). |
| 💬 **Chat with Document** | Ask contextual questions — Gemini 2.5 Pro retrieves relevant chunks and answers only from your uploaded document. |
| 💾 **Session Management** | Keeps conversation history for contextual responses. |
| 🎨 **Frontend** | Single-page HTML + CSS interface (no JavaScript). |
| ⚙️ **Robust Error Handling** | Handles large files, missing tokens, summarization retries, and model timeouts gracefully. |

---

## 🏗️ Project Structure
```
AI-Chatbot-using-Gemini/
│
├── app/
│ ├── main.py # FastAPI entry point
│ ├── utils.py # Core NLP, embeddings, and Gemini logic
│ ├── templates/
│ │ └── index.html # Single-page web UI
│ └── static/
│ └── style.css # Minimal responsive styling
│
├── .env # API keys and environment variables
├── requirements.txt # Dependencies
└── README.md # Project documentation
```

---

## ⚙️ Tech Stack

| Layer | Tools / Libraries |
|-------|--------------------|
| **Frontend** | HTML, CSS (no JS) |
| **Backend** | FastAPI, Uvicorn |
| **Text Extraction** | pdfplumber, python-docx |
| **Summarization** | Hugging Face Transformers (`facebook/bart-large-cnn`) |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Vector Search** | NumPy cosine similarity |
| **LLM** | Gemini 2.5 Pro (`google-generativeai`) |
| **Env Management** | python-dotenv |

---

## 🔑 Environment Variables (`.env`)

Create a `.env` file in the project root:

```env
# Google Gemini API key
GEMINI_API_KEY=your_gemini_api_key_here

# Hugging Face token for model downloads
HF_TOKEN=your_huggingface_token_here

# Optional custom model name
GEMINI_MODEL_NAME=gemini-2.5-pro
```

---

## 🧰 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/AlenKJ01/GeminiDocQA.git
cd GeminiDocQA
```

### 2️⃣ Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate     # Windows
# or
source venv/bin/activate  # macOS/Linux
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

If you don’t have a requirements.txt, use:

pip install fastapi uvicorn jinja2 python-multipart pdfplumber python-docx transformers torch sentence-transformers langchain google-generativeai python-dotenv typing_extensions

### 4️⃣ Set up environment variables

Add your .env file (as shown above) with valid API keys.

### 5️⃣ Run the app
```bash
uvicorn app.main:app --port 8000
```

Visit 👉 
## http://127.0.0.1:8000

---

## 🧮 Workflow

1. Upload Document
  - Choose a .pdf, .docx, or .txt file.
  - FastAPI extracts its text and displays it.

2. Generate Summary
  - Summarization is performed using the BART transformer model.
  - Large texts are chunked, summarized per part, and combined.

3. Create Vectorstore
  - Text is divided into overlapping chunks.
  - Embeddings generated via all-MiniLM-L6-v2.
  - Stored locally in .npy for semantic retrieval.

4. Chat with Document
  - User question → semantic search retrieves relevant chunks → Gemini 2.5 Pro forms the answer using only that context.
  - The conversation remains contextual for multiple queries.

---

## 🧠 Example Flow

1. Upload a project report PDF.

2. Click Generate Summary — see concise summary of all sections.

3. Click Create Vectorstore.

4. Ask questions like:
  - "What are the key results discussed in the report?"
  - "Which technologies were used?"
  - "Summarize the conclusions."

---
## 🛠️ Troubleshooting

| Problem                                        | Solution                                                               |
| ---------------------------------------------- | ---------------------------------------------------------------------- |
| **`Unauthorized 401` when loading BART model** | Add your `HF_TOKEN` in `.env`                                          |
| **`ModuleNotFoundError: faiss`**               | FAISS removed — ensure `utils.py` version uses NumPy search            |
| **Gemini returns empty / `MAX_TOKENS`**        | New parser retries automatically; adjust `max_output_tokens` if needed |
| **Windows crash on shutdown**                  | Use `asyncio.WindowsSelectorEventLoopPolicy()` (already applied)       |
