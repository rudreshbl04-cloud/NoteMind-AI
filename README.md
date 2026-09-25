# NoteMind AI

> Chat with your own PDF notes using Google Gemini — strictly grounded with exact page citations.

---

## What it does

**NoteMind AI** is a lightweight, beginner-friendly web application designed for students and researchers. You simply enter your own free Google Gemini API key, upload your lecture or course PDF notes, and ask questions.

The AI answers **only** using the content found in your notes. If an answer cannot be found in your uploaded documents, NoteMind AI responds strictly with:
```
Not found in your notes
```
Every valid answer displays the exact PDF file name and page number where the information originated.

---

## Features

- **Gemini API Key Connection**: Connect securely using your own API key. The key lives solely in browser session memory and is never saved to disk or databases.
- **PDF Upload**: Upload one or multiple course or lecture notes simultaneously.
- **Page-by-Page Extraction**: Extracted page by page using `pypdf`, preserving exact 1-indexed page numbers.
- **Text Chunking**: Splits notes into readable, overlapping chunks without losing source or page references.
- **Local Vector Search**: Uses ChromaDB locally to store chunk embeddings generated via Gemini (`text-embedding-004`).
- **Strict Grounding & Hallucination Prevention**: Prompt engineering and cosine distance filtering guarantee answers come strictly from note context.
- **Source Citations**: Displays exact source file names and page numbers (e.g. `lecture_notes.pdf — Page 5`).
- **Session Chat History**: Review questions and answers in an interactive chat stream.
- **Clear Chat**: Clear your active conversation anytime.
- **Remove Files**: Clear all indexed notes from local vector storage with a single click.

---

## Screenshots

> **Note**: Screenshot placeholders are shown below. Place your actual screenshots into the `screenshots/` directory named `setup.png` and `chat.png`.

### API Key Setup
![API Key Setup](screenshots/setup.png)

### Notes Chat
![Notes Chat](screenshots/chat.png)

---

## Requirements

- **Python**: Version 3.10 or higher (tested with Python 3.11).
- **Internet Connection**: Required for Google Gemini API calls.
- **Gemini API Key**: Free tier from Google AI Studio.

---

## How to get a Gemini API key

1. Visit [Google AI Studio](https://aistudio.google.com/).
2. Sign in with your Google account.
3. Click **"Get API key"** in the top navigation or sidebar.
4. Select **"Create API key"** and choose an existing Google Cloud project or create a new one.
5. Copy your API key and keep it safe.

---

## Windows Installation

Follow these steps using PowerShell in your project folder:

```powershell
# 1. Navigate to the project directory
cd path\to\NoteMind-AI

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
.venv\Scripts\Activate.ps1

# 4. Set UTF-8 encoding for smooth terminal output
$env:PYTHONUTF8 = "1"

# 5. Upgrade pip
python -m pip install --upgrade pip

# 6. Install dependencies
pip install -r requirements.txt

# 7. Run NoteMind AI
streamlit run app.py
```

### Safe PowerShell Execution Policy Workaround
If PowerShell blocks virtual environment script activation (`Activate.ps1 cannot be loaded because running scripts is disabled`), run the venv executables directly **without** changing global system security policies:

```powershell
# Install dependencies directly via venv python
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Launch Streamlit directly via venv
.\.venv\Scripts\streamlit.exe run app.py
```

---

## Windows Troubleshooting

- **Python not found**:
  Ensure Python 3.10+ is installed and check **"Add Python to PATH"** in the installer, or run via `py -3.11 app.py`.
- **pip not found**:
  Run `python -m ensurepip --upgrade` or call `python -m pip install ...`.
- **Streamlit not starting / Port conflict**:
  If port 8501 is occupied, launch on a custom port:
  ```powershell
  .\.venv\Scripts\streamlit.exe run app.py --server.port 8502
  ```
- **Missing C++ build tools error with ChromaDB**:
  The modern ChromaDB wheel (`chromadb>=0.5.0`) provides pre-built binaries on Windows for Python 3.10, 3.11, and 3.12. Ensure you are using a standard 64-bit Python release.

---

## Running the App

Once installed, simply run:
```powershell
streamlit run app.py
```
Your default browser will automatically open to `http://localhost:8501`.

---

## Security

- **In-Memory API Key**: Your Gemini API key is entered in a password-style field (`type="password"`) and kept exclusively in `st.session_state`.
- **No Disk or Database Storage**: The key is never written to disk, SQLite, ChromaDB, environment files, or server logs.
- **Git Protection**: `.gitignore` is pre-configured to ignore `.env`, `.streamlit/secrets.toml`, and the local ChromaDB database directory (`data/chroma/`).
- **Disconnect Button**: Disconnecting immediately flushes the key and all chat history from memory.

---

## Tech Stack

- **Python 3.11** — Core runtime
- **Streamlit** — Web user interface & session state
- **Google GenAI Python SDK (`google-genai`)** — Gemini embeddings & text generation
- **ChromaDB** — Local vector database with cosine similarity
- **pypdf** — Page-by-page PDF extraction and structure analysis

---

## Version 1 Limitations

- **No Google Sign-In or Accounts**: All sessions are local and temporary.
- **No OCR**: Image-only or scanned PDFs are detected and rejected with a friendly message. Only text-based PDFs are supported in Version 1.
- **Local ChromaDB**: Vectors are stored in a local directory (`data/chroma/`).
- **Session-Scoped Key**: Closing your browser tab requires re-entering your key.
- **Ephemeral Storage on Cloud**: Cloud deployments (like Streamlit Community Cloud) have ephemeral disk storage; reboots reset indexed vector data.

---

## Manual Grounding & Hallucination Test Checklist

Use this checklist to verify that the RAG pipeline is working as intended:

| Test Case | Steps | Expected Outcome |
| :--- | :--- | :--- |
| **1. Grounded Fact** | Upload notes containing *"Python was created by Guido van Rossum."* Ask: *"Who created Python?"* | Answers *"Python was created by Guido van Rossum"* with exact page citation (e.g. `Page 1`). |
| **2. Unrelated Query** | Ask: *"What is the capital of France?"* (when notes do not mention France). | Returns exactly: `Not found in your notes`. |
| **3. Insufficient Context** | Ask a question where only partial or unrelated keywords match. | Returns exactly: `Not found in your notes`. |
| **4. Citation Accuracy** | Ask a question answered on multiple pages or across multiple PDFs. | Displays all matching unique sources and page numbers. |
| **5. Scanned / Empty PDF** | Upload a blank or scanned image-only PDF. | Shows friendly warning: *"This PDF appears to be scanned or image-only. OCR is not supported in Version 1."* |
| **6. Remove Files** | Click "Remove all files" and confirm. | All indexed notes are cleared; subsequent questions state that no notes are indexed. |

---

## GitHub Safety Workflow

Before committing changes to Git, always verify that no credentials or private data are staged:

```powershell
# 1. Inspect status
git status

# 2. Review staged changes for sensitive keys or passwords
git diff --cached
```

### Git Initialization & Push
```powershell
# Initialize local repository
git init

# Stage all project files (safe files governed by .gitignore)
git add .

# Verify staged files
git status

# Create initial commit
git commit -m "Initial NoteMind AI version"

# Set branch name to main
git branch -M main

# Add your remote repository URL (replace with your repository link)
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>

# Push to GitHub
git push -u origin main
```

> ⚠️ **CRITICAL SECURITY NOTE**: If you accidentally commit a real API key to Git, **do not simply delete the file in a new commit**—the key remains in your Git commit history! Immediately go to [Google AI Studio](https://aistudio.google.com/) and **revoke / delete the compromised API key**.

---

## Streamlit Community Cloud Deployment Guide

You can deploy NoteMind AI for free on Streamlit Community Cloud:

1. Push your clean code to a GitHub repository (following the GitHub Safety steps above).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New app"**.
4. Select your **Repository**, **Branch** (`main`), and set **Main file path** to `app.py`.
5. Click **"Deploy!"**.
6. **API Key Handling**: NoteMind AI Version 1 does **NOT** require any server secrets. Each student enters their own Gemini API key in the web interface upon opening the app.
7. **Persistence Note**: Streamlit Community Cloud operates with ephemeral file systems. Any indexed notes in `data/chroma/` will be reset if the container restarts or reboots. This is expected behavior for Version 1.

---

## License

MIT License. Built for educational and research use.
