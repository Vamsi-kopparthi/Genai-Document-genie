# GENAI - LANGCHAIN - PROJECT
🧞 Document Genie: Multi-PDF Summarizer &amp; Q&amp;A
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-brightgreen.svg)
![LangChain](https://img.shields.io/badge/LangChain-Orchestration-orange.svg)
![Google Gemini](https://img.shields.io/badge/Google-Gemini%20LLM-red.svg)
![Status](https://img.shields.io/badge/Open%20Source-Yes-purple.svg)
# 🧞 Document Genie: Multi-PDF Summarizer & Q&A

Document Genie is a powerful web application built with **Streamlit** and **LangChain** that allows you to process multiple PDF documents at once. It generates a consolidated summary of all documents and enables you to ask specific questions about the content using Google's state-of-the-art Gemini models.

---

## ✨ Features

- 📂 **Multi-File Upload**: Upload one or more PDF documents simultaneously.  
- 🤖 **AI-Powered Summarization**: Generates a single, coherent summary from all provided documents using a map-reduce technique.  
- 💬 **Interactive Q&A**: Ask questions in natural language and get precise answers sourced directly from the document content.  
- ⚡ **High-Speed Processing**: Powered by the efficient `gemini-1.5-flash-latest` model for fast results.  
- 🖥️ **Simple Web Interface**: Easy-to-use interface built with Streamlit.  

---

## 🛠️ Tech Stack

- **Backend:** Python  
- **Web Framework:** Streamlit  
- **LLM Orchestration:** LangChain  
- **LLM:** Google Gemini (`1.5-flash-latest`, `1.5-pro-latest`)  
- **Embeddings:** Google Generative AI Embeddings  
- **Vector Store:** FAISS (Facebook AI Similarity Search)  
- **PDF Processing:** PyPDF2  

---

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.  

### 1. Prerequisites
- Python **3.8+**  
- A **Google API Key** with access to the Gemini API  

---

### 2. Setup and Installation

#### Step 1: Clone or Download the Project
Get the project files onto your computer. Place `app.py` in a new project folder.

#### Step 2: Create Project Files
Inside your project folder, create two new files:  
`requirements.txt` and `.env`.  
Your folder structure should look like this:

---
- document-genie/
- ├── .env
- ├── app.py
- └── requirements.txt


---

#### Step 3: Populate `requirements.txt`
Copy and paste the following libraries:

- streamlit
- langchain
- langchain-google-genai
- pypdf2
- python-dotenv
- faiss-cpu
- langchain-community
- nest-asyncio

---

#### Step 4: Set Up Your Google API Key
- Obtain your API key from **Google AI Studio**.  
- Open `.env` and add:


⚠️ Make sure **Vertex AI API** is enabled in your Google Cloud project to avoid `404` or `429` errors.  


---

#### Step 5: Create a Virtual Environment & Install Dependencies
Run these commands in your terminal:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install all required libraries
pip install -r requirements.txt
```
---

#### step 6. Running the Application

Start the Streamlit app:

- streamlit run app.py


-  Your browser will open with Document Genie. Upload PDFs, generate summaries, and query content instantly! 🚀
---

📜 License

- This project is Open Source and created by Vamsi Kopparthi.

- This project is open-source. Feel free to modify and enhance it!
