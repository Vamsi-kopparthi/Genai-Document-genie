import streamlit as st
import nest_asyncio
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

# Apply the patch for asyncio
nest_asyncio.apply()

# Load environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# --- Helper Functions ---

def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    """Splits text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Creates a FAISS vector store from text chunks."""
    try:
        # Use the recommended embedding model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# --- Main Application Logic ---

def main():
    st.set_page_config(page_title="Document Genie 🧞", layout="wide")
    st.title("Document Genie: Summarize & Ask 🧞")
    st.markdown("---")

    # Initialize session state variables
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # --- Sidebar for Instructions and File Upload ---
    with st.sidebar:
        st.header("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDF files and click 'Process'",
            accept_multiple_files=True,
            type="pdf"
        )
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing... This may take a moment."):
                    # 1. Extract and chunk text
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)

                    # 2. Create vector store for Q&A
                    st.session_state.vector_store = get_vector_store(text_chunks)
                    
                    if st.session_state.vector_store:
                        # 3. Create the Q&A chain
                        llm_qa = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)
                        prompt_template = """
                        Answer the question as detailed as possible from the provided context.
                        If the answer is not in the provided context, just say, "The answer is not available in the documents."
                        Don't provide a wrong answer.\n\n
                        Context:\n{context}\n
                        Question:\n{question}\n

                        Answer:
                        """
                        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                        st.session_state.qa_chain = load_qa_chain(llm_qa, chain_type="stuff", prompt=prompt)

                        # 4. Create and run the summarization chain
                        docs_for_summary = [Document(page_content=chunk) for chunk in text_chunks]
                        llm_summary = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
                        summary_chain = load_summarize_chain(llm_summary, chain_type="map_reduce")
                        st.session_state.summary = summary_chain.run(docs_for_summary)
                        
                        st.success("Documents processed! You can now view the summary and ask questions.")
            else:
                st.warning("Please upload at least one PDF file.")

    # --- Main Content Area for Summary and Q&A ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Summary of Your Documents")
        if st.session_state.summary:
            st.write(st.session_state.summary)
        else:
            st.info("Upload documents and click 'Process' to see the summary here.")

    with col2:
        st.subheader("Ask a Question")
        if st.session_state.qa_chain:
            user_question = st.text_input("What would you like to know from the documents?")
            if user_question:
                with st.spinner("Finding an answer..."):
                    # Use the vector store to find relevant docs
                    docs = st.session_state.vector_store.similarity_search(user_question)
                    # Run the Q&A chain with the found docs
                    response = st.session_state.qa_chain(
                        {"input_documents": docs, "question": user_question},
                        return_only_outputs=True
                    )
                    st.write("#### Answer:")
                    st.write(response["output_text"])
        else:
            st.info("Your Q&A assistant will be ready once documents are processed.")

if __name__ == "__main__":
    main()