import streamlit as st
from dotenv import load_dotenv
import os
import traceback
import tempfile

# Updated LangChain imports for latest version
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables from .env file
load_dotenv()

# ---------------------------
# Core RAG Chain Function
# ---------------------------
def create_rag_chain(uploaded_file):
    """
    Builds a Retrieval-Augmented Generation (RAG) chain from an uploaded PDF.
    """
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name

    try:
        # 1. Load and split the PDF
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        
        # Check if documents were loaded successfully
        if not documents:
            raise ValueError("No content could be extracted from the PDF")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)

        # 2. Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(split_docs, embeddings)

        # 3. Create the LLM with updated syntax
        llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-xxl",
            temperature=0.1,
            max_new_tokens=512,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )

        # 4. Create the prompt template
        prompt_template = """
        Answer the user's question based only on the following context. If you cannot find the answer in the context, say "I cannot find this information in the provided document."

        <context>
        {context}
        </context>

        Question: {input}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # 5. Create the document chain
        document_chain = create_stuff_documents_chain(llm, prompt)

        # 6. Create the retrieval chain
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        return retrieval_chain

    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.set_page_config(
        page_title="Quantum Analyst", 
        page_icon="⚛️",
        layout="wide"
    )
    
    st.title("Quantum Analyst ⚛️")
    st.markdown("Upload a financial or legal PDF and ask questions in plain English.")

    # Verify Hugging Face API token is available
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        st.error("⚠️ Missing Hugging Face API token. Please add `HUGGINGFACEHUB_API_TOKEN` to your .env file.")
        st.markdown("""
        To get your API token:
        1. Go to [Hugging Face](https://huggingface.co/settings/tokens)
        2. Create a new token with 'Read' permissions
        3. Add it to your .env file as: `HUGGINGFACEHUB_API_TOKEN=your_token_here`
        """)
        st.stop()

    # Create two columns for better layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf",
            help="Upload a PDF document to analyze"
        )
        
        if uploaded_file:
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            st.write(f"File size: {len(uploaded_file.getvalue())} bytes")

    with col2:
        st.subheader("💬 Ask Questions")
        
        if uploaded_file:
            question = st.text_input(
                "Enter your question about the document:",
                placeholder="e.g., What are the main financial highlights?",
                help="Ask specific questions about the content of your PDF"
            )
            
            if st.button("🔍 Analyze", type="primary"):
                if question.strip():
                    with st.spinner("🤖 Analyzing your document..."):
                        try:
                            rag_chain = create_rag_chain(uploaded_file)
                            result = rag_chain.invoke({"input": question})
                            
                            st.success("✅ Analysis Complete!")
                            
                            # Display the answer
                            st.subheader("📝 Answer:")
                            answer = result.get("answer", "No answer returned.")
                            st.markdown(answer)
                            
                            # Display source documents (optional)
                            if "context" in result:
                                with st.expander("📚 Source Context"):
                                    context_docs = result.get("context", [])
                                    for i, doc in enumerate(context_docs, 1):
                                        st.write(f"**Source {i}:**")
                                        st.write(doc.page_content[:500] + "...")
                                        st.write("---")
                                        
                        except Exception as e:
                            st.error("❌ An error occurred during processing.")
                            with st.expander("Error Details"):
                                st.code(traceback.format_exc())
                else:
                    st.warning("Please enter a question.")
        else:
            st.info("👆 Please upload a PDF file first to ask questions.")

    # Add footer with usage instructions
    st.markdown("---")
    st.markdown("""
    ### 📋 How to use:
    1. **Upload** a PDF document using the file uploader
    2. **Ask** specific questions about the document content
    3. **Get** AI-powered answers based on the document context
    
    ### 💡 Example questions:
    - "What are the key financial metrics mentioned?"
    - "Summarize the main conclusions"
    - "What risks are identified in this document?"
    """)

if __name__ == "__main__":
    main()