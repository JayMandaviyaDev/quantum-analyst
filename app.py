import streamlit as st
from dotenv import load_dotenv
import os
import traceback
import tempfile

# Updated LangChain imports for latest version
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables from .env file
load_dotenv()

# ---------------------------
# LLM Selection Functions
# ---------------------------
def create_llm(llm_choice):
    """Create LLM based on user choice"""
    
    if llm_choice == "OpenAI GPT":
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        except ImportError:
            st.error("OpenAI package not installed. Run: pip install langchain-openai")
            return None
        except Exception as e:
            st.error(f"OpenAI setup failed: {str(e)}")
            return None
    
    elif llm_choice == "Ollama (Local)":
        try:
            from langchain_community.llms import Ollama
            return Ollama(
                model="llama2",  # or "mistral", "codellama"
                temperature=0.1
            )
        except Exception as e:
            st.error(f"Ollama setup failed: {str(e)}. Make sure Ollama is installed and running.")
            return None
    
    elif llm_choice == "Hugging Face":
        try:
            # Try multiple models in order of preference
            models_to_try = [
                "google/flan-t5-base",
                "microsoft/DialoGPT-medium",
                "distilbert-base-uncased"
            ]
            
            for model in models_to_try:
                try:
                    from langchain_huggingface import HuggingFaceEndpoint
                    return HuggingFaceEndpoint(
                        repo_id=model,
                        temperature=0.1,
                        max_new_tokens=512,
                        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
                    )
                except:
                    continue
            
            # Final fallback to community version
            from langchain_community.llms import HuggingFaceHub
            return HuggingFaceHub(
                repo_id="google/flan-t5-base",
                model_kwargs={"temperature": 0.1, "max_length": 512},
                huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
            )
            
        except Exception as e:
            st.error(f"Hugging Face setup failed: {str(e)}")
            return None
    
    return None

# ---------------------------
# Core RAG Chain Function
# ---------------------------
def create_rag_chain(uploaded_file, llm_choice):
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

        # 3. Create the LLM
        llm = create_llm(llm_choice)
        if llm is None:
            raise ValueError(f"Failed to initialize {llm_choice} LLM")

        # 4. Create the prompt template
        prompt_template = """You are a helpful AI assistant. Answer the question based on the provided context. If you cannot find the answer in the context, say "I cannot find this information in the provided document."

Context: {context}

Question: {input}

Answer:"""
        
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

    # LLM Selection in sidebar
    with st.sidebar:
        st.header("🤖 LLM Configuration")
        llm_choice = st.selectbox(
            "Choose your LLM:",
            ["Hugging Face", "OpenAI GPT", "Ollama (Local)"],
            help="Select which Large Language Model to use"
        )
        
        # Show API key requirements based on selection
        if llm_choice == "OpenAI GPT":
            if not os.getenv("OPENAI_API_KEY"):
                st.error("⚠️ OpenAI API key required in .env file")
                st.code("OPENAI_API_KEY=your_openai_key_here")
        elif llm_choice == "Hugging Face":
            if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
                st.error("⚠️ Hugging Face token required in .env file")
                st.code("HUGGINGFACEHUB_API_TOKEN=your_hf_token_here")
        elif llm_choice == "Ollama (Local)":
            st.info("💡 Make sure Ollama is installed and running locally")
            st.markdown("Install: `curl -fsSL https://ollama.ai/install.sh | sh`")
            st.markdown("Run: `ollama pull llama2`")

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
                    with st.spinner(f"🤖 Analyzing with {llm_choice}..."):
                        try:
                            rag_chain = create_rag_chain(uploaded_file, llm_choice)
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
                            st.error(f"❌ Error with {llm_choice}: {str(e)}")
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
    1. **Choose** your preferred LLM from the sidebar
    2. **Upload** a PDF document using the file uploader
    3. **Ask** specific questions about the document content
    4. **Get** AI-powered answers based on the document context
    
    ### 🔧 LLM Options:
    - **Hugging Face**: Free, requires HF token, cloud-based
    - **OpenAI GPT**: Paid, requires API key, high quality
    - **Ollama**: Free, local installation required, private
    
    ### 💡 Example questions:
    - "What are the key financial metrics mentioned?"
    - "Summarize the main conclusions"
    - "What risks are identified in this document?"
    """)

if __name__ == "__main__":
    main()