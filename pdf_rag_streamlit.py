import streamlit as st
import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"


def ingest_pdf(uploaded_file):
    """Load PDF document from uploaded file."""
    if uploaded_file is not None:
        temp_path = os.path.join("./data", uploaded_file.name)
        os.makedirs("./data", exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        loader = UnstructuredPDFLoader(file_path=temp_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data, temp_path
    else:
        logging.error("No file uploaded.")
        st.error("Please upload a PDF file.")
        return None, None


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


def load_vector_db(documents):
    """Create and load the vector database."""
    ollama.pull(EMBEDDING_MODEL)
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    chunks = split_documents(documents)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name=VECTOR_STORE_NAME,
        persist_directory=PERSIST_DIRECTORY,
    )
    vector_db.persist()
    logging.info("Vector database created and persisted.")
    return vector_db


def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI assistant. Generate five
        different versions of the given user question to retrieve relevant documents
        from a vector database. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever


def create_chain(retriever, llm):
    """Create the chain with preserved syntax."""
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    logging.info("Chain created with preserved syntax.")
    return chain


def main():
    st.title("📄 RAG Document Assistant 🤖")

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            documents, file_path = ingest_pdf(uploaded_file)
            if documents is None:
                return

            vector_db = load_vector_db(documents)

            llm = ChatOllama(model=MODEL_NAME)
            retriever = create_retriever(vector_db, llm)
            chain = create_chain(retriever, llm)

        st.success("File processed successfully! You can now ask questions.")

        # User input
        user_input = st.text_input("Enter your question:")

        if user_input:
            with st.spinner("Generating response..."):
                try:
                    response = chain.invoke(input=user_input)
                    st.markdown("**Assistant:**")
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload a PDF file to begin.")


if __name__ == "__main__":
    main()