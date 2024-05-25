# Import necessary libraries for the Streamlit application
import streamlit as st
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma

# Define constants for file paths
CHROMA_PATH = "chroma"
DATA_PATH = "data"

# Function to clear the database by removing the Chroma directory
def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

# Function to load documents from the specified data directory
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# Function to split documents into chunks for processing
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

# Function to add chunks to the Chroma database
def add_to_chroma(chunks):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    st.write(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        st.write(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        st.write("No new documents to add")

# Function to calculate unique chunk IDs
def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

# Main function to create the Streamlit UI
def main():
    st.title("Document Management with Chroma")
    st.write("This app allows you to manage documents using Chroma and LangChain.")

    # Reset database button
    if st.button("Reset Database"):
        st.write("Clearing Database...")
        clear_database()
        st.success("Database cleared!")

    # Upload documents button
    if st.button("Upload and Process Documents"):
        st.write("Loading documents...")
        documents = load_documents()
        st.write(f"Loaded {len(documents)} documents.")
        
        st.write("Splitting documents into chunks...")
        chunks = split_documents(documents)
        st.write(f"Split into {len(chunks)} chunks.")
        
        st.write("Adding chunks to Chroma...")
        add_to_chroma(chunks)
        st.success("Documents processed and added to database!")

# Entry point for the Streamlit app
if __name__ == "__main__":
    main()

