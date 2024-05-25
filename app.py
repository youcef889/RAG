import streamlit as st  # Importing the Streamlit library for creating the UI
#from langchain.vectorstores.chroma import Chroma  # Importing Chroma for vector store operations
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate  # Importing ChatPromptTemplate for prompt formatting
from langchain_community.llms.ollama import Ollama  # Importing Ollama for language model operations
from get_embedding_function import get_embedding_function  # Importing the custom embedding function

# Constants
CHROMA_PATH = "chroma"  # Path to the Chroma vector store
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    """
    Function to query the RAG (Retrieval-Augmented Generation) system.
    
    Args:
    - query_text (str): The input query text.

    Returns:
    - response_text (str): The generated response text.
    """
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    # Construct the context text from the search results.
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Format the prompt using the template.
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Initialize the model and get the response.
    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    # Extract sources from the search results.
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response


# Streamlit UI
st.title("RAG Query Interface")  # Setting the title of the app
st.write("Enter a query text to get a response based on the context from the vector store.")  # Description

# Input field for the query text
query_text = st.text_input("Query Text:", value="", help="Enter the query text you want to ask.")

# Button to submit the query
if st.button("Submit Query"):
    if query_text:  # Check if query_text is not empty
        with st.spinner('Processing...'):  # Show a spinner while processing
            response = query_rag(query_text)  # Call the query_rag function with the input query
            st.success("Query processed!")  # Show a success message
            st.write(response)  # Display the response
         
    else:
        st.error("Please enter a query text.")  # Show an error message if the input is empty

# Add a sidebar with additional information
st.sidebar.title("About")
st.sidebar.write("""
This is a simple RAG (Retrieval-Augmented Generation) query interface built with Streamlit.
Enter your query and get a response based on the context retrieved from a vector store.
""")

