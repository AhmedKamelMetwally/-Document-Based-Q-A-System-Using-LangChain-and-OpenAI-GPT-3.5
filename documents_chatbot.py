import streamlit as st  
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader  
from langchain.embeddings import HuggingFaceEmbeddings  
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.vectorstores import FAISS  
from langchain.schema import Document, BaseRetriever  
from langchain.retrievers.document_compressors import EmbeddingsFilter 
from langchain.retrievers import ContextualCompressionRetriever 
from langchain.chains import ConversationalRetrievalChain  
from langchain.chains.base import Chain  
from langchain_community.chat_models import ChatOllama 
from langchain.memory import ConversationBufferMemory  
import tempfile  
import os  
import pathlib 
from typing import List  
# Setting up  Ollama model to be used for the chatbot
llm = ChatOllama(model="gemma")  

# Defining an exception for data files that are not .pdf or .txt or .docx or .doc
class loader_exception(Exception):  
    pass

# a class to use document loader suitable for file extension
class Document_loader:
    
    supported_extensions = {
        ".pdf": PyPDFLoader,  
        ".txt": TextLoader,  
        ".docx": UnstructuredWordDocumentLoader,  
        ".doc": UnstructuredWordDocumentLoader  
    }

# Loading documents from file path
def load_doc(filepath: str) -> List[Document]:
    # Getting the file extension of the uploaded file
    extension = pathlib.Path(filepath).suffix  
     # Getting the corresponding loader based on the file extension
    loader_class = Document_loader.supported_extensions.get(extension) 
    if not loader_class:
        raise loader_exception(f"Unsupported file type: {extension}")  
    # Initializing the appropriate loader for the file type
    loader = loader_class(filepath)  
    # Loading the document and return it as a list of Document objects
    return loader.load()  

# Configuring retriever using FAISS for vector search
def retriever_config(docs: List[Document]) -> BaseRetriever:
    # Splitting the documents into chunks of 1500 characters
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500)  
    # Applying the splitter to the documents
    splits = splitter.split_documents(docs)  
    # Creating embeddings using HuggingFace  model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
    # Storing the document embeddings in a FAISS vector 
    vec_db = FAISS.from_documents(splits, embeddings)  
    # Converting the vector  into a retriever for querying
    retriever = vec_db.as_retriever()  
    # Using embeddings filter for  filtering
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings) 
    # Returning the configured retriever 
    return ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)  

# Configuring the conversational chain 
def config_chain(retriever: BaseRetriever) -> Chain:
    # Initializing memory to store the history of conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  
    return ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)  

# Processing uploaded files and configuring the  chain
def configure_QA_chain(uploaded_files) -> Chain:
    # List to store the loaded documents
    docs = []  
    # Creating a temporary directory to store the uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:  
        # Looping over each uploaded file
        for file in uploaded_files:
            # Getting the path to save the uploaded file in the directory  
            temp_path = os.path.join(temp_dir, file.name)
            # Openning the file in write-binary mode  
            with open(temp_path, "wb") as f:
                # Writting the file content to the temporary file  
                f.write(file.getvalue())
                # Loading the document and extending the docs list with the content  
            docs.extend(load_doc(temp_path))
    # Configuring the retriever with the loaded documents  
    retriever = retriever_config(docs)  
    return config_chain(retriever)  

#deployment

st.set_page_config(page_title="Ahmed Kamel Chatbot", layout="wide")  
st.title("Ahmed Kamel Chatbot")  
st.markdown("Upload PDF, TXT, or DOCX files and ask questions ")  
# Upload section
uploaded_files = st.file_uploader("Upload your documents", type=["pdf", "txt", "docx", "doc"], accept_multiple_files=True)  


if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None  
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  


if uploaded_files and st.button(" Process Documents"):  
    try:
        st.session_state.qa_chain = configure_QA_chain(uploaded_files)  
        st.success("Documents processed. You can now ask questions.")  
    except Exception as e:
        st.error(f" Error: {str(e)}")  


if st.session_state.qa_chain:
    user_question = st.text_input(" Ask a question:")  
    if user_question:  
        try:
            
            response = st.session_state.qa_chain.invoke({"question": user_question})
            
            answer = response["result"]  
            
            st.session_state.chat_history.append(("You", user_question))
            st.session_state.chat_history.append(("AI", answer))
        except Exception as e:
            st.error(f"Error: {str(e)}")  


if st.session_state.chat_history:
    st.markdown("---") 
    st.markdown("###  Conversation")  
    for speaker, message in st.session_state.chat_history:  
        st.markdown(f"**{speaker}:** {message}") 


