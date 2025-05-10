# importing functions to load various types of documents
# To load pdf documents
from langchain.document_loaders import PyPDFLoader
#to load .txt files
from langchain.document_loaders import TextLoader
# to load .doc and .docx files
from langchain.document_loaders import UnstructuredWordDocumentLoader
#such that I can use Void type in functions
from typing import Any
#to extract the extension of the file from file path
import pathlib
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.schema import Document, BaseRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import os
import tempfile
#importing my openai api key (I removed my api key for privacy)
os.environ["OPENAI_API_KEY"] = ""
class Document_loader(object):
    #calling the suitable loader function depending on file extension
    supported_extensions={".pdf":PyPDFLoader,".txt":TextLoader,".docx":UnstructuredWordDocumentLoader,".doc":UnstructuredWordDocumentLoader}
#exception if file extension is not pdf or .txt or .doc or .docx
class loader_exception(Exception):
    pass
#Document loader function (takes file path and return this file as list of documents)
def load_doc(filepath:str)->list[Document]:
    # getting the extension of the file to call the suitable function
    extension = pathlib.Path(filepath).suffix
    #calling the suitable function for the extension
    loader=Document_loader.supported_extensions.get(extension)
    #raising exception if file extension is not pdf or .txt or .doc or .docx
    if not loader:
        raise loader_exception(
            "cannot load this type of file"
        )
    #calling loader function to load the file through the file path
    load_doc=loader(filepath)
    docs=load_doc.load()
    #returning list of documents
    return docs
# a function to split the documents ,converting to embeddings , stores them in an in-memory vector database , and returns a retriever that uses Max Marginal Relevance to find the most relevant results
def retriever_config (docs:list[Document])->BaseRetriever:
    #splitting documents into splits and each split will be composed of 1500 characters
    splitter=RecursiveCharacterTextSplitter(chunk_size=1500)
    splits=splitter.split_documents(docs)
    #converting splits into embeddings using all-MiniLM-L6-v2 model from HUGGINGFACE , and creating an in-memory vector database (vec_DB) from  documents embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vec_DB=DocArrayInMemorySearch.from_documents(splits, embeddings)
    #Converting the vector database into a LangChain retriever such that I can retrieve data from documents 
    retriever = vec_DB.as_retriever()
    #filtering the documents retrieved by the  retriever based on their  similarity to the  query , using embedding vectors to compare the query  retrieved documents
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings)
    return ContextualCompressionRetriever(base_compressor=embeddings_filter,base_retriever=retriever)
#defining a chain that uses the provided retriever to fetch relevant documents and then passes them to gpt-3.5-turbo LLM to generate  answer
def config_chain(retriever: BaseRetriever) -> Chain:
    #ConversationBufferMemory function stores the full history of the conversation in memory
    #the chat history will be in history variable
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    #preparing LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, streaming=True)
    return ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)
#defining a function to read loaded documents , configure retriever and configure chain
def configure_QA_chain(uploaded_files):
    # Initializing an empty list to store loaded documents
    docs = []

    # Creating a temporary directory to store uploaded files during processing
    temp_dir = tempfile.TemporaryDirectory()

    # Looping through each uploaded file
    for file in uploaded_files:
        # Defining a temporary file path inside the temp directory
        temp_filepath = os.path.join(temp_dir.name, file.name)

        # Writing the uploaded files content to the temporary file
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())

        # Loading the document from the temporary file and add to docs list
        docs.extend(load_doc(temp_filepath))

    # Creating a retriever from the processed documents
    retriever = retriever_config(docs=docs)

    # Configuring and returning the full conversational QA chain
    return config_chain(retriever=retriever)


file_paths = input("Enter comma separated paths to your documents: ").split(",")

# Striping any  whitespace from each file path
file_paths = [path.strip() for path in file_paths]

# Initializing an empty list to store all loaded documents
all_docs = []

# Looping over each file path provided by the user
for path in file_paths:
    try:
        #  loading the document and adding it to the list
        all_docs.extend(load_doc(path))
    except loader_exception :
        
        print("loading error")

# Checking if no documents were successfully loaded
if not all_docs:
    # If list is empty exit the program
    exit()
    


retriever = retriever_config(all_docs)


qa_chain = config_chain(retriever)

print("\nYou can now ask questions about your documents. Type exit to quit.")


while True:
    
    question = input("\nQuestion: ")
    
    # If the user types exit end the program
    if question.lower().strip() == "exit":
        print("Program ended.")
        break

    response = qa_chain.run(question)

    # Printing the response returned by the QA system
    print(f"Answer: {response}")










