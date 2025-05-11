# -Document-Based-Q-A-System-Using-LangChain-and-OpenAI-GPT-3.5
 Built an AI Chatbot with Streamlit and Langchain! 

I'm excited to share a project I've been working on — an AI-powered Chatbot built with Streamlit and Langchain that allows users to upload PDF, DOC, and TXT documents and ask questions based on the content of those documents! 

 What Makes This Chatbot Special?
Document Upload and Interaction: Users can upload their documents in PDF, DOCX, or TXT format and then interact with the chatbot to ask detailed questions about the document content. Whether it’s a contract, research paper, or text file, the AI can extract and provide relevant answers.

Semantic Understanding: Powered by HuggingFace embeddings and FAISS for fast semantic search, the chatbot understands the context and content of the documents to provide accurate answers to user queries. It doesn’t just match keywords; it understands the meaning behind the text, thanks to the language model.

Contextual Memory: The chatbot leverages Langchain's conversational memory to maintain the context of the conversation. It can recall previous interactions and seamlessly integrate new information from the document for better user engagement.

 Tools and Technologies Used:
Streamlit: Used to build the interactive frontend for the application. It allows users to easily upload documents and interact with the chatbot in real-time. Streamlit makes it easy to deploy and share the app with others.

Langchain: Langchain is the core engine behind the document processing. It allows for effective document splitting, semantic search, and interaction with large language models. Langchain's Contextual Compression Retriever helps refine document retrieval and provide relevant results based on the user's queries.

Ollama (Gemma): I used Ollama's Gemma model, a state-of-the-art conversational model that powers the chat functionality. It can handle dynamic user queries with excellent natural language understanding and generation capabilities.

FAISS: For efficient document search and retrieval, I used FAISS (Facebook AI Similarity Search). FAISS is perfect for large-scale vector search and helps quickly retrieve the most relevant sections from large documents.

HuggingFace Embeddings: The system leverages HuggingFace embeddings to convert the document text into vector representations, enabling semantic search capabilities. The embeddings provide a high-quality, deep understanding of the content.

 How It Works:
User Uploads Document: The user uploads a document (PDF, DOC, or TXT).

Document Processing: The document is loaded and split into smaller chunks for processing. Each chunk is then embedded using HuggingFace's embeddings model.

Search and Retrieval: When the user asks a question, the chatbot searches through the document chunks using FAISS to find the most relevant information.

Contextual Conversations: The chatbot keeps track of the conversation history using Langchain’s ConversationBufferMemory so it can reference previous interactions and provide more context-aware responses.

 What Can You Do?
Upload a document and ask questions about the content — whether it's summarizing sections, extracting key points, or answering detailed inquiries.

Explore how semantic search can significantly improve the way we interact with documents and data.

Applications:
Legal: Quickly review contracts and documents, extracting key clauses and insights.

Academic: Help students and researchers by answering questions from research papers, textbooks, and lecture notes.

Business: Provide insights from reports, market analysis, and business proposals.

Customer Support: Chatbot-driven customer support for document-heavy industries (e.g., insurance, HR).


