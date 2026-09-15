# pdf_rag_application
A lightweight, local RAG (Retrieval-Augmented Generation) application built with Streamlit and LangChain. It lets you upload multiple PDF documents and chat with them naturally, keeping track of your conversation history so you can ask follow-up questions without losing context.

Multi-PDF support: Upload one or multiple PDFs and query them all at once.

Context-aware: Uses vector similarity search so answers are pulled directly from your documents (no hallucinations).

Chat memory: Remembers the conversation flow, making back-and-forth Q&A feel natural.

Clean UI: Styled with custom CSS in Streamlit to give it a modern dark-mode look instead of the default plain layout.

Frontend: Streamlit

Orchestration & RAG: LangChain

Vector Store: FAISS

LLM & Embeddings: Google Gemini (gemini-3.5-flash-lite & Google Generative AI Embeddings)

PDF Parsing: PyPDF2

Extraction: Reads the text out of your uploaded PDFs page by page.

Chunking: Splits the text into manageable chunks using a recursive character splitter with overlap.

Embedding: Turns those chunks into vector embeddings and indexes them locally using FAISS.

Retrieval & Generation: When you ask a question, it finds the most relevant chunks from the vector store and passes them to Gemini along with your chat history to construct a precise answer.
## Live Demo
[https://sahel-project-3.streamlit.app/]
