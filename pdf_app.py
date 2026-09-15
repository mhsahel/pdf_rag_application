from multiprocessing import process

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import GoogleRateLimitError


from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


def get_raw_data(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


def get_text_chunk(text):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks


def get_vector_store(chunks):
    embedd_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectors = FAISS.from_texts(texts=chunks, embedding=embedd_model)
    return vectors


def conversation_chain(vector_store):
    embed_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.3,timeout = 30)
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using the context below. "
                   "If the answer isn't in the context, say you don't know.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    questions_answers_chain = create_stuff_documents_chain(embed_llm, prompt)
    rag_chain = create_retrieval_chain(retriever, questions_answers_chain)
 
    return rag_chain


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process a PDF first.")
        return

    response = st.session_state.conversation.invoke({
        "input": user_question,
        "chat_history": st.session_state.messages
    })

    st.session_state.messages.append(HumanMessage(content=user_question))
    st.session_state.messages.append(AIMessage(content=response["answer"]))


 
    for message in st.session_state.messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)
        
def css_file():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(180deg, #0f1117 0%, #1a1d29 100%);
        }
        [data-testid="stSidebar"] {
            background-color: #161925;
            border-right: 1px solid #2a2e3f;
        }
        .main-title {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }
        .subtitle {
            color: #9ca3af;
            font-size: 0.95rem;
            margin-top: -8px;
        }
        [data-testid="stChatMessage"] {
            border-radius: 14px;
            padding: 4px 8px;
            margin-bottom: 6px;
        }
        .stButton button {
            background: linear-gradient(90deg, #6366f1, #a855f7);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            transition: transform 0.15s ease;
        }
        .stButton button:hover {
            transform: scale(1.02);
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title='Multiple chat with pdfs', page_icon=':books:')
    css_file()
    

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    st.markdown('<p class="main-title">📚 DocChat AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Chat with your PDFs </p>', unsafe_allow_html=True)
    st.divider()
    

   

    user_question = st.chat_input("Ask a question about your documents:")
    with st.spinner("process"):
        if user_question:
            handle_userinput(user_question)

    with st.sidebar:
        
        st.markdown("### 📁 Document Manager")
        st.caption("Upload PDFs, then click Process to build your knowledge base.")
        pdf_docs = st.file_uploader(
           "Drop your PDF files here",
            accept_multiple_files=True,
            type = ['pdf']
        )
        if st.button('Process Documents'):
            if not pdf_docs:
                st.warning("Please upload at least one PDF first.")
            else:
                try:
                    with st.spinner('Reading, chunking, and embedding your documents...'):
                        raw_text_data = get_raw_data(pdf_docs)
                        text_chunks = get_text_chunk(raw_text_data)
                        vector_store = get_vector_store(text_chunks)
                        st.session_state.conversation = conversation_chain(vector_store)
                        st.success("Ask anything about your documents.")
                        
                
                except GoogleRateLimitError:
                    st.error("Google's API rate limit was hit. Please wait a bit and try again.")
                except Exception as e:
                    st.error(f"Something went wrong while processing: {e}")

if __name__ == '__main__':
    main()






    
    


    