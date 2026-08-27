"""Module to handle text splitting, vector indexing, and QA chain execution."""

from typing import List
from langchain_core.documents import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import config

def build_vector_store(documents: List[Document]) -> FAISS:
    """Splits documents and stores embeddings in a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    split_docs = splitter.split_documents(documents)
    
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store

def create_rag_chain(vector_store: FAISS):
    """Sets up the retrieval and prompt chain with Ollama."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Answer the question based only on the following context from the video:
{context}

Question: {question}
Answer:"""
    )
    
    def answer_query(query: str):
        docs = retriever.invoke(query)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        chain = prompt | llm
        return chain.invoke({"context": context_text, "question": query})
        
    return answer_query