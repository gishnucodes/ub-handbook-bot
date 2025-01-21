#%%
import re
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from markitdown import MarkItDown
import streamlit as st

load_dotenv()

os.environ["GOOGLE_API_KEY"] = 'AIzaSyCO_JGvibGVMWHZNI6mxv-qc2wmdChLABk'

def load_file():
    md = MarkItDown()
    result = md.convert("grad-handbook-2024.pdf")
    docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", result.text_content)]
    documents = [item['page_content'] for item in docs]
    return documents

def get_vector_store(documents):
    # Initialize Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # texts = text_splitter.split_texts(documents)
    docstest = text_splitter.create_documents(documents)
    # Create the vector store
    vectorstore = FAISS.from_documents(docstest, embeddings)
    return vectorstore

def init_vector_store():
    documents = load_file()
    vectorstore = get_vector_store(documents)
    return vectorstore

#%%
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
#%%
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key='AIzaSyCO_JGvibGVMWHZNI6mxv-qc2wmdChLABk')
vectorstore = init_vector_store()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)
#%%
# Streamlit App
st.title("Graduate Student Handbook Bot - CSE UB")


# User input for research topic
topic = st.text_input("Enter the topic to search:")

# Generate and display markdown
if st.button("Search"):
  if topic:
    markdown_content = qa_chain.invoke({"query": topic})
    st.markdown(markdown_content.get('result'))

    # Copy to clipboard button
    # st.button("Copy to Clipboard", on_click=lambda: clipboard.copy(markdown_content))
  else:
    st.warning("Please enter the topic to search")