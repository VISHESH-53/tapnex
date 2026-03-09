import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

import os

# -------------------------
# OpenAI Key
# -------------------------
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# -------------------------
# Streamlit UI
# -------------------------
st.title("🤖 TapNex AI Customer Support Agent")

st.write(
    "Ask questions about TapNex services such as NFC payments, recharge systems, "
    "event technology services, and refund policies."
)

# -------------------------
# Load Knowledge Base
# -------------------------
@st.cache_resource
def load_vector_store():

    with open("knowledge_base.txt", "r", encoding="utf-8") as f:
        text = f.read()

    documents = [Document(page_content=text)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    vector_store = FAISS.from_documents(docs, embeddings)

    return vector_store


vector_store = load_vector_store()

retriever = vector_store.as_retriever(search_kwargs={"k":3})

# -------------------------
# LLM
# -------------------------
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# -------------------------
# Chat Memory
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------
# User Query
# -------------------------
if prompt := st.chat_input("Ask about TapNex..."):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    result = qa(prompt)

    answer = result["result"]
    sources = result["source_documents"]

    response = answer + "\n\n**Sources:**\n"

    for doc in sources:
        response += "- TapNex Knowledge Base\n"

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
