import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# ---------- Load Knowledge Base ----------
with open("knowledge_base.txt", "r") as f:
    text = f.read()

# ---------- Split Documents ----------
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=300,
    chunk_overlap=50
)

docs = text_splitter.create_documents([text])

# ---------- Create Embeddings ----------
embeddings = OpenAIEmbeddings()

# ---------- Vector Database ----------
vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()

# ---------- LLM ----------
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# ---------- Streamlit UI ----------
st.title("🤖 TapNex AI Customer Support")

st.write("Ask questions about TapNex services")

query = st.text_input("Enter your question")

if st.button("Ask AI"):

    if query:
        response = qa_chain.run(query)

        st.write("### AI Response")
        st.write(response)
