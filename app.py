import streamlit as st
from langchain_aws import ChatBedrock
from langchain_community.vectorstores.pgvector import PGVector
from langchain_aws import BedrockEmbeddings
import os

# Aurora connection
CONNECTION_STRING = os.getenv("AURORA_DB_URL")

# Embeddings
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2",
    region_name=os.getenv("BEDROCK_REGION")
)

# Vector store
vectorstore = PGVector(
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
    collection_name="bedrock_kb"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# LLM
llm = ChatBedrock(
    model_id="amazon.nova-pro-v1:0",
    region="us-east-1"
)

st.title("🔐 Nexus Security Policy Chatbot")

query = st.text_input("Ask a question about Nexus Security Policy")

if query:
    with st.spinner("Thinking..."):
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
        You are a Nexus assistant.
        Use ONLY the context below to answer.

        Context:
        {context}

        Question: {query}
        """

        response = llm.invoke(prompt)
        st.write(response.content)

        st.subheader("Sources")
        for d in docs:
            st.write(f"- {d.metadata['source']}")
