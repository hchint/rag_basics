import os
import streamlit as st
import boto3
from langchain_aws import ChatBedrock
from langchain_aws import AmazonKnowledgeBasesRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID", "BLQQSKMUC8")
MODEL_ID = os.getenv("MODEL_ID", "amazon.nova-pro-v1:0")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
TEMPRATURE = float(os.getenv("TEMPRATURE", "0.3"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

#============================================
# AWS clients & LangChain componenets (cached per session)
#=============================================

@st.cache_resource
def get_llm():
    """Intitalize Amazon Bedrock Model via Bedrock"""
    return ChatBedrock(
        model_id= MODEL_ID,
        region_name=AWS_REGION,
        model_config={
            "max_tokens": MAX_TOKENS,
            "temprature": TEMPRATURE,
        },
    )

@st.cache_resource
def get_retriever():
    """Initialize Bedrock KNowledge Base Retriever (Aurora Vector DB backed)."""
    return AmazonKnowledgeBaseRetriver(
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        region_name=AWS_REGION,
        retrieval_config={
            "vectorSearchConfiguration": {
                "numberOfResults": TOP_K_RESULTS
            }
        },
    )

def get_memory():
    """Conversation Memory - keeps last 5 exchanges for context."""
    return ConversationBufferWindowMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer",
        k=5,
    )

def get_chain(memory):
    """ Build the conversational retrieval chain"""
    llm = get_llm()
    retriever = get_retriever()
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return chain

#=======================================
# Steamlit UI
#=======================================

st.set_page_config(
    page_title="knwoledge Base Chatbot",
    page_icon=":robot_face:",
    layout="wide",
)

st.title("Chat with Your Knowledge Base")
st.caption("Powered by Amazon Bedrock (Nova) + Aurora Vector DB Knwoledge Base")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    st.text_input(
        "KnowledgeBAse ID",
        value=KNOWLEDGE_BASE_ID,
        key="kb_id_input",
        help="Your Bedrock Knowledge Base ID",
    )
    st.text_input(
        "Model ID",
        value=MODEL_ID,
        key="model_id_input",
        help="Bedrocl Model ID (e.g., amazon.nova-pro-v1:0, amazon.nova-lite-v1:0)",
    )
    st.slider(
        "Temprature",
        min_value=0.0,
        max_value=1.0,
        value=TEMPRATURE,
        step=1.0,
        key="temprature_input",
    )
    st.slider(
        "Number of results to retrieve",
        min_value=1,
        max_value=20,
        value=TOP_K_RESULTS,
        key="top_k_input",
    )
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.pop("memory", None)
        st.rerun()

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = get_memory()

# Display Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            sources = message["sources"]
            if sources:
                with st.expander("Sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source{i}:** {source}")

# Chat Input
if prompt := st.chat_input("Ask a question about Your Knowledge Base:"):
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate a response from the AI model
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                chain = get_chain(st.session_state.memory)
                result = chain.invoke({"question": prompt})

                answer = result["answer"]
                source_docs = result.get("source_documents", [])

                # Extract source metadata
                sources = []
                for doc in source_docs:
                    metadata = doc.metadata
                    source_info = metadata.get("source", metadata.get("location",{}).get("s3Location",{}).get("uri", "unknown"))
                    sources.append(source_info)
                
                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source{i}:** {source}")

                # Store in history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})