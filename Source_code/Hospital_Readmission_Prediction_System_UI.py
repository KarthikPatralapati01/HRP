# Import required libraries
import streamlit as st
import base64
import shelve
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain import PromptTemplate
import pandas as pd
import shelve
import uuid
from dotenv import load_dotenv

load_dotenv()

# Streamlit app configuration
st.set_page_config(page_title="Hospital Readmission Prediction System", layout="wide")


# CSS for background and highlighting
page_bg_img = f"""
<style>
/* Background image with a light overlay */
[data-testid="stAppViewContainer"] {{
    background-image: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)),
                      url("https://img.pikbest.com/backgrounds/20200804/protection-doctor-abstract-neon-background-v_2450396jpg!sw800");
    background-size: cover;
    background-position: right;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

/* Header styling */
[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    color: #000;
}}

/* Chat message boxes */
[data-testid="stChatMessage"] {{
    background: rgba(255, 255, 255, 0.9); /* Slightly transparent white */
    border: 2px solid #008080; /* Blue border to highlight */
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2); /* Add shadow for emphasis */
    margin-bottom: 10px; /* Spacing between messages */
}}

/* Chat input styling */
[data-testid="stTextInput"] {{
    background: rgba(255, 255, 255, 0); /* Slightly opaque input box */
    color: #000; /* Dark text for readability */
    border-radius: 8px; /* Rounded corners */
    padding: 10px;
    border: 1px solid #ccc; /* Subtle border */
}}

/* Sidebar styling */
[data-testid="stSidebar"] {{
    background-color: rgba(255, 255, 255, 0.5); /* Light gray for contrast */
    border-right: 1px solid #ddd; /* Subtle divider */
}}

/* Toolbar adjustments */
[data-testid="stToolbar"] {{
    right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Fixed header with responsive title
st.markdown("""
    <style>
    .header {
        position: fixed;
        top: 0px;
        left: 40px;
        width: 100%;
        background-color: rgba(255, 255, 255, 0.5);
        padding: 4px 0;
        box-shadow: 0 4px 2px -2px gray;
        z-index: 1000;
    }
    .header h1 {
        font-size: 2.5rem;
        text-align: center;
        margin: 0;
        padding: 0;
    }
    .header p {
        font-size: 1.2rem;
        text-align: center;
        color: #555;
    }
    .main-container {
        margin-top: 70px;  /* Adjust based on header height */
    }
    </style>
    <div class="header">
        <h1>Hospital Readmission Prediction System</h1>
        <p>Leveraging RAG and LLMs to predict hospital readmissions.</p>
    </div>
""", unsafe_allow_html=True)
# Sidebar for model selection and settings
with st.sidebar:
    st.title("Settings")
    st.markdown("### Select a Model:")
    model_choice = st.selectbox(
        "Choose a model for prediction:",
        ["Llama 3.1", "Llama 3", "Llama 2", "BioMistral", "Mistral"]
    )
    st.markdown("---")

# Chat functionality
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat history management
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Sidebar for chat history management
with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])

# Functions for Mistral model setup
# Load the CSV file
transformed_train_df = pd.read_csv("drive/transformed_data.csv")
text_content = " ".join(transformed_train_df["question_prompt_answer"].astype(str))
@st.cache_resource
def prepare_vector_store(text_content):
    # Split text into chunks
    chunk_size = 1000
    chunk_overlap = 20
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = text_splitter.split_text(text_content)
    
    # Class to hold text document chunks with unique ids
    class TextDocument:
        def __init__(self, page_content):
            self.page_content = page_content
            self.metadata = {}
            self.id = str(uuid.uuid4())  # Assign a unique ID to each document
    
    # Create document objects from text chunks
    documents = [TextDocument(page_content=chunk) for chunk in text_chunks]
    
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create FAISS index in memory
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    
    print("Created the vector store in memory.")
    
    return vector_store

vector_store = prepare_vector_store(text_content)

@st.cache_resource
def load_llm():
    llm = LlamaCpp(
        streaming=True,
        model_path="drive/mistral-7b-instruct-v0.1.Q5_K_S.gguf",
        temperature=0.75,
        top_p=1,
        verbose=True,
        n_ctx=4096
    )
    return llm
    
@st.cache_resource
def create_qa_chain(_vector_store):
    llm = load_llm()
    retriever = _vector_store.as_retriever(search_kwargs={"k": 2})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa

mistral_qa_chain = create_qa_chain(vector_store)

# Functions for BioMistral model setup
# Load CSV and prepare the documents
@st.cache_data
def load_data(file_path):
    sampled_df = pd.read_csv(file_path)
    sampled_df['combined_text'] = sampled_df.apply(
        lambda row: f"Clinical Note: {row['TEXT']}, Readmission Status: {row['readmitted']}", axis=1
    )
    return sampled_df
# Function to create and persist Chroma vectorstore
@st.cache_resource
def create_biomistral_vectorstore(df):
    docs = [Document(page_content=text) for text in df['combined_text']]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    chroma_directory = "Bio_Mistral_Model_HF"
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=chroma_directory)

    return vectorstore
    
# Load data and create vector store
file_path = "drive/sample_model_df_balanced.csv"
df = load_data(file_path)
biomistral_vectorstore = create_biomistral_vectorstore(df)

# Initialize LLM (BioMistral)
@st.cache_resource
def load_biomistral_llm():
    llm = LlamaCpp(
        model_path="drive/BioMistral-7B.Q4_K_M.gguf",
        temperature=0.3,
        max_tokens=2048,
        top_p=1
    )
    return llm

biomistral_llm = load_biomistral_llm()

# RAG Chain Setup
@st.cache_resource
def setup_biomistral_rag_chain(_vectorstore, _llm):
    retriever = _vectorstore.as_retriever(search_kwargs={'k': 5})

    template = """
    <|context|>
    You are an AI assistant that follows instructions extremely well.
    Please be truthful and give direct answers. Also, tell if the patient will be readmitted or not.
    </s>
    <|user|>
    {query}
    </s>
    <|assistant|>
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | _llm
        | StrOutputParser()
    )

    return rag_chain

biomistral_rag_chain = setup_biomistral_rag_chain(biomistral_vectorstore, biomistral_llm)


# Functions for Llama 2 model setup

# Step 1: Load and Preprocess Data
@st.cache_data
def load_llama_data(llama_file_path):
    llama_df = pd.read_csv(llama_file_path).head(10)  # Load first 10 records for a subset of data
    documents = [Document(page_content=text) for text in llama_df['TEXT'].tolist()]
    return documents

# Step 2: Create Document Chunks
@st.cache_data
def split_documents(_documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    return text_splitter.split_documents(_documents)

# Step 3: Initialize Embeddings
@st.cache_resource
def create_embeddings():
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings

# Step 4: Initialize FAISS Vector Store
@st.cache_resource
def create_llama_vectorstore(_embeddings, texts):
    return FAISS.from_texts(texts, _embeddings)

# Step 5: Load and Set up Llama 3 Model with PEFT
@st.cache_resource
def load_llama_llm():
    tokenizer = AutoTokenizer.from_pretrained("bhsai2709/T7_Llama_readmission_prediction")
    #model = AutoModelForCausalLM.from_pretrained("bhsai2709/T7_Llama3_readmission_prediction")
    model = AutoModelForCausalLM.from_pretrained(
        "bhsai2709/T7_Llama_readmission_prediction",
        torch_dtype=torch.float16,  # or torch.bfloat16 if your CPU supports it
        low_cpu_mem_usage=True, 
        device_map="cpu"
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    return HuggingFacePipeline(pipeline=pipe)    
    
# Load data, split documents, and create vector store
llama_file_path = 'drive/hospital_data.csv'  # Specify the path to your CSV file
documents = load_llama_data(llama_file_path)
splits = split_documents(documents)
texts = [doc.page_content for doc in splits]
embeddings = create_embeddings()
llama_vectorstore = create_llama_vectorstore(embeddings, texts)

# Llama 3 model setup
@st.cache_resource
def load_llama3_llm():
    tokenizer = AutoTokenizer.from_pretrained("bhsai2709/T7_Llama3_readmission_prediction")
    #model = AutoModelForCausalLM.from_pretrained("bhsai2709/T7_Llama3_readmission_prediction")
    model = AutoModelForCausalLM.from_pretrained(
        "bhsai2709/T7_Llama3_readmission_prediction",
        torch_dtype=torch.float16,  # or torch.bfloat16 if your CPU supports it
        low_cpu_mem_usage=True, 
        device_map="cpu"
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    return HuggingFacePipeline(pipeline=pipe)  

# Llama 3.1 model setup 
@st.cache_resource
def load_llama31_llm():
    tokenizer = AutoTokenizer.from_pretrained("bhsai2709/T7_Llama3.1_readmission_prediction")
    #model = AutoModelForCausalLM.from_pretrained("bhsai2709/T7_Llama3_readmission_prediction")
    model = AutoModelForCausalLM.from_pretrained(
        "bhsai2709/T7_Llama3.1_readmission_prediction",
        torch_dtype=torch.float16,  # or torch.bfloat16 if your CPU supports it
        low_cpu_mem_usage=True, 
        device_map="cpu"
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    return HuggingFacePipeline(pipeline=pipe)  

llama_llm2 = load_llama_llm()
llama_llm3 = load_llama3_llm()
llama_llm31 = load_llama31_llm()

# Set up Llama RAG Chain
@st.cache_resource
def setup_rag_chain(_vectorstore, _llm):
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 1})  # Retrieve the top document
    prompt_template = """
    Context: {context}
    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

llama2_qa_chain = setup_rag_chain(llama_vectorstore, llama_llm2)
llama3_qa_chain = setup_rag_chain(llama_vectorstore, llama_llm3)
llama31_qa_chain = setup_rag_chain(llama_vectorstore, llama_llm31)

# Display chat messages
for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Main chat interface
if query := st.chat_input("Enter your query"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    # Process user query and get response from the selected model
    with st.chat_message("assistant", avatar="🤖"):
        response_placeholder = st.empty()
        
        # Log start time to measure response delay
        import time
        #start_time = time.time()
        with st.spinner("Processing..."):   
            if model_choice == "Llama 3":
                response = llama3_qa_chain.run(query)
            elif model_choice == "BioMistral":
                response = biomistral_rag_chain.invoke(query)
            elif model_choice == "Mistral":
                response = mistral_qa_chain.run(query)
            elif model_choice == "Llama 3.1":
                response = llama31_qa_chain.run(query)
            elif model_choice == "Llama 2":
                response = llama2_qa_chain.run(query)
            else:
                response = "Please select a model to proceed"
        # Log end time and print to help debug
        #end_time = time.time()
        #st.write(f"{model_choice} Average response time: {end_time - start_time:.2f} seconds")
        
        # Display response in the chat
        response_placeholder.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Save chat history after each interaction
save_chat_history(st.session_state.messages)