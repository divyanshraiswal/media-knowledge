import os
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone and embeddings
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("my-index")
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

# Streamlit app
st.title(" :blue[HealthScope] ")
st.header("Unlocking medical knowledge with AI")

query = st.text_input("Enter your query")

if query:
    # Embed the query and retrieve unique document sources
    query_embedding = embeddings.embed_query(query)
    query_results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    
    retrieved_docs = list({match['metadata']['source'] for match in query_results['matches']})[:3]  # Unique sources

    # Prepare and invoke Groq
    groq_chain = ChatGroq(model="llama3-8b-8192", temperature=1)  
    groq_prompt = f"Use the following documents to answer the question: {query}. \nDocuments: {retrieved_docs}"
    groq_response = groq_chain.invoke(groq_prompt)
    
    # Display the answer
    st.write(f"**Answer:** {groq_response.content}")
