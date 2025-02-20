from fastapi import FastAPI, UploadFile, File, Form, HTTPException,BackgroundTasks
from typing import List
import os
from pydantic import BaseModel
import pdfplumber
from docx import Document
from PIL import Image
import pytesseract
from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter
import lancedb
import numpy as np
import pyarrow as pa
from numpy.linalg import norm
import pandas as pd
import openai
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Initialize LanceDB
DB_PATH = "lancedb_store"
os.makedirs(DB_PATH, exist_ok=True)
db = lancedb.connect(DB_PATH)
embeddings = OpenAIEmbeddings()
class PCA_Response(BaseModel):
    sentiment_score: int
    knowledge_gap:List[str]
    topic: List[str]

def get_knowledgebase_context(query:str, kb_id):
    knowledgebase_table = db.open_table('knowledge_bases').to_pandas()
    filtered_data = knowledgebase_table[knowledgebase_table["kb_id"] == kb_id]['data']
    # print(filtered_data)
    if not filtered_data.empty:
        table_name = filtered_data.iloc[0]  # Get the first matching value
        # chunk_table = db.open_table(table_name).to_pandas()
        # chunk_embeddings = chunk_table["embeddings"]
    else:
        print("No matching knowledge base found.")
    
    query_embedding = embed_text(query)
    print("###2")
    # array of chunks
    relevant_chunks = relevant_documents(query_embedding=query_embedding,table_name=table_name,top_k=5)
    return relevant_chunks

def get_conversation_history(conversation_id):
    conversation_table = db.open_table("conversation_table").to_pandas()
    record = conversation_table[conversation_table["conversation_id"] == conversation_id]
    # print(record)
    
    # get list of conversation and send this as documents.
    if not record.empty:
        conversation = record.iloc[0]['conversation']
        # print(conversation) 
        return conversation
    return []


def process_image(image_file: UploadFile):
    image = Image.open(image_file.file)
    text = pytesseract.image_to_string(image)
    return text

def embed_text(text:str):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"  # Use the latest embedding model
         )
    return(response.data[0].embedding)
    
def relevant_documents(query_embedding,table_name,top_k):
    # Search in LanceDB using cosine similarity
    df = db.open_table(table_name).to_pandas()
    df["similarity"] = df["embeddings"].apply(lambda emb: cosine_similarity(emb, query_embedding))
    print(type(top_k))
    # Retrieve top_k similar chunks (e.g., top 2)
    top_chunks = df.sort_values(by = "similarity",ascending=False).head(top_k)
    # Print results
    return top_chunks['text'].to_numpy()
    
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
# Function to fetch conversation history from LanceDB

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)
def extract_text(file_path, file_type):
    
    
    if file_type == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text =  (f.read())
    
    elif file_type == "pdf":
        with pdfplumber.open(file_path) as pdf:
            text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    
    elif file_type == "docx":
        doc = Document(file_path)
        text = "".join([para.text for para in doc.paragraphs])
    
    elif file_type in ["jpg", "jpeg", "png"]:
        image = Image.open(file_path)
        text  = pytesseract.image_to_string(image)
    
    return text.strip()

async def process_task(kb_id,files_text,knowledgebase_name):
    # allowed_types = ["txt", "pdf", "docx", "jpg", "jpeg", "png"]
    chunk_table_name = f"chunk_data_{kb_id[:3]}"  # Table name for chunk data
    
    if "knowledge_bases" not in db.table_names():
        # Define the schema for the knowledge base table
        schema = pa.schema([
            pa.field("kb_id", pa.string()),  # The unique ID for the knowledge base (string)
            pa.field("kb_name", pa.string()),  # The name of the knowledge base (string)
            pa.field("data", pa.string()),  # Reference to the chunk data table (string for table name)
            pa.field("status",pa.string()),
            pa.field("total_chunk",pa.string())
        ])
        db.create_table("knowledge_bases", schema=schema)
    schema = pa.schema([
            pa.field("chunk_id", pa.string()),  # The unique ID for the knowledge base (string)
            pa.field("text", pa.string()),  # The name of the knowledge base (string)
            pa.field("embeddings", pa.list_(pa.float32()))  # Reference to the chunk data table (string for table name)
        ])
    chunk_table = db.create_table(chunk_table_name, schema=schema)
    total_chunks = 0
    chunks = chunk_text(files_text)
    chunk_data = []
    for chunk in chunks:
        chunk_id = str(uuid4())
        chunk_embedding = embeddings.embed_query(chunk)  # Generate vector embeddings
        chunk_data.append({
            "chunk_id": chunk_id,
            "text": chunk,
            "embeddings": np.array(chunk_embedding)
        })
    chunk_table.add(chunk_data)
    total_chunks += len(chunk_data)
    # Clean up temp file
    kbt = db.open_table("knowledge_bases")
    kbt.add([
        {
            "kb_id" :kb_id,
            "kb_name":knowledgebase_name,
            "data":chunk_table_name[:14],
            "status":"success",
            "total_chunk":total_chunks
        }
    ])