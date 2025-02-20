from fastapi import FastAPI, UploadFile, File, Form, HTTPException,BackgroundTasks
from typing import List
import os
import shutil
import jsonify
import json
from docx import Document
from uuid import uuid4
from langchain_openai import OpenAIEmbeddings
import lancedb
import pyarrow as pa
from numpy.linalg import norm
import openai
from collections import OrderedDict
from dotenv import load_dotenv
from utils import (
    PCA_Response,chunk_text,embed_text,cosine_similarity,get_conversation_history,get_knowledgebase_context,
    process_task,extract_text,relevant_documents
)
load_dotenv()
app = FastAPI()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Initialize LanceDB
DB_PATH = "lancedb_store"
os.makedirs(DB_PATH, exist_ok=True)
db = lancedb.connect(DB_PATH)
embeddings = OpenAIEmbeddings()
@app.post("/upload/")
async def upload_files(
    background_task:BackgroundTasks,
    knowledgebase_name: str = Form(...),
    files: List[UploadFile] = File(...),
    
):
    allowed_types = ["txt", "pdf", "docx", "jpg", "jpeg", "png"]
    # print(db.table_names)
    files_text =""
    for file in files:
        file_ext = file.filename.split(".")[-1].lower()
        
        if file_ext not in allowed_types:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file.filename}")

        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text from the uploaded file
        extracted_text = extract_text(temp_file_path, file_ext)
        files_text+= extracted_text
        
        if not extracted_text:
            os.remove(temp_file_path)
            continue
        os.remove(temp_file_path)
    kb_id = str(uuid4())
    background_task.add_task(process_task,kb_id,files_text,knowledgebase_name)

    return {
        "kb_id": kb_id,
        "status":"pending",
    }
@app.get("/check_status")
async def check_status(knowledgebase_id: str):
    kbt = db.open_table("knowledge_bases")
    results = kbt.search().where(f"kb_id = '{str(knowledgebase_id)}'").to_pandas()
    if results.empty:
        return{
            'kb_id':knowledgebase_id,
            'status':"pending"
        }
    else:
        status = results['status'].iloc[0]
        kb_name = kbt.to_pandas()['kb_name'].to_list()
        return{
        'kb_id':knowledgebase_id,
        'status':status,
        'kb_names':kb_name}
@app.post("/conversations")
async def create_conversation(kb_id: str,
    query: str
):
    # Create a unique conversation ID
    conversation = []
    conversation_schema = pa.schema([
    pa.field("conversation_id", pa.string()),  # Unique conversation ID
    pa.field("knowledge_base_id", pa.string()),  # Knowledge base name
    pa.field("conversation", pa.list_(pa.struct([
          # User's query
        pa.field("answer", pa.string()),
        pa.field("created_at", pa.int64()),  # User's query
        pa.field("input_tokens", pa.int64()),
        pa.field("output_tokens", pa.int64()),
        pa.field("query", pa.string()),
        pa.field("source", pa.list_(pa.string())),
        pa.field("total_tokens", pa.int64()),

    ])))
    ])

    # Create the table if it doesn't exis
    if "conversation_table" not in db.table_names():
        conversation_table = db.create_table("conversation_table", schema=conversation_schema)
    else:
        conversation_table = db.open_table("conversation_table") 
    # # print(knowledgebase_id)
    knowledgebase_table = db.open_table('knowledge_bases').to_pandas()
    print(knowledgebase_table)
    print(kb_id)
    filtered_data = knowledgebase_table[knowledgebase_table["kb_id"] == kb_id]['data']
    # kb_id  = knowledgebase_table[knowledgebase_table["kb_name"] == kb_name]['kb_id'].iloc[0]
    print(filtered_data)
    if not filtered_data.empty:
        table_name = filtered_data.iloc[0]  # Get the first matching value
        
    else:
        print("No matching knowledge base found.")
    
    query_embedding = embed_text(query)
    print("###2")
    # array of chunks
    relevant_chunks = relevant_documents(query_embedding=query_embedding,table_name=table_name,top_k=5)

    documents = " ".join(chunk for chunk in relevant_chunks if relevant_chunks is not None)
    prompt = f"""Use the following context to answer the question:
    Context:
    {documents}
    Question: {query}
    Answer:"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an AI assistant."},
                {"role": "user", "content": prompt}],      
    )
    # print(response)
    answer = response.choices[0].message.content
    created_at = response.created
    total_token = response.usage.total_tokens
    input_token = response.usage.prompt_tokens
    output_token = response.usage.completion_tokens
    conv_id =str(uuid4())
    chat = OrderedDict([ 
        ('answer', answer),
        ('created_at', created_at),
        ('input_tokens', input_token),
        ('output_tokens', output_token),
        ('query', query), 
        ('source', relevant_chunks.tolist()),
        ('total_tokens', total_token),
        
    ])

    conversation_list = conversation_table.to_pandas()['conversation_id'].tolist()

    # conversation.append(chat)
    conversation_table.add([{
        'conversation_id':conv_id,
        'knowledge_base_id' :kb_id,
        'conversation':[chat]}])

    # print(query_embedding)
    return {
        'conversation_id':conv_id,
        'kb_id' :kb_id,
        'query' :query,
        'answer':answer,
        'created_at':created_at,
        'total_tokens':total_token,
        'source':relevant_chunks.tolist(),
        "conv_list":conversation_list
    }

@app.post("/continue_conversations")
async def continue_conversation(conversation_id: str, kb_id: str,
    query: str):
    # Retrieve existing conversation history
    # knowledgebase_table = db.open_table('knowledge_bases').to_pandas()
    # filtered_data = knowledgebase_table[knowledgebase_table["kb_id"] == kb_id]['kb_id']
    # kb_id = filtered_data.iloc[0]
    relevant_chunks = get_knowledgebase_context(query, kb_id)
    # print(relevant_chunks)
    conversation_record = get_conversation_history(conversation_id=conversation_id)
    print(f"lenght {len(conversation_record)}")
    # print(conversation_record)
    messages = [{"role": "system", "content": "You are an intelligent AI assistant that accurately tracks conversation history and integrates relevant knowledge for better responses."}]
    # print("00")
# Add conversation history
    for chat in conversation_record:
        # print(chat["query"])
        messages.append({"role": "user", "content": chat["query"]})
        messages.append({"role": "assistant", "content": chat["answer"]})

    # Combine relevant documents if available
    documents = " ".join(relevant_chunk for relevant_chunk in relevant_chunks if relevant_chunk)


    # Construct query with history and knowledge base context
    prompt = f"""You are responding to a user query using both previous conversation history and relevant external knowledge.

    Conversation History:
    {messages[-6:] if len(messages) > 6 else messages}  # Using the last few interactions to keep context fresh

    External Context:
    {documents}

    User's Current Question: {query}

    Answer:"""

    # Append the new query to messages
    messages.append({"role": "user", "content": prompt})

    # Generate AI response
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    answer = response.choices[0].message.content
    created_at = response.created
    total_tokens = response.usage.total_tokens
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    # Append new interaction to conversation history
    new_chat = {
        'query' :query,
        'answer':answer,
        'created_at':created_at,
        'total_tokens':total_tokens,
        'input_tokens':input_tokens,
        'output_tokens':output_tokens,
        'source':relevant_chunks.tolist()
    }

    conversation_table = db.open_table("conversation_table")

    conversation_df = conversation_table.to_pandas()
    print(conversation_df.shape)
    record = conversation_df[conversation_df["conversation_id"] == conversation_id]
    if record.empty:
        raise ValueError(f"No conversation found for ID: {conversation_id}")

    # Extract the first matching record
    record_data = record.iloc[0]
    existing_conversation = record_data["conversation"]

    # Append the new chat
    updated_conversation = list(existing_conversation)  # Copy the existing list
    updated_conversation.append(new_chat) 
    # Delete the old record
    conversation_table.delete(f"conversation_id = '{conversation_id}'")

    # Insert the updated record
    conversation_table.add([
        {
            "conversation_id": conversation_id,
            "knowledge_base_id": record_data["knowledge_base_id"],
            "conversation": updated_conversation
        }
    ])
        # conversation_table.update(where=f"conversation_id = '{str(conversation_id)}", values={"conversation": existing_conversation})
    return {
        "status":"conversation added succesfully",
        "conversation_id":conversation_id,
        'query': query,
        'answer': answer,
        'query' :query,
        'answer':answer,
        'created_at':created_at,
        'total_tokens':total_tokens,
        'source':relevant_chunks.tolist()
    }
@app.post("/pca_conversations")
async def continue_conversation(conversation_id: str):
    conversation = get_conversation_history(conversation_id).tolist
    print(conversation)
    symptoms_system_prompt = '''
    You will be given the context and knowledgebase and you will output a json object containing the following information:

    {
        "sentiment_analysis": {  # Object containing sentiment details
            "score":  List[int] - list of scores
        },
        "knowledge_gap": {  # Object containing content gap details
            "miss_information":  List[str] - list of misinformation strings
        },
        "tags_topics": {  # Object containing query categorization details
            "query_category": List[str] - list of query categories
        },
        
       
    '''

    description = """The choosen metrics cover essential aspects of bot performance,
      user experience, and content quality. Here's a brief details on each metric:
      Sentiment Analysis [ Number of Negative,Number of Neutral, Number of Positive]: 
      This is crucial for understanding user emotions and satisfaction during interactions. 
      The score from the range of 0-10. It helps in gauging the overall user experience.0 means negative, 5 means neutral 
      10 means positive.
      Knowledge Gap:content gaps helps in continuously improving the bot's responses by addressing areas 
      where it frequently fails to provide accurate or helpful information. Those answer from the bot which are not in knowledge bases are inlcuded as list.
     Query Categorization:This metric helps you understand what types of questions are most common, 
    allowing you to focus on improving those areas or expanding the bot's knowledge base.It is a list of topics that query and answer belongs in knowledge base.
    """

    response = openai.beta.chat.completions.parse(
        response_format=PCA_Response,
        model='gpt-4o-mini',
        messages= [
                        {'role': 'system', 'content': "You are the conversation analysis system. You will be given a conversation. Keep in mind that the bot is tuned to give answer only from the provided"},
                        {'role': 'system', 'content': f"{description}"},
                        {'role': 'system', 'content': "Analyze the conversation and Provide the response in the Json format."},
                        {'role': 'user', 'content': f"Conversation : {conversation}"},
                        {"role": "user", "content": f"{symptoms_system_prompt}"},
                        ],
    )
        
    # print(response)
    json_string = response.choices[0].message.content
    json_data = json.loads(json_string)
    print(json_data)
    return json_data
        


@app.get("/convert")
async def convert(kb_name:str):
    tbl = db.open_table('knowledge_bases').to_pandas()
    filtered_data = tbl[tbl["kb_name"] == kb_name]['kb_id']
    if not filtered_data.empty:
        kb_id = filtered_data.iloc[0]
        return {"kb_id":kb_id}
    else:
        return []