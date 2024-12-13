import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
from pymilvus import connections, Collection, AnnSearchRequest, WeightedRanker
from spellchecker import SpellChecker
import requests
import json
import streamlit_shadcn_ui as ui

st.set_page_config(
    page_title="Navy Chat",
    page_icon="🚢",
    layout="centered"
)

# Set up SpellChecker
spell = SpellChecker()

# Function to interact with Ollama API
def query_ollama(prompt, model="llama3.2:latest"):
    print("hey1")
    url = "http://ollama:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
    }
    print("hey1")

    try:
        print("getting response...")
        response = requests.post(url, headers=headers, data=json.dumps(payload), stream=True)
        print("Got response.")
                
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {e}")
        return None

# Streamlit Sidebar for configuration
with st.sidebar:
    st.title("🚢 Medical Navy Chatbot 💬")
    st.write("Configure your chatbot settings below.")

# Connect to Zilliz Cloud cluster
CLUSTER_ENDPOINT = "https://in03-cf607103ea8262d.serverless.gcp-us-west1.cloud.zilliz.com"
TOKEN = st.secrets["ZILLIZ_TOKEN"]
connections.connect(uri=CLUSTER_ENDPOINT, token=TOKEN)

# Load model and tokenizer from Hugging Face Hub
tokenizer_hyb = AutoTokenizer.from_pretrained('BAAI/llm-embedder')
model = AutoModel.from_pretrained('BAAI/llm-embedder')
model.eval()
tokenizer_hyb.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer_hyb))

# Function to perform hybrid search
def hybrid_search(query_text):
    encoded_input = tokenizer_hyb([query_text], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
        query_embedding = torch.nn.functional.normalize(model_output[0][:, 0], p=2, dim=1)

    search_param_poster = {
        "data": query_embedding,
        "anns_field": "posterVector",
        "param": {"metric_type": "L2", "params": {"nprobe": 10}},
        "limit": 5
    }

    request_poster = AnnSearchRequest(**search_param_poster)
    rerank = WeightedRanker(1.0)
    collection_name = "Oct_31"
    collection = Collection(collection_name)
    collection.load()
    res = collection.hybrid_search(reqs=[request_poster], rerank=rerank, limit=5, output_fields=["metadata", "chunks"])

    summary_list = []
    for hit in res:
        for entity in hit:
            metadata = entity.entity.get("metadata")
            chunks = entity.entity.get("chunks")
            combined_string = f"{chunks}. This information is from {metadata.get('source')} on page {metadata.get('page')}."
            summary_list.append(combined_string)
    return summary_list

# Function to update query with spell correction
def updateQuery(query):
    return query

# Function to get answer from QA pipeline
def get_answer(question):
    search_results = hybrid_search(question)
    context = " ".join(search_results)
    prompt = f"  Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Try to utilize the input to answer this question, but dont mention to me if it is convoluded information and act as though it makes sense:\n\nQuestion:\n{question} \n\nContext:\n{context}"
    result = query_ollama(prompt)
    return result

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Please enter your question."}]

# Main Streamlit app
st.title("💬 Chatbot")
st.caption("This chatbot is trained on the following PDF's shown below.")

ui.badges(badge_list=[("Humanitarian Assistance and Disaster Relief Aboard the USNS Mercy", "secondary"), 
                      ("US Navy Ship-Based Disaster Response", "secondary"), 
                      ("Sea Power: The US Navy and Foreign Policy", "secondary"), 
                      ("A Decade of Surgery Abroad the US Naval Ship Comfort", "secondary"), 
                      ("Hospital Ships Adrift?", "secondary"),], 
                      class_name="flex gap-2", key="badges1")


# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input with spell correction and response display
user_input = st.chat_input("Enter your question:")
if user_input:
    corrected_input = updateQuery(user_input)
    st.session_state.messages.append({"role": "user", "content": corrected_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(corrected_input)
    
    # Get chatbot response
    answer = get_answer(corrected_input)
 

    # Create a placeholder inside the assistant chat message
    with st.chat_message("assistant"):
        placeholder = st.empty()  # Initialize a placeholder

        # Variable to hold cumulative text
        result=''
        for line in answer.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                token = data.get('response', '')
                result += token
                placeholder.write(result)
 
        # Add the completed answer to session state chat history
        st.session_state["messages"].append({"role": "assistant", "content": result})

                
        print("Ran streaming code")
