import streamlit as st
import sys
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from pymilvus import connections, Collection, AnnSearchRequest, WeightedRanker
from spellchecker import SpellChecker
import requests
import json
import streamlit_shadcn_ui as ui

# sets tab name and icon
st.set_page_config(
    page_title="Navy Chat",
    page_icon="🚢",
    layout="centered"
)
st.title("Quiz Mode")

score = 0
total_questions = 0
# Streamlit Sidebar for configuration
with st.sidebar:
    st.title("Learning Mode")
    st.markdown("**Note**: You must have a working zilliz token so that our model can run, otherwise you will get an error.")
    zilliz_token = st.text_input("Zilliz Token", None, type="password")
    st.markdown('''
    **How It Works**
                
    1. Ask your question
    2. We give you 4 multiple choice answers to your question
    3. Make your selection
    4. We give you feedback and an explanation for the answer.
    5. Keep learning! :brain: 
    ''')


# Set up SpellChecker
spell = SpellChecker()

# Function to interact with Ollama API
def query_ollama(prompt, model="llama3.2:latest"):
    print("sending request to ollama...\n")
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
    }
    print("sent details to model...\n")

    try:
        print("awaiting response...\n")
        response = requests.post(url, headers=headers, data=json.dumps(payload), stream=False)
        print("Response received.\n")
        # response.raise_for_status()
                
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {e}")
        return None

# checks for token first
if zilliz_token is None:
    st.error("No Zilliz token provided. Cannot run model.")

# Connect to Zilliz Cloud cluster
CLUSTER_ENDPOINT = "https://in03-cf607103ea8262d.serverless.gcp-us-west1.cloud.zilliz.com"
TOKEN = zilliz_token
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
    prompt = f"  Generate a question based on the context. Provide 4 multiple choices for the correct answer and an explanation for the correct answer. Provide the response as a json containing question, choices, correct_answer, explanation:\n\nQuestion:\n{question} \n\nContext:\n{context}"
    result = query_ollama(prompt)
    return result

def initialize_session_state():
    session_state = st.session_state
    session_state.form_count = 0
    session_state.quiz_data = get_answer(corrected_input)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Please enter your question."}]

# Main Streamlit app
st.caption("Take a quiz covering content on the following PDFs:")

ui.badges(badge_list=[("Humanitarian Assistance and Disaster Relief Aboard the USNS Mercy", "secondary"), 
                      ("US Navy Ship-Based Disaster Response", "secondary"), 
                      ("Sea Power: The US Navy and Foreign Policy", "secondary"), 
                      ("A Decade of Surgery Abroad the US Naval Ship Comfort", "secondary"), 
                      ("Hospital Ships Adrift?", "secondary"),], 
                      class_name="flex gap-2", key="badges1")

# Display chat history
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.write(msg["content"])

# Chat input with spell correction and response display
user_input = st.chat_input("Enter your question:")

if user_input:
    corrected_input = updateQuery(user_input)
    st.session_state.messages.append({"role": "user", "content": corrected_input})
    
    # Display user's query
    with st.chat_message("user"):
        st.write(corrected_input)

    with st.spinner("Generating response..."):
        answer = get_answer(corrected_input)

    # code that breaks down response and prints it out
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
 
        # Add the completed answer to session state chat history
        # st.session_state["messages"].append({"role": "assistant", "content": result})
                
        print("Ran streaming code\n")

        json_result = json.loads(result)

        try:
            json_result = json.loads(result)
            st.session_state["messages"].append({"role": "assistant", "content": json_result})

            # Display parsed content if JSON parsing was successful
            st.write("**Question:**")
            st.write(json_result['question'])
            user_choice = st.radio("Choose an answer", json_result['choices'], index=None)
            
            if user_choice == json_result['correct_answer']:
                st.success("Correct!")
            else:
                st.error("Incorrect :(")
            st.write(f"**Explanation:** {json_result['explanation']}")

        except:
            st.error("Something went wrong 😞 Please try another question or ask again.")