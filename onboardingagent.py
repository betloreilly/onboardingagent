import streamlit as st
import os
import tempfile
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import AstraDB
from langchain.schema.runnable import RunnableMap
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image
from langchain.document_loaders import PyPDFLoader

# Start with empty messages and chat history, stored in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_history = []  # Initialize chat history

# Streaming call back handler for responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

# Function for Vectorizing uploaded data into Astra DB
def vectorize_text(uploaded_file, vector_store):
    if uploaded_file is not None:
        
        # Write to temporary file
        temp_dir = tempfile.TemporaryDirectory()
        file = uploaded_file
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, 'wb') as f:
            f.write(file.getvalue())

        # Create the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1500,
            chunk_overlap  = 100
        )

        # Load and vectorize the PDF
        docs = []
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

        pages = text_splitter.split_documents(docs)
        vector_store.add_documents(pages)  
        st.info(f"{len(pages)} pages loaded.")

# Cache prompt for future runs
@st.cache_data()
def load_prompt():
    template = """ You are a chatbot for giving recommendations for the best practices in data modelling in Cassandra and you provide best practices recommendations and you provide a CQL with the best possible partition key for their table in the prompt.
        You need to ask questions related to how many rows do they expect in a partition and an initial draft for their table and you will calculate the partition size
          Here an example how to calculate the partition size: CREATE TABLE temperature_readings (device_id UUID,timestamp TIMESTAMP,temperature FLOAT,PRIMARY KEY (device_id, timestamp)); 
          They will have 500K rows for each device per day: Using the given data type size you can calculate the partition size:Partition size = (column1 type size + column2 type size + column3 type size) x Number of rows. 
          In this example; Partition size = (16+8+4)*500000 = 14000000 bytes = 13MB. This is a guiding example, adjust the calculation based on the actual data provided and show the new calculation as a response.
          One of the best practices is to have a partition size less than 10MB. If it is above that, you can recommend using another column in the partition key or add time bucketing and show the new table in CQL.
          Provide other best practices for Cassandra Data Modelling like using TTL, using a new denormalized table for each query access instead of using Secondary Index and Materialized Views.
          For Migration questions, check the relevant information in vector database and suggest Cassandra Data Migrator or Zero Down Time (ZDM) only while migrating from cassandra based databases not for the other databases like sql server, oracle , mysql. These tools don't support the databases that are not Cassandra.
      :

RECENT QUESTIONS:
{chat_history}

CONTEXT:
{context}

QUESTION:
{question}

YOUR ANSWER:
"""
    return ChatPromptTemplate.from_messages([("system", template)])
prompt = load_prompt()

# Cache OpenAI Chat Model for future runs
@st.cache_resource()
def load_chat_model():
    return ChatOpenAI(
        temperature=0.3,
        model='gpt-4',
        streaming=True,
        verbose=True
    )
chat_model = load_chat_model()

# Cache the Astra DB Vector Store for future runs
@st.cache_resource(show_spinner='Connecting to Astra')
def load_vector_store():
    # Connect to the Vector Store
    vector_store = AstraDB(
        embedding=OpenAIEmbeddings(),
        collection_name="datamodelagent",
        api_endpoint=st.secrets['ASTRA_API_ENDPOINT'],
        token=st.secrets['ASTRA_TOKEN']
    )
    return vector_store
vector_store = load_vector_store()

# Cache the Retriever for future runs
@st.cache_resource(show_spinner='Getting retriever')
def load_retriever():
    # Get the retriever for the Chat Model
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )
    return retriever
retriever = load_retriever()

# Start with empty messages, stored in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Draw a title and some markdown
#image = Image.open('/Users/betuloreilly/demos/datamodelaiagent/datastaxlogo.png')
#st.image(image)
st.title("Data Model Agent")
st.markdown("""I am a Data Model Agent bot. What would you like to know about data modelling?""")

# Include the upload form for new data to be Vectorized
with st.sidebar:
    with st.form('upload'):
        uploaded_file = st.file_uploader('Upload a document for additional context', type=['pdf'])
        submitted = st.form_submit_button('Save to Astra DB')
        if submitted:
            vectorize_text(uploaded_file, vector_store)

# Draw all messages, both user and bot so far (every time the app reruns)
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Draw the chat input box
if question := st.chat_input("How may I help you today?"):
    
    # Store the user's question in a session object for redrawing next time
    st.session_state.messages.append({"role": "human", "content": question})

    # Update chat history with the new question
    st.session_state.chat_history.append(question)
    if len(st.session_state.chat_history) > 3:
        st.session_state.chat_history.pop(0)  # Keep only last 3 questions
    
    # Create a string with the last three questions
    chat_history_str = '\n'.join(st.session_state.chat_history)
    
    # Draw the user's question
    with st.chat_message('human'):
        st.markdown(question)

    # UI placeholder to start filling with agent response
    with st.chat_message('assistant'):
        response_placeholder = st.empty()

    # Generate the answer by calling OpenAI's Chat Model
    inputs = RunnableMap({
        'context': lambda x: retriever.get_relevant_documents(x['question']),
        'chat_history': lambda x: chat_history_str,  # Add this line
        'question': lambda x: x['question']
    })
 
    chain = inputs | prompt | chat_model
    response = chain.invoke({'question': question}, config={'callbacks': [StreamHandler(response_placeholder)]})
    answer = response.content

    # Store the bot's answer in a session object for redrawing next time
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Write the final answer without the cursor
    response_placeholder.markdown(answer)