# Installing Dependencies
import os
import json
import streamlit as st
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import ConfigurableField
from neo4j import GraphDatabase
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from neo4j import GraphDatabase
from typing import Tuple, List, Optional
from pydantic import BaseModel, Field, field_validator
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_nomic import NomicEmbeddings
# from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

st.set_page_config(
    page_title="PPKS | Chat Bot",
    page_icon="assets/logo-PPKS.png",
    layout="centered",
)

# Load environtment app
load_dotenv()

# Load llm model using ollama
# @st.cache_resource
# def load_llm_ollama():
#     return ChatOllama(
#         model='llama3.1:8b-instruct-fp16',
#         temprature=0
#     )
# llm = load_llm_ollama()

# Load llm model using Groq
@st.cache_resource
def load_llm_groq():
    return ChatGroq(
        model='llama-3.1-70b-versatile', #llama-3.1-70b-versatile, llama-3.1-8b-instant
        # temprature=0
    )
llm = load_llm_groq()

# load llm model using gemini
# @st.cache_resource
# def load_llm_groq():
#     return ChatGoogleGenerativeAI(
#         model="gemini-pro",
#         convert_system_message_to_human=True
#         )
# llm = load_llm_groq()

# Load knowledge graph fron neo4j
@st.cache_resource
def load_knowledge_graph():
    return Neo4jGraph()

graph = load_knowledge_graph()

# Create vector space from graph
@st.cache_resource
def create_vector_space_from_graph():
    vector_index = Neo4jVector.from_existing_graph(
        NomicEmbeddings(model="nomic-embed-text-v1.5"),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )

    return vector_index

vector_index = create_vector_space_from_graph()

# Create retrival flow
## Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, product, or business entities that "
        "appear in the text",
    )

    @field_validator("names", mode='before')
    def parse_stringified_list(cls, value):
        if isinstance(value, str):
            try:
                # Attempt to parse the string as JSON
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("Invalid list format; unable to parse string as list.")
        if not isinstance(value, list):
            raise ValueError("items must be a list of strings.")
        return value

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization, product, and person entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)

entity_chain = prompt | llm.with_structured_output(Entities)


# Generate Query
def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

# Fulltext index query and retirieve context
def structured_retriever(question: str) -> str:
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
           """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL(node) {
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result

# Retrival knowledge
def retriever(question: str):
    # print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ". join(unstructured_data)}
    """
    return final_data


_template = """
You are an assistant skilled in paraphrasing questions, ensuring they align with the current conversation context. Every time a new question appears, check the recent chat history to decide if it’s on the same topic or if there’s a new topic shift. 

Guidelines:
1. If the latest question is vague (e.g., "What is its capital?"), identify the most recent *explicitly mentioned topic* in the chat history and use it as context.
2. When a new complete question introduces a different topic, assume it’s a topic shift and use this new topic in the next responses until another shift occurs.
3. Prioritize the most recent complete topic if multiple topics are discussed in history.

**Examples:**

Example 1:
**Chat History:**
- User: "Who is the president of Indonesia?"
- AI: "The president of Indonesia is Joko Widodo."

**Latest Question:**  
User: "When did it gain independence?"

**Paraphrased Question:**  
"When did Indonesia gain independence?"

---

Example 2 (Topic Shift):
**Chat History:**
- User: "Who is the president of Indonesia?"
- AI: "The president of Indonesia is Joko Widodo."
- User: "What is its capital?"
- AI: "The capital of Indonesia is Jakarta."
- User: "Who is the president of Vietnam?"
- AI: "The president of Vietnam is Tran Dai Quang."

**Latest Question:**  
User: "What is its capital?"

**Paraphrased Question:**  
"What is the capital of Vietnam?"

---

Example 3:
**Chat History:**
- User: "Who is the CEO of Apple?"
- AI: "The CEO of Apple is Tim Cook."
  
**Latest Question:**  
User: "How many employees does it have?"

**Paraphrased Question:**  
"How many employees does Apple have?"

---

Example 4 (Topic Shift):
**Chat History:**
- User: "Who is the CEO of Apple?"
- AI: "The CEO of Apple is Tim Cook."
- User: "What is the companys revenue?"
- AI: "Apple's revenue is $274.5 billion."

**Latest Question:**  
User: "What is its revenue?"

**Paraphrased Question:**  
"What is the revenue of CEO Microsoft?"

---

Now, parafrase the latest question based on the recent topic or topic shift, using the latest chat history provided.
But don't explain in  output. just give the parafrased question as output.

**Chat History:**
{chat_history}

**Latest Question:**
{question}

**Paraphrased Question:**
"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# Chat history fromatter
def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

# Extract chat history if exists
_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(lambda x : x["question"]),
)

# Prompt to real prompt
template = """You are a great, friendly and professional AI chat bot about product from the "Pusat Penelitian Pusat Penelitian Minyak Sawit Indonesia (PPKS)". The website (https://iopri.co.id/).
Answer the question based only on the following context
{context}
        
Question: {question}
Use Indonesian that is easy to understand. And answer maximum in one paragaf and constantly be efficient and direct base what user ask.
Answer: """

prompt = ChatPromptTemplate.from_template(template)

# Creating chain for llm
chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)


# Create title for chat APP
col = st.columns([0.15, 0.85], vertical_alignment="center")

with col[0]:
    st.image(image="asset/logo-PPKS.png", use_column_width=True)
with col[1]:
    st.header("| Chat Bot PPKS 🤖")

st.divider()

# Setup a session state to hold up all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'need_greetings' not in st.session_state:
    st.session_state.need_greetings = True

# Displaying all historical messages
for message in st.session_state.messages:
    st.chat_message(message['role'], avatar= "asset/logo-PPKS.png" if message['role'] == "assistant" else None).markdown(message['content'])

if st.session_state.need_greetings :

    # greet users
    greetings = "Selamat Datang, ada yang bisa saya bantu?"
    st.chat_message("assistant", avatar="asset/logo-PPKS.png").markdown(greetings)

    st.session_state.messages.append({'role' : 'assistant', 'content': greetings})

    st.session_state.need_greetings = False


# Getting chat input from user
prompt = st.chat_input("e.g. apa itu produk Marfu-p?")

# Displaying chat prompt
if prompt:
    # Displaying user chat prompt
    st.chat_message("user").markdown(prompt)

    # Saving user prompt to session state
    st.session_state.messages.append({'role' : 'user', 'content': prompt})

    # Getting response from llm model
    response = chain.invoke({
        "chat_history" : st.session_state.chat_history, 
        "question" : prompt
    })


    # Displaying response
    st.chat_message("assistant", avatar="asset/logo-PPKS.png").markdown(response)

    # Saving response to chat history in session state
    st.session_state.messages.append({'role' : 'assistant', 'content': response})

    # Saving user and llm response to chat history
    st.session_state.chat_history.append((prompt, response))

