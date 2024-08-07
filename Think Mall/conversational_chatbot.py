from tqdm import tqdm
import re
import os
import json
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.agents import Tool
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import pinecone
from langchain.vectorstores import Pinecone
import os
from langchain.chains import RetrievalQA

openai_api_key = os.environ["THINK_MALL_GPT_KEY"]

PINECONE_API_KEY = os.environ["PINE_API_KEY"]

PINECONE_API_ENV = os.environ["PINE_ENV"]



pinecone.init(
            api_key= PINECONE_API_KEY, 
            environment= PINECONE_API_ENV
)
index_name = "think-mall"
if index_name not in pinecone.list_indexes():
    raise ValueError(
        f"No '{index_name}' index exists. You must create the index before "
        "running this notebook. Please refer to the walkthrough at "
        "'github.com/pinecone-io/examples'."  # TODO add full link
    )

index = pinecone.Index(index_name)
embeddings = OpenAIEmbeddings(model_name='gpt-3.5-turbo' , openai_api_key=openai_api_key)

vectordb = Pinecone.from_existing_index(index_name , embeddings)
llm = ChatOpenAI(model_name='gpt-3.5-turbo',openai_api_key=openai_api_key , temperature=0)

# retriever = VectorStoreRetriever(vectorstore=vectordb, search_kwargs={"k":20})
retriever = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
    search_kwargs={"k":20}
)
tool_desc = """Use this tool to answer user questions about the shops in Ibne Batuta Mall. If the user ask about shops information of some shopping related things then use this tool to get
the answer. This tool can also be used for follow up questions from the user."""




tools = [Tool(
    func=retriever.run,
    description=tool_desc,
    name='Ibne Batuta Mall'
)]

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",  # important to align with agent prompt (below)
    k=5,
    return_messages=True
)

conversational_agent = initialize_agent(
    agent='chat-conversational-react-description', 
    tools=tools, 
    llm=llm,
    verbose=True,
    max_iterations=2,
    early_stopping_method="generate",
    memory=memory,
)

# print(conversational_agent.agent.llm_chain.prompt)


sys_msg = """You are a helpful chatbot that answers the user's questions about the shops in the Ibne Batuta Mall.
"""

prompt = conversational_agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
)
conversational_agent.agent.llm_chain.prompt = prompt

# print(conversational_agent.agent.llm_chain.prompt)


zz = conversational_agent("where can find the gaming accessoeires")

print("----------------")
print(zz["output"])
print("----------------")
zz = conversational_agent("similar shops to that")
print(zz["output"])
print("----------------")