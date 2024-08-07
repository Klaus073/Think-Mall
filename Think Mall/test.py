from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
import pinecone
from langchain.vectorstores import Pinecone

openai_api_key = os.environ["THINK_MALL_GPT_KEY"]

PINECONE_API_KEY = os.environ["PINE_API_KEY"]

PINECONE_API_ENV = os.environ["PINE_ENV"]

pinecone.init(
            api_key= PINECONE_API_KEY, 
            environment= PINECONE_API_ENV
)
index_name = "think-mall"

llm = OpenAI(model_name='gpt-3.5-turbo',openai_api_key=openai_api_key , temperature=0)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key) 

embeddings = OpenAIEmbeddings()
vectordb = Pinecone.from_existing_index(index_name , embeddings)
retriever = VectorStoreRetriever(vectorstore=vectordb, search_kwargs={"k":20})

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0,model_name="gpt-3.5-turbo"), vectordb.as_retriever(), memory=memory)

query = "My name's Bob. How are you?"
result = qa({"question": query})

print(result)

query = "What's my name?"
result = qa({"question": query})

print("NEW MESSAGE:",result)