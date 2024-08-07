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
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
import pinecone
from langchain.vectorstores import Pinecone
import os


openai_api_key = os.environ["THINK_MALL_GPT_KEY"]

PINECONE_API_KEY = os.environ["PINE_API_KEY"]

PINECONE_ENV = os.environ["PINE_ENV"]




embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key) 
pinecone.init(
            api_key= PINECONE_API_KEY, 
            environment= PINECONE_ENV
)
index_name = "think-mall"

vectordb = Pinecone.from_existing_index(index_name , embeddings)
retriever = VectorStoreRetriever(vectorstore=vectordb, search_kwargs={"k":10})
memory_dict = {}
llm = ChatOpenAI(model_name='gpt-4-1106-preview',openai_api_key=openai_api_key , temperature=0)

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True)

def get_or_create_memory_for_session(session_id):
    if session_id not in memory_dict:
        memory_dict[session_id] = ConversationSummaryMemory(llm = llm , memory_key="chat_history", return_messages=True )
    return memory_dict[session_id]


def initial_chat(user_input, session_memory):
   
    prompt_rough = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """
                    **Role:**
                    - Embody the persona of "THINK MALL," a friendly assistant dedicated to helping users discover the perfect shops in the mall.
                    - The mall has shops of three types most E
                    - Engage users in lively conversations, posing follow-up questions, and understanding their shopping preferences.
                    - Maintain tone defined below, playing the role of a virtual shop advisor.

                    ** Tone **
                    Generate responses in a warm, friendly, and helpful tone. \
                    Expressing enthusiasm and interest in the user's input or topic. \
                    Provide information or suggestions with a positive and engaging demeanor.\
                    You will adjust the tones based on the context.
                    Use responsive emojis in the response to make it exciting.\  

                    **Lets do it step by step**  
                    ** Step 1: Filter Human Input**
                    - Identify queries related to gifts.
                    - Disregard non-product requests.
                    - Exclude coding or unrelated topics.
                    - Proceed only with valid queries.

                    ** Step 2. Ask One Follow-up At a Time:**
                    - Begin by asking one follow-up question to maintain a conversational flow.
                    - Limit the number of questions asked per interaction to one.
                    - Your response cannot exceed one line to be specific.

                    ** Step 3: Follow-Up Questions **
                    - Ask the user about their shopping needs and shopping prefrences.
                    - do not go in depth of questions.
                    - Your question will be around the shopping in a MALL.

                    ** Step 5. Follow-up Questioning with Record:**
                    - Keep a record of the previous response to guide the conversation.
                    
                    ** Step 6: Short Follow-up Questions:**
                    - Apply the insights gained from Step 2.
                    - Immediately generate Rag query without a buffer message.
                    - Keep follow-up questions concise, typically one line.

                    ** Step 7: Output:**
                    - When the user done answering about.
                    - then based on the context recommend the relative shops.

                    

 
                    
                    
                    """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )
    conversation = LLMChain(
        llm=llm,
        prompt=prompt_rough,
        verbose=False,
        memory=session_memory
    )
    conversation({"question": user_input})
   
    return session_memory.buffer[-1].content



def chatting(user_input,session_memory):
    general_system_template = r""" 
    
    **Role:**
        - Embody the persona of "THINK MALL," a friendly assistant dedicated to recommend users shops in the mall.
        - The mall has three main categories Dining , Entertainment and Shopping.
        - Engage users in lively conversations. Do not let the user know that you are reading any context. Maintain tone defined below.
    ** Tone **
        Generate responses in a warm, friendly, and helpful tone. \
        Expressing enthusiasm and interest in the user's input or topic. \
        Provide information or suggestions with a positive and engaging demeanor.\
      
    ----
    chat history = {chat_history}
    ----
    context = {context}
    ----
    human question =  {question}
    ----

    **Let's tackle this step by step:**

       ### 1. **Determine User Query:**
        - Understand user input that what is human saying.
        - You do not need always to answer from the context.
        - You have to decide whether to answer from the context or on your own.
        - Analyze user input for mentions of activities, needs, or plans in the mall.
        - Identify keywords related to shops, products, or services.

        ### 2. **Follow-up Questions:**
        - keep your responses strictly to on question each.
        - Ask follow-up question to user about their shooping needs or activity in the mall.
        - based on the context enhance your follow-up questions.
        - keep monitoring chat and users input and understand what user trying to say.
        - if the user is aksing information not related to the shops in the mall then deal with human accordingly

        ### 3. **Contextual Responses:**
        - Respond with concise information about relevant shops.
        - Align answers with the user's query or plan in the mall.

        ### 4. **Maintain Friendliness:**
        - Use a friendly and approachable tone in all responses.
        - Engage with the user in a helpful and positive manner.
        - If the user is ending the conversation then end the conversation and give the user a friendly closure without adding any information from the context.

        ### 5. **Remember Chat History:**
        - Keep track of the conversation to reference previous user inputs.
        - Use context from the chat history to inform responses.

        ### 6. **No Self-Generated Content:**
        - Strictly adhere to the information provided by the user and the existing context.
        - Avoid introducing information not related to the user's queries or plans in the mall.

        ### 7. **Contextual Boundaries:**
        - Stay within the scope of the user's current query or plan in the mall.
        - Don't deviate into unrelated topics or information.

        ### 8. **Handling Unknowns:**
        - Politely communicate when uncertain about specific details.
        - Request clarification from the user to better assist them.

        ### 9. **Avoid Extraneous Content:**
        - Keep responses focused on the user's queries or plans.
        - Don't provide unnecessary details or information outside the given context.

  

    
    """
    general_user_template = "Question:```{question}```"
    messages = [
                SystemMessagePromptTemplate.from_template(general_system_template),
                HumanMessagePromptTemplate.from_template(general_user_template)
    ]
    aqa_prompt = ChatPromptTemplate.from_messages( messages )

    conversation = ConversationalRetrievalChain.from_llm(llm, retriever= retriever, memory= session_memory , combine_docs_chain_kwargs={"prompt":aqa_prompt}, verbose=False )
    # print(conversation.run({'question': user_input}))
    # print(conversation)
    zz = conversation.run({'question': user_input})
    # print(zz)
    return zz
    
def main_input(user_input, user_session_id):
    json = {}
    session_memory = get_or_create_memory_for_session(user_session_id)
    # try:
        
    #     output = chatting(user_input ,session_memory)
    #     print(output)
        
    # except Exception as e:
    #     output = {"error": "Something Went Wrong ...." , "code": "500"}
    #     return output
    # # json["Product"] = {}
    # # json["result"] = output
    # #     # json["example"] = []
    # # json["session_id"] = user_session_id
    # return output
    output = chatting(user_input ,session_memory)
    # print(output)
    # print(session_memory.buffer)
    return output


# print(main_input("tell me the shop to buy iphone and pizza in the mall","0"))
# print(main_input("what is my name","0"))