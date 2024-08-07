import pinecone      
import os
import json

from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
import pinecone
from langchain.document_loaders import JSONLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

openai_api_key = os.environ["THINK_MALL_GPT_KEY"]
PINECONE_API_KEY = os.environ["PINE_API_KEY"]

PINECONE_API_ENV = os.environ["PINE_ENV"]

pinecone.init(
    api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV
)

file_path = 'entertainments.json'
with open(file_path, 'r') as file:
    json_data = json.load(file)


def generate_description(shop_name, shop_data):
    description = shop_data.get('Description', '')
    working_hours = shop_data.get('Working Hours', '')
    contact_number = shop_data.get('Contact Number', '')
    location = shop_data.get('Location', '')

    similar_brands = shop_data.get('Similar Brands', '')
    other_info = shop_data.get('Other Info', '')

    # Create a generic description sentence
    generic_description = f"{shop_name.capitalize()} is located in {location}. "
    if description:
        generic_description += f"{description}. "

    if working_hours:
        generic_description += f"We are open {working_hours}. "

    if contact_number:
        generic_description += f"You can reach us at {contact_number}. "

    if similar_brands:
        generic_description += f"We are similar to brands like {similar_brands}. "

    if other_info:
        generic_description += f"Additional information: {other_info}"

    return generic_description.strip()


# Assuming 'data' contains your JSON structure

# Lists to store information
headings = []
descriptions = []
docs = []

for category, shops in json_data.items():
    for shop_name, shop_data in shops.items():
        description = generate_description(shop_name, shop_data)
        location = shop_data.get('Location', 'Unknown Location')
        
        # Create the heading
        heading = f"{category} - {shop_name.capitalize()} - {location.capitalize()}"
        headings.append(heading)
        
        # Create the description
        descriptions.append(description)
        
        docs.append(Document(
        page_content=description,
        metadata={"category": category, "shop name": shop_name, "location": location.capitalize()}))
    

# Print the information in separate lists

# for heading , description in (zip(headings,descriptions)):
#     print(heading)
#     print(description)
#     print("\n")

# print("\nDescriptions:")
# for description in descriptions:
#     print(description)

print(docs)

embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002",openai_api_key=openai_api_key) 

vectorstore = Pinecone.from_documents(
    docs, embeddings, index_name="think-mall"
)
