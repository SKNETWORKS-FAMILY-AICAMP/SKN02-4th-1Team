from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough

OPENAI_KEY = "your key"

loader = PyPDFLoader("../../data/sk-채용정보.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
splits = text_splitter.split_documents(pages)

# 문서를 디스크에 저장합니다. 저장시 persist_directory에 저장할 경로를 지정합니다.ectorstore = Chroma.from_documents(splits, OpenAIEmbeddings(), persist_directory='../ChromaDB', collection_name="my_db")


vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=OPENAI_KEY), persist_directory='../ChromaDB', collection_name="my_db")