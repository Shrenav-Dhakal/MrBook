from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
import os 
from langchain_community.embeddings import HuggingFaceEmbeddings

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

def create(pdf):
    name = pdf.split("\\")[-1][:-4]
    name = name.split(" ")
    final_name = "".join(name)
    print(final_name)
    loader = PyMuPDFLoader(pdf)
    pages = loader.load_and_split()
    chromadb = Chroma.from_documents(pages, embedding=embeddings, persist_directory="chrom1", collection_name=final_name)






