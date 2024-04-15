from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
load_dotenv()
import os 

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

pdfname = r"C:\Users\Shreenav Dhakal\OneDrive\Desktop\Attention is all you need.pdf"

loader = PyMuPDFLoader(pdfname)
pages = loader.load_and_split()

print(pages[0])
chromadb = Chroma.from_documents(pages, OpenAIEmbeddings(), persist_directory="chrom1")







