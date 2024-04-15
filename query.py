from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from gtts import gTTS
import langchain_google_genai as genai
from dotenv import load_dotenv
load_dotenv()
import os
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap
from langchain_community.embeddings import HuggingFaceEmbeddings

os.getenv("OPENAI_API_KEY")


api = os.getenv("GOOGLE_API_KEY")

def send_data(output: str):
    model = genai.ChatGoogleGenerativeAI(
        google_api_key=api,
        model="gemini-1.0-pro",
        temperature=0.2
    )


    template = """
    You are a Bookworm named Mr. Book who has read all the books. Your 
    knowledge is based in the context provided.
    You answer the questions based on the context as your domain of 
    knowledge. 
    Question inside double backticks :
    ``{question}``
    Context inside triple backticks:
    ```{context}```
    Provide human like helpful and concise answer with word 
    count not exceeding 80 and Do not provide any tags or symbols like
    backwards.
    """

    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

    db = Chroma(persist_directory="chrom1", embedding_function=embeddings)

    prompt = ChatPromptTemplate.from_template(template=template)

    chain = RunnableMap(
        {
            "question":lambda x: x["question"],
            "context": lambda x: db.similarity_search(x['question'], k=4)
        }
    ) | prompt | model | StrOutputParser()

    answer = chain.invoke({'question':output})
    tts = gTTS(text=answer)
    tts.save("upload/a.mp3")







    



