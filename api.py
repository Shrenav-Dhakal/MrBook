from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import assemblyai as aai
from dotenv import load_dotenv
load_dotenv()
import os

aai.settings.api_key = os.getenv("ASSEMBLY_API_KEY")
app = FastAPI()


app.mount("/upload/{filename}", StaticFiles(directory="upload"), name="upload")

@app.get("/text_to_speech")
async def text_to_speech(data: str):
    
    return FileResponse("upload/a.mp3")


# @app.get("/speech_to_text")
# async def speech_to_text(url:str = "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0019_8k.wav"):
#     if url:
#         transcriber = aai.Transcriber()
#         transcript = transcriber.transcribe(url)
#     return transcript.text




