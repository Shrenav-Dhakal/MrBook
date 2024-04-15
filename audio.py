import streamlit as st
from audio_recorder_streamlit import audio_recorder
from query import send_data
import os
from dotenv import load_dotenv
load_dotenv()
import assemblyai as aai
aai.settings.api_key = os.getenv("ASSEMBLY_API_KEY")


audio_bytes = audio_recorder()
if audio_bytes:
    with open("audio.mp3", "wb") as f:
        f.write(audio_bytes)
        


def getdata():
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe("audio.mp3")
    send_data(transcript.text)


button = st.button("Click here to get answer", on_click=getdata)

if button:
    st.audio("upload/a.mp3", format="audio/mp3")


    