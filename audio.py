import streamlit as st
from audio_recorder_streamlit import audio_recorder
from query import send_data, send_name
import os
from dotenv import load_dotenv
load_dotenv()
import assemblyai as aai
from create_chroma import create
import translators as ts


aai.settings.api_key = os.getenv("ASSEMBLY_API_KEY")

st.title("Mr Book")


language = st.selectbox(label="Choose Your Language", options=['en', 'fr', 'hi'])

uploaded_file = st.file_uploader("Upload your pdf", type='pdf')


if st.button("Submit"):
    with st.spinner("Processing....Please Wait"):
        if uploaded_file is not None:
            filename = os.path.basename(uploaded_file.name)
            savepath = os.path.join("uploaded_files",filename)
            with open(savepath, "wb") as f:
                f.write(uploaded_file.getvalue())
            send_name(savepath)
            create(savepath)
         
temperature = st.slider(label="Precision preference: 0 for precise, 1 for creative.", min_value=0.0, max_value=1.0, step=0.1)

audio_bytes = audio_recorder()
if audio_bytes:
    with open("audio.mp3", "wb") as f:
        f.write(audio_bytes)
        


def getdata():
    config = aai.TranscriptionConfig(language_code=language)
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe("audio.mp3")
    print("transcribed text = ", transcript.text)
    new_text = ts.translate_text(transcript.text, from_language="auto", to_language="en")
    print("New Text =", new_text)
    send_data(output=new_text,temp=temperature,lang=language)


button = st.button("Click here to get answer", on_click=getdata)

if button:
    st.audio("upload/a.mp3", format="audio/mp3")


    