# Importing necessary libraries and modules
from langchain_core.tools.base import BaseTool
import whisper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from pathlib import Path
import os
from transformers import pipeline
import torch
from langchain_openai import ChatOpenAI
import time

# Defining the AudioTool class which extends BaseTool
class AudioTool(BaseTool):
    name : str = "answer_question_audio_tool"
    description: str = "This tool will reply to a query based on the audio given the path of a locally stored file. This file DOES NOT DOWNLOAD the file from the web. Run the download_file_tool first" 

    def _run(self, query: str, file_path: str) -> str:
        # Method to transcribe the provided audio file and answer the query using LLM
        try:
            #pipe = pipeline(
            #    task="automatic-speech-recognition",
            #    model="openai/whisper-base",
            #    torch_dtype=torch.float32,
            #    device=0,
            #    return_timestamps=True
            #)         
            #result = pipe(str(Path("./") / Path(file_path)), return_timestamps=True)
            model = whisper.load_model("base")
            result = model.transcribe(audio=str(Path("./") / Path(file_path)), language='en')  # Transcribing the audio using Whisper model
        except Exception as e:
            print("Exception", e)

        print(result["text"])

        human_message = HumanMessage([{"type": "text", "text": query},
            {"type": "text", "text": f"\n\nTranscript: {result['text']}"}])

        system_message = SystemMessage("""You are a helpful assistant. Whenever you receive a transcript of an audio recording along with a user's query:

1. Carefully read the query multiple times to ensure you fully grasp what is being asked.

2. Start by thinking, in clear bullet points, each precise requirement implied by the user's instructions (e.g., which portions of the transcript to use, what to include or exclude, and any specific formatting). 

3. After thinking more about the requirements, fulfill the request exactly as specified. Follow all content and formatting rules without deviation (for instance, “list only names,” “omit quantities,” “use comma-separated values,” “alphabetize,” etc.). 

4. Ensure that your final answer adheres strictly to the user's criteria and contains nothing beyond what was requested.

Always prioritize accuracy and strict adherence to the user's stated needs before providing the answer. REPLY ONLY WITH WHAT THE HUMAN ASKED. Return only the final answer!""")

        time.sleep(5)
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        
        response = llm.invoke([system_message, human_message])  # Getting the response from the LLM

        return response  # Returning the response from the LLM



