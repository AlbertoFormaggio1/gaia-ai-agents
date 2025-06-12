# Importing necessary libraries and modules
from langchain_core.tools.base import BaseTool
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import PrivateAttr
import os
from dotenv import load_dotenv
import whisper
import base64

load_dotenv(".env", override=True)  # Loading environment variables

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # Fetching Azure OpenAI endpoint from environment
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION_GEN", "2023-12-01-preview") # Default API version
# AZURE_OPENAI_DEPLOYMENT_NAME will be used as the 'model' for API calls
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4.1"


# Defining the AnswerQuestionFromFileTool class which extends BaseTool
class AnswerQuestionFromFileTool(BaseTool):
    name: str = "answer_question_from_file_tool"
    description: str = """
        This tool allows you to answer a question taking into account information that were provided inside a file. 
        You must provide the file in b64 when processing here.

        Args:
            The question that needs to be answered.
            The file extension of the file that is being processed.
        """
    _llm = PrivateAttr()

    def __init__(self):
        # Initializing the AnswerQuestionFromFileTool
        super().__init__()
        self._llm = ChatGoogleGenerativeAI(  # Setting up the LLM with specific parameters
            model="gemini-2.0-flash",
            temperature=0)


    def _run(self, question: str, file_name: str, file_extension: str) -> str:

        with open(file_name, "rb") as f:
            file = f.read()

        if file_extension in ["png", "jpg"]:
            encoded_file = base64.b64encode(file).decode("utf-8")

            message = {"type": "image_url", "image_url": f"data:image/png;base64,{encoded_file}"}
        elif file_extension == "pdf":
            encoded_file = base64.b64encode(file).decode("utf-8")
            message = {"type": "image_url", 
                    "image_url": f"data:application/pdf;base64,{encoded_file}"
                  }
        else:
            message = {"type": "text", "text": "The file is not supported."}

        message_local = HumanMessage(
            content=[
                {"type": "text", "text": question + "\nLet's think step by step."},
                message,
            ]
        )

        response = self._llm.invoke([message_local])

        return response
    
    