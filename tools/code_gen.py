from langchain_core.tools.base import BaseTool
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv(".env", override=True)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT_GEN")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY_GEN")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION_GEN", "2023-12-01-preview") # Default API version
# AZURE_OPENAI_DEPLOYMENT_NAME will be used as the 'model' for API calls
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4.1"

class CodeGenTool(BaseTool):
    name : str = "code_generator_tool"
    description: str = "Given the instructions provided, it generates Python code as text. It's important that the instructions provide: which args must be provided in input, the content of the function and what is the desired output."

    def _run(self, function_description: str, input: str, output: str) -> str:
        if not function_description:
            return "You need to pass in a function description. Retry providing the right parameters."

        system = SystemMessage("""You are an expert software engineer, your goal is to generate a piece of code.
                               YOU MUST GENERATE A **PYTHON** FUNCTION. 
                               You will be given a description of what the function needs to do, for example "Generate a function that retrieves a web page from the internet".
                               Then you will be given information about what the input parameters are and the output.

                               In the output code you must list the imports as well.
                               It's VERY IMPORTANT that you stick to the contraints given for input and output.
                               If you believe there is a better way to do things, IGNORE THIS IDEA and stick to what is given in input.
                                """)
        
        human = HumanMessage(f"Description of the function:\n{function_description}\n\nInput parameters:\n{input}\n\nOutput result:\n{output}\n\n")

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.5)
        
        response = llm.invoke([system, human])

        return response
