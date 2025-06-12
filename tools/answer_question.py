# Importing necessary libraries and modules
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import PrivateAttr
from langchain_core.tools.base import BaseTool
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import time
from openai import OpenAI

# Defining the AnswerQuestionTool class which extends BaseTool
class AnswerQuestionTool(BaseTool):
    name : str = "answer_question_tool"
    description: str = "Use this tool to answer any elementary question that you can solve without needing access to any external tool. Simply provide the question in input, reporting the whole question including desired output format. You can use this tool for example for vegetable classification."
    _llm = PrivateAttr()
    _system_prompt = PrivateAttr()

    def __init__(self):
        # Initializing the AnswerQuestionTool
        super().__init__()
        #self._llm = ChatGoogleGenerativeAI(
        #    model="gemini-2.0-flash",
        #    temperature=0)
        #self._llm = ChatOpenAI(model="o4-mini", temperature=0)


        self._system_prompt = SystemMessage("""You are a helpful assistant.
                                            You will be given a question and you will have to answer that question.
                                            Provide also the reasoning for your answer as well as your final answer.

                                            When provided with a list you must stick with the exact terms provided in the list and not make any modification.
                                            Green beans, corn and zucchini are NOT VEGEATABLES BOTANICALLY!
                                            Let's think step by step.
                                            """)
        
    def _run(self, question: str) -> str:
        # Method to run the tool and get an answer for the given question
        human_message = HumanMessage(
            # Creating a human message with the question content
            content=[
                {"type": "text", "text": question},
            ]
        )

        time.sleep(5)  # Adding a delay for rate limits
        client = OpenAI()  # Initializing the OpenAI client
        response = client.responses.create(
            # Creating a response using OpenAI's API
            model="o4-mini",
            messages = [
                {
                    "role": "system", "content": self._system_prompt.text()
                },
                {
                    "role": "user", "content": question
                }]
            )
        #response = self._llm.invoke([self._system_prompt, human_message])

        return response  # Returning the response from the OpenAI API