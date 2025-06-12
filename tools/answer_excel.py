# Importing necessary libraries and modules
from langchain_core.tools.base import BaseTool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType

# Defining the AnswerExcelTool class which extends BaseTool
class AnswerExcelTool(BaseTool):
    name : str = "answer_excel_tool"
    description: str = "Given the path to a file containing an excel file and a query, this tool tries to get an answer by querying the excel file. Provide the whole question in input. Another agent will later break down the task."

    def _run(self, query: str, file_path: str) -> str:
        # Method to run the tool, using a query and the file path to an Excel file
        df = pd.read_excel(file_path)  # Reading the Excel file into a DataFrame

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)  # Configuring the LLM

        agent_executor = create_pandas_dataframe_agent(
            # Creating a Pandas DataFrame agent with the LLM and DataFrame
            llm,
            df,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            allow_dangerous_code=True  # IMPORTANT: Understand the risks
        )

        return agent_executor(query)  # Executing the query using the agent