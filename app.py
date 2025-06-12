# Importing necessary libraries and modules
import os
import gradio as gr
import requests
import inspect
import pandas as pd
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import List, TypedDict, Annotated, Optional
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode, tools_condition
from PIL import Image
import requests
from io import BytesIO
import PyPDF2
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
import time
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import BraveSearch
from tools.answer_question_from_file import AnswerQuestionFromFileTool
from tools.answer_question import AnswerQuestionTool
from tools.download_file import DownloadFile
from tools.reverse_string import ReverseString
from tools.web_search import WebSearchTool
from tools.wikipedia import WikipediaTool
from tools.youtube_transcript import YoutubeTranscriptTool
from tools.code_exec import PythonExecutionTool
from tools.code_gen import CodeGenTool
from tools.answer_excel import AnswerExcelTool
from contextlib import redirect_stdout
from tools.chess_tool import ChessTool
from tools.audio_tool import AudioTool
from tools.fetch_web_page import FetchWebPageTool

load_dotenv(".env", override=True)
BRAVE_API_KEY = os.getenv("BRAVE_API")

# Defining the State class which will hold various parameters related to the agent's state
class State(TypedDict):
    file_path : str
    file: Optional[str]
    parsed_file: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]
    parsed_file_message: dict
    question: str
    response: str

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Basic Agent Definition ---
# Defining the BasicAgent class which contains the logic for the AI agent
class BasicAgent:
    def __init__(self):
        # Initializing the BasicAgent with tools and an LLM (Large Language Model)
        tools = [CodeGenTool(), PythonExecutionTool(temp_dir="./"), YoutubeTranscriptTool(), 
                 AnswerQuestionFromFileTool(), AnswerQuestionTool(), DownloadFile(), 
                 ReverseString(), WebSearchTool(), WikipediaTool(), AnswerExcelTool(), ChessTool(), AudioTool(), FetchWebPageTool()]

        #llm = ChatGoogleGenerativeAI(
        #    model="gemini-2.0-flash",
        #    temperature=0)
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)  # Configuring the LLM
        self.llm_with_tools = llm.bind_tools(tools)

        builder = StateGraph(State)  # Building a state graph for handling agent's state transitions

        builder.add_node("assistant", self.assistant)  # Adding nodes to the state graph
        builder.add_node("tools", ToolNode(tools))
        builder.add_node("final_answer", BasicAgent.final_answer)
        #builder.add_node("download_file", BasicAgent.download_file_node)
        #builder.add_node("parse_img", BasicAgent.parse_image)
        #builder.add_node("parse_pdf", BasicAgent.parse_pdf)
        #builder.add_node("parse_audio", BasicAgent.parse_audio)
        #builder.add_node("extract_data", BasicAgent.extract_data_from_file)

        builder.add_edge(START, "assistant")
        #builder.add_conditional_edges("download_file", BasicAgent.determine_file_type,
        #    {"img": "parse_img", "pdf": "parse_pdf", "audio": "parse_audio", "end": END})
        #builder.add_edge("parse_img", "assistant")
        #builder.add_edge("parse_pdf", "assistant")
        #builder.add_edge("parse_audio", "assistant")
        builder.add_conditional_edges(  # Adding conditional edges to manage state transitions based on tools' availability
            "assistant",  # Starting with the assistant node
            tools_condition,
            path_map={
                "tools":   "tools",
                "__end__": "final_answer"
            }   
        )

        builder.add_edge("tools", "assistant")  # Defining edges for state transitions
        builder.add_edge("final_answer", END)

        self.react_graph = builder.compile()  # Compiling the state graph into a reactive graph


    def __call__(self, question: str, task_id: str, file_name: Optional[str]) -> str:
        # Handling the agent's main call
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        
        messages = [HumanMessage(question)]  # Creating a list of human messages
        messages = self.react_graph.invoke({"messages": messages, "file_path": file_name, "question": question})  # Invoking the reactive graph with the current state

        with open(f'messages_{task_id}.txt', 'w', encoding='utf-8') as out:  # Writing the messages to a file
            with redirect_stdout(out):
                for m in messages['messages']:
                    m.pretty_print()

        final_answer = messages["messages"][-1].content.strip()  # Extracting the final answer from the messages
        print(f"Final answer is {final_answer}")
        return final_answer


    def assistant(self, state: State):
        # Defining the assistant node which processes the state
        if state["file_path"]:
            file_name = state["file_path"].split(".")[0]
            file_extension = state["file_path"].split(".")[1]
        else:
            file_extension = None
            file_name = None
        
        prompt = f"""  # Constructing the prompt for the language model
        You are a general AI assistant. When I ask you a question:

        Share your reasoning process clearly.

        End with the exact template:
        FINAL ANSWER: [YOUR FINAL ANSWER]

        -------------------------------------------
        Guidelines for FINAL ANSWER:

        - Use a single number, a minimal phrase, or a comma-separated list of numbers and/or strings.

        - For numbers, do not use commas, currency symbols, or percentage signs unless explicitly requested.

        - For strings, avoid articles and abbreviations (e.g., no city abbreviations). Write digits in full text unless otherwise specified.

        - Do not change capitalization of the terms you see unless it explicitly specified.

        NEVER REPEAT THE SAME SEARCH MORE THAN ONCE, EVEN WITH SIMILAR TERMS. If you didn't find anything on the first go, it means there's nothing with that search query available.
        If you can't find an answer just say you can't find it without repeating the same thing over and over.

        Always read the prompt carefully.

        Start with Wikipedia when searching for information. If Wikipedia doesn't have the answer, then use the web search tool. Use every available resource to find the correct answer.

        IMPORTANT: Never make assumptions. Always use the provided tools!! If you are asked a question you think you know without using any tool, do not answer but invoke the answer_question_tool provided the WHOLE question in input.

        NOTE: the question about the actor is tricky: they want to know who Bartłomiej played in Magda M.

        If a file is provided (named {file_name} with extension {file_extension}), your first action MUST BE TO CALL the download_file tool with the URL:
        {DEFAULT_API_URL}/files/{file_name}
        Do **NOT** include the file extension in the URL and send WITHOUT MODIFICATION.
        """

        sys_msg = SystemMessage(content=prompt)  # Creating a system message with the prompt
        
        time.sleep(40)  # Simulating a delay for processing
        return {"messages": [self.llm_with_tools.invoke([sys_msg] + state["messages"])]}

    def final_answer(state: State):
        # Defining the final answer node which processes the state and returns an answer
        system_prompt = f"""
        You will be given an answer and a question. You MUST remove EVERYTHING not needed from the answer and answer the question exactly without reporting "FINAL ANSWER". 
        That is if you are being asked the number of something, you must not return the thought process, but just the number X.

        You must be VERY CAREFUL of what the question asks!!!
        For example:
            if they ask you to give the full name of a city without abbreviations you should stick to it (for example, St. Petersburg should be Saint Petersburg).
            if the first name is asked, you MUST return the first name only (Claus and not Claus Peter)!
        Remove full stops at the end, they are not needed. If you return something comma separated, there must always be a space between the comma and the next letter. Always!!
        """

        human_prompt = f"""
        Question: {state['question']}

        Answer: {state['messages'][-1]}
        """

        human_msg = HumanMessage(content=human_prompt)

        sys_msg = SystemMessage(content=system_prompt)

        time.sleep(1)

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0)

        response = llm.invoke([sys_msg, human_msg]) 

        return {"messages": state["messages"] + [response]}

def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # Running the agent on the fetched questions
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(question_text, task_id, file_name)

            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}  # Preparing the data for submission
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    # Submitting the answers
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# Building the Gradio interface using Blocks
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")  # Title of the interface
    gr.Markdown(
        """
        # Instructions for the interface usage

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()  # Login button for Hugging Face account

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    # Launching Gradio interface for the application
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)