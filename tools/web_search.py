# Importing necessary libraries and modules
from langchain_core.tools.base import BaseTool
from dotenv import load_dotenv
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import TavilySearchResults, DuckDuckGoSearchResults
from langchain_tavily import TavilySearch
import os
from pydantic import PrivateAttr
from langchain_community.document_loaders import WebBaseLoader
import json
import requests

load_dotenv(".env", override=True)  # Loading environment variables

# Defining the WebSearchTool class which extends BaseTool
class WebSearchTool(BaseTool):
    name: str = "web_search_tool"
    description: str = "Perform a web search and extract concise factual answers. The query should be concise, below 400 characters. Use for online facts not in GAIA/Wikipedia—e.g. sports stats, Olympic participation, published papers, museum specimen locations, competition winners, and other up-to-date info."
    _search: TavilySearch = PrivateAttr()

    def __init__(self):
        # Initializing the WebSearchTool
        super().__init__()
        self._search = TavilySearch(max_results=3, topic="general")  # Setting up the TavilySearch with specific parameters
    
    def _run(self, query: str) -> dict:
        # Method to run the web search tool with the given query
        search_results = []  # Initializing the list for search results
        search_results.append(self._search.run(query))  # Performing the search and adding the results to the list

        return search_results  # Returning the search results
