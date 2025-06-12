# Importing necessary libraries and modules
from langchain_core.tools.base import BaseTool
from typing import List
import requests

# Defining the FetchWebPageTool class which extends BaseTool
class FetchWebPageTool(BaseTool):
    name : str = "fetch_web_page_tool"
    description: str = "Provided the urls of 1 or more web pages, this tool returns the full content of the web page. This tool needs to be called AFTER calling the web_page_tool. It's important to fetch only pages which are useful to your task!"

    def _run(self, urls: List[str]) -> List[str]:
        # Method to fetch the full content of the provided web pages
        pages = [requests.get(url).text for url in urls]  # Fetching the content of each URL

        return pages  # Returning the fetched content of the web pages
