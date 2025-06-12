from langchain_core.tools.base import BaseTool
import requests
import base64
import pandas as pd
import os
import tempfile
import whisper

class DownloadFile(BaseTool):
    name : str = "download_file_tool"
    description: str = """
        This tool downloads a file (image, pdf, python code, excel, etc.) given the name of the file. The url for the request will be composed in the function so ONLY the name of the file should be passed in.

        You may have to download a file in 2 different scenarios:
        - A file given already as part of the task. In this case the format of the url must be: {DEFAULT_API_URL}/files/{file_name} THE EXTENSION OF THE FILE MUST NOT(!!) BE INCLUDED!
        - A url retrieved from the internet in the format https://some_url. In that case, you simply need to provide the url of the file that needs to be retrieved.

        Args: 
            file_name: the name of the file to be retrieved DEFAULT_API_URL/files/task_id
            file_extension: the extension of the file, without the dot. So for example "pdf", "img", "py", "xlsx", etc.

        Output:
        IF the file is a document, image or audio:
        A string with the path to the file.
        
        IF the file is a piece of code:
            A dict made of:
                The text of the image

        IF the file is an excel:
            A dict made of:
            A pandas dataframe
        """

    def _run(self, file_url: str, file_extension: str) -> dict:
        response = requests.get(file_url)
        
        if response.status_code == 200:
            msg = "File downloaded successfully!!"
            if file_extension in ["png", "jpg", "pdf"]:
                file = response.content
                
                with open("downloaded_files/image.png", "wb") as f:
                    f.write(file)

                return "downloaded_files/image.png"
            elif file_extension in ["mp3", "wav"]:
                res = response.content
                with open("downloaded_files/audio.mp3", mode="wb") as f:
                    f.write(res)

                return f"./downloaded_files/audio.{file_extension}"

            elif file_extension == "py":
                return {"text": response.text}
            elif file_extension == "xlsx":
                file_name = file_url.split("/")[-1]

                with open(f"./downloaded_files/{file_name}.xlsx", "wb") as f:
                    f.write(response.content)

                return f"./downloaded_files/{file_name}.xlsx"
            else:
                return "The file extension is not valid."
        else:
            msg = "There was an error downloading the file."

            return msg

        




