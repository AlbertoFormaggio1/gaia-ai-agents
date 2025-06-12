from langchain_core.tools.base import BaseTool

class ReverseString(BaseTool):
    name: str = "reverse_string_tool"
    description: str = ("This tool inverts the order of the characters within a sentence. It is particularly useful if you can't understand the content in any language.")

    def _run(self, string: str) -> str:
        return string[::-1]