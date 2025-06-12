# Importing necessary libraries and modules
from langchain_core.tools.base import BaseTool, ToolException
from typing import Optional
import subprocess
import tempfile
import os
from pydantic import PrivateAttr

# Defining the PythonExecutionTool class which extends BaseTool
class PythonExecutionTool(BaseTool):
    # A LangChain “tool” that takes a string of Python code,
    # writes it to a temporary .py file, executes it in a fresh
    # Python subprocess, captures stdout/stderr, and returns the result.

    name : str = "python_execution"
    description : str = (
        "Executes a string of Python code in an isolated subprocess. "
        "Returns stdout on success, or stderr (with exit code) on failure."
    )
    _python_executable: str = PrivateAttr()
    _timeout: int = PrivateAttr()
    _temp_dir: str = PrivateAttr()

    def __init__(
        self,
        python_executable: str = "C:\\Users\\FORMAGGA\\Documents\\personal\\Final_Assignment_Template\\.venv\\Scripts\\python.exe",
        timeout: int = 5,
        *,
        temp_dir: Optional[str] = None
    ):
        
        """
        :param python_executable: Path to the Python interpreter to invoke.
        :param timeout: Maximum seconds to allow the code to run.
        :param temp_dir: Optional directory in which to create the temp file.
        """
        super().__init__()
        self._python_executable = python_executable
        self._timeout = timeout
        self._temp_dir = temp_dir

    def _run(self, code: str) -> str:
        """
        Synchronously execute the provided Python code.
        :param code: The complete Python source to run.
        :return: Captured stdout if exit code is 0; otherwise stderr + exit code.
        :raises ToolException: On internal error (e.g. unable to write temp file).
        """
        # 1. Write code to a temporary file on disk to avoid shell-quoting issues.
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".py", delete=False, dir=self._temp_dir, mode="w", encoding="utf-8"
            ) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
        except Exception as e:
            raise ToolException(f"Failed to write temp file: {e!r}")

        # 2. Invoke a fresh Python process on that file, capturing stdout & stderr.
        try:
            result = subprocess.run(
                [self._python_executable, "-u", tmp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            return f"⚠️ Execution timed out after {self._timeout} seconds."
        except Exception as e:
            raise ToolException(f"Failed to launch subprocess: {e!r}")
        finally:
            # 3. Clean up the temp file no matter what
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        # 4. Process the result
        if result.returncode != 0:
            return (
                f"❌ Process exited with code {result.returncode}.\n"
                f"stderr:\n{result.stderr}"
            )
        return result.stdout 