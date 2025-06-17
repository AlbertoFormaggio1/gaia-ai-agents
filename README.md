# AI Agent for GAIA Dataset

This repository contains the code and solution developed for the final test of the Hugging Face course on AI Agents. It demonstrates the capabilities of an AI Agent using the GAIA dataset.

## Repository Structure

- **`app.py`**: Fetches test questions from the Hugging Face server and executes the AI Agent.
- **`tools/`**: Contains custom tools created specifically to assist the Agent in solving tasks.

## Performance and Considerations

The implemented solution achieved a **70% score** on the GAIA benchmark using the following models:

- Gemini-2.0-flash
- GPT-4.1-mini

However, due to the inherent non-deterministic behavior of these large language models (LLMs), outputs can occasionally vary even when using a temperature setting of `0`. Batched inference is likely the main contributor to this variability.

Additionally, a substantial number of tools were necessary to enhance performance given the limitations of these more economical models. For instance, a dedicated chess tool was implemented due to consistent inaccuracies in parsing image data by GPT-4.1-mini and Gemini models.

## Installation

1\. **Clone the repository**

```bash
git clone <repository-url>
```

2\. **Set up Python virtual environment**

**Unix or MacOS**:
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows**:
```batch
python -m venv .venv
.\.venv\Scripts\activate
```

3\. **Install dependencies**

```bash
pip install -r requirements.txt
```

4\. **Configure environment variables**

**Unix or MacOS**:
```bash
export HF_TOKEN="your_huggingface_token"
export OPENAI_API_KEY="your_openai_api_key"
export GEMINI_API_KEY="your_gemini_api_key"
```

**Windows**:
```batch
set HF_TOKEN=your_huggingface_token
set OPENAI_API_KEY=your_openai_api_key
set GEMINI_API_KEY=your_gemini_api_key
```

5\. **Run the application**

```bash
python app.py
```

Upon starting, log in as prompted and run the provided questions.
