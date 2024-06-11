## LLM with RAG

In this repo you can find 2 llm based apps:

1. Basic chatbot with an LLM. Ask a question, the LLM will provide an answer.
2. A document based Q&A chatbot. You can upload PDF or TXT file and then ask questions related to the content
of the file

### Tech Stack

- Langchain
- Chainlit
- Chroma DB
- OpenAI API
- Llama 3

### Langchain

- Framework for simplifying the creation of applications that use LLMs.

### Chainlit

- Framework which allows the easy creation of chatbot interfaces

### Chroma DB

- Open source database
- General-purpose database (can be used both for local experiments and production deployments)
- Can be used in both in-memory mode and persistent mode
- Embedding functions can be changed

### How to run the application?

Go to the top level folder and run the following commands:

- Make sure you have set your OPENAI_API_KEY as environment variable
- Make sure you have downloaded **Llama3** with [Ollama](https://ollama.com/) 
- [OPTIONAL] Update chainlit.md
- Install required packages:

```shell
    python -m venv venv/
    source venv/bin/activate
    pip install -r requirements.txt
```

LLM without RAG
```shell
   chainlit run llm_no_rag.py
```

LLM with RAG
```shell
  chainlit run llm_with_rag.py 
```