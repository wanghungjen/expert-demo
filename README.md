## How To Run
1. Create a .env file and type the following (for local llm)
```
OPENAI_API_BASE=http://localhost:11434/v1
OPENAI_MODEL_NAME=llama3
OPENAI_API_KEY=NA
```
2. Open up terminal and type in the following to install
```
cd expert-demo
poetry install --no-root
poetry shell
```
3. Type in the interpreter path into VSCode
4. Run 1_crew.py
```
python 1_crew.py
```

## References
RAG: https://www.youtube.com/watch?v=7GhWXODugWM

Local LLM: https://www.youtube.com/watch?v=0ai-L50VCYU

## Notes
Llama 3 and all-minilm used
