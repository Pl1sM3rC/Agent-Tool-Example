# Setup
1. Install the dependancies (preferrably in a conda environment)
2. Create a .env file and create the key DEEPSEEK_API_KEY="Your Key here"


# Struktur der Methode 
Grundsätzlich brauche ich nur eine API-Methode mit der ich die wichtigen Funktionen an einem Ort aufrufen kann. 
Die Methode wird dann so wie du es im Beispiel sehen kannst eingebunden. 
Wichtig wäre nur, dass jede Methode eine kurze, prägnante Beschreibung hat, was sie macht und welche inputs sie erwartet. 
* https://github.com/langchain-ai/langchain-mcp-adapters

# Troubleshooting
* Falls du keinen Deepseek API key hast, kannst du stattdessen LM Studio benutzen, um eine LLM lokal auf deinem Rechner laufen zu lassen: 
```python
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
  base_url="http://localhost:8001/v1",
  temperature=0,
  api_key="not-needed"
)
```
