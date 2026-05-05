from src.flat_rag import query_flat_rag

questions = [
    "Who founded OpenAI?",
    "Which company acquired DeepMind?",
    "How is Google related to AlphaGo?",
    "What is the connection between Microsoft and GitHub Copilot?",
    "Who founded NVIDIA?"
]

for question in questions:
    query_flat_rag(question, top_k=3, verbose=True)