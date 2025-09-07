from enum import Enum


class ModelProvider(str, Enum):
    AZURE_OPENAI = "AzureOpenAI"
    GOOGLE_VERTEX = "Vertex"
    HUGGINGFACE = "HuggingFace"
    GROQ = "Groq"
