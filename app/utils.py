"""Utilities for backend service."""
from llama_index.core import Settings as LlamaIndexSettings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.llms.vertex import Vertex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.huggingface import HuggingFaceLLM
# from llama_index.llms.groq import Groq

from config import settings


def init_azure_openai_models():
    """Inits embedding model and LLM for Azure OpenAI and configures in llama-index
    settings.

    Example:

    .. code-block:: python

        from llama_index.core import Settings

        init_azure_openai_models()
        Settings.embed_model.get_text_embedding("this is a test")

        Settings.llm.chat.completions.create(
            messages=[{
                "role": "user",
                "content": "How many toes do dogs have?",
            }],
            model="gpt-35-turbo"
        )
    """
    embed_model = AzureOpenAIEmbedding(
        deployment_name=settings.azure_openai_embed_model_deployment_name,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )
    LlamaIndexSettings.embed_model = embed_model
    llm = AzureOpenAI(
        azure_deployment=settings.azure_openai_llm_deployment_name,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )
    LlamaIndexSettings.llm = llm


def init_vertexai_models():
    """Inits embedding model and LLM for Google Vertex AI and configures in llama-index
    settings.

    To authenticate with GCP interactively, here are the two methods.
    Method 1: with a user account
    Make sure you have gcloud-cli installed and run the following commands:

        gcloud auth application-default login
        gcloud auth application-default set-quota-project <you-project-name>

    Method 2: with a service account
    Make sure you have the service account credential file available via env var:

        export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credential.json

    Example:

    .. code-block:: python

        from llama_index.core import Settings
        from llama_index.core.llms import ChatMessage, MessageRole

        init_vertexai_models()
        Settings.embed_model.get_text_embedding("this is a test")

        Settings.llm.completion("hello")
        Settings.llm.chat(
            messages=[
                ChatMessage(role=MessageRole.SYSTEM, content="Reply everything in Japanese"),
                ChatMessage(role=MessageRole.USER, content="Hello"),
            ]
        )
    """

    embed_model = VertexTextEmbedding(
        project=settings.gcp_project,
        model_name=settings.vertexai_embed_model,
    )
    LlamaIndexSettings.embed_model = embed_model
    llm = Vertex(
        project=settings.gcp_project,
        model=settings.vertexai_llm,
    )
    LlamaIndexSettings.llm = llm

def init_models():
    """Inits embedding model from huggingface and LLM model from groq for configures in llama-index
    settings.
    """
    embed_model = HuggingFaceEmbedding(
        model_name=settings.hf_embedding
    )
    LlamaIndexSettings.embed_model = embed_model
    llm = Groq(
        model=settings.groq_model_name,
        api_key=settings.groq_api_key
    )
    LlamaIndexSettings.llm = llm