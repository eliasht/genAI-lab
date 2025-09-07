import sys

sys.path.append("../app/")

import streamlit as st
from utils import init_models

from typing import Any, Dict, List
import os
import time
import logging
from dataclasses import dataclass
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.groq import Groq

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# (opsional) aktifkan log bawaan LlamaIndex juga
from llama_index.core import Settings as _LISettings
_LISettings.log_level = logging.INFO


# =====================
# Konfigurasi
# =====================
ITE_INDEX_ID = os.getenv("ITE_INDEX_ID", "ite_vector_chunked_index")
PAJAK_INDEX_ID = os.getenv("PAJAK_INDEX_ID", "pajak_vector_chunked_index")
ITE_PERSIST_DIR = os.getenv("ITE_PERSIST_DIR", "storage_ite")
PAJAK_PERSIST_DIR = os.getenv("PAJAK_PERSIST_DIR", "storage_pajak")
DEFAULT_SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 5))

# Set embedding model

init_models()

# ===============================old====================================
# Load index 
# from llama_index.core import StorageContext, load_index_from_storage

# rebuild storage context
# storage_context = StorageContext.from_defaults(persist_dir="storage")

# # load index
# index = load_index_from_storage(storage_context, index_id="vector_chunked_index")

# Define query engine
# query_engine = index.as_query_engine(similarity_top_k=5)
# =======================end old==========================================

# =====================
# Helper: load dua index (persist dir terpisah)
# =====================


def load_indexes(
    ite_persist_dir: str = ITE_PERSIST_DIR,
    pajak_persist_dir: str = PAJAK_PERSIST_DIR,
    ite_index_id: str = ITE_INDEX_ID,
    pajak_index_id: str = PAJAK_INDEX_ID,
    ):
    # Per index, kita buat StorageContext terpisah
    sc_ite = StorageContext.from_defaults(persist_dir=ite_persist_dir)
    sc_pajak = StorageContext.from_defaults(persist_dir=pajak_persist_dir)

    index_ite = load_index_from_storage(sc_ite, index_id=ite_index_id)
    index_pajak = load_index_from_storage(sc_pajak, index_id=pajak_index_id)
    return index_ite, index_pajak

class LoggingQueryEngine:
    def __init__(self, name: str, engine):
        self._name = name
        self._engine = engine
    def query(self, q):
        logging.info("[ENGINE] Running tool: %s", self._name)
        return self._engine.query(q)
    async def aquery(self, q):
        logging.info("[ENGINE] Running tool (async): %s", self._name)
        return await self._engine.aquery(q)

def build_query_engine_tools(index_ite, index_pajak, similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K) -> List[QueryEngineTool]:
    ite_engine = index_ite.as_query_engine(similarity_top_k=similarity_top_k+2, similarity_cutoff=0.5, resonse_mode="tree_summarize")
    pajak_engine = index_pajak.as_query_engine(similarity_top_k=similarity_top_k+2, similarity_cutoff=0.5, resonse_mode="tree_summarize")

    ite_engine = LoggingQueryEngine("uu_ite_tool", ite_engine)
    pajak_engine = LoggingQueryEngine("pajak_tool", pajak_engine)

    ite_tool = QueryEngineTool.from_defaults(
        name="uu_ite_tool",
        query_engine=ite_engine,
        description=(
            """Gunakan tool ini untuk pertanyaan terkait UU ITE (UU No. 11/2008 jo. 19/2016): 
            informasi/dokumen elektronik, akses ilegal, penghinaan/pencemaran nama baik daring, 
            penyadapan, transaksi elektronik, tanda tangan elektronik, dll. Kata kunci: ITE, Informasi Elektronik, Transaksi Elektronik."""
        )
    )
    pajak_tool = QueryEngineTool.from_defaults(
        name="pajak_tool",
        query_engine=pajak_engine,
        description=(
            """Gunakan tool ini untuk pertanyaan terkait Pajak (terutama PPh/PPN/KUP): objek pajak, tarif, 
            penghasilan kena pajak, pengecualian, pemotongan/pemungutan, sanksi, ketentuan umum perpajakan, dll. 
            Kata kunci: pajak, PPh, PPN, DJP, tarif."""
        )
    )
    return [ite_tool, pajak_tool]

# =====================
# Pembungkus agar bisa dipanggil seperti fungsi (engine(prompt))
# =====================

class RouterWrapper:
    """Wrap RouterQueryEngine agar bisa dipanggil langsung dan mengembalikan string."""
    def __init__(self, router: RouterQueryEngine, tools: List[QueryEngineTool], selector: PydanticSingleSelector):
        self.router = router
        self.tools = tools
        self.selector = selector  

    def __call__(self, query: str) -> str:
        resp = self.query(query)
        return str(resp)

    def query(self, query: str):
        try:
            sel_result = self.selector.select(self.tools, query)
            # di beberapa versi nama field bisa 'index' atau 'ind'; cover keduanya
            sel_index = getattr(sel_result, "index", getattr(sel_result, "ind", None))
            sel_reason = getattr(sel_result, "reason", None)

            if sel_index is not None and 0 <= sel_index < len(self.tools):
                chosen_tool = self.tools[sel_index]
                logging.info(
                    "[ROUTER] Selected tool: %s | reason: %s",
                    getattr(chosen_tool, "name", "UNKNOWN"),
                    sel_reason if sel_reason else "(no reason provided)"
                )
            else:
                logging.info("[ROUTER] Selector returned invalid index: %s", sel_index)
        except Exception as e:
            logging.exception("[ROUTER] Failed to run selector diagnosis: %s", e)

        # --- Eksekusi router sebenarnya
        return self.router.query(query)

# =====================
# Build RouterQueryEngine
# =====================

def build_router_query_engine(init_models: bool = True) -> RouterWrapper:
    if init_models and init_models is not None:
        init_models()

    if Settings.llm is None:
        raise RuntimeError(
            "Settings.llm belum diinisiasi. Pastikan utils.init_models() meng-assign Settings.llm."
        )

    index_ite, index_pajak = load_indexes()
    tools = build_query_engine_tools(index_ite, index_pajak)

    selector = PydanticSingleSelector.from_defaults()
    router = RouterQueryEngine(
        selector=selector,
        query_engine_tools=tools,
        verbose=True,   
    )
    return RouterWrapper(router, tools=tools, selector=selector)


def display_chat_messages() -> None:
    """Print message history
    @returns None
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# Logo 
st.image("https://eproc.unitedtractors.com/Content/assets/images/logo-ut-baru.png", width=350)

# Title
st.title("genAI Chatbot")

# Side bar content

with st.sidebar:
    st.title("AI Chatbot")
    st.markdown(
        """Knowledge information retrieval"""
    )


mode_descriptions = {
    "OpenAI": [
        "OpenAI LLMs.",
        30,
    ],
    "Vertex AI": [
        "Vertex AI LLMs.",
        15,
    ],
}


# User Configuration Sidebar
# with st.sidebar:
#     mode = st.radio(
#         "**LLM options for answer generation**", options=["OpenAI", "Vertex AI"], index=1
#     )
#     st.info(mode_descriptions[mode][0])

st.divider()

# Chat area
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False

# Display chat messages from history on app rerun
display_chat_messages()

# Greet user
if not st.session_state.greetings:
    with st.chat_message("assistant"):
        intro = "Hey! I am your chat assistant, help you to answer questions about Indonesian Taxation Law and Electronic information and transactions!"
        st.markdown(intro)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.greetings = True

# Example prompts
# example_prompts = [
#     "Apa yang dimaksud dengan informasi elektronik menurut UU ITE?",
#     "Apa pengertian dokumen elektronik dalam UU ITE?",
#     "Bagaimana UU ITE mendefinisikan transaksi elektronik?",
#     "Siapa saja yang termasuk subjek pajak luar negeri dalam UU PPh ini?",
#     "Apa pengertian dari bentuk usaha tetap (BUT) menurut undang-undang ini?",
#     "Kapan kewajiban pajak subjektif orang pribadi dianggap dimulai dan berakhir?"
# ]

button_cols_1 = st.columns(3)
button_cols_2 = st.columns(3)
button_all = button_cols_1 + button_cols_2

button_pressed = ""

# for i in range(6):
#     if button_all[i].button(example_prompts[i]):
#         button_pressed = example_prompts[i]
#         break


if prompt := (st.chat_input("What is the question you want to answer?") or button_pressed):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if prompt != "":
        query = prompt.strip().lower()
        with st.chat_message("assistant"):
            ### will be removed
            # Create a chat completion, will be removed after talking to backend
            query_engine = build_router_query_engine()
            response = query_engine(prompt)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
            st.rerun()
