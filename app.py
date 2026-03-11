"""Chainlit entry point for the Sukhibava medical RAG chatbot."""

from contextvars import ContextVar
from importlib import import_module

import chainlit as cl
from dotenv import load_dotenv
from langchain_core.callbacks.base import AsyncCallbackHandler

from rag.pipeline import build_rag_chain
from rag.vectorstore import load_vectorstore


chainlit_context_module = import_module("chainlit.context")
chainlit_step_module = import_module("chainlit.step")
chainlit_message_module = import_module("chainlit.message")
chainlit_openai_module = import_module("chainlit.openai")

safe_local_steps = ContextVar("local_steps", default=[])
chainlit_context_module.local_steps = safe_local_steps
chainlit_step_module.local_steps = safe_local_steps
chainlit_message_module.local_steps = safe_local_steps
chainlit_openai_module.local_steps = safe_local_steps
local_steps = safe_local_steps

load_dotenv()
local_steps.set([])


class ChainlitStreamHandler(AsyncCallbackHandler):
    """Stream new LLM tokens into the active Chainlit message."""

    def __init__(self, message: cl.Message) -> None:
        self.message = message

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        await self.message.stream_token(token)


def create_chain():
    """Load the FAISS store and construct a conversational RAG chain."""
    vectorstore = load_vectorstore()
    return build_rag_chain(vectorstore)


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize the vector store and conversational RAG chain for a session."""
    local_steps.set([])
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    chain = build_rag_chain(vectorstore)
    cl.user_session.set("chain", chain)
    cl.user_session.set("retriever", retriever)

    await cl.Message(
        content=(
            "👋 Hi! I'm Sukhibava, your medical assistant.\n"
            "Ask me anything about symptoms, diseases, or treatments.\n"
            "I'll also show you which sources I used to answer each question.\n"
            "⚠️ Always consult a doctor for personal medical advice."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    retriever = cl.user_session.get("retriever")

    if chain is None or retriever is None:
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        chain = build_rag_chain(vectorstore)
        cl.user_session.set("chain", chain)
        cl.user_session.set("retriever", retriever)

    # Get relevant source chunks
    docs = retriever.get_relevant_documents(message.content)

    # Run the chain
    response = await chain.acall({"question": message.content})
    answer = response["answer"]

    # Build sources text
    sources_text = "📚 **Sources used:**\n"
    for i, doc in enumerate(docs[:3], 1):
        snippet = doc.page_content[:150].replace("\n", " ")
        sources_text += f"• Chunk {i}: \"{snippet}...\"\n"

    # Send answer
    await cl.Message(content=answer).send()

    # Send sources as separate message
    await cl.Message(
        content=sources_text,
        elements=[],
        author="Sources"
    ).send()
