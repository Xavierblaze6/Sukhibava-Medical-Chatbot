"""RAG pipeline assembly for the Sukhibava chatbot."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.schema import Document
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI


SYSTEM_PROMPT = (
    "You are Sukhibava, a helpful and empathetic medical assistant. "
    "Answer questions based on the provided medical context. "
    "Always remind users to consult a doctor for personal medical advice. "
    "If you don't know the answer, say so clearly."
)


@dataclass
class SukhibavaRAGChain:
    """Small conversational RAG runner that returns both answer text and retrieved docs."""

    retriever: Any
    llm: ChatOpenAI
    memory: ConversationBufferWindowMemory
    prompt: ChatPromptTemplate

    @staticmethod
    def _format_context(documents: List[Document]) -> str:
        return "\n\n".join(document.page_content for document in documents)

    async def ainvoke(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Retrieve relevant chunks, answer with the LLM, and persist chat memory."""
        question = inputs["question"]
        chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
        source_documents = await self.retriever.ainvoke(question)
        prompt_value = self.prompt.invoke(
            {
                "chat_history": chat_history,
                "context": self._format_context(source_documents),
                "question": question,
            }
        )
        response = await self.llm.ainvoke(prompt_value.to_messages(), config=config)
        answer = response.content if isinstance(response.content, str) else str(response.content)
        await self.memory.asave_context({"question": question}, {"answer": answer})
        return {
            "answer": answer,
            "source_documents": source_documents,
        }

    async def acall(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """LangChain-style async call compatibility wrapper."""
        return await self.ainvoke(inputs)


def build_rag_chain(vectorstore):
    """Create the conversational RAG runner with memory and source document exposure."""
    load_dotenv()

    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )

    return SukhibavaRAGChain(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.3, streaming=True),
        memory=memory,
        prompt=prompt,
    )
