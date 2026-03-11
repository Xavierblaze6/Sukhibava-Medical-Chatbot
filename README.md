# Sukhibava

Sukhibava is a medical Q&A chatbot built with Retrieval Augmented Generation (RAG), Chainlit, LangChain, FAISS, and OpenAI.

## What RAG Means

RAG stands for Retrieval Augmented Generation. Instead of asking the language model to answer from its general training alone, the application first retrieves relevant passages from a curated medical knowledge base. Those passages are then sent to the model as context so the response is grounded in the local dataset.

In Sukhibava, the flow is:

1. Medical reference text is split into chunks.
2. Each chunk is converted into an embedding using `sentence-transformers/all-MiniLM-L6-v2`.
3. The embeddings are stored in a FAISS index for fast similarity search.
4. For each user question, the app retrieves the top 4 relevant chunks.
5. `gpt-4o-mini` answers using that retrieved context plus the recent conversation history.
6. The UI shows the top retrieved chunks in a dedicated Sources panel after every answer.

This design improves relevance, helps the chatbot stay focused on the provided medical content, and supports conversational follow-up questions.

## Project Structure

```text
sukhibava/
├── app.py
├── rag/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── embeddings.py
│   └── vectorstore.py
├── data/
│   └── medical_data.txt
├── ingest.py
├── requirements.txt
└── README.md
```

## How Sukhibava Works

- `ingest.py` reads the medical knowledge base and builds a FAISS vector store in `faiss_index/`.
- `rag/vectorstore.py` handles splitting text, embedding chunks, and saving or loading the index.
- `rag/pipeline.py` builds a custom conversational RAG runner with:
  - `ChatOpenAI(model="gpt-4o-mini", temperature=0.3)`
  - a FAISS retriever with `k=4`
  - `ConversationBufferWindowMemory(k=5)` for short conversational memory
  - a custom medical system prompt
- `app.py` provides the Chainlit chat interface, streams generated tokens back to the user, and shows the retrieved source chunks after every answer.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Add your OpenAI API key to `.env`:

   ```env
   OPENAI_API_KEY=your_key_here
   ```

3. Build the vector store:

   ```bash
   python ingest.py
   ```

4. Run the chatbot:

   ```bash
   chainlit run app.py
   ```

## Notes

- The chatbot keeps the last 5 turns in memory to support conversational follow-up.
- Responses are grounded in the local medical knowledge base through retrieval.
- The app shows up to 3 retrieved chunks after every response so users can see which sources were used.
- The assistant reminds users to consult a doctor for personal medical advice.
- `.env` and `faiss_index/` are excluded through `.gitignore`.