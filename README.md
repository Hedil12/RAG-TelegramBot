# 🤖 Local RAG Bot: CS & Python Expert

An asynchronous Telegram bot that uses a **Retrieval-Augmented Generation (RAG)** pipeline to answer technical questions about Computer Science and Python development. 

This project was built to explore the limits of **Small Language Models (SLMs)** running on consumer-grade CPU hardware, utilizing local vector embeddings to ground the model in specific technical datasets.

---

## 🛠️ Technical Architecture

*   **Model:** 0.5B Parameter LLM (e.g., Qwen2 / TinyLlama) 
*   **Inference:** CPU-based (optimized via `torch` and `accelerate`)
*   **Vector Store:** `FAISS` for high-speed similarity search
*   **Data Orchestration:** `LangChain` & `LangChain-Community`
*   **Interface:** `aiogram` / `python-telegram-bot` (Async)

---

## 📚 Knowledge Base (The Data)
To reduce hallucinations in a small model, the RAG pipeline is fed with specialized datasets:
1.  **Python Documentation:** Core syntax, standard libraries, and best practices.
2.  **CS Fundamentals:** Data structures, algorithms, and system design principles.

The bot doesn't just "guess"—it retrieves the most relevant snippets from these datasets before generating a response.

---

## 🚀 Key Features

*   **Resource Efficient:** Designed to run on standard CPUs without requiring a dedicated GPU (though GPU support is toggleable).
*   **Hybrid Framework:** Uses `nest_asyncio` to bridge the gap between heavy AI processing and the Telegram event loop.
*   **Scalable Parameters:** The architecture is model-agnostic; it can be easily updated to 3B, 7B, or 14B models by changing a single config line.
*   **Local & Private:** No data leaves the local environment; all embeddings and generation happen on-device.

---

## 🧠 Technical Challenges & Learning Journey

Building this RAG pipeline in mid-2024 provided several key insights into the constraints of local AI:

*   **The "Small Model" Challenge:** Working with a 0.5B parameter model required high-quality data augmentation. I learned that the retrieval step (RAG) is the "brain" of the operation when the generator is lightweight.
*   **Hardware Optimization:** I specifically targeted CPU-based inference to make the bot accessible. This involved tuning `torch` and `accelerate` settings to keep response times under a few seconds.
*   **Async Complexity:** Integrating the Telegram API's polling with heavy AI inference taught me how to manage non-blocking code and handle the common "loop already running" errors using `nest_asyncio`.
*   **Data Specificity:** By feeding the model Python and CS-specific datasets, I observed a significant drop in hallucinations compared to using the base model alone.
