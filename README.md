# VectorMind

My teams project for the National AI Hackathon

# 🚀 RAG Backend for Document Q&A

This repository contains the backend for a Retrieval-Augmented Generation (RAG) system designed to answer questions based on your uploaded documents. It leverages **Pinecone** for vector storage, **Hugging Face Transformers.js** for embeddings, and **OpenAI's GPT-4o** for generating answers.

## ✨ Features

- **PDF Document Upload & Processing**: Upload PDF files, which are parsed, semantically chunked, and then converted into vector embeddings.
- **Vector Database Integration (Pinecone)**: Stores document chunks and their embeddings in a Pinecone index for efficient similarity search.
- **Semantic Search**: Uses query embeddings to find the most relevant document chunks in your knowledge base.
- **Context-Aware AI Generation**: Leverages retrieved document chunks as context for OpenAI's GPT-4o to generate accurate and relevant answers.
- **Scalable & Modular Architecture**: Built with Express.js, featuring separate routes and controllers for better organization.
- **Environment Variable Configuration**: Securely manages API keys and sensitive information.

## 🛠️ Technologies Used

- **Node.js**: JavaScript runtime.
- **Express.js**: Web framework for Node.js.
- **Pinecone**: Vector database for similarity search.
- **`@xenova/transformers`**: For generating embeddings using pre-trained models (e.g., `bge-m3`) client-side.
- **OpenAI**: For the `gpt-4o` large language model to generate answers.
- **`pdf-parse`**: To extract text from PDF documents.
- **`langchain/text_splitter`**: For intelligent semantic chunking of text.
- **`cors`**: Middleware to enable Cross-Origin Resource Sharing.
- **`morgan`**: HTTP request logger middleware (for development).
- **`dotenv`**: To load environment variables.
- **`uuid`**: For generating unique IDs.
- **`multer`**: (Implied, but not explicitly shown in `index.js` for file upload handling) A Node.js middleware for handling `multipart/form-data`.

## ⚙️ Installation & Setup

1.  **Clone the repository:**

    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install dependencies:**

    ```bash
    npm install
    ```

3.  **Configure Environment Variables:**
    Create a `.env` file in the root directory of your project, based on `.env.example`.

    ```env
    PORT=5000
    MONGO_URI=your_mongodb_connection_string # If using MongoDB
    PINECONE_API_KEY=your_pinecone_api_key
    OPENAI_API_KEY=your_openai_api_key
    # Add other environment variables as needed (e.g., for user authentication secrets)
    ```

    - **`PORT`**: The port your server will run on (default 5000).
    - **`MONGO_URI`**: Your MongoDB connection string (if `db.js` connects to MongoDB).
    - **`PINECONE_API_KEY`**: Your API key from [Pinecone](https://www.pinecone.io/).
    - **`OPENAI_API_KEY`**: Your API key from [OpenAI](https://platform.openai.com/).

4.  **Database Setup (if applicable):**
    Ensure your `db.js` file correctly connects to your chosen database (e.g., MongoDB). The provided `index.js` expects a `connectDB()` function from `./db.js`.

5.  **Pinecone Index (Optional, handled automatically):**
    The `uploadController.js` includes `ensurePineconeIndexExists()` which will automatically create the Pinecone index named `georag1` with a dimension of `1024` (for `bge-m3` embeddings) and `cosine` similarity metric in the `us-east-1` region if it doesn't already exist.

## ▶️ Running the Backend

To start the development server:

```bash
npm run dev
# Or using nodemon (if installed globally for auto-restarts):
# nodemon index.js
```
