# 🌾 Kerala Farming RAG Chatbot 🌾

A Retrieval-Augmented Generation (RAG) based chatbot designed specifically for Kerala farmers. This system uses your PDF knowledge base to provide accurate, context-aware answers to farming questions.

## ✨ Features

- **RAG-based responses**: Answers based on your PDF knowledge base
- **ChromaDB vector storage**: Efficient similarity search and retrieval
- **Multiple LLM options**: Support for OpenAI GPT models and local Ollama models
- **Kerala-specific**: Tailored for Kerala's farming conditions and practices
- **Simple CLI interface**: Easy to use command-line interface
- **Source citations**: Shows which documents were used for answers

## 📁 Project Structure

```
krishi/
├── main.py                     # Main application script
├── requirements.txt            # Python dependencies
├── .env.example               # Environment configuration template
├── README.md                  # This file
├── data/
│   └── kb/                    # Knowledge base PDFs
│       ├── 92-pest-and-diseases.pdf
│       ├── Farmguide-2023.pdf
│       ├── Farmguide-2024.pdf
│       ├── crop-advisories.pdf
│       ├── package_of_practices_2016.pdf
│       ├── soil-management.pdf
│       ├── soil_conservation_measure.txt
│       └── welfare-schemes.pdf
└── src/
    ├── chatbot/
    │   └── chatbot.py         # Main chatbot logic
    ├── embeddings/
    │   └── build_vector_db.py # Vector database builder
    ├── loaders/
    │   └── load_kb.py         # PDF and document loaders
    └── utils/
        └── __init__.py
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file and configure it:

```bash
copy .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your_actual_api_key_here
```

> **Alternative**: You can use Ollama for local models (no API key required). Install Ollama from https://ollama.ai/

### 3. Check Prerequisites

```bash
python main.py --check
```

### 4. Build Vector Database and Start Chatbot

```bash
python main.py --setup-db --chat
```

That's it! Your Kerala farming chatbot is ready to use! 🎉

## 🔧 Detailed Usage

### Building the Vector Database

```bash
# Build database from PDFs in data/kb/
python main.py --setup-db

# Force rebuild even if database exists
python main.py --setup-db --force-rebuild

# Use custom knowledge base directory
python main.py --setup-db --kb-dir path/to/your/pdfs
```

### Running the Chatbot

```bash
# Basic chatbot with OpenAI GPT-3.5
python main.py --chat

# Use GPT-4
python main.py --chat --model gpt-4

# Use local Ollama model
python main.py --chat --llm-type ollama --model llama2

# Retrieve more context documents
python main.py --chat --top-k 10
```

### Advanced Options

```bash
# Complete workflow with custom settings
python main.py --setup-db --chat --model gpt-4 --temperature 0.2 --top-k 7

# Check system status
python main.py --check
```

## 🤖 Using the Chatbot

Once started, you can ask questions like:

- "What are the best rice varieties for Kerala?"
- "How to control pest in coconut trees?"
- "When is the best time to plant pepper in Kerala?"
- "What government schemes are available for farmers?"
- "How to prepare soil for vegetable cultivation?"

Type `quit`, `exit`, or `bye` to end the conversation.

## ⚙️ Configuration Options

### Environment Variables (.env)

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required for OpenAI models |
| `LLM_TYPE` | LLM type ("openai" or "ollama") | openai |
| `LLM_MODEL` | Model name | gpt-3.5-turbo |
| `LLM_TEMPERATURE` | Response randomness (0.0-1.0) | 0.1 |
| `TOP_K_DOCUMENTS` | Documents to retrieve | 5 |
| `CHUNK_SIZE` | Text chunk size | 1000 |
| `CHUNK_OVERLAP` | Text chunk overlap | 200 |

### Command Line Options

```bash
python main.py --help
```

## 🔍 Troubleshooting

### Common Issues

1. **"Vector database not found"**
   - Run `python main.py --setup-db` first

2. **"OPENAI_API_KEY not found"**
   - Add your API key to the `.env` file
   - Or use Ollama: `--llm-type ollama --model llama2`

3. **"No PDF files found"**
   - Make sure PDFs are in the `data/kb/` directory
   - Check file extensions (.pdf)

4. **"Error loading vector store"**
   - Delete the `chroma_db` folder and rebuild: `--setup-db --force-rebuild`

### Performance Tips

- **For faster responses**: Use smaller models like `gpt-3.5-turbo`
- **For better accuracy**: Use `gpt-4` or increase `--top-k`
- **For privacy**: Use local Ollama models
- **For large documents**: Increase chunk size in `.env`

## 📚 Adding New Documents

1. Copy new PDF files to `data/kb/` directory
2. Rebuild the vector database:
   ```bash
   python main.py --setup-db --force-rebuild
   ```

## 🛠️ Development

### Project Architecture

- **Document Loading**: `src/loaders/load_kb.py` handles PDF text extraction
- **Vector Database**: `src/embeddings/build_vector_db.py` manages ChromaDB
- **Chatbot**: `src/chatbot/chatbot.py` implements the RAG pipeline
- **Main App**: `main.py` ties everything together

### Adding New Features

The codebase is modular and easy to extend:

- Add new document types in `load_kb.py`
- Modify the prompt template in `chatbot.py`
- Add new LLM providers in the chatbot initialization

## 📖 How It Works

1. **Document Processing**: PDFs are loaded and split into chunks
2. **Embedding**: Text chunks are converted to vector embeddings
3. **Storage**: Embeddings are stored in ChromaDB for fast retrieval
4. **Query Processing**: User questions are embedded and matched against stored chunks
5. **Answer Generation**: Relevant chunks are provided as context to the LLM
6. **Response**: The LLM generates contextual answers based on the retrieved information

## 🤝 Contributing

This is a project for Kerala farmers! Feel free to:
- Add more agricultural documents
- Improve the prompt templates
- Add support for regional languages
- Enhance the user interface

## 📝 License

This project is created for educational and agricultural support purposes.

---

**Happy farming! 🌱** If you have questions or need help, the chatbot is here to assist with your Kerala farming needs!
