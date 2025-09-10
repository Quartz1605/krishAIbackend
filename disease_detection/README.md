# Plant Disease Detection with RAG-based Solutions

This system combines computer vision for plant disease detection with a RAG (Retrieval-Augmented Generation) system to provide comprehensive treatment and prevention advice.

## 🌟 Features

- **Image-based Disease Detection**: Uses a trained CNN model to identify plant diseases from images
- **RAG-powered Solutions**: Generates treatment plans using agricultural knowledge base
- **Interactive Chat**: Follow-up Q&A system for detailed plant care advice  
- **Multiple Output Formats**: Single analysis, interactive mode, prediction-only mode
- **Confidence Scores**: Provides top-3 predictions with confidence levels
- **Source Attribution**: Shows which knowledge sources were used for recommendations

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment activated (`venv/Scripts/Activate.ps1`)
- All dependencies installed (`pip install -r requirements.txt`)
- ChromaDB database (copied from `../rag_chatbot/chroma_db`)
- API keys set in `.env` file

### Basic Usage

**Simple Analysis:**
```bash
python main.py --image virus4.JPG
```

**Interactive Mode:**
```bash
python main.py --image virus4.JPG --interactive
```

**Prediction Only (No RAG):**
```bash
python predict_with_rag.py --image virus4.JPG --no-rag
```

## 📋 System Architecture

```
Image Input → CNN Disease Detection → RAG Solution Generation → Formatted Output
     ↓                    ↓                      ↓                    ↓
  virus4.JPG         Tomato___TYLCV        Treatment Plan        User Interface
```

## 🔧 Main Components

### 1. `main.py` - CLI Entry Point
The primary interface similar to `rag_chatbot/main.py`:

```bash
python main.py --image <path> [options]
```

**Options:**
- `--image`: Path to image file (required)
- `--interactive`: Enable interactive chat mode
- `--model`: LLM model ('gemini' or 'llama2')
- `--db-path`: Path to ChromaDB database
- `--no-sources`: Don't include source documents

### 2. `predict_with_rag.py` - Complete Analysis Pipeline
Full-featured script with multiple modes:

```bash
python predict_with_rag.py --image <path> [options]
```

**Options:**
- `--interactive`: Interactive mode with follow-up chat
- `--no-rag`: Skip RAG, only show disease prediction
- `--no-sources`: Don't include source documents
- `--model`: Choose LLM model

### 3. `src/chatbot/disease_chatbot.py` - RAG System
Core RAG implementation that:
- Loads ChromaDB vector database
- Generates treatment solutions for diseased plants
- Provides prevention advice for healthy plants
- Handles interactive follow-up questions

## 🎯 Disease Detection Classes

The system can detect **38 different plant conditions**:

### Crops Supported:
- **Tomato**: 9 conditions (healthy + 8 diseases)
- **Potato**: 3 conditions (healthy + 2 diseases)  
- **Apple**: 4 conditions (healthy + 3 diseases)
- **Corn**: 4 conditions (healthy + 3 diseases)
- **Grape**: 4 conditions (healthy + 3 diseases)
- **Others**: Cherry, Blueberry, Peach, Pepper, Raspberry, Soybean, Squash, Strawberry, Orange

### Example Diseases:
- Tomato Yellow Leaf Curl Virus
- Potato Late Blight
- Apple Scab
- Corn Northern Leaf Blight
- And many more...

## 📊 Output Format

### Analysis Results Display:
```
🌿 PLANT DISEASE ANALYSIS RESULTS 🌿
======================================================================
📷 Image: virus4.JPG

🔬 Disease Detection:
   Primary Diagnosis: Tomato___Tomato_Yellow_Leaf_Curl_Virus
   Confidence: 100.0%

   Top 3 Predictions:
      1. Tomato___Tomato_Yellow_Leaf_Curl_Virus (100.0%)
      2. Tomato___Late_blight (0.0%)
      3. Tomato___Bacterial_spot (0.0%)

🏥 Analysis:
   Plant Type: Tomato
   Disease: Tomato Yellow Leaf Curl Virus
   Status: DISEASED
   Advice Type: Treatment

💡 Treatment Recommendations:
[Detailed treatment plan generated using RAG...]

📚 Knowledge Sources (5):
   1. crop-advisories.pdf
   2. [... more sources]
======================================================================
```

## 🤖 Interactive Mode

After initial analysis, you can ask follow-up questions:

```
🤖 Ask me follow-up questions about Tomato care, or type 'quit' to exit.

🧑‍🌾 You: What organic treatments are available?
🤖 Assistant: [RAG-generated response about organic treatments]

🧑‍🌾 You: How long does treatment take?
🤖 Assistant: [RAG-generated response about treatment duration]
```

## 🗂️ File Structure

```
disease_detection/
├── main.py                  # CLI entry point
├── predict_with_rag.py      # Complete analysis pipeline  
├── predict.py              # Original prediction script
├── src/
│   └── chatbot/
│       ├── __init__.py
│       └── disease_chatbot.py  # RAG system implementation
├── model/
│   └── plant_disease_cnn_scripted.pt  # Trained CNN model
├── chroma_db/              # Vector database (copied from rag_chatbot)
├── requirements.txt        # Dependencies
├── .env                   # API keys
└── README.md             # This file
```

## 🔑 Environment Variables

Ensure your `.env` file contains:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
# Add other API keys as needed
```

## 📦 Dependencies

Key packages (see `requirements.txt` for complete list):
- `torch` & `torchvision` - Deep learning framework
- `langchain` - RAG framework
- `langchain-google-genai` - Gemini integration
- `langchain-chroma` - Vector database
- `langchain-huggingface` - Embeddings
- `pillow` - Image processing
- `python-dotenv` - Environment variables

## 🎨 Usage Examples

### 1. Quick Disease Analysis
```bash
python main.py --image tomato_leaf.jpg
```

### 2. Interactive Consultation
```bash  
python main.py --image diseased_plant.jpg --interactive
```

### 3. Prediction Only (Fast)
```bash
python predict_with_rag.py --image plant.jpg --no-rag
```

### 4. Custom Model Selection
```bash
python main.py --image plant.jpg --model llama2
```

## 🌱 Treatment Categories

The system provides different types of advice based on plant health:

### For Diseased Plants:
- **Immediate treatment steps**
- **Chemical/organic treatments** 
- **Cultural practices**
- **Application timing**
- **Safety precautions**
- **Disease progression info**
- **Monitoring techniques**

### For Healthy Plants:
- **Prevention strategies**
- **Cultural best practices**
- **Resistant varieties**
- **Soil management**
- **Preventive spraying**
- **Environmental conditions**
- **Seasonal care**
- **IPM approaches**

## 🔧 Troubleshooting

### Common Issues:

**1. "Vector database not found"**
- Ensure `chroma_db` folder exists in disease_detection
- Copy from `../rag_chatbot/chroma_db` if missing

**2. "GEMINI_API_KEY not found"**
- Check your `.env` file
- Ensure API key is valid

**3. Module import errors**
- Activate virtual environment: `venv/Scripts/Activate.ps1`
- Install dependencies: `pip install -r requirements.txt`

**4. Image not found**
- Check image path
- Ensure image format is supported (JPG, PNG, etc.)

## 🤝 Integration

This system can be integrated with:
- **Web applications** (Flask/FastAPI)
- **Mobile apps** (via REST API)
- **Agricultural monitoring systems**
- **IoT devices**
- **Farm management software**

## 📈 Performance

- **Disease Detection**: ~1-2 seconds per image
- **RAG Solution Generation**: ~3-5 seconds
- **Interactive Responses**: ~2-3 seconds per query
- **Supported Image Formats**: JPG, PNG, JPEG, BMP
- **Max Image Size**: Automatically resized to 256x256

## 🔮 Future Enhancements

- [ ] Support for more crop types
- [ ] Real-time image capture
- [ ] Batch processing
- [ ] API endpoints  
- [ ] Mobile app interface
- [ ] Multi-language support
- [ ] Weather integration
- [ ] GPS-based regional advice

---

**Happy Farming! 🌾** 

For questions or issues, consult the original `rag_chatbot` documentation or modify the RAG knowledge base to add more agricultural information.
