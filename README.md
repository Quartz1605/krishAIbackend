# krishAIbackend Setup (Windows)

End-to-end setup for the Kerala Farming RAG Chatbot + Plant Disease Detection.

## 1. Prerequisites
- Python 3.13 (preferred; repo pycache shows 3.13)
- Git
- (Optional) Google Gemini API key for better answers
- (Alternative) [Ollama](https://ollama.com/) installed + `ollama pull llama2`
- ~6 GB free disk (models + embeddings + deps)

## 2. Clone & Enter
```powershell
git clone <repo-url>
cd krishAIbackend
```

## 3. Virtual Environment
```powershell
py -3.13 -m venv .venv
./.venv/Scripts/Activate.ps1
```
If PowerShell blocks scripts: run (once, as admin):
```powershell
Set-ExecutionPolicy RemoteSigned
```

## 4. Environment Variables
Create `.env` (or copy from example):
```
GEMINI_API_KEY=your_key_here   # leave blank to use Ollama llama2
```
If no key: ensure Ollama service running and model pulled:
```powershell
ollama pull llama2
```

## 5. Install Dependencies
Both subprojects share nearly identical requirements. Install from one list (choose the larger — disease_detection):
```powershell
pip install --upgrade pip
pip install -r disease_detection/requirements.txt
```
(First install may take several minutes: torch, sentence-transformers, chromadb.)

## 6. Build Vector Database (RAG Knowledge Base)
This uses PDFs in `rag_chatbot/data/kb` and persists embeddings to `rag_chatbot/chroma_db`.
```powershell
python rag_chatbot/main.py --setup-db
```
Rebuild (e.g., after adding PDFs):
```powershell
python rag_chatbot/main.py --setup-db --force-rebuild
```

## 7. Run Chatbot API (Flask)
```powershell
python flask_app.py
```
Test endpoints:
```powershell
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"query":"Best paddy practices in monsoon?"}'
```

## 8. CLI Chat (Optional)
```powershell
python rag_chatbot/main.py --chat
```
Combine build + chat:
```powershell
python rag_chatbot/main.py --setup-db --chat
```

## 9. Plant Disease Detection (Image + RAG Advice)
Image model: `disease_detection/model/plant_disease_cnn_scripted.pt` (CPU inference).
Run analysis (uses disease_detection/chroma_db by default—point it to the one you built if needed):
```powershell
python disease_detection/main.py --image disease_detection/test_images/CornCommonRust1.JPG --db-path ../rag_chatbot/chroma_db
```
Interactive mode:
```powershell
python disease_detection/main.py --image disease_detection/test_images/PotatoHealthy2.JPG --interactive --db-path ../rag_chatbot/chroma_db
```

## 10. Common Issues
| Problem | Fix |
|---------|-----|
| Module import errors | Ensure venv active & deps installed |
| Chatbot says DB missing | Re-run vector build step |
| Gemini errors | Check `GEMINI_API_KEY` and network |
| Ollama fallback fails | Start Ollama app/service; pull model |
| Slow first answer | Embedding model + DB warm-up; improves after |

## 11. Updating Knowledge Base
1. Drop new PDFs into `rag_chatbot/data/kb/`
2. Rebuild: `python rag_chatbot/main.py --setup-db --force-rebuild`
3. Restart Flask server (if running)

## 12. Minimal Smoke Tests
```powershell
# Health
curl http://127.0.0.1:8000/health
# Simple chat
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"query":"Soil conservation methods"}'
# Disease detection
python disease_detection/main.py --image disease_detection/test_images/virus4.JPG --db-path ../rag_chatbot/chroma_db
```

## 13. Folder Roles (Quick)
- `flask_app.py` : REST API wrapper over RAG chatbot
- `rag_chatbot/` : RAG pipeline, vector DB builder, PDFs
- `disease_detection/` : Image classifier + RAG treatment adviser

## 14. Performance Notes
All configured for CPU; set `model_kwargs={'device':'cuda'}` in `build_vector_db.py` & adjust torch install for GPU acceleration.

## 15. Next Steps (Optional Enhancements)
- Unify requirements into root `requirements.txt`
- Add Dockerfile
- Add basic pytest health tests
- Share single chroma_db between both subsystems via config

---
Happy farming 🌱
