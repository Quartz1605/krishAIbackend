"""
Malayalam Voice Assistant orchestrator service
"""
import asyncio
import base64
import time
from typing import Dict, Any, Optional
from pathlib import Path

from ..utils.logger import logger
from ..utils.helpers import AudioUtils, format_duration
from ..config.settings import settings
from .voice_processing import VoiceProcessingService
from .translation import TranslationService
from .rag_chatbot import RAGChatbotService


class MalayalamVoiceAssistant:
    """
    Orchestrates ASR, translation, RAG processing, and TTS for Malayalam voice assistant
    """
    
    def __init__(self):
        self.voice_service = VoiceProcessingService()
        self.translation_service = TranslationService()
        self.rag_service = RAGChatbotService()
        self.audio_utils = AudioUtils()
    
    async def process_audio_query(self, audio_path: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a Malayalam audio query end-to-end
        
        Args:
            audio_path: Path to input audio file
            user_id: Optional user identifier
            
        Returns:
            Dictionary with detailed processing information
        """
        start_time = time.time()
        
        response_data: Dict[str, Any] = {
            "success": False,
            "steps": [],
            "timings": {},
            "original_audio_path": audio_path,
            "original_audio_b64": None,
            "malayalam_text": None,
            "english_text": None,
            "english_response": None,
            "malayalam_response": None,
            "response_audio_b64": None,
            "sources": [],
            "metadata": {}
        }
        
        # Add original audio as base64 for response (as per requirements)
        try:
            response_data["original_audio_b64"] = self.audio_utils.encode_audio_to_base64(audio_path)
        except Exception as e:
            logger.warning(f"Failed to encode original audio: {str(e)}")
        
        # Step 1: ASR
        t0 = time.time()
        asr_result = await self.voice_service.process_voice_query(audio_path, source_lang="ml")
        t1 = time.time()
        response_data["timings"]["asr"] = t1 - t0
        response_data["steps"].append("asr")
        
        if not asr_result["success"] or not asr_result["transcription"]:
            response_data["error"] = asr_result.get("error", "ASR failed")
            response_data["timings"]["total"] = time.time() - start_time
            return response_data
        
        malayalam_text = asr_result["transcription"]
        response_data["malayalam_text"] = malayalam_text
        
        # Step 2: Translation ML->EN
        t2 = time.time()
        english_text = await self.translation_service.translate(malayalam_text, source_lang="ml", target_lang="en")
        t3 = time.time()
        response_data["timings"]["translation_ml_en"] = t3 - t2
        response_data["steps"].append("translate_ml_en")
        response_data["english_text"] = english_text
        
        # Step 3: RAG chatbot
        t4 = time.time()
        english_response = await self.rag_service.query_with_fallback(english_text, user_id=user_id)
        t5 = time.time()
        response_data["timings"]["rag"] = t5 - t4
        response_data["steps"].append("rag")
        
        english_response = self.rag_service.format_agricultural_response(english_response)
        response_data["english_response"] = english_response
        
        # Step 4: Translation EN->ML
        t6 = time.time()
        malayalam_response = await self.translation_service.translate(english_response, source_lang="en", target_lang="ml")
        t7 = time.time()
        response_data["timings"]["translation_en_ml"] = t7 - t6
        response_data["steps"].append("translate_en_ml")
        response_data["malayalam_response"] = malayalam_response
        
        # Step 5: TTS
        t8 = time.time()
        tts_path = await self.voice_service.text_to_speech(malayalam_response, language="ml")
        t9 = time.time()
        response_data["timings"]["tts"] = t9 - t8
        response_data["steps"].append("tts")
        
        if not tts_path or not Path(tts_path).exists():
            response_data["error"] = "Failed to generate TTS audio"
            response_data["timings"]["total"] = time.time() - start_time
            return response_data
        
        # Encode TTS audio to base64
        try:
            response_data["response_audio_b64"] = self.audio_utils.encode_audio_to_base64(tts_path)
        except Exception as e:
            logger.error(f"Failed to encode TTS audio: {str(e)}")
            response_data["error"] = "Failed to encode TTS audio"
            response_data["timings"]["total"] = time.time() - start_time
            return response_data
        
        # Success
        response_data["success"] = True
        response_data["timings"]["total"] = time.time() - start_time
        return response_data
    
    async def process_text_query(self, malayalam_text: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a Malayalam text query end-to-end
        
        Args:
            malayalam_text: Malayalam input text
            user_id: Optional user identifier
            
        Returns:
            Dictionary with detailed processing information
        """
        start_time = time.time()
        
        response_data: Dict[str, Any] = {
            "success": False,
            "steps": [],
            "timings": {},
            "malayalam_text": malayalam_text,
            "english_text": None,
            "english_response": None,
            "malayalam_response": None,
            "response_audio_b64": None,
            "sources": [],
            "metadata": {}
        }
        
        # Step 1: Translation ML->EN
        t2 = time.time()
        english_text = await self.translation_service.translate(malayalam_text, source_lang="ml", target_lang="en")
        t3 = time.time()
        response_data["timings"]["translation_ml_en"] = t3 - t2
        response_data["steps"].append("translate_ml_en")
        response_data["english_text"] = english_text
        
        # Step 2: RAG chatbot
        t4 = time.time()
        english_response = await self.rag_service.query_with_fallback(english_text, user_id=user_id)
        t5 = time.time()
        response_data["timings"]["rag"] = t5 - t4
        response_data["steps"].append("rag")
        
        english_response = self.rag_service.format_agricultural_response(english_response)
        response_data["english_response"] = english_response
        
        # Step 3: Translation EN->ML
        t6 = time.time()
        malayalam_response = await self.translation_service.translate(english_response, source_lang="en", target_lang="ml")
        t7 = time.time()
        response_data["timings"]["translation_en_ml"] = t7 - t6
        response_data["steps"].append("translate_en_ml")
        response_data["malayalam_response"] = malayalam_response
        
        # Step 4: TTS
        t8 = time.time()
        tts_path = await self.voice_service.text_to_speech(malayalam_response, language="ml")
        t9 = time.time()
        response_data["timings"]["tts"] = t9 - t8
        response_data["steps"].append("tts")
        
        if not tts_path or not Path(tts_path).exists():
            response_data["error"] = "Failed to generate TTS audio"
            response_data["timings"]["total"] = time.time() - start_time
            return response_data
        
        # Encode TTS audio to base64
        try:
            response_data["response_audio_b64"] = self.audio_utils.encode_audio_to_base64(tts_path)
        except Exception as e:
            logger.error(f"Failed to encode TTS audio: {str(e)}")
            response_data["error"] = "Failed to encode TTS audio"
            response_data["timings"]["total"] = time.time() - start_time
            return response_data
        
        # Success
        response_data["success"] = True
        response_data["timings"]["total"] = time.time() - start_time
        return response_data

