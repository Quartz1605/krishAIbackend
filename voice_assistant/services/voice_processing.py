"""
Voice Processing Service for Malayalam ASR and TTS
Uses free APIs: Hugging Face for ASR and gTTS for TTS
"""
import os
import asyncio
import tempfile
from typing import Optional, Dict, Any
from pathlib import Path
import aiohttp
import aiofiles
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
import base64
import json

from ..utils.logger import logger
from ..utils.helpers import AudioUtils, async_retry
from ..config.settings import settings


class VoiceProcessingService:
    """
    Service for handling voice processing operations
    Supports multiple ASR/TTS providers with fallback mechanisms
    """
    
    def __init__(self):
        """Initialize voice processing service"""
        self.recognizer = sr.Recognizer()
        self.audio_utils = AudioUtils()
        self.temp_dir = settings.TEMP_DIR
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        
        # Hugging Face API setup (free tier)
        self.hf_api_url = "https://api-inference.huggingface.co/models/"
        self.hf_headers = {}
        if settings.HUGGINGFACE_API_KEY:
            self.hf_headers["Authorization"] = f"Bearer {settings.HUGGINGFACE_API_KEY}"
    
    @async_retry(max_attempts=3, delay=1.0)
    async def speech_to_text_huggingface(self, audio_path: str, language: str = "ml") -> Optional[str]:
        """
        Convert speech to text using Hugging Face API (Free)
        
        Args:
            audio_path: Path to audio file
            language: Language code (ml for Malayalam)
            
        Returns:
            Transcribed text or None if failed
        """
        try:
            # Use multilingual ASR model from Hugging Face
            model_id = "openai/whisper-small"  # Free model that supports Malayalam
            url = f"{self.hf_api_url}{model_id}"
            
            async with aiofiles.open(audio_path, 'rb') as f:
                audio_data = await f.read()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=self.hf_headers,
                    data=audio_data,
                    params={"language": language}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        text = result.get("text", "")
                        logger.info(f"Hugging Face ASR successful: {text[:50]}...")
                        return text
                    else:
                        error_text = await response.text()
                        logger.error(f"Hugging Face ASR failed: {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"Hugging Face ASR error: {str(e)}")
            return None
    
    async def speech_to_text_google(self, audio_path: str, language: str = "ml-IN") -> Optional[str]:
        """
        Convert speech to text using Google Speech Recognition (Free)
        
        Args:
            audio_path: Path to audio file
            language: Language code (ml-IN for Malayalam)
            
        Returns:
            Transcribed text or None if failed
        """
        try:
            # Convert audio to WAV format if needed
            audio = AudioSegment.from_file(audio_path)
            wav_path = audio_path.replace(Path(audio_path).suffix, ".wav")
            audio.export(wav_path, format="wav")
            
            # Use speech_recognition library
            with sr.AudioFile(wav_path) as source:
                audio_data = self.recognizer.record(source)
                
            # Try Google's free speech recognition
            text = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.recognizer.recognize_google(audio_data, language=language)
            )
            
            logger.info(f"Google ASR successful: {text[:50]}...")
            
            # Cleanup temporary WAV file
            if wav_path != audio_path:
                os.remove(wav_path)
            
            return text
            
        except sr.UnknownValueError:
            logger.warning("Google ASR could not understand the audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Google ASR request error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Google ASR error: {str(e)}")
            return None
    
    async def speech_to_text(self, audio_path: str, language: str = "ml") -> Optional[str]:
        """
        Convert speech to text with fallback mechanisms
        
        Args:
            audio_path: Path to audio file
            language: Language code
            
        Returns:
            Transcribed text or None if all methods fail
        """
        logger.info(f"Starting speech-to-text for {audio_path}")
        
        # Try Hugging Face first (if API key is available)
        if settings.HUGGINGFACE_API_KEY:
            text = await self.speech_to_text_huggingface(audio_path, language)
            if text:
                return text
        
        # Fallback to Google's free API
        language_code = "ml-IN" if language == "ml" else f"{language}-IN"
        text = await self.speech_to_text_google(audio_path, language_code)
        if text:
            return text
        
        logger.error("All ASR methods failed")
        return None
    
    async def text_to_speech_gtts(self, text: str, language: str = "ml", output_path: Optional[str] = None) -> Optional[str]:
        """
        Convert text to speech using gTTS (Free)
        
        Args:
            text: Text to convert
            language: Language code
            output_path: Optional output path for audio file
            
        Returns:
            Path to generated audio file or None if failed
        """
        try:
            if not output_path:
                timestamp = Path(tempfile.mktemp()).stem
                output_path = os.path.join(self.temp_dir, f"tts_{timestamp}.mp3")
            
            # Generate speech using gTTS
            tts = gTTS(text=text, lang=language, slow=False)
            
            # Save to file asynchronously
            await asyncio.get_event_loop().run_in_executor(
                None,
                tts.save,
                output_path
            )
            
            logger.info(f"gTTS successful, saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"gTTS error: {str(e)}")
            return None
    
    async def text_to_speech_edge_tts(self, text: str, language: str = "ml", output_path: Optional[str] = None) -> Optional[str]:
        """
        Convert text to speech using Edge TTS (Free, high quality)
        
        Args:
            text: Text to convert
            language: Language code
            output_path: Optional output path for audio file
            
        Returns:
            Path to generated audio file or None if failed
        """
        try:
            import edge_tts
            
            if not output_path:
                timestamp = Path(tempfile.mktemp()).stem
                output_path = os.path.join(self.temp_dir, f"tts_{timestamp}.mp3")
            
            # Map language to Edge TTS voice
            voice_map = {
                "ml": "ml-IN-SobhanaNeural",  # Malayalam female voice
                "en": "en-IN-NeerjaNeural",   # English (India) female voice
            }
            
            voice = voice_map.get(language, "ml-IN-SobhanaNeural")
            
            # Generate speech
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
            
            logger.info(f"Edge TTS successful, saved to {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("edge-tts not installed, falling back to gTTS")
            return None
        except Exception as e:
            logger.error(f"Edge TTS error: {str(e)}")
            return None
    
    async def text_to_speech(self, text: str, language: str = "ml", output_path: Optional[str] = None) -> Optional[str]:
        """
        Convert text to speech with fallback mechanisms
        
        Args:
            text: Text to convert
            language: Language code
            output_path: Optional output path for audio file
            
        Returns:
            Path to generated audio file or None if all methods fail
        """
        logger.info(f"Starting text-to-speech for text: {text[:50]}...")
        
        # Try Edge TTS first (highest quality free option)
        audio_path = await self.text_to_speech_edge_tts(text, language, output_path)
        if audio_path:
            return audio_path
        
        # Fallback to gTTS
        audio_path = await self.text_to_speech_gtts(text, language, output_path)
        if audio_path:
            return audio_path
        
        logger.error("All TTS methods failed")
        return None
    
    async def convert_audio_format(self, input_path: str, output_format: str = "wav") -> str:
        """
        Convert audio file to different format
        
        Args:
            input_path: Path to input audio file
            output_format: Target format (wav, mp3, etc.)
            
        Returns:
            Path to converted audio file
        """
        try:
            audio = AudioSegment.from_file(input_path)
            output_path = input_path.replace(Path(input_path).suffix, f".{output_format}")
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                audio.export,
                output_path,
                output_format
            )
            
            logger.info(f"Converted {input_path} to {output_format}")
            return output_path
            
        except Exception as e:
            logger.error(f"Audio conversion error: {str(e)}")
            raise
    
    async def process_voice_query(self, audio_path: str, source_lang: str = "ml") -> Dict[str, Any]:
        """
        Process complete voice query: ASR -> Translation -> Response -> TTS
        
        Args:
            audio_path: Path to input audio file
            source_lang: Source language code
            
        Returns:
            Dictionary with transcription and processing status
        """
        result = {
            "success": False,
            "transcription": None,
            "error": None
        }
        
        try:
            # Perform speech-to-text
            transcription = await self.speech_to_text(audio_path, source_lang)
            
            if transcription:
                result["success"] = True
                result["transcription"] = transcription
            else:
                result["error"] = "Failed to transcribe audio"
                
        except Exception as e:
            logger.error(f"Voice query processing error: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            for file_path in Path(self.temp_dir).glob("tts_*.mp3"):
                try:
                    file_path.unlink()
                except:
                    pass
            logger.info("Cleaned up temporary audio files")
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
