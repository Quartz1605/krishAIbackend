"""
Translation Service for Malayalam-English bidirectional translation
Uses free translation APIs with fallback mechanisms
"""
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List
from deep_translator import GoogleTranslator
import json
import hashlib

from ..utils.logger import logger
from ..utils.helpers import async_retry, cache_manager
from ..config.settings import settings


class TranslationService:
    """
    Service for handling Malayalam-English translations
    Supports multiple translation providers with caching
    """
    
    def __init__(self):
        """Initialize translation service"""
        self.google_translator = GoogleTranslator(source='auto', target='en')
        self.cache_enabled = settings.ENABLE_CACHE
        
        # LibreTranslate API (free, self-hosted option)
        self.libre_translate_url = "https://libretranslate.de/translate"
        
        # MyMemory Translation API (free tier: 5000 chars/day)
        self.mymemory_url = "https://api.mymemory.translated.net/get"
        
        # Language codes mapping
        self.language_map = {
            "ml": "ml",  # Malayalam
            "en": "en",  # English
            "malayalam": "ml",
            "english": "en"
        }
    
    def _get_cache_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generate cache key for translation"""
        return cache_manager._generate_key(
            f"translation:{source_lang}:{target_lang}",
            text
        )
    
    @async_retry(max_attempts=3, delay=1.0)
    async def translate_google(self, text: str, source_lang: str = "ml", target_lang: str = "en") -> Optional[str]:
        """
        Translate text using Google Translate (Free, unofficial)
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text or None if failed
        """
        try:
            # Use deep-translator library (Google Translate API)
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                translator.translate,
                text
            )
            
            if result:
                logger.info(f"Google translation successful: {result[:50]}...")
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Google translation error: {str(e)}")
            return None
    
    async def translate_libre(self, text: str, source_lang: str = "ml", target_lang: str = "en") -> Optional[str]:
        """
        Translate text using LibreTranslate (Free, open-source)
        Note: Malayalam support may be limited
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text or None if failed
        """
        try:
            payload = {
                "q": text,
                "source": source_lang,
                "target": target_lang,
                "format": "text"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.libre_translate_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        translated_text = result.get("translatedText")
                        if translated_text:
                            logger.info(f"LibreTranslate successful: {translated_text[:50]}...")
                            return translated_text
                    else:
                        error_text = await response.text()
                        logger.warning(f"LibreTranslate failed: {error_text}")
            
            return None
            
        except Exception as e:
            logger.error(f"LibreTranslate error: {str(e)}")
            return None
    
    async def translate_mymemory(self, text: str, source_lang: str = "ml", target_lang: str = "en") -> Optional[str]:
        """
        Translate text using MyMemory API (Free tier: 5000 chars/day)
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text or None if failed
        """
        try:
            # Prepare language pair
            lang_pair = f"{source_lang}|{target_lang}"
            
            params = {
                "q": text,
                "langpair": lang_pair
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.mymemory_url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("responseStatus") == 200:
                            translated_text = result.get("responseData", {}).get("translatedText")
                            if translated_text:
                                logger.info(f"MyMemory translation successful: {translated_text[:50]}...")
                                return translated_text
                    else:
                        error_text = await response.text()
                        logger.warning(f"MyMemory translation failed: {error_text}")
            
            return None
            
        except Exception as e:
            logger.error(f"MyMemory translation error: {str(e)}")
            return None
    
    async def translate_huggingface(self, text: str, source_lang: str = "ml", target_lang: str = "en") -> Optional[str]:
        """
        Translate text using Hugging Face models (Free with API key)
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text or None if failed
        """
        if not settings.HUGGINGFACE_API_KEY:
            return None
        
        try:
            # Use Helsinki-NLP models for translation
            if source_lang == "ml" and target_lang == "en":
                model_id = "Helsinki-NLP/opus-mt-ml-en"
            elif source_lang == "en" and target_lang == "ml":
                model_id = "Helsinki-NLP/opus-mt-en-ml"
            else:
                logger.warning(f"No Hugging Face model for {source_lang} to {target_lang}")
                return None
            
            url = f"https://api-inference.huggingface.co/models/{model_id}"
            headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}
            payload = {"inputs": text}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, list) and len(result) > 0:
                            translated_text = result[0].get("translation_text")
                            if translated_text:
                                logger.info(f"Hugging Face translation successful: {translated_text[:50]}...")
                                return translated_text
                    else:
                        error_text = await response.text()
                        logger.warning(f"Hugging Face translation failed: {error_text}")
            
            return None
            
        except Exception as e:
            logger.error(f"Hugging Face translation error: {str(e)}")
            return None
    
    async def translate(self, text: str, source_lang: str = "ml", target_lang: str = "en") -> str:
        """
        Translate text with caching and fallback mechanisms
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text or original text if all methods fail
        """
        # Normalize language codes
        source_lang = self.language_map.get(source_lang.lower(), source_lang)
        target_lang = self.language_map.get(target_lang.lower(), target_lang)
        
        # Check cache first
        if self.cache_enabled:
            cache_key = self._get_cache_key(text, source_lang, target_lang)
            cached_translation = cache_manager.get(cache_key)
            if cached_translation:
                logger.info(f"Translation cache hit for: {text[:30]}...")
                return cached_translation
        
        logger.info(f"Translating from {source_lang} to {target_lang}: {text[:50]}...")
        
        # Try translation methods in order of preference
        translation_methods = [
            ("Hugging Face", self.translate_huggingface),
            ("Google", self.translate_google),
            ("MyMemory", self.translate_mymemory),
            ("LibreTranslate", self.translate_libre)
        ]
        
        for method_name, method in translation_methods:
            try:
                translated = await method(text, source_lang, target_lang)
                if translated:
                    # Cache successful translation
                    if self.cache_enabled:
                        cache_manager.set(cache_key, translated)
                    
                    logger.info(f"Translation successful using {method_name}")
                    return translated
            except Exception as e:
                logger.warning(f"{method_name} translation failed: {str(e)}")
                continue
        
        logger.error("All translation methods failed, returning original text")
        return text
    
    async def translate_batch(self, texts: List[str], source_lang: str = "ml", target_lang: str = "en") -> List[str]:
        """
        Translate multiple texts in batch
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of translated texts
        """
        tasks = [self.translate(text, source_lang, target_lang) for text in texts]
        results = await asyncio.gather(*tasks)
        return results
    
    async def detect_language(self, text: str) -> str:
        """
        Detect language of given text
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (ml, en, etc.)
        """
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.google_translator.detect,
                text
            )
            
            if result and result.lang:
                logger.info(f"Detected language: {result.lang} (confidence: {result.confidence})")
                return result.lang
            
            # Default to Malayalam if detection fails
            return "ml"
            
        except Exception as e:
            logger.error(f"Language detection error: {str(e)}")
            return "ml"
    
    def preprocess_malayalam_text(self, text: str) -> str:
        """
        Preprocess Malayalam text for better translation
        
        Args:
            text: Malayalam text
            
        Returns:
            Preprocessed text
        """
        # Remove extra whitespaces
        text = " ".join(text.split())
        
        # Handle common Malayalam punctuation
        text = text.replace("।", ".")
        
        # Preserve agricultural terms by marking them
        agricultural_terms = [
            "നെല്ല്", "കൃഷി", "വിള", "കീടനാശിനി", "വളം",
            "മണ്ണ്", "ജലസേചനം", "കാലാവസ്ഥ", "വിത്ത്"
        ]
        
        # This is a placeholder for more sophisticated preprocessing
        # In production, you might want to use NLP libraries for Malayalam
        
        return text
    
    def postprocess_translation(self, text: str, target_lang: str = "en") -> str:
        """
        Postprocess translated text to improve quality
        
        Args:
            text: Translated text
            target_lang: Target language code
            
        Returns:
            Postprocessed text
        """
        # Capitalize first letter of sentences
        sentences = text.split(". ")
        sentences = [s.capitalize() for s in sentences]
        text = ". ".join(sentences)
        
        # Fix common translation artifacts
        text = text.replace(" ,", ",")
        text = text.replace(" .", ".")
        text = text.replace(" ?", "?")
        text = text.replace(" !", "!")
        
        return text
