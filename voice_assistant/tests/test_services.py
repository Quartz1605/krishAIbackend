"""
Test suite for Malayalam Voice Assistant services
"""
import pytest
import asyncio
from pathlib import Path
import tempfile
import os

# Import services
from voice_assistant.services import (
    VoiceProcessingService,
    TranslationService,
    RAGChatbotService,
    MalayalamVoiceAssistant
)

@pytest.fixture
def voice_service():
    """Create voice processing service instance"""
    return VoiceProcessingService()

@pytest.fixture
def translation_service():
    """Create translation service instance"""
    return TranslationService()

@pytest.fixture
def rag_service():
    """Create RAG chatbot service instance"""
    return RAGChatbotService()

@pytest.fixture
def assistant():
    """Create Malayalam voice assistant instance"""
    return MalayalamVoiceAssistant()


class TestTranslationService:
    """Test translation service"""
    
    @pytest.mark.asyncio
    async def test_translate_malayalam_to_english(self, translation_service):
        """Test Malayalam to English translation"""
        malayalam_text = "എന്റെ പേര് കൃഷ്ണൻ"
        english_text = await translation_service.translate(
            malayalam_text, 
            source_lang="ml", 
            target_lang="en"
        )
        assert english_text is not None
        assert len(english_text) > 0
        print(f"Malayalam: {malayalam_text}")
        print(f"English: {english_text}")
    
    @pytest.mark.asyncio
    async def test_translate_english_to_malayalam(self, translation_service):
        """Test English to Malayalam translation"""
        english_text = "My crops are affected by disease"
        malayalam_text = await translation_service.translate(
            english_text,
            source_lang="en",
            target_lang="ml"
        )
        assert malayalam_text is not None
        assert len(malayalam_text) > 0
        print(f"English: {english_text}")
        print(f"Malayalam: {malayalam_text}")
    
    @pytest.mark.asyncio
    async def test_language_detection(self, translation_service):
        """Test language detection"""
        malayalam_text = "എന്റെ വിളകൾ രോഗബാധിതമാണ്"
        detected_lang = await translation_service.detect_language(malayalam_text)
        assert detected_lang == "ml"
        
        english_text = "My crops are healthy"
        detected_lang = await translation_service.detect_language(english_text)
        assert detected_lang == "en"
    
    @pytest.mark.asyncio
    async def test_batch_translation(self, translation_service):
        """Test batch translation"""
        texts = [
            "എന്റെ നെല്ല് വയലിൽ പ്രശ്നമുണ്ട്",
            "മഴക്കാലത്ത് എന്താണ് ചെയ്യേണ്ടത്?",
            "വളം എപ്പോൾ ഇടണം?"
        ]
        translations = await translation_service.translate_batch(
            texts,
            source_lang="ml",
            target_lang="en"
        )
        assert len(translations) == len(texts)
        for orig, trans in zip(texts, translations):
            print(f"{orig} -> {trans}")


class TestVoiceProcessingService:
    """Test voice processing service"""
    
    @pytest.mark.asyncio
    async def test_text_to_speech(self, voice_service):
        """Test TTS generation"""
        text = "നമസ്കാരം, ഞാൻ നിങ്ങളുടെ കൃഷി സഹായിയാണ്"
        audio_path = await voice_service.text_to_speech(text, language="ml")
        assert audio_path is not None
        assert Path(audio_path).exists()
        assert Path(audio_path).stat().st_size > 0
        print(f"Generated audio: {audio_path}")
        # Cleanup
        voice_service.cleanup()


class TestRAGChatbotService:
    """Test RAG chatbot service"""
    
    @pytest.mark.asyncio
    async def test_fallback_responses(self, rag_service):
        """Test fallback response generation"""
        queries = [
            "My rice plants have yellow leaves",
            "How to apply fertilizer?",
            "When should I water my crops?"
        ]
        
        for query in queries:
            response = rag_service._get_fallback_response(query)
            assert response is not None
            assert len(response) > 0
            print(f"Query: {query}")
            print(f"Fallback: {response[:100]}...\n")
    
    @pytest.mark.asyncio
    async def test_health_check(self, rag_service):
        """Test health check (may fail if RAG not running)"""
        is_healthy = await rag_service.health_check()
        print(f"RAG service health: {is_healthy}")
        # Don't assert as RAG may not be running


class TestMalayalamVoiceAssistant:
    """Test complete voice assistant pipeline"""
    
    @pytest.mark.asyncio
    async def test_text_query_pipeline(self, assistant):
        """Test text query processing pipeline"""
        malayalam_query = "എന്റെ നെല്ലിൽ മഞ്ഞ ഇലകൾ ഉണ്ട്. എന്താണ് കാരണം?"
        
        result = await assistant.process_text_query(malayalam_query)
        
        assert "malayalam_text" in result
        assert "english_text" in result
        assert "english_response" in result
        assert "malayalam_response" in result
        
        print(f"Original: {result['malayalam_text']}")
        print(f"English: {result['english_text']}")
        print(f"Response (EN): {result['english_response'][:100]}...")
        print(f"Response (ML): {result['malayalam_response'][:100]}...")
        
        if result['success']:
            assert result['response_audio_b64'] is not None
            print(f"Audio generated: {len(result['response_audio_b64'])} bytes (base64)")


# Integration test example
@pytest.mark.asyncio
async def test_end_to_end_integration():
    """Test complete integration"""
    assistant = MalayalamVoiceAssistant()
    
    # Test agricultural queries
    queries = [
        "എന്റെ പച്ചക്കറി തോട്ടത്തിൽ കീടങ്ങൾ ഉണ്ട്",
        "മഴക്കാലത്ത് നെല്ല് കൃഷി എങ്ങനെ?",
        "ജൈവ വളം എങ്ങനെ ഉണ്ടാക്കാം?"
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"Testing: {query}")
        print('='*50)
        
        result = await assistant.process_text_query(query)
        
        if result['success']:
            print(f"✓ Translation: {result['english_text']}")
            print(f"✓ Response: {result['malayalam_response'][:100]}...")
            print(f"✓ Audio: {'Generated' if result['response_audio_b64'] else 'Failed'}")
        else:
            print(f"✗ Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    # Run basic tests
    print("Running Malayalam Voice Assistant Tests\n")
    
    # Test translation
    print("Testing Translation Service...")
    translation_service = TranslationService()
    
    async def run_translation_test():
        ml_text = "എന്റെ കൃഷിയിൽ പ്രശ്നമുണ്ട്"
        en_text = await translation_service.translate(ml_text, "ml", "en")
        print(f"ML->EN: {ml_text} -> {en_text}")
        
        en_text = "I need help with farming"
        ml_text = await translation_service.translate(en_text, "en", "ml")
        print(f"EN->ML: {en_text} -> {ml_text}")
    
    asyncio.run(run_translation_test())
    
    # Test TTS
    print("\nTesting Text-to-Speech...")
    voice_service = VoiceProcessingService()
    
    async def run_tts_test():
        text = "നമസ്കാരം കർഷകരേ"
        audio_path = await voice_service.text_to_speech(text, "ml")
        if audio_path:
            print(f"TTS Success: {audio_path}")
        else:
            print("TTS Failed")
    
    asyncio.run(run_tts_test())
    
    print("\nTests completed!")
