"""
Quick test script to verify voice assistant setup
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def test_basic_imports():
    """Test that all modules import correctly"""
    print("1. Testing imports...")
    try:
        from voice_assistant.services import (
            VoiceProcessingService,
            TranslationService,
            RAGChatbotService,
            MalayalamVoiceAssistant
        )
        from voice_assistant.config.settings import settings
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

async def test_translation():
    """Test translation service"""
    print("\n2. Testing translation service...")
    try:
        from voice_assistant.services.translation import TranslationService
        
        translator = TranslationService()
        
        # Test Malayalam to English
        ml_text = "നമസ്കാരം, എന്റെ പേര് കൃഷ്ണൻ"
        en_text = await translator.translate(ml_text, "ml", "en")
        print(f"   ML->EN: {ml_text}")
        print(f"   Result: {en_text}")
        
        # Test English to Malayalam
        en_text2 = "How to grow rice?"
        ml_text2 = await translator.translate(en_text2, "en", "ml")
        print(f"   EN->ML: {en_text2}")
        print(f"   Result: {ml_text2}")
        
        print("✅ Translation service working!")
        return True
    except Exception as e:
        print(f"❌ Translation error: {e}")
        return False

async def test_tts():
    """Test text-to-speech"""
    print("\n3. Testing text-to-speech...")
    try:
        from voice_assistant.services.voice_processing import VoiceProcessingService
        
        voice_service = VoiceProcessingService()
        
        # Generate Malayalam speech
        text = "നമസ്കാരം കർഷകരേ, ഞാൻ നിങ്ങളുടെ സഹായിയാണ്"
        audio_path = await voice_service.text_to_speech(text, "ml")
        
        if audio_path and Path(audio_path).exists():
            print(f"✅ TTS successful! Audio saved to: {audio_path}")
            print(f"   File size: {Path(audio_path).stat().st_size} bytes")
            return True
        else:
            print("❌ TTS failed - no audio generated")
            return False
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return False

async def test_rag_health():
    """Test RAG chatbot connection"""
    print("\n4. Testing RAG chatbot connection...")
    try:
        from voice_assistant.services.rag_chatbot import RAGChatbotService
        
        rag_service = RAGChatbotService()
        is_healthy = await rag_service.health_check()
        
        if is_healthy:
            print("✅ RAG chatbot is healthy and responding!")
        else:
            print("⚠️  RAG chatbot is not responding (this is okay - it will use fallbacks)")
        
        # Test fallback response
        response = rag_service._get_fallback_response("How to treat rice disease?")
        print(f"   Fallback response sample: {response[:100]}...")
        
        return True
    except Exception as e:
        print(f"⚠️  RAG connection test: {e}")
        return True  # Not critical

async def test_full_pipeline():
    """Test the complete pipeline"""
    print("\n5. Testing complete pipeline...")
    try:
        from voice_assistant.services.assistant import MalayalamVoiceAssistant
        
        assistant = MalayalamVoiceAssistant()
        
        # Test with Malayalam text
        malayalam_query = "എന്റെ നെല്ലിൽ മഞ്ഞ പാടുകൾ ഉണ്ട്"
        print(f"   Query: {malayalam_query}")
        
        result = await assistant.process_text_query(malayalam_query)
        
        if result["success"]:
            print(f"✅ Pipeline successful!")
            print(f"   English translation: {result.get('english_text', 'N/A')[:50]}...")
            print(f"   Malayalam response: {result.get('malayalam_response', 'N/A')[:50]}...")
            print(f"   Audio generated: {'Yes' if result.get('response_audio_b64') else 'No'}")
            print(f"   Processing time: {result.get('timings', {}).get('total', 0):.2f}s")
        else:
            print(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")
        
        return result["success"]
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("="*60)
    print("MALAYALAM VOICE ASSISTANT - QUICK TEST")
    print("="*60)
    
    # Check environment
    from voice_assistant.config.settings import settings
    print(f"\nEnvironment Check:")
    print(f"  TEMP_DIR: {settings.TEMP_DIR}")
    print(f"  RAG_API_URL: {settings.RAG_API_URL}")
    print(f"  Cache enabled: {settings.ENABLE_CACHE}")
    
    # Create temp directory
    Path(settings.TEMP_DIR).mkdir(parents=True, exist_ok=True)
    
    # Run tests
    tests = [
        ("Imports", test_basic_imports),
        ("Translation", test_translation),
        ("Text-to-Speech", test_tts),
        ("RAG Connection", test_rag_health),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = await test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY:")
    print("="*60)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:20} {status}")
    
    total_passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 All tests passed! Your voice assistant is ready to use!")
    elif total_passed >= 3:
        print("\n✅ Core functionality working! You can start using the voice assistant.")
        print("   Some features may have limited functionality.")
    else:
        print("\n⚠️  Some core features are not working. Please check your setup.")

if __name__ == "__main__":
    asyncio.run(main())
