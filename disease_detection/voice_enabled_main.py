#!/usr/bin/env python3
"""
Voice-Enabled Disease Detection System

This script integrates voice input capabilities with the disease detection
and RAG systems, allowing Malayalam farmers to interact using voice.
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_assistant import VoiceAssistant
from predict_with_rag import PlantDiseaseAnalyzer

def get_audio_files_list(audio_source: str):
    """
    Get list of audio files from source (file or directory)
    
    Args:
        audio_source: Path to audio file or directory containing audio files
        
    Returns:
        List of audio file paths
    """
    audio_path = Path(audio_source)
    
    if audio_path.is_file():
        # Single audio file
        return [str(audio_path)]
    elif audio_path.is_dir():
        # Directory of audio files
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(audio_path.glob(f"*{ext}"))
            audio_files.extend(audio_path.glob(f"*{ext.upper()}"))
        
        # Sort files alphabetically for consistent processing order
        return sorted([str(f) for f in audio_files])
    else:
        raise FileNotFoundError(f"Audio source not found: {audio_source}")

async def process_audio_file(voice_assistant, audio_file_path: str):
    """
    Process a single audio file and return the voice result
    
    Args:
        voice_assistant: VoiceAssistant instance
        audio_file_path: Path to audio file
        
    Returns:
        Voice processing result
    """
    print(f"\n📁 Processing audio file: {Path(audio_file_path).name}")
    return await voice_assistant.process_voice_input(audio_file_path)

async def voice_disease_detection(image_path: str = None, audio_source: str = None):
    """Voice-enabled disease detection with image analysis"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize components
    voice_assistant = VoiceAssistant()
    disease_analyzer = PlantDiseaseAnalyzer()
    
    try:
        # Get audio files list if audio_source is provided
        audio_files = []
        if audio_source:
            try:
                audio_files = get_audio_files_list(audio_source)
                print(f"📁 Found {len(audio_files)} audio files for processing")
                for i, file in enumerate(audio_files, 1):
                    print(f"   {i}. {Path(file).name}")
                print()
            except FileNotFoundError as e:
                print(f"❌ {e}")
                return
        
        if image_path:
            # Image analysis with voice follow-up
            logger.info(f"Analyzing image: {image_path}")
            
            # Perform disease detection
            result = disease_analyzer.analyze_image(image_path, include_sources=True)
            disease_analyzer.print_analysis_results(result)
            
            # Voice interaction for follow-up questions
            if audio_files:
                # Process pre-recorded audio files
                print(f"\n🎤 Processing {len(audio_files)} pre-recorded audio files for follow-up questions:")
                
                for audio_file in audio_files:
                    try:
                        voice_result = await process_audio_file(voice_assistant, audio_file)
                        
                        if not voice_result["success"]:
                            print(f"❌ Voice processing failed: {voice_result['error']}")
                            continue
                        
                        print(f"🗣️  Malayalam: {voice_result['malayalam_text']}")
                        print(f"🔄  English: {voice_result['english_text']}")
                        
                        # Get RAG response for follow-up question
                        context = f"Previous analysis: {result.get('prediction', 'Unknown')}. Question: {voice_result['english_text']}"
                        
                        # Use the existing RAG system from disease analyzer - get relevant documents and ask the LLM
                        docs = disease_analyzer.chatbot.retriever.get_relevant_documents(voice_result['english_text'])
                        context_text = "\n\n".join([doc.page_content for doc in docs])
                        
                        # Create a simple prompt for general agricultural questions
                        prompt = f"""You are an agricultural expert. Answer the following question based on the provided context.

Previous Analysis: {result.get('prediction', 'Unknown')}

Context:
{context_text}

Question: {voice_result['english_text']}

Answer:"""
                        response = disease_analyzer.chatbot.llm.invoke(prompt).content
                        print(f"🤖  Assistant: {response}")
                        
                        # Translate response back to Malayalam
                        malayalam_response = await voice_assistant.translator.translate_to_malayalam(response)
                        print(f"🔄  Malayalam Response: {malayalam_response}")
                        
                        # Generate voice response
                        try:
                            audio_file = await voice_assistant.generate_voice_response(malayalam_response, "malayalam")
                            print(f"🔊  Audio response saved: {audio_file}")
                            
                            # Optionally play the audio
                            print("🎵  Playing audio response...")
                            await voice_assistant.tts_service.play_speech(audio_file)
                            
                        except Exception as e:
                            logger.warning(f"Voice response generation failed: {e}")
                        
                        print("-" * 50)  # Separator between audio files
                        
                    except Exception as e:
                        print(f"❌ Error processing {Path(audio_file).name}: {e}")
                        continue
                
                print("✅ Finished processing all audio files.")
                
            else:
                # Live microphone interaction (original behavior)
                print(f"\n🎤 Ask follow-up questions in Malayalam about the plant (or press Ctrl+C to exit):")
                
                while True:
                    try:
                        print("\n📢 Speak your question now...")
                        voice_result = await voice_assistant.process_voice_input()
                        
                        if not voice_result["success"]:
                            print(f"❌ Voice processing failed: {voice_result['error']}")
                            continue
                        
                        print(f"🗣️  Malayalam: {voice_result['malayalam_text']}")
                        print(f"🔄  English: {voice_result['english_text']}")
                        
                        # Get RAG response for follow-up question
                        context = f"Previous analysis: {result.get('prediction', 'Unknown')}. Question: {voice_result['english_text']}"
                        
                        # Use the existing RAG system from disease analyzer - get relevant documents and ask the LLM
                        docs = disease_analyzer.chatbot.retriever.get_relevant_documents(voice_result['english_text'])
                        context_text = "\n\n".join([doc.page_content for doc in docs])
                        
                        # Create a simple prompt for general agricultural questions
                        prompt = f"""You are an agricultural expert. Answer the following question based on the provided context.

Previous Analysis: {result.get('prediction', 'Unknown')}

Context:
{context_text}

Question: {voice_result['english_text']}

Answer:"""
                        response = disease_analyzer.chatbot.llm.invoke(prompt).content
                        print(f"🤖  Assistant: {response}")
                        
                        # Translate response back to Malayalam
                        malayalam_response = await voice_assistant.translator.translate_to_malayalam(response)
                        print(f"🔄  Malayalam Response: {malayalam_response}")
                        
                        # Generate voice response
                        try:
                            audio_file = await voice_assistant.generate_voice_response(malayalam_response, "malayalam")
                            print(f"🔊  Audio response saved: {audio_file}")
                            
                            # Optionally play the audio
                            print("🎵  Playing audio response...")
                            await voice_assistant.tts_service.play_speech(audio_file)
                            
                        except Exception as e:
                            logger.warning(f"Voice response generation failed: {e}")
                    
                    except KeyboardInterrupt:
                        print("\n👋 Voice interaction ended.")
                        break
        
        else:
            # Voice-only RAG mode (no image)
            if audio_files:
                # Process pre-recorded audio files
                print(f"🎤 Processing {len(audio_files)} pre-recorded audio files for RAG queries:")
                
                for audio_file in audio_files:
                    try:
                        voice_result = await process_audio_file(voice_assistant, audio_file)
                        
                        if not voice_result["success"]:
                            print(f"❌ Voice processing failed: {voice_result['error']}")
                            continue
                        
                        print(f"🗣️  Malayalam: {voice_result['malayalam_text']}")
                        print(f"🔄  English: {voice_result['english_text']}")
                        
                        # Get RAG response
                        docs = disease_analyzer.chatbot.retriever.get_relevant_documents(voice_result['english_text'])
                        context_text = "\n\n".join([doc.page_content for doc in docs])
                        
                        # Create a simple prompt for general agricultural questions
                        prompt = f"""You are an agricultural expert. Answer the following question based on the provided context.

Context:
{context_text}

Question: {voice_result['english_text']}

Answer:"""
                        response = disease_analyzer.chatbot.llm.invoke(prompt).content
                        print(f"🤖  Assistant: {response}")
                        
                        # Translate response back to Malayalam
                        malayalam_response = await voice_assistant.translator.translate_to_malayalam(response)
                        print(f"🔄  Malayalam Response: {malayalam_response}")
                        
                        # Generate voice response
                        try:
                            audio_output_file = await voice_assistant.generate_voice_response(malayalam_response, "malayalam")
                            print(f"🔊  Audio response saved: {audio_output_file}")
                            
                            # Play the audio response
                            print("🎵  Playing audio response...")
                            await voice_assistant.tts_service.play_speech(audio_output_file)
                            
                        except Exception as e:
                            logger.warning(f"Voice response generation failed: {e}")
                        
                        print("-" * 50)  # Separator between audio files
                        
                    except Exception as e:
                        print(f"❌ Error processing {Path(audio_file).name}: {e}")
                        continue
                
                print("✅ Finished processing all audio files.")
                
            else:
                # Live microphone interaction (original behavior)
                print("🎤 Voice-only mode: Ask questions about plant diseases in Malayalam")
                print("Press Ctrl+C to exit\n")
                
                while True:
                    try:
                        print("📢 Speak your question now...")
                        voice_result = await voice_assistant.process_voice_input()
                        
                        if not voice_result["success"]:
                            print(f"❌ Voice processing failed: {voice_result['error']}")
                            continue
                        
                        print(f"🗣️  Malayalam: {voice_result['malayalam_text']}")
                        print(f"🔄  English: {voice_result['english_text']}")
                        
                        # Get RAG response
                        docs = disease_analyzer.chatbot.retriever.get_relevant_documents(voice_result['english_text'])
                        context_text = "\n\n".join([doc.page_content for doc in docs])
                        
                        # Create a simple prompt for general agricultural questions
                        prompt = f"""You are an agricultural expert. Answer the following question based on the provided context.

Context:
{context_text}

Question: {voice_result['english_text']}

Answer:"""
                        response = disease_analyzer.chatbot.llm.invoke(prompt).content
                        print(f"🤖  Assistant: {response}")
                        
                        # Translate response back to Malayalam
                        malayalam_response = await voice_assistant.translator.translate_to_malayalam(response)
                        print(f"🔄  Malayalam Response: {malayalam_response}")
                        
                        # Generate voice response
                        try:
                            audio_file = await voice_assistant.generate_voice_response(malayalam_response, "malayalam")
                            print(f"🔊  Audio response saved: {audio_file}")
                            
                            # Play the audio response
                            print("🎵  Playing audio response...")
                            await voice_assistant.tts_service.play_speech(audio_file)
                            
                        except Exception as e:
                            logger.warning(f"Voice response generation failed: {e}")
                    
                    except KeyboardInterrupt:
                        print("\n👋 Voice chat ended.")
                        break

    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Error: {e}")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Voice-Enabled Plant Disease Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Voice-only mode (ask questions about plants)
  python voice_enabled_main.py
  
  # Voice-only mode with pre-recorded audio files
  python voice_enabled_main.py --audio-file path/to/audio/directory/
  python voice_enabled_main.py --audio-file path/to/single/audio.wav
  
  # Image analysis with voice follow-up
  python voice_enabled_main.py --image test_images/virus4.JPG
  
  # Image analysis with pre-recorded audio follow-up
  python voice_enabled_main.py --image test_images/virus4.JPG --audio-file audio_questions/
  
  # Image analysis only (no voice interaction)
  python voice_enabled_main.py --image test_images/virus4.JPG --no-voice
        """
    )

    parser.add_argument("--image", type=str, help="Path to plant image for analysis")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice interaction")
    parser.add_argument("--audio-file", type=str, help="Use pre-recorded audio file or directory of audio files instead of live recording")

    args = parser.parse_args()

    # Validate image path if provided
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"❌ Image not found: {args.image}")
            sys.exit(1)
    
    # Handle no-voice mode
    if args.no_voice and args.image:
        # Run traditional disease detection without voice
        from predict_with_rag import PlantDiseaseAnalyzer
        analyzer = PlantDiseaseAnalyzer()
        result = analyzer.analyze_image(args.image, include_sources=True)
        analyzer.print_analysis_results(result)
        return
    
    # Run voice-enabled system
    try:
        asyncio.run(voice_disease_detection(args.image, args.audio_file))
    except KeyboardInterrupt:
        print("\n\n👋 Thank you for using Voice-Enabled Disease Detection!")
    except Exception as e:
        print(f"\n❌ Application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
