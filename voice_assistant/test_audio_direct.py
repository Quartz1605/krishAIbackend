"""
Direct test of Malayalam audio file without running the server
"""
import asyncio
import sys
from pathlib import Path
import base64

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def test_malayalam_audio(audio_file_path: str):
    """
    Test Malayalam audio file directly
    
    Args:
        audio_file_path: Path to your Malayalam audio file
    """
    print(f"\n{'='*60}")
    print("TESTING MALAYALAM AUDIO FILE")
    print(f"{'='*60}")
    print(f"Audio file: {audio_file_path}\n")
    
    # Check if file exists
    if not Path(audio_file_path).exists():
        print(f"❌ Error: File not found: {audio_file_path}")
        return
    
    try:
        from voice_assistant.services.assistant import MalayalamVoiceAssistant
        
        # Initialize assistant
        print("Initializing voice assistant...")
        assistant = MalayalamVoiceAssistant()
        
        # Process the audio file
        print("Processing audio file...")
        result = await assistant.process_audio_query(audio_file_path)
        
        print(f"\n{'='*60}")
        print("RESULTS:")
        print(f"{'='*60}")
        
        if result["success"]:
            print(f"✅ Success!\n")
            
            # Show transcription
            if result.get("malayalam_text"):
                print(f"1. Malayalam Transcription:")
                print(f"   {result['malayalam_text']}\n")
            
            # Show English translation
            if result.get("english_text"):
                print(f"2. English Translation:")
                print(f"   {result['english_text']}\n")
            
            # Show English response
            if result.get("english_response"):
                print(f"3. English Response:")
                print(f"   {result['english_response'][:200]}...\n")
            
            # Show Malayalam response
            if result.get("malayalam_response"):
                print(f"4. Malayalam Response:")
                print(f"   {result['malayalam_response'][:200]}...\n")
            
            # Save audio response
            if result.get("response_audio_b64"):
                output_file = "response_malayalam.mp3"
                audio_data = base64.b64decode(result["response_audio_b64"])
                with open(output_file, "wb") as f:
                    f.write(audio_data)
                print(f"5. Audio Response:")
                print(f"   ✅ Saved to: {output_file}")
                print(f"   File size: {len(audio_data)} bytes\n")
            
            # Show timing
            if result.get("timings"):
                print(f"6. Processing Times:")
                for step, time in result["timings"].items():
                    print(f"   {step}: {time:.2f}s")
            
        else:
            print(f"❌ Processing failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        # Use provided audio file
        audio_file = sys.argv[1]
    else:
        # Ask for audio file
        print("\nMALAYALAM AUDIO FILE TESTER")
        print("-" * 30)
        audio_file = input("Enter path to Malayalam audio file: ").strip()
        
        # Remove quotes if present
        audio_file = audio_file.strip('"').strip("'")
    
    # Run the test
    asyncio.run(test_malayalam_audio(audio_file))

if __name__ == "__main__":
    main()
