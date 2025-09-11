"""
Example usage of Malayalam Voice Assistant API
"""
import asyncio
import aiohttp
import json
import base64
from pathlib import Path


# API Configuration
API_BASE_URL = "http://localhost:8080"


async def test_text_query():
    """Example: Process Malayalam text query"""
    print("\n" + "="*50)
    print("Testing Text Query Endpoint")
    print("="*50)
    
    # Sample Malayalam agricultural queries
    queries = [
        {
            "text": "എന്റെ നെല്ലിൽ ഇലകൾ മഞ്ഞയാകുന്നു. എന്താണ് പ്രശ്നം?",
            "user_id": "farmer_001"
        },
        {
            "text": "മഴക്കാലത്ത് പച്ചക്കറി കൃഷി എങ്ങനെ ചെയ്യും?",
            "user_id": "farmer_002"
        },
        {
            "text": "ജൈവ വളം എങ്ങനെ തയ്യാറാക്കാം?",
            "user_id": "farmer_003"
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        for query_data in queries:
            print(f"\nQuery: {query_data['text']}")
            
            async with session.post(
                f"{API_BASE_URL}/text-query",
                json=query_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if result.get("success"):
                        print(f"✓ English Translation: {result.get('english_text', 'N/A')}")
                        print(f"✓ Malayalam Response: {result.get('malayalam_response', 'N/A')[:100]}...")
                        
                        # Save audio if available
                        if result.get("audio_response_base64"):
                            audio_data = base64.b64decode(result["audio_response_base64"])
                            filename = f"response_{query_data['user_id']}.mp3"
                            with open(filename, "wb") as f:
                                f.write(audio_data)
                            print(f"✓ Audio saved as: {filename}")
                        
                        print(f"✓ Processing Time: {result.get('processing_time', 0):.2f}s")
                    else:
                        print(f"✗ Error: {result.get('error', 'Unknown error')}")
                else:
                    print(f"✗ HTTP Error: {response.status}")


async def test_voice_query():
    """Example: Process audio file"""
    print("\n" + "="*50)
    print("Testing Voice Query Endpoint")
    print("="*50)
    
    # Create a sample audio file path (you need to provide an actual Malayalam audio file)
    audio_file_path = "sample_malayalam_audio.mp3"
    
    if not Path(audio_file_path).exists():
        print(f"⚠ Audio file not found: {audio_file_path}")
        print("Please provide a Malayalam audio file for testing")
        return
    
    async with aiohttp.ClientSession() as session:
        with open(audio_file_path, 'rb') as audio_file:
            data = aiohttp.FormData()
            data.add_field('audio_file',
                          audio_file,
                          filename='query.mp3',
                          content_type='audio/mpeg')
            data.add_field('user_id', 'farmer_voice_001')
            
            async with session.post(
                f"{API_BASE_URL}/voice-query",
                data=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if result.get("success"):
                        print(f"✓ Transcription: {result.get('malayalam_text', 'N/A')}")
                        print(f"✓ English Translation: {result.get('english_text', 'N/A')}")
                        print(f"✓ Response: {result.get('malayalam_response', 'N/A')[:100]}...")
                        
                        # Save response audio
                        if result.get("audio_response_base64"):
                            audio_data = base64.b64decode(result["audio_response_base64"])
                            filename = "voice_response.mp3"
                            with open(filename, "wb") as f:
                                f.write(audio_data)
                            print(f"✓ Response audio saved as: {filename}")
                        
                        print(f"✓ Processing Time: {result.get('processing_time', 0):.2f}s")
                    else:
                        print(f"✗ Error: {result.get('error', 'Unknown error')}")
                else:
                    error_text = await response.text()
                    print(f"✗ HTTP Error {response.status}: {error_text}")


async def test_health_check():
    """Example: Check service health"""
    print("\n" + "="*50)
    print("Testing Health Check")
    print("="*50)
    
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_BASE_URL}/health") as response:
            if response.status == 200:
                health_data = await response.json()
                print(f"Status: {health_data.get('status', 'unknown')}")
                print(f"Timestamp: {health_data.get('timestamp', 'N/A')}")
                print("Services:")
                for service, status in health_data.get('services', {}).items():
                    status_icon = "✓" if status else "✗"
                    print(f"  {status_icon} {service}: {'healthy' if status else 'unhealthy'}")
            else:
                print(f"✗ Health check failed with status: {response.status}")


async def test_supported_formats():
    """Example: Get supported formats"""
    print("\n" + "="*50)
    print("Getting Supported Formats")
    print("="*50)
    
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_BASE_URL}/supported-formats") as response:
            if response.status == 200:
                formats_data = await response.json()
                print(f"Audio Formats: {', '.join(formats_data.get('audio_formats', []))}")
                print(f"Max File Size: {formats_data.get('max_file_size_mb', 0)} MB")
                print(f"Max Duration: {formats_data.get('max_duration_seconds', 0)} seconds")
                
                languages = formats_data.get('languages', {})
                print(f"Source Language: {languages.get('source', 'N/A')}")
                print(f"Target Language: {languages.get('target', 'N/A')}")
                print(f"TTS Voices: {', '.join(languages.get('tts_voices', []))}")
            else:
                print(f"✗ Failed to get supported formats: {response.status}")


async def main():
    """Run all examples"""
    print("Malayalam Voice Assistant API Examples")
    print("Make sure the API is running at", API_BASE_URL)
    
    # Test health first
    await test_health_check()
    
    # Get supported formats
    await test_supported_formats()
    
    # Test text queries
    await test_text_query()
    
    # Test voice query (if audio file exists)
    await test_voice_query()
    
    print("\n" + "="*50)
    print("Examples completed!")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main())
