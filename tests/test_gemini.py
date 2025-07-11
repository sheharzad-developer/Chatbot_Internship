#!/usr/bin/env python3
"""
Test script for Google Gemini API integration
Demonstrates the two examples provided by the user
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from services.gemini_service import GeminiService

# Load environment variables
load_dotenv()

async def test_basic_example():
    """Test the basic example from user"""
    print("=== Testing Basic Gemini Example ===")
    
    gemini_service = GeminiService()
    
    try:
        response = await gemini_service.generate_simple("Explain how AI works in a few words")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

async def test_advanced_example():
    """Test the advanced example with thinking config disabled"""
    print("\n=== Testing Advanced Gemini Example (Thinking Disabled) ===")
    
    gemini_service = GeminiService()
    
    try:
        response = await gemini_service.generate_content(
            prompt="Explain how AI works in a few words",
            model="gemini-2.5-flash",
            disable_thinking=True  # This disables thinking
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

async def test_connection():
    """Test if the API connection works"""
    print("\n=== Testing API Connection ===")
    
    gemini_service = GeminiService()
    
    if gemini_service.test_connection():
        print("✅ Gemini API connection successful!")
    else:
        print("❌ Gemini API connection failed!")

async def main():
    print("Google Gemini API Test Script")
    print("=" * 40)
    
    # Check if API key is set
    api_key = os.getenv("GOOGLE_AI_GENERATIVE")
    if not api_key:
        print("❌ GOOGLE_AI_GENERATIVE environment variable not set!")
        print("Please set your API key in .env file:")
        print("GOOGLE_AI_GENERATIVE=your_api_key_here")
        return
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    # Run tests
    await test_connection()
    await test_basic_example()
    await test_advanced_example()

if __name__ == "__main__":
    asyncio.run(main()) 