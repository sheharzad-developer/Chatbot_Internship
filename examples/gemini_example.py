#!/usr/bin/env python3
"""
Standalone example demonstrating Google Gemini API integration
This recreates the exact examples provided by the user
"""

import os
import asyncio
from google import genai
from google.genai import types

# Set the API key from user's request
API_KEY = ""
os.environ['GEMINI_API_KEY'] = API_KEY

def example_1_basic():
    """First example from user - basic usage"""
    print("=== Example 1: Basic Gemini Usage ===")
    
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents="Explain how AI works in a few words"
    )
    print(response.text)

def example_2_with_config():
    """Second example from user - with thinking config disabled"""
    print("\n=== Example 2: Gemini with Thinking Disabled ===")
    
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Explain how AI works in a few words",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disables thinking
        ),
    )
    print(response.text)

def main():
    """Run both examples"""
    print("Google Gemini API Examples")
    print("=" * 50)
    
    try:
        example_1_basic()
        example_2_with_config()
        print("\n✅ Both examples completed successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have the google-genai package installed:")
        print("pip install google-genai")

if __name__ == "__main__":
    main() 