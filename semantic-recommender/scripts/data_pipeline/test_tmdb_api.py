#!/usr/bin/env python3
"""
Quick test script to verify TMDB API key and connection.
Tests API access with a known movie (Inception).
"""

import os
import sys
import json
import requests

def test_tmdb_api():
    """Test TMDB API key and connection."""

    # Get API key
    api_key = os.getenv("TMDB_API_KEY")

    print("=" * 70)
    print("TMDB API Test")
    print("=" * 70)
    print()

    # Check API key
    if not api_key:
        print("❌ TMDB_API_KEY environment variable not set")
        print()
        print("To get your API key:")
        print("  1. Visit: https://www.themoviedb.org/settings/api")
        print("  2. Register for free account")
        print("  3. Request API key (instant approval)")
        print("  4. Copy 'API Key (v3 auth)'")
        print()
        print("Then set it:")
        print("  export TMDB_API_KEY='your_api_key_here'")
        print()
        return False

    print(f"✅ API key found: {api_key[:8]}...{api_key[-8:]}")
    print()

    # Test API request
    print("Testing API connection...")
    print("Fetching movie: Inception (ID: 27205)")
    print()

    url = "https://api.themoviedb.org/3/movie/27205"
    params = {
        'api_key': api_key,
        'append_to_response': 'keywords,credits'
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()

            print("✅ API connection successful!")
            print()
            print("Movie Details:")
            print(f"  Title: {data.get('title')}")
            print(f"  Tagline: {data.get('tagline')}")
            print(f"  Overview: {data.get('overview')[:100]}...")
            print()

            # Genres
            genres = [g['name'] for g in data.get('genres', [])]
            print(f"  Genres: {', '.join(genres)}")
            print()

            # Keywords
            keywords = [k['name'] for k in data.get('keywords', {}).get('keywords', [])[:5]]
            print(f"  Keywords: {', '.join(keywords)}")
            print()

            # Cast
            cast = data.get('credits', {}).get('cast', [])[:5]
            cast_names = [actor['name'] for actor in cast]
            print(f"  Cast: {', '.join(cast_names)}")
            print()

            # Director
            crew = data.get('credits', {}).get('crew', [])
            directors = [c['name'] for c in crew if c.get('job') == 'Director']
            if directors:
                print(f"  Director: {directors[0]}")
                print()

            print("=" * 70)
            print("✅ All tests passed! Ready to run enrichment pipeline.")
            print("=" * 70)

            return True

        elif response.status_code == 401:
            print("❌ Invalid API key")
            print()
            print("Please verify your API key:")
            print("  1. Visit: https://www.themoviedb.org/settings/api")
            print("  2. Copy 'API Key (v3 auth)' (NOT the Read Access Token)")
            print("  3. Set it: export TMDB_API_KEY='your_key'")
            print()
            return False

        elif response.status_code == 429:
            print("⚠️  Rate limit exceeded")
            print("This shouldn't happen on first request. Wait a moment and try again.")
            print()
            return False

        else:
            print(f"❌ API error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            print()
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        print()
        print("Check your internet connection and try again.")
        return False


if __name__ == "__main__":
    success = test_tmdb_api()
    sys.exit(0 if success else 1)
