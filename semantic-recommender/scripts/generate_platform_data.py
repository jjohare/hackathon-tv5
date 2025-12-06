#!/usr/bin/env python3
"""
Generate synthetic platform availability data
Simulates where content is available and how
"""

import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

# Ensure output directory
(SYNTHETIC_DIR / "platforms").mkdir(parents=True, exist_ok=True)

# Platform definitions
PLATFORMS = [
    {
        'id': 'tv5monde',
        'name': 'TV5MONDE',
        'type': 'free',
        'regions': ['FR', 'BE', 'CH', 'CA', 'US', 'Global'],
        'quality': ['HD', '4K'],
        'subtitle_langs': ['fr', 'en', 'es', 'de'],
        'availability_rate': 0.15  # 15% of movies
    },
    {
        'id': 'netflix',
        'name': 'Netflix',
        'type': 'subscription',
        'regions': ['US', 'FR', 'UK', 'DE', 'ES', 'IT', 'CA', 'AU', 'Global'],
        'quality': ['HD', '4K'],
        'subtitle_langs': ['en', 'fr', 'es', 'de', 'it', 'pt', 'ja', 'ko'],
        'availability_rate': 0.35
    },
    {
        'id': 'amazon',
        'name': 'Amazon Prime Video',
        'type': 'subscription',
        'regions': ['US', 'UK', 'DE', 'FR', 'IT', 'ES', 'CA', 'AU'],
        'quality': ['HD', '4K'],
        'subtitle_langs': ['en', 'fr', 'es', 'de', 'it'],
        'availability_rate': 0.30
    },
    {
        'id': 'mubi',
        'name': 'MUBI',
        'type': 'subscription',
        'regions': ['Global'],
        'quality': ['HD'],
        'subtitle_langs': ['en', 'fr', 'es', 'de', 'it'],
        'availability_rate': 0.10
    },
    {
        'id': 'criterion',
        'name': 'Criterion Channel',
        'type': 'subscription',
        'regions': ['US', 'CA'],
        'quality': ['HD'],
        'subtitle_langs': ['en'],
        'availability_rate': 0.08
    },
    {
        'id': 'disney',
        'name': 'Disney+',
        'type': 'subscription',
        'regions': ['US', 'FR', 'UK', 'DE', 'ES', 'IT', 'CA', 'AU', 'Global'],
        'quality': ['HD', '4K'],
        'subtitle_langs': ['en', 'fr', 'es', 'de', 'it', 'pt'],
        'availability_rate': 0.20
    },
    {
        'id': 'hbo',
        'name': 'HBO Max',
        'type': 'subscription',
        'regions': ['US', 'UK', 'DE', 'FR', 'ES'],
        'quality': ['HD', '4K'],
        'subtitle_langs': ['en', 'es', 'fr', 'de'],
        'availability_rate': 0.25
    },
    {
        'id': 'apple',
        'name': 'Apple TV+',
        'type': 'subscription',
        'regions': ['Global'],
        'quality': ['4K'],
        'subtitle_langs': ['en', 'fr', 'es', 'de', 'it', 'pt', 'ja', 'ko'],
        'availability_rate': 0.15
    }
]


def deterministic_random(media_id: str, seed_suffix: str = "") -> random.Random:
    """Create deterministic random generator from media ID"""
    seed = int(hashlib.md5(f"{media_id}{seed_suffix}".encode()).hexdigest()[:8], 16)
    return random.Random(seed)


def should_be_on_platform(media_id: str, platform: Dict) -> bool:
    """Deterministically decide if content is on platform"""
    rng = deterministic_random(media_id, platform['id'])
    return rng.random() < platform['availability_rate']


def generate_platform_entry(media_id: str, platform: Dict) -> Dict:
    """Generate availability entry for a specific platform"""
    rng = deterministic_random(media_id, f"{platform['id']}_details")

    # Select region(s)
    num_regions = rng.randint(1, min(3, len(platform['regions'])))
    regions = rng.sample(platform['regions'], k=num_regions)

    # Select quality options
    quality = rng.sample(platform['quality'], k=rng.randint(1, len(platform['quality'])))

    # Select subtitle languages
    max_subs = min(6, len(platform['subtitle_langs']))
    min_subs = min(2, max_subs)  # Ensure min <= max
    num_subs = rng.randint(min_subs, max_subs)
    subtitles = rng.sample(platform['subtitle_langs'], k=num_subs)

    # Audio tracks (fewer than subtitles)
    max_audio = min(3, len(platform['subtitle_langs']))
    num_audio = rng.randint(1, max_audio)
    audio_tracks = rng.sample(platform['subtitle_langs'], k=num_audio)

    return {
        'platform_id': platform['id'],
        'platform_name': platform['name'],
        'regions': regions,
        'availability_type': platform['type'],
        'price': None if platform['type'] == 'subscription' else rng.choice([3.99, 4.99, 5.99, 9.99, 14.99]),
        'quality_options': quality,
        'subtitle_languages': subtitles,
        'audio_tracks': audio_tracks,
        'hdr_available': '4K' in quality and rng.random() < 0.7,
        'dolby_atmos': '4K' in quality and rng.random() < 0.5
    }


def generate_availability_for_movie(media_id: str) -> Dict:
    """Generate platform availability for a single movie"""
    available_platforms = []

    for platform in PLATFORMS:
        if should_be_on_platform(media_id, platform):
            platform_entry = generate_platform_entry(media_id, platform)
            available_platforms.append(platform_entry)

    return {
        'media_id': media_id,
        'platforms': available_platforms,
        'total_platforms': len(available_platforms),
        'updated_at': '2024-12-06T00:00:00Z'
    }


def load_movie_ids() -> List[str]:
    """Load movie IDs from processed data"""
    print("📋 Loading movie IDs...")

    movie_ids = []
    movies_path = PROCESSED_DIR / "media" / "movies.jsonl"

    if not movies_path.exists():
        print(f"⚠️  Movies file not found: {movies_path}")
        print("   Run parse_movielens.py first!")
        return []

    with open(movies_path) as f:
        for line in f:
            data = json.loads(line)
            movie_ids.append(data['media_id'])

    print(f"✅ Loaded {len(movie_ids):,} movies")
    return movie_ids


def generate_all_platform_data():
    """Generate platform availability for all movies"""
    print("=" * 60)
    print("🎬 Platform Availability Generator")
    print("=" * 60)
    print()

    # Load movie IDs
    movie_ids = load_movie_ids()
    if not movie_ids:
        print("❌ No movies found. Exiting.")
        return

    print(f"🌐 Generating platform data for {len(movie_ids):,} movies...")
    print(f"   Platforms: {len(PLATFORMS)}")
    print()

    # Generate availability data
    availability_data = []
    platform_counts = {p['id']: 0 for p in PLATFORMS}
    movies_with_platforms = 0

    for i, media_id in enumerate(movie_ids):
        avail = generate_availability_for_movie(media_id)
        availability_data.append(avail)

        if avail['total_platforms'] > 0:
            movies_with_platforms += 1

        for platform_entry in avail['platforms']:
            platform_counts[platform_entry['platform_id']] += 1

        if (i + 1) % 5000 == 0:
            print(f"   Processed {i+1:,} movies...")

    # Save as JSONL
    output_path = SYNTHETIC_DIR / "platforms" / "availability.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for avail in availability_data:
            f.write(json.dumps(avail) + '\n')

    print(f"\n✅ Generated platform data for {len(availability_data):,} movies → {output_path}")
    print(f"   Movies with at least one platform: {movies_with_platforms:,} ({movies_with_platforms/len(movie_ids)*100:.1f}%)")

    # Show platform distribution
    print("\n📊 Platform Distribution:")
    for platform_id, count in sorted(platform_counts.items(), key=lambda x: x[1], reverse=True):
        platform_name = next(p['name'] for p in PLATFORMS if p['id'] == platform_id)
        percentage = (count / len(movie_ids)) * 100
        print(f"   {platform_name:30s}: {count:6,} ({percentage:5.2f}%)")

    # Save summary
    summary = {
        'total_movies': len(movie_ids),
        'movies_with_platforms': movies_with_platforms,
        'platform_distribution': platform_counts,
        'avg_platforms_per_movie': sum(a['total_platforms'] for a in availability_data) / len(availability_data)
    }

    summary_path = SYNTHETIC_DIR / "platforms" / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Summary saved to {summary_path}")
    print(f"   Average platforms per movie: {summary['avg_platforms_per_movie']:.2f}")

    print("\n" + "=" * 60)
    print("✅ PLATFORM DATA GENERATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    try:
        generate_all_platform_data()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
