#!/usr/bin/env python3
"""
Generate synthetic user demographic profiles
Enhances anonymous MovieLens users with realistic attributes
"""

import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List
from faker import Faker

# Multi-locale faker for international users
fake = Faker(['en_US', 'fr_FR', 'es_ES', 'de_DE', 'it_IT', 'en_GB', 'en_CA'])

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

# Ensure output directory exists
(SYNTHETIC_DIR / "users").mkdir(parents=True, exist_ok=True)

# User archetypes
ARCHETYPES = {
    'cinephile': {
        'description': 'Film enthusiast, art house lover',
        'cultural_context': ['art_house', 'film_festivals', 'criterion', 'auteur_theory'],
        'device_prefs': {'tv': 0.7, 'laptop': 0.8, 'mobile': 0.3},
        'viewing_times': {'weekday_evening': 0.8, 'weekend_afternoon': 0.7, 'late_night': 0.6}
    },
    'casual': {
        'description': 'Mainstream viewer, popular content',
        'cultural_context': ['mainstream', 'popular_culture', 'trending'],
        'device_prefs': {'tv': 0.9, 'laptop': 0.5, 'mobile': 0.7},
        'viewing_times': {'weekday_evening': 0.7, 'weekend_afternoon': 0.9, 'late_night': 0.3}
    },
    'family': {
        'description': 'Family-oriented content consumer',
        'cultural_context': ['family_friendly', 'animated', 'educational', 'wholesome'],
        'device_prefs': {'tv': 0.95, 'laptop': 0.4, 'mobile': 0.5},
        'viewing_times': {'weekday_evening': 0.9, 'weekend_afternoon': 0.95, 'late_night': 0.1}
    },
    'young_adult': {
        'description': 'Young viewer, social media influenced',
        'cultural_context': ['trending', 'social_media', 'binge', 'viral', 'youth_culture'],
        'device_prefs': {'tv': 0.5, 'laptop': 0.7, 'mobile': 0.95},
        'viewing_times': {'weekday_evening': 0.6, 'weekend_afternoon': 0.5, 'late_night': 0.9}
    },
    'senior': {
        'description': 'Older viewer, classic and documentary',
        'cultural_context': ['classic', 'documentary', 'historical', 'educational', 'nostalgia'],
        'device_prefs': {'tv': 0.95, 'laptop': 0.3, 'mobile': 0.2},
        'viewing_times': {'weekday_evening': 0.8, 'weekend_afternoon': 0.9, 'late_night': 0.2}
    },
    'international': {
        'description': 'World cinema enthusiast',
        'cultural_context': ['world_cinema', 'multilingual', 'festivals', 'subtitled', 'foreign'],
        'device_prefs': {'tv': 0.7, 'laptop': 0.8, 'mobile': 0.5},
        'viewing_times': {'weekday_evening': 0.7, 'weekend_afternoon': 0.6, 'late_night': 0.7}
    },
    'genre_specialist': {
        'description': 'Deep expertise in specific genres',
        'cultural_context': ['genre_expert', 'niche', 'deep_knowledge', 'community'],
        'device_prefs': {'tv': 0.7, 'laptop': 0.7, 'mobile': 0.6},
        'viewing_times': {'weekday_evening': 0.75, 'weekend_afternoon': 0.65, 'late_night': 0.8}
    }
}

AGE_RANGES = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
COUNTRIES = ['US', 'FR', 'DE', 'ES', 'IT', 'CA', 'UK', 'AU', 'BE', 'NL']
LANGUAGES = ['en', 'fr', 'es', 'de', 'it', 'pt', 'nl']
TIMEZONES = ['UTC-8', 'UTC-7', 'UTC-6', 'UTC-5', 'UTC+0', 'UTC+1', 'UTC+2']


def deterministic_random(user_id: str, seed_suffix: str = "") -> random.Random:
    """Create deterministic random generator from user ID"""
    seed = int(hashlib.md5(f"{user_id}{seed_suffix}".encode()).hexdigest()[:8], 16)
    return random.Random(seed)


def select_archetype(user_id: str) -> str:
    """Deterministically select archetype based on user ID"""
    rng = deterministic_random(user_id, "archetype")
    return rng.choice(list(ARCHETYPES.keys()))


def generate_demographics(user_id: str, archetype: str) -> Dict:
    """Generate demographic information"""
    rng = deterministic_random(user_id, "demographics")

    # Age distribution based on archetype
    if archetype == 'young_adult':
        age_range = rng.choice(['18-24', '25-34'])
    elif archetype == 'senior':
        age_range = rng.choice(['55-64', '65+'])
    elif archetype == 'family':
        age_range = rng.choice(['35-44', '45-54'])
    else:
        age_range = rng.choice(AGE_RANGES)

    # Location
    country = rng.choice(COUNTRIES)
    timezone = rng.choice(TIMEZONES)

    # Languages (1-3 languages)
    num_langs = rng.randint(1, 3)
    languages = rng.sample(LANGUAGES, k=num_langs)
    # Ensure primary language matches country when possible
    if country == 'FR' and 'fr' not in languages:
        languages[0] = 'fr'
    elif country == 'DE' and 'de' not in languages:
        languages[0] = 'de'
    elif country in ['ES'] and 'es' not in languages:
        languages[0] = 'es'

    return {
        'age_range': age_range,
        'location': {
            'country': country,
            'timezone': timezone,
            'city': fake.city()
        },
        'language_preferences': languages,
        'cultural_context': ARCHETYPES[archetype]['cultural_context']
    }


def generate_preferences(user_id: str, archetype: str) -> Dict:
    """Generate viewing preferences"""
    rng = deterministic_random(user_id, "preferences")
    archetype_data = ARCHETYPES[archetype]

    # Add noise to archetype preferences
    def add_noise(base_value: float, variance: float = 0.2) -> float:
        return max(0.0, min(1.0, base_value + rng.gauss(0, variance)))

    viewing_times = {
        k: add_noise(v) for k, v in archetype_data['viewing_times'].items()
    }

    device_prefs = {
        k: add_noise(v) for k, v in archetype_data['device_prefs'].items()
    }

    return {
        'viewing_time_patterns': viewing_times,
        'device_preferences': device_prefs,
        'content_intensity': rng.choice(['light', 'medium', 'intense']),
        'attention_span': rng.choice(['short', 'medium', 'long', 'binge'])
    }


def generate_behavioral_traits(user_id: str, archetype: str) -> Dict:
    """Generate behavioral characteristics"""
    rng = deterministic_random(user_id, "behavior")

    return {
        'openness_to_new': rng.uniform(0.3, 0.9),
        'genre_diversity': rng.uniform(0.2, 0.9),
        'rating_frequency': rng.uniform(0.1, 0.9),
        'social_sharing': rng.uniform(0.0, 0.8),
        'completion_rate': rng.uniform(0.5, 0.95),
        'rewatch_tendency': rng.uniform(0.1, 0.7)
    }


def generate_user_profile(user_id: str) -> Dict:
    """Generate complete synthetic user profile"""

    # Select archetype deterministically
    archetype = select_archetype(user_id)

    profile = {
        'user_id': user_id,
        'archetype': archetype,
        'archetype_description': ARCHETYPES[archetype]['description'],
        'demographics': generate_demographics(user_id, archetype),
        'preferences': generate_preferences(user_id, archetype),
        'behavioral_traits': generate_behavioral_traits(user_id, archetype),
        'created_at': '2024-01-01T00:00:00Z',  # Placeholder
        'last_active': '2024-12-06T00:00:00Z'  # Placeholder
    }

    return profile


def extract_user_ids_from_ratings() -> List[str]:
    """Extract unique user IDs from processed ratings"""
    print("📋 Extracting user IDs from ratings...")

    user_ids = set()
    ratings_path = PROCESSED_DIR / "interactions" / "ratings.jsonl"

    if not ratings_path.exists():
        print(f"⚠️  Ratings file not found: {ratings_path}")
        print("   Run parse_movielens.py first!")
        return []

    with open(ratings_path) as f:
        for line in f:
            data = json.loads(line)
            user_ids.add(data['user_id'])

    user_list = sorted(list(user_ids))
    print(f"✅ Found {len(user_list):,} unique users")
    return user_list


def generate_all_profiles():
    """Generate profiles for all users"""
    print("=" * 60)
    print("👥 Synthetic User Profile Generator")
    print("=" * 60)
    print()

    # Get user IDs
    user_ids = extract_user_ids_from_ratings()
    if not user_ids:
        print("❌ No user IDs found. Exiting.")
        return

    print(f"🎭 Generating profiles for {len(user_ids):,} users...")
    print()

    # Generate profiles
    profiles = []
    archetype_counts = {k: 0 for k in ARCHETYPES.keys()}

    for i, user_id in enumerate(user_ids):
        profile = generate_user_profile(user_id)
        profiles.append(profile)
        archetype_counts[profile['archetype']] += 1

        if (i + 1) % 10000 == 0:
            print(f"   Generated {i+1:,} profiles...")

    # Save as JSONL
    output_path = SYNTHETIC_DIR / "users" / "demographics.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for profile in profiles:
            f.write(json.dumps(profile, ensure_ascii=False) + '\n')

    print(f"\n✅ Generated {len(profiles):,} user profiles → {output_path}")

    # Show archetype distribution
    print("\n📊 Archetype Distribution:")
    for archetype, count in sorted(archetype_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(profiles)) * 100
        print(f"   {archetype:20s}: {count:6,} ({percentage:5.2f}%)")

    # Save summary
    summary = {
        'total_users': len(profiles),
        'archetype_distribution': archetype_counts,
        'age_distribution': {},
        'country_distribution': {}
    }

    # Calculate age and country distributions
    for profile in profiles:
        age = profile['demographics']['age_range']
        country = profile['demographics']['location']['country']

        summary['age_distribution'][age] = summary['age_distribution'].get(age, 0) + 1
        summary['country_distribution'][country] = summary['country_distribution'].get(country, 0) + 1

    summary_path = SYNTHETIC_DIR / "users" / "profile_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Summary saved to {summary_path}")
    print("\n" + "=" * 60)
    print("✅ USER PROFILE GENERATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    try:
        generate_all_profiles()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
