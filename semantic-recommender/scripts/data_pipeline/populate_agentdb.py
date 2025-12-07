#!/usr/bin/env python3
"""
Populate AgentDB with RL policies and training episodes
Creates user learning models and experience replay buffers
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
AGENTDB_DIR = DATA_DIR / "agentdb"

# Ensure output directory
AGENTDB_DIR.mkdir(parents=True, exist_ok=True)

# RL configuration
LEARNING_ALGORITHMS = [
    'q_learning',
    'sarsa',
    'dqn',
    'actor_critic',
    'ppo',
    'decision_transformer'
]

# Episode parameters
MAX_EPISODES_PER_USER = 200  # Synthetic training episodes


def load_user_ids():
    """Load all user IDs"""
    print("📋 Loading user IDs...")

    # Try from embeddings first
    ids_path = DATA_DIR / "embeddings" / "users" / "user_ids.json"

    if ids_path.exists():
        with open(ids_path) as f:
            user_ids = json.load(f)
    else:
        # Fall back to demographics
        demo_path = SYNTHETIC_DIR / "users" / "demographics.jsonl"
        user_ids = []
        with open(demo_path) as f:
            for line in f:
                data = json.loads(line)
                user_ids.append(data['user_id'])

    print(f"✅ Loaded {len(user_ids):,} user IDs")
    return user_ids


def load_user_ratings():
    """Load user ratings for episode generation"""
    print("📁 Loading user ratings...")

    user_ratings = {}
    ratings_path = PROCESSED_DIR / "interactions" / "ratings.jsonl"

    with open(ratings_path) as f:
        for line in tqdm(f, desc="Loading ratings"):
            rating = json.loads(line)
            uid = rating['user_id']

            if uid not in user_ratings:
                user_ratings[uid] = []

            user_ratings[uid].append(rating)

    print(f"✅ Loaded ratings for {len(user_ratings):,} users")
    return user_ratings


def load_demographics():
    """Load user demographics for archetype info"""
    print("📁 Loading demographics...")

    demographics = {}
    demo_path = SYNTHETIC_DIR / "users" / "demographics.jsonl"

    if demo_path.exists():
        with open(demo_path) as f:
            for line in f:
                data = json.loads(line)
                demographics[data['user_id']] = data

    print(f"✅ Loaded demographics for {len(demographics):,} users")
    return demographics


def select_algorithm(user_id: str, archetype: str) -> str:
    """Select learning algorithm based on user archetype"""
    archetype_algorithms = {
        'cinephile': 'decision_transformer',  # Complex preferences
        'casual': 'q_learning',  # Simple, fast
        'family': 'actor_critic',  # Multi-objective
        'young_adult': 'dqn',  # Exploration-focused
        'senior': 'sarsa',  # Conservative
        'international': 'ppo',  # Diverse content
        'genre_specialist': 'decision_transformer'  # Sequential
    }

    return archetype_algorithms.get(archetype, 'q_learning')


def create_policy_entry(user_id: str, algorithm: str, archetype: str, rating_count: int) -> Dict:
    """Create RL policy entry for user"""

    # Initialize policy parameters based on algorithm
    if algorithm == 'q_learning':
        policy_params = {
            'learning_rate': 0.1,
            'discount_factor': 0.9,
            'epsilon': 0.1,
            'q_table_size': rating_count
        }
    elif algorithm == 'sarsa':
        policy_params = {
            'learning_rate': 0.1,
            'discount_factor': 0.9,
            'epsilon': 0.05,
            'q_table_size': rating_count
        }
    elif algorithm == 'dqn':
        policy_params = {
            'learning_rate': 0.001,
            'discount_factor': 0.99,
            'epsilon': 0.2,
            'replay_buffer_size': 10000,
            'batch_size': 32,
            'target_update_freq': 100
        }
    elif algorithm == 'actor_critic':
        policy_params = {
            'actor_lr': 0.001,
            'critic_lr': 0.002,
            'discount_factor': 0.99,
            'gae_lambda': 0.95
        }
    elif algorithm == 'ppo':
        policy_params = {
            'learning_rate': 0.0003,
            'clip_range': 0.2,
            'n_epochs': 10,
            'batch_size': 64
        }
    else:  # decision_transformer
        policy_params = {
            'learning_rate': 0.0001,
            'context_length': 20,
            'n_layers': 4,
            'n_heads': 4,
            'embed_dim': 128
        }

    return {
        'user_id': user_id,
        'algorithm': algorithm,
        'archetype': archetype,
        'policy_params': policy_params,
        'training_episodes': min(rating_count * 2, MAX_EPISODES_PER_USER),
        'created_at': '2024-12-06T00:00:00Z',
        'last_updated': '2024-12-06T00:00:00Z',
        'version': 1
    }


def generate_episodes_from_ratings(user_id: str, ratings: List[Dict], max_episodes: int) -> List[Dict]:
    """Generate training episodes from user ratings"""
    episodes = []

    # Sort by timestamp
    sorted_ratings = sorted(ratings, key=lambda r: r['timestamp'])

    # Create episodes (state, action, reward, next_state)
    for i in range(len(sorted_ratings)):
        rating = sorted_ratings[i]

        # State: recent viewing history (last 5 items)
        state_history = sorted_ratings[max(0, i-5):i]
        state = {
            'recent_items': [r['media_id'] for r in state_history],
            'recent_ratings': [r['rating'] for r in state_history],
            'time_of_day': rating['context']['time_of_day'],
            'step': i
        }

        # Action: selected this item
        action = {
            'media_id': rating['media_id'],
            'action_type': 'select'
        }

        # Reward: normalized rating (0-1)
        reward = (rating['rating'] - 1) / 4.0

        # Next state: includes this item
        next_state_history = sorted_ratings[max(0, i-4):i+1]
        next_state = {
            'recent_items': [r['media_id'] for r in next_state_history],
            'recent_ratings': [r['rating'] for r in next_state_history],
            'time_of_day': rating['context']['time_of_day'],
            'step': i + 1
        }

        # Terminal if last rating or big time gap
        terminal = (i == len(sorted_ratings) - 1)
        if i < len(sorted_ratings) - 1:
            time_gap = sorted_ratings[i+1]['timestamp'] - rating['timestamp']
            terminal = terminal or (time_gap > 86400 * 30)  # 30 days

        episode = {
            'episode_id': f'{user_id}_ep_{i}',
            'user_id': user_id,
            'step': i,
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'terminal': terminal,
            'timestamp': rating['timestamp']
        }

        episodes.append(episode)

        if len(episodes) >= max_episodes:
            break

    return episodes


def create_policies():
    """Create RL policies for all users"""
    print("\n" + "=" * 60)
    print("🧠 Creating RL Policies")
    print("=" * 60)

    # Load data
    user_ids = load_user_ids()
    demographics = load_demographics()
    user_ratings = load_user_ratings()

    # Create policies
    print(f"\n💾 Generating policies...")
    policies = []

    for user_id in tqdm(user_ids, desc="Creating policies"):
        # Get user info
        demo = demographics.get(user_id, {})
        archetype = demo.get('archetype', 'casual')

        # Get rating count
        ratings = user_ratings.get(user_id, [])
        rating_count = len(ratings)

        if rating_count == 0:
            continue  # Skip users with no ratings

        # Select algorithm
        algorithm = select_algorithm(user_id, archetype)

        # Create policy
        policy = create_policy_entry(user_id, algorithm, archetype, rating_count)
        policies.append(policy)

    # Save policies
    policies_path = AGENTDB_DIR / "policies.jsonl"
    with open(policies_path, 'w') as f:
        for policy in policies:
            f.write(json.dumps(policy) + '\n')

    print(f"✅ Created {len(policies):,} policies → {policies_path}")

    # Algorithm distribution
    algo_counts = {}
    for policy in policies:
        algo = policy['algorithm']
        algo_counts[algo] = algo_counts.get(algo, 0) + 1

    print("\n📊 Algorithm Distribution:")
    for algo, count in sorted(algo_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(policies)) * 100
        print(f"   {algo:25s}: {count:6,} ({percentage:5.2f}%)")

    return policies


def create_episodes():
    """Create training episodes from ratings"""
    print("\n" + "=" * 60)
    print("🎯 Creating Training Episodes")
    print("=" * 60)

    # Load data
    user_ratings = load_user_ratings()

    # Generate episodes
    print(f"\n💾 Generating episodes...")
    all_episodes = []
    total_episodes = 0

    for user_id, ratings in tqdm(user_ratings.items(), desc="Generating episodes"):
        episodes = generate_episodes_from_ratings(user_id, ratings, MAX_EPISODES_PER_USER)
        all_episodes.extend(episodes)
        total_episodes += len(episodes)

    # Save episodes
    episodes_path = AGENTDB_DIR / "episodes.jsonl"
    with open(episodes_path, 'w') as f:
        for episode in all_episodes:
            f.write(json.dumps(episode) + '\n')

    print(f"✅ Created {total_episodes:,} episodes → {episodes_path}")

    # Calculate stats
    avg_episodes = total_episodes / len(user_ratings)
    avg_reward = np.mean([ep['reward'] for ep in all_episodes])
    terminal_count = sum(1 for ep in all_episodes if ep['terminal'])

    print(f"\n📊 Episode Statistics:")
    print(f"   Total episodes: {total_episodes:,}")
    print(f"   Unique users: {len(user_ratings):,}")
    print(f"   Avg episodes/user: {avg_episodes:.1f}")
    print(f"   Avg reward: {avg_reward:.3f}")
    print(f"   Terminal episodes: {terminal_count:,}")

    return all_episodes


def create_replay_buffers():
    """Create experience replay buffers for each algorithm"""
    print("\n" + "=" * 60)
    print("💾 Creating Replay Buffers")
    print("=" * 60)

    # Load episodes
    episodes_path = AGENTDB_DIR / "episodes.jsonl"

    if not episodes_path.exists():
        print("   ⚠️  Episodes not found, run create_episodes first")
        return

    # Group episodes by user
    user_episodes = {}
    with open(episodes_path) as f:
        for line in f:
            episode = json.loads(line)
            uid = episode['user_id']
            if uid not in user_episodes:
                user_episodes[uid] = []
            user_episodes[uid].append(episode)

    # Load policies to get algorithms
    policies_path = AGENTDB_DIR / "policies.jsonl"
    user_algorithms = {}
    with open(policies_path) as f:
        for line in f:
            policy = json.loads(line)
            user_algorithms[policy['user_id']] = policy['algorithm']

    # Create buffers by algorithm
    algo_buffers = {algo: [] for algo in LEARNING_ALGORITHMS}

    for user_id, episodes in user_episodes.items():
        algo = user_algorithms.get(user_id)
        if algo in algo_buffers:
            algo_buffers[algo].extend(episodes)

    # Save buffers
    for algo, episodes in algo_buffers.items():
        if episodes:
            buffer_path = AGENTDB_DIR / f"replay_buffer_{algo}.jsonl"
            with open(buffer_path, 'w') as f:
                for episode in episodes:
                    f.write(json.dumps(episode) + '\n')
            print(f"   ✓ {algo:25s}: {len(episodes):8,} episodes → {buffer_path.name}")

    print("✅ Replay buffers created")


def generate_stats():
    """Generate final statistics"""
    print("\n" + "=" * 60)
    print("📊 Final Statistics")
    print("=" * 60)

    # Count files
    policies_path = AGENTDB_DIR / "policies.jsonl"
    episodes_path = AGENTDB_DIR / "episodes.jsonl"

    policy_count = 0
    with open(policies_path) as f:
        policy_count = sum(1 for _ in f)

    episode_count = 0
    with open(episodes_path) as f:
        episode_count = sum(1 for _ in f)

    # Count replay buffers
    buffer_sizes = {}
    for algo in LEARNING_ALGORITHMS:
        buffer_path = AGENTDB_DIR / f"replay_buffer_{algo}.jsonl"
        if buffer_path.exists():
            with open(buffer_path) as f:
                buffer_sizes[algo] = sum(1 for _ in f)

    stats = {
        'policies': policy_count,
        'episodes': episode_count,
        'replay_buffers': buffer_sizes,
        'total_buffer_episodes': sum(buffer_sizes.values()),
        'algorithms_used': len(buffer_sizes)
    }

    print(f"\nPolicies: {stats['policies']:,}")
    print(f"Training Episodes: {stats['episodes']:,}")
    print(f"Replay Buffer Episodes: {stats['total_buffer_episodes']:,}")
    print(f"Algorithms: {stats['algorithms_used']}")

    # Save stats
    stats_path = AGENTDB_DIR / "agentdb_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n💾 Stats saved to {stats_path}")

    return stats


def main():
    """Main execution"""
    print("=" * 60)
    print("🚀 AgentDB Population Script")
    print("=" * 60)
    print()

    try:
        # Create policies
        policies = create_policies()

        # Create episodes
        episodes = create_episodes()

        # Create replay buffers
        create_replay_buffers()

        # Generate stats
        stats = generate_stats()

        print("\n" + "=" * 60)
        print("✅ AGENTDB POPULATION COMPLETE")
        print("=" * 60)
        print(f"\nRL Infrastructure Created:")
        print(f"  Policies: {stats['policies']:,}")
        print(f"  Episodes: {stats['episodes']:,}")
        print(f"  Algorithms: {stats['algorithms_used']}")
        print(f"\nReady for reinforcement learning training")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
