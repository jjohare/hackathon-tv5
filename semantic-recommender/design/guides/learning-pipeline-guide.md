# Learning Pipeline Implementation Guide

**Target Audience**: ML Engineers, Data Scientists
**Prerequisites**: Understanding of reinforcement learning, Python/Rust
**Estimated Implementation Time**: 3-4 weeks

---

## Table of Contents

1. [AgentDB Integration](#1-agentdb-integration)
2. [RLHF Implementation](#2-rlhf-implementation)
3. [A/B Testing Framework](#3-ab-testing-framework)
4. [Online Learning](#4-online-learning)
5. [Monitoring & Evaluation](#5-monitoring--evaluation)
6. [Production Deployment](#6-production-deployment)

---

## 1. AgentDB Integration

### 1.1 AgentDB Setup

```bash
# Install AgentDB
pip install agentdb-python

# Or use Docker
docker run -d \
  --name agentdb \
  -p 8888:8888 \
  -v $(pwd)/agentdb_data:/data \
  agentdb/agentdb:latest
```

### 1.2 Initialize Learning System

```python
# src/learning/agentdb_integration.py
from agentdb import AgentDB, Episode, Trajectory
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

@dataclass
class UserInteraction:
    """Single user interaction with recommendation system"""
    user_id: str
    item_id: str
    timestamp: int
    action: str  # 'click', 'watch', 'skip', 'rate'
    duration: Optional[int]  # Watch duration in seconds
    rating: Optional[float]  # Explicit rating (1-5)
    context: Dict[str, any]  # Session context

@dataclass
class RecommendationState:
    """State representation for RL agent"""
    user_embedding: np.ndarray  # 1024-dim user vector
    context_vector: np.ndarray  # 512-dim context (time, device, etc.)
    recent_history: List[str]   # Last 10 item IDs
    psychographic_state: str    # 'energized', 'contemplative', etc.

@dataclass
class RecommendationAction:
    """Action taken by recommendation agent"""
    item_id: str
    strategy: str  # 'exploit', 'explore', 'serendipity'
    confidence: float

class AgentDBLearningSystem:
    def __init__(self, connection_url: str = "http://localhost:8888"):
        self.db = AgentDB(connection_url)
        self.db.create_collection("recommendation_episodes")

    def log_interaction(
        self,
        state: RecommendationState,
        action: RecommendationAction,
        reward: float,
        next_state: RecommendationState,
        done: bool
    ):
        """Log single step to AgentDB"""
        step = {
            'state': {
                'user_embedding': state.user_embedding.tolist(),
                'context': state.context_vector.tolist(),
                'history': state.recent_history,
                'psychographic': state.psychographic_state,
            },
            'action': {
                'item_id': action.item_id,
                'strategy': action.strategy,
                'confidence': action.confidence,
            },
            'reward': reward,
            'next_state': {
                'user_embedding': next_state.user_embedding.tolist(),
                'context': next_state.context_vector.tolist(),
                'history': next_state.recent_history,
                'psychographic': next_state.psychographic_state,
            },
            'done': done,
        }

        self.db.log_step("recommendation_episodes", step)

    def create_trajectory(
        self,
        user_id: str,
        session_id: str,
        steps: List[Dict]
    ) -> Trajectory:
        """Create trajectory from session"""
        return Trajectory(
            id=f"{user_id}_{session_id}",
            steps=steps,
            total_reward=sum(step['reward'] for step in steps),
            metadata={
                'user_id': user_id,
                'session_id': session_id,
            }
        )

    def compute_reward(
        self,
        interaction: UserInteraction,
        predicted_rating: float
    ) -> float:
        """
        Compute reward from user interaction

        Reward components:
        - Engagement: 0-1 (watch duration / total duration)
        - Explicit feedback: -1 to 1 (rating normalized)
        - Implicit signals: click (+0.2), skip (-0.3)
        """
        reward = 0.0

        # Engagement reward
        if interaction.action == 'watch' and interaction.duration:
            # Assume 2-hour max duration
            engagement = min(interaction.duration / 7200, 1.0)
            reward += engagement

        # Explicit feedback
        if interaction.rating:
            # Normalize to [-1, 1]
            normalized_rating = (interaction.rating - 3.0) / 2.0
            reward += normalized_rating * 2.0  # Weight explicit feedback higher

        # Implicit signals
        if interaction.action == 'click':
            reward += 0.2
        elif interaction.action == 'skip':
            reward -= 0.3

        # Prediction accuracy bonus
        if interaction.rating and predicted_rating:
            error = abs(interaction.rating - predicted_rating)
            accuracy_bonus = max(0, 1.0 - error / 5.0)
            reward += accuracy_bonus * 0.5

        return np.clip(reward, -1.0, 1.0)

    def sample_experience_replay(
        self,
        batch_size: int,
        prioritized: bool = True
    ) -> List[Dict]:
        """Sample batch for training with optional prioritization"""
        if prioritized:
            # Prioritize high-reward and surprising experiences
            episodes = self.db.query_episodes(
                collection="recommendation_episodes",
                filter={
                    'total_reward': {'$gt': 0.5}  # High-reward episodes
                },
                limit=batch_size
            )
        else:
            # Uniform sampling
            episodes = self.db.sample_episodes(
                collection="recommendation_episodes",
                n=batch_size
            )

        return episodes
```

---

## 2. RLHF Implementation

### 2.1 Contextual Bandit Agent

```python
# src/learning/contextual_bandit.py
import numpy as np
from scipy.stats import beta
from typing import List, Dict, Tuple

class ThompsonSamplingBandit:
    """
    Thompson Sampling for contextual bandits

    Uses Beta distribution for exploration/exploitation:
    - Each item has alpha, beta parameters
    - Sample from Beta(alpha, beta) to select item
    - Update parameters based on reward
    """

    def __init__(
        self,
        num_items: int,
        context_dim: int,
        learning_rate: float = 0.1
    ):
        self.num_items = num_items
        self.context_dim = context_dim

        # Beta distribution parameters (successes, failures)
        self.alpha = np.ones(num_items)
        self.beta = np.ones(num_items)

        # Context-aware weight matrix
        self.W = np.random.randn(num_items, context_dim) * 0.01

        self.learning_rate = learning_rate

    def select_item(
        self,
        context: np.ndarray,
        candidate_items: List[int],
        explore_rate: float = 0.1
    ) -> Tuple[int, float]:
        """
        Select item using Thompson Sampling

        Args:
            context: User/session context vector
            candidate_items: List of candidate item IDs
            explore_rate: Probability of pure exploration

        Returns:
            (selected_item_id, confidence)
        """
        # Pure exploration with probability explore_rate
        if np.random.random() < explore_rate:
            item_id = np.random.choice(candidate_items)
            return item_id, 0.0

        # Thompson Sampling
        samples = []
        for item_id in candidate_items:
            # Sample from Beta distribution
            theta = np.random.beta(
                self.alpha[item_id],
                self.beta[item_id]
            )

            # Context-aware adjustment
            context_score = np.dot(self.W[item_id], context)
            adjusted_theta = theta + context_score

            samples.append((item_id, adjusted_theta))

        # Select item with highest sampled value
        selected_item, confidence = max(samples, key=lambda x: x[1])

        return selected_item, confidence

    def update(
        self,
        item_id: int,
        context: np.ndarray,
        reward: float
    ):
        """
        Update model based on observed reward

        Args:
            item_id: Selected item ID
            context: Context vector
            reward: Observed reward (0-1)
        """
        # Update Beta distribution
        if reward > 0.5:
            self.alpha[item_id] += 1
        else:
            self.beta[item_id] += 1

        # Update context weights (gradient descent)
        predicted = np.dot(self.W[item_id], context)
        error = reward - predicted
        gradient = error * context
        self.W[item_id] += self.learning_rate * gradient

    def get_expected_rewards(
        self,
        context: np.ndarray,
        candidate_items: List[int]
    ) -> Dict[int, float]:
        """Get expected reward for each candidate item"""
        expected = {}
        for item_id in candidate_items:
            # Expected value of Beta distribution
            mean = self.alpha[item_id] / (self.alpha[item_id] + self.beta[item_id])

            # Context adjustment
            context_score = np.dot(self.W[item_id], context)

            expected[item_id] = mean + context_score

        return expected


class LinUCB:
    """
    Linear Upper Confidence Bound for contextual bandits

    Maintains confidence bounds on reward predictions:
    - Select item with highest UCB = predicted_reward + uncertainty
    - More efficient exploration than Thompson Sampling
    """

    def __init__(
        self,
        num_items: int,
        context_dim: int,
        alpha: float = 1.0  # Exploration parameter
    ):
        self.num_items = num_items
        self.context_dim = context_dim
        self.alpha = alpha

        # Per-item linear model: A^{-1} and b
        self.A = [
            np.identity(context_dim) for _ in range(num_items)
        ]
        self.b = [
            np.zeros(context_dim) for _ in range(num_items)
        ]

    def select_item(
        self,
        context: np.ndarray,
        candidate_items: List[int]
    ) -> Tuple[int, float]:
        """Select item with highest UCB"""
        ucb_scores = []

        for item_id in candidate_items:
            # Compute UCB
            A_inv = np.linalg.inv(self.A[item_id])
            theta = A_inv @ self.b[item_id]

            predicted_reward = theta @ context
            uncertainty = np.sqrt(context @ A_inv @ context)

            ucb = predicted_reward + self.alpha * uncertainty
            ucb_scores.append((item_id, ucb))

        selected_item, confidence = max(ucb_scores, key=lambda x: x[1])
        return selected_item, confidence

    def update(
        self,
        item_id: int,
        context: np.ndarray,
        reward: float
    ):
        """Update linear model for item"""
        self.A[item_id] += np.outer(context, context)
        self.b[item_id] += reward * context
```

### 2.2 Deep Reinforcement Learning (Optional)

```python
# src/learning/deep_rl.py
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class RecommendationPolicyNetwork(nn.Module):
    """
    Neural network policy for recommendation

    Architecture:
    - Input: concatenated [user_embedding, context, item_embedding]
    - Hidden layers: 512 -> 256 -> 128
    - Output: predicted reward (regression)
    """

    def __init__(
        self,
        user_dim: int = 1024,
        context_dim: int = 512,
        item_dim: int = 1024
    ):
        super().__init__()

        input_dim = user_dim + context_dim + item_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Predicted reward
        )

    def forward(
        self,
        user_embedding: torch.Tensor,
        context: torch.Tensor,
        item_embedding: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([user_embedding, context, item_embedding], dim=-1)
        return self.network(x)


class DeepQLearningAgent:
    """
    Deep Q-Learning for recommendation

    Uses experience replay and target network for stable training
    """

    def __init__(
        self,
        policy_net: RecommendationPolicyNetwork,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 100000
    ):
        self.policy_net = policy_net
        self.target_net = type(policy_net)().to(policy_net.device)
        self.target_net.load_state_dict(policy_net.state_dict())

        self.optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

    def store_experience(
        self,
        state: Dict,
        action: int,
        reward: float,
        next_state: Dict,
        done: bool
    ):
        """Store experience in replay buffer"""
        self.replay_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
        })

    def sample_batch(self, batch_size: int) -> List[Dict]:
        """Sample random batch from replay buffer"""
        return random.sample(self.replay_buffer, min(batch_size, len(self.replay_buffer)))

    def train_step(self, batch_size: int = 64):
        """Perform one training step"""
        if len(self.replay_buffer) < batch_size:
            return

        batch = self.sample_batch(batch_size)

        # Prepare batch tensors
        states = torch.stack([torch.tensor(exp['state']['user_embedding']) for exp in batch])
        actions = torch.tensor([exp['action'] for exp in batch])
        rewards = torch.tensor([exp['reward'] for exp in batch])
        next_states = torch.stack([torch.tensor(exp['next_state']['user_embedding']) for exp in batch])
        dones = torch.tensor([exp['done'] for exp in batch])

        # Compute Q-values
        current_q = self.policy_net(states, actions)

        # Compute target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss and backprop
        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy policy network weights to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
```

---

## 3. A/B Testing Framework

### 3.1 Experiment Manager

```python
# src/learning/ab_testing.py
from dataclasses import dataclass
from typing import List, Dict, Optional
import hashlib
import numpy as np

@dataclass
class Variant:
    """A/B test variant"""
    name: str
    strategy: str  # 'contextual_bandit', 'deep_rl', 'baseline'
    allocation: float  # Traffic allocation (0-1)
    config: Dict  # Variant-specific configuration

@dataclass
class ABTestConfig:
    """A/B test configuration"""
    experiment_id: str
    name: str
    description: str
    variants: List[Variant]
    start_timestamp: int
    end_timestamp: Optional[int]
    success_metric: str  # 'ctr', 'watch_time', 'rating'
    guardrail_metrics: List[str]  # Metrics that shouldn't degrade

class ABTestManager:
    def __init__(self):
        self.experiments: Dict[str, ABTestConfig] = {}
        self.user_assignments: Dict[str, str] = {}  # user_id -> variant_name

    def create_experiment(self, config: ABTestConfig):
        """Create new A/B test experiment"""
        # Validate allocations sum to 1.0
        total_allocation = sum(v.allocation for v in config.variants)
        assert abs(total_allocation - 1.0) < 1e-6, "Allocations must sum to 1.0"

        self.experiments[config.experiment_id] = config

    def assign_variant(
        self,
        experiment_id: str,
        user_id: str
    ) -> str:
        """
        Assign user to variant using consistent hashing

        Ensures:
        - Same user always gets same variant
        - Traffic splits match configured allocations
        """
        config = self.experiments[experiment_id]

        # Check existing assignment
        cache_key = f"{experiment_id}_{user_id}"
        if cache_key in self.user_assignments:
            return self.user_assignments[cache_key]

        # Consistent hashing
        hash_value = int(
            hashlib.md5(f"{experiment_id}{user_id}".encode()).hexdigest(),
            16
        )
        normalized = (hash_value % 1000000) / 1000000.0

        # Select variant based on cumulative allocation
        cumulative = 0.0
        for variant in config.variants:
            cumulative += variant.allocation
            if normalized < cumulative:
                self.user_assignments[cache_key] = variant.name
                return variant.name

        # Fallback (shouldn't reach here)
        return config.variants[0].name

    def get_variant_config(
        self,
        experiment_id: str,
        variant_name: str
    ) -> Dict:
        """Get configuration for specific variant"""
        config = self.experiments[experiment_id]
        for variant in config.variants:
            if variant.name == variant_name:
                return variant.config
        raise ValueError(f"Variant {variant_name} not found")


class MetricsCollector:
    """Collect metrics for A/B test analysis"""

    def __init__(self):
        self.metrics: Dict[str, List[Dict]] = {}

    def log_metric(
        self,
        experiment_id: str,
        variant_name: str,
        user_id: str,
        metric_name: str,
        value: float,
        timestamp: int
    ):
        """Log metric value for analysis"""
        key = f"{experiment_id}_{variant_name}"
        if key not in self.metrics:
            self.metrics[key] = []

        self.metrics[key].append({
            'user_id': user_id,
            'metric': metric_name,
            'value': value,
            'timestamp': timestamp,
        })

    def compute_statistics(
        self,
        experiment_id: str,
        metric_name: str
    ) -> Dict[str, Dict]:
        """
        Compute statistics for each variant

        Returns:
            {
                'variant_a': {'mean': 0.15, 'std': 0.05, 'n': 1000},
                'variant_b': {'mean': 0.18, 'std': 0.06, 'n': 1000},
            }
        """
        config = self.experiments[experiment_id]
        stats = {}

        for variant in config.variants:
            key = f"{experiment_id}_{variant.name}"
            if key not in self.metrics:
                continue

            values = [
                m['value'] for m in self.metrics[key]
                if m['metric'] == metric_name
            ]

            if values:
                stats[variant.name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'n': len(values),
                }

        return stats

    def run_statistical_test(
        self,
        experiment_id: str,
        metric_name: str,
        alpha: float = 0.05
    ) -> Dict:
        """
        Run t-test to determine statistical significance

        Returns:
            {
                'significant': True/False,
                'p_value': 0.03,
                'winner': 'variant_b',
                'lift': 0.15  # 15% improvement
            }
        """
        from scipy import stats as scipy_stats

        statistics = self.compute_statistics(experiment_id, metric_name)

        if len(statistics) < 2:
            return {'error': 'Need at least 2 variants for comparison'}

        # Compare first two variants (extend for multi-arm)
        variants = list(statistics.keys())
        a_data = [
            m['value'] for m in self.metrics[f"{experiment_id}_{variants[0]}"]
            if m['metric'] == metric_name
        ]
        b_data = [
            m['value'] for m in self.metrics[f"{experiment_id}_{variants[1]}"]
            if m['metric'] == metric_name
        ]

        # Two-sample t-test
        t_stat, p_value = scipy_stats.ttest_ind(a_data, b_data)

        significant = p_value < alpha
        mean_a = statistics[variants[0]]['mean']
        mean_b = statistics[variants[1]]['mean']
        lift = (mean_b - mean_a) / mean_a if mean_a > 0 else 0
        winner = variants[1] if mean_b > mean_a else variants[0]

        return {
            'significant': significant,
            'p_value': p_value,
            'winner': winner if significant else None,
            'lift': lift,
            'statistics': statistics,
        }
```

---

## 4. Online Learning

### 4.1 Streaming Update Pipeline

```python
# src/learning/online_learning.py
import asyncio
from kafka import KafkaConsumer, KafkaProducer
import json

class OnlineLearningPipeline:
    """
    Real-time learning from streaming user interactions

    Pipeline:
    1. Consume interactions from Kafka
    2. Compute rewards
    3. Update models
    4. Publish updated predictions
    """

    def __init__(
        self,
        kafka_bootstrap_servers: List[str],
        model: Union[ThompsonSamplingBandit, LinUCB],
        agentdb: AgentDBLearningSystem
    ):
        self.consumer = KafkaConsumer(
            'user_interactions',
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda m: json.dumps(m).encode('utf-8')
        )

        self.model = model
        self.agentdb = agentdb

    async def run(self):
        """Main processing loop"""
        for message in self.consumer:
            interaction = UserInteraction(**message.value)

            # Compute reward
            reward = self.agentdb.compute_reward(
                interaction,
                predicted_rating=message.value.get('predicted_rating')
            )

            # Update model
            context = self._extract_context(interaction)
            item_id = self._get_item_index(interaction.item_id)

            self.model.update(item_id, context, reward)

            # Log to AgentDB
            state = self._build_state(interaction)
            action = RecommendationAction(
                item_id=interaction.item_id,
                strategy='online_learning',
                confidence=1.0
            )
            next_state = self._build_next_state(interaction)

            self.agentdb.log_interaction(
                state, action, reward, next_state, done=True
            )

            # Publish updated model statistics (for monitoring)
            self.producer.send('model_updates', {
                'user_id': interaction.user_id,
                'item_id': interaction.item_id,
                'reward': reward,
                'timestamp': interaction.timestamp,
            })

    def _extract_context(self, interaction: UserInteraction) -> np.ndarray:
        """Extract context vector from interaction"""
        # Placeholder - implement actual feature engineering
        context = np.random.randn(512)
        return context

    def _get_item_index(self, item_id: str) -> int:
        """Map item ID to index"""
        # Placeholder - use actual item ID mapping
        return hash(item_id) % self.model.num_items
```

---

## 5. Monitoring & Evaluation

### 5.1 Metrics Dashboard

```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

class RecommendationMetrics:
    """Prometheus metrics for recommendation system"""

    def __init__(self):
        # Request metrics
        self.recommendations_total = Counter(
            'recommendations_total',
            'Total recommendations served',
            ['variant', 'strategy']
        )

        self.recommendation_latency = Histogram(
            'recommendation_latency_seconds',
            'Recommendation serving latency',
            ['variant'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        )

        # Engagement metrics
        self.click_through_rate = Gauge(
            'click_through_rate',
            'CTR by variant',
            ['variant']
        )

        self.avg_watch_time = Gauge(
            'avg_watch_time_minutes',
            'Average watch time by variant',
            ['variant']
        )

        # Learning metrics
        self.model_updates = Counter(
            'model_updates_total',
            'Total model updates',
            ['model_type']
        )

        self.reward_mean = Gauge(
            'reward_mean',
            'Mean reward per variant',
            ['variant']
        )

        self.exploration_rate = Gauge(
            'exploration_rate',
            'Exploration rate per variant',
            ['variant']
        )

    def observe_recommendation(
        self,
        variant: str,
        strategy: str,
        latency: float
    ):
        """Record recommendation served"""
        self.recommendations_total.labels(variant=variant, strategy=strategy).inc()
        self.recommendation_latency.labels(variant=variant).observe(latency)

    def update_engagement_metrics(
        self,
        variant: str,
        ctr: float,
        avg_watch_time: float
    ):
        """Update engagement metrics"""
        self.click_through_rate.labels(variant=variant).set(ctr)
        self.avg_watch_time.labels(variant=variant).set(avg_watch_time)

# Start Prometheus metrics server
start_http_server(9090)
```

---

## 6. Production Deployment

### 6.1 Full Pipeline Integration

```python
# src/main.py
import asyncio
from learning.agentdb_integration import AgentDBLearningSystem
from learning.contextual_bandit import ThompsonSamplingBandit
from learning.ab_testing import ABTestManager, ABTestConfig, Variant
from learning.online_learning import OnlineLearningPipeline

async def main():
    # Initialize components
    agentdb = AgentDBLearningSystem("http://localhost:8888")

    # Create bandit models for each variant
    thompson_sampling = ThompsonSamplingBandit(
        num_items=100000,
        context_dim=512
    )

    # Set up A/B test
    ab_manager = ABTestManager()
    ab_manager.create_experiment(ABTestConfig(
        experiment_id="recommendation_v2",
        name="Thompson Sampling vs Baseline",
        description="Test new contextual bandit model",
        variants=[
            Variant(
                name="control",
                strategy="baseline",
                allocation=0.5,
                config={"model": "matrix_factorization"}
            ),
            Variant(
                name="treatment",
                strategy="contextual_bandit",
                allocation=0.5,
                config={"model": "thompson_sampling", "explore_rate": 0.1}
            ),
        ],
        start_timestamp=1700000000,
        end_timestamp=None,
        success_metric="watch_time",
        guardrail_metrics=["ctr", "rating"]
    ))

    # Start online learning pipeline
    pipeline = OnlineLearningPipeline(
        kafka_bootstrap_servers=["localhost:9092"],
        model=thompson_sampling,
        agentdb=agentdb
    )

    await pipeline.run()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Performance Targets

**Target Metrics:**
- **Model update latency**: <10ms
- **A/B test assignment**: <1ms
- **Reward computation**: <5ms
- **Statistical significance**: Within 7 days
- **Exploration rate**: 10-15%

**Optimization Checklist:**
- [ ] AgentDB integration working
- [ ] Contextual bandits trained
- [ ] A/B testing framework deployed
- [ ] Online learning pipeline streaming
- [ ] Metrics dashboard live
- [ ] Statistical testing automated

---

**Next Steps:**
1. Integrate with recommendation API
2. Set up Kafka streaming
3. Deploy monitoring dashboards
4. Run initial A/B tests

**Related Guides:**
- [GPU Setup Guide](gpu-setup-guide.md)
- [Vector Search Implementation](vector-search-implementation.md)
- [Deployment Guide](deployment-guide.md)
