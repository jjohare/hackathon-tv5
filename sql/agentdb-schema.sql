-- AgentDB PostgreSQL Schema for TV5 Monde Media Gateway
-- Multi-modal memory: episodic (trajectories), semantic (patterns), procedural (RL policies)
-- Target: 5ms cached lookups, async RL updates, pgvector for semantic search

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";

-- Agent episodic memory: trajectories (state-action-reward sequences)
CREATE TABLE IF NOT EXISTS agent_episodes (
    episode_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    state_vector VECTOR(768),  -- User behavioral embedding
    action_taken JSONB NOT NULL,  -- {"media_id": "...", "rank": 1, "method": "contextual_bandit"}
    reward FLOAT NOT NULL,  -- Engagement score (watch_time / duration, 0-1)
    context JSONB NOT NULL,  -- {"device": "mobile", "time": "evening", "location": "FR"}
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    episode_complete BOOLEAN DEFAULT FALSE,

    -- Metadata for analysis
    media_id VARCHAR(255),
    interaction_type VARCHAR(50),  -- 'view', 'click', 'watch', 'complete', 'skip'
    engagement_duration_sec INTEGER
);

-- Indexes for fast episode retrieval
CREATE INDEX IF NOT EXISTS idx_agent_episodes_agent ON agent_episodes(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_episodes_user ON agent_episodes(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_episodes_session ON agent_episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_episodes_timestamp ON agent_episodes(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_agent_episodes_media ON agent_episodes(media_id) WHERE media_id IS NOT NULL;

-- IVFFlat index for vector similarity (approximate nearest neighbor)
-- Lists=100 for ~1M episodes, adjust based on scale
CREATE INDEX IF NOT EXISTS idx_agent_episodes_vector
    ON agent_episodes
    USING ivfflat (state_vector vector_cosine_ops)
    WITH (lists = 100);

-- RL policies: Q-values and contextual bandit parameters
CREATE TABLE IF NOT EXISTS rl_policies (
    policy_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    state_hash VARCHAR(64) NOT NULL,  -- Hash of state vector for fast lookup

    -- Thompson Sampling parameters
    q_values JSONB NOT NULL,  -- {"action_1": {"mean": 0.85, "variance": 0.02}, ...}
    action_counts JSONB NOT NULL DEFAULT '{}',  -- {"action_1": 42, "action_2": 38}
    total_visits INTEGER DEFAULT 1,

    -- Exploration parameters
    exploration_rate FLOAT DEFAULT 0.1,  -- ε-greedy
    ucb_constant FLOAT DEFAULT 1.414,  -- UCB confidence

    -- Performance metrics
    average_reward FLOAT DEFAULT 0.0,
    cumulative_reward FLOAT DEFAULT 0.0,
    last_updated TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(agent_id, user_id, state_hash)
);

-- Indexes for policy lookups (5ms target)
CREATE INDEX IF NOT EXISTS idx_rl_policies_agent_user ON rl_policies(agent_id, user_id);
CREATE INDEX IF NOT EXISTS idx_rl_policies_agent_state ON rl_policies(agent_id, state_hash);
CREATE INDEX IF NOT EXISTS idx_rl_policies_user ON rl_policies(user_id);
CREATE INDEX IF NOT EXISTS idx_rl_policies_updated ON rl_policies(last_updated DESC);

-- Learned patterns: semantic memory (user preferences, trends)
CREATE TABLE IF NOT EXISTS learned_patterns (
    pattern_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),  -- NULL for global patterns
    pattern_type VARCHAR(50) NOT NULL,  -- 'user_preference', 'seasonal_trend', 'genre_affinity', 'mood_correlation'
    pattern_data JSONB NOT NULL,  -- Type-specific pattern data

    -- Pattern metadata
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    evidence_count INTEGER DEFAULT 1,
    support_ratio FLOAT,  -- Proportion of data supporting this pattern

    -- Temporal validity
    valid_from TIMESTAMPTZ DEFAULT NOW(),
    valid_until TIMESTAMPTZ,  -- NULL = indefinite
    last_validated TIMESTAMPTZ DEFAULT NOW(),

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for pattern retrieval
CREATE INDEX IF NOT EXISTS idx_learned_patterns_agent ON learned_patterns(agent_id);
CREATE INDEX IF NOT EXISTS idx_learned_patterns_user ON learned_patterns(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_learned_patterns_type ON learned_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_learned_patterns_confidence ON learned_patterns(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_learned_patterns_valid ON learned_patterns(valid_from, valid_until);

-- User state snapshots: current user embeddings for cold-start
CREATE TABLE IF NOT EXISTS user_states (
    state_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    agent_id VARCHAR(255) NOT NULL,

    -- Behavioral embeddings
    state_embedding VECTOR(768),  -- Current user representation
    preference_distribution JSONB,  -- {"documentary": 0.35, "drama": 0.28, ...}
    mood_profile JSONB,  -- VAD scores: {"valence": 0.7, "arousal": 0.5, "dominance": 0.6}
    cultural_context JSONB,  -- {"language": "fr", "region": "FR", "themes": ["francophone"]}

    -- Session context
    device_type VARCHAR(50),
    location VARCHAR(100),
    time_of_day VARCHAR(20),  -- 'morning', 'afternoon', 'evening', 'night'
    session_start TIMESTAMPTZ DEFAULT NOW(),
    last_interaction TIMESTAMPTZ DEFAULT NOW(),

    -- Metadata
    interaction_count INTEGER DEFAULT 0,
    total_watch_time_sec INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(user_id, agent_id)
);

-- Indexes for user state lookups
CREATE INDEX IF NOT EXISTS idx_user_states_user ON user_states(user_id);
CREATE INDEX IF NOT EXISTS idx_user_states_agent ON user_states(agent_id);
CREATE INDEX IF NOT EXISTS idx_user_states_last_interaction ON user_states(last_interaction DESC);

-- Vector similarity index for user states (find similar users)
CREATE INDEX IF NOT EXISTS idx_user_states_embedding
    ON user_states
    USING ivfflat (state_embedding vector_cosine_ops)
    WITH (lists = 100);

-- Reward signals: raw feedback data before aggregation
CREATE TABLE IF NOT EXISTS reward_signals (
    signal_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    episode_id UUID REFERENCES agent_episodes(episode_id),
    user_id VARCHAR(255) NOT NULL,
    media_id VARCHAR(255) NOT NULL,

    -- Signal metrics
    signal_type VARCHAR(50) NOT NULL,  -- 'implicit', 'explicit', 'engagement'
    signal_value FLOAT NOT NULL,
    signal_strength FLOAT DEFAULT 1.0,  -- Confidence in signal (0-1)

    -- Engagement details
    watch_duration_sec INTEGER,
    media_duration_sec INTEGER,
    completion_rate FLOAT,

    -- Explicit feedback (optional)
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    liked BOOLEAN,
    shared BOOLEAN DEFAULT FALSE,

    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for reward aggregation
CREATE INDEX IF NOT EXISTS idx_reward_signals_episode ON reward_signals(episode_id);
CREATE INDEX IF NOT EXISTS idx_reward_signals_user ON reward_signals(user_id);
CREATE INDEX IF NOT EXISTS idx_reward_signals_media ON reward_signals(media_id);
CREATE INDEX IF NOT EXISTS idx_reward_signals_timestamp ON reward_signals(timestamp DESC);

-- Agent metadata: configuration and performance tracking
CREATE TABLE IF NOT EXISTS agent_metadata (
    agent_id VARCHAR(255) PRIMARY KEY,
    agent_type VARCHAR(50) NOT NULL,  -- 'recommendation', 'personalization', 'contextual_bandit'

    -- Configuration
    config JSONB NOT NULL DEFAULT '{}',
    hyperparameters JSONB NOT NULL DEFAULT '{}',

    -- Performance metrics
    total_episodes INTEGER DEFAULT 0,
    total_users INTEGER DEFAULT 0,
    average_reward FLOAT DEFAULT 0.0,
    reward_variance FLOAT DEFAULT 0.0,

    -- A/B testing
    version VARCHAR(50),
    experiment_id VARCHAR(255),

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Performance metrics rollup (hourly aggregates)
CREATE TABLE IF NOT EXISTS performance_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(255) NOT NULL,

    -- Time window
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,

    -- Aggregate metrics
    episodes_count INTEGER DEFAULT 0,
    unique_users INTEGER DEFAULT 0,
    avg_reward FLOAT DEFAULT 0.0,
    p50_reward FLOAT,
    p95_reward FLOAT,
    p99_reward FLOAT,

    -- Latency metrics (microseconds)
    avg_policy_lookup_us INTEGER,
    p99_policy_lookup_us INTEGER,

    -- Cache performance
    cache_hit_rate FLOAT,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(agent_id, window_start)
);

CREATE INDEX IF NOT EXISTS idx_performance_metrics_agent ON performance_metrics(agent_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_window ON performance_metrics(window_start DESC);

-- Partitioning strategy for episodes (by month)
-- Uncomment for production scale (millions of episodes)
-- CREATE TABLE agent_episodes_y2025m12 PARTITION OF agent_episodes
--     FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- Function: Calculate state hash from embedding (for fast policy lookup)
CREATE OR REPLACE FUNCTION calculate_state_hash(embedding VECTOR(768))
RETURNS VARCHAR(64) AS $$
DECLARE
    hash_input TEXT;
BEGIN
    -- Use first 16 dimensions as hash input (dimensionality reduction)
    hash_input := embedding::TEXT;
    RETURN encode(sha256(hash_input::bytea), 'hex');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function: Find similar episodes (semantic memory retrieval)
CREATE OR REPLACE FUNCTION find_similar_episodes(
    query_embedding VECTOR(768),
    query_agent_id VARCHAR(255),
    limit_count INTEGER DEFAULT 10
) RETURNS TABLE (
    episode_id UUID,
    user_id VARCHAR(255),
    action_taken JSONB,
    reward FLOAT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.episode_id,
        e.user_id,
        e.action_taken,
        e.reward,
        1 - (e.state_vector <=> query_embedding) AS similarity
    FROM agent_episodes e
    WHERE e.agent_id = query_agent_id
        AND e.state_vector IS NOT NULL
    ORDER BY e.state_vector <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Function: Update RL policy (Thompson Sampling)
CREATE OR REPLACE FUNCTION update_policy_thompson_sampling(
    p_agent_id VARCHAR(255),
    p_user_id VARCHAR(255),
    p_state_hash VARCHAR(64),
    p_action VARCHAR(255),
    p_reward FLOAT
) RETURNS VOID AS $$
DECLARE
    current_q JSONB;
    action_data JSONB;
    new_mean FLOAT;
    new_variance FLOAT;
    visit_count INTEGER;
BEGIN
    -- Get current Q-values or initialize
    SELECT q_values, total_visits INTO current_q, visit_count
    FROM rl_policies
    WHERE agent_id = p_agent_id
        AND user_id = p_user_id
        AND state_hash = p_state_hash;

    IF NOT FOUND THEN
        -- Initialize new policy
        INSERT INTO rl_policies (agent_id, user_id, state_hash, q_values, action_counts, total_visits, average_reward, cumulative_reward)
        VALUES (
            p_agent_id,
            p_user_id,
            p_state_hash,
            jsonb_build_object(p_action, jsonb_build_object('mean', p_reward, 'variance', 0.1)),
            jsonb_build_object(p_action, 1),
            1,
            p_reward,
            p_reward
        );
    ELSE
        -- Bayesian update of Q-value
        action_data := current_q -> p_action;

        IF action_data IS NULL THEN
            -- New action for this state
            current_q := current_q || jsonb_build_object(p_action, jsonb_build_object('mean', p_reward, 'variance', 0.1));

            UPDATE rl_policies
            SET q_values = current_q,
                action_counts = action_counts || jsonb_build_object(p_action, 1),
                total_visits = total_visits + 1,
                average_reward = (average_reward * total_visits + p_reward) / (total_visits + 1),
                cumulative_reward = cumulative_reward + p_reward,
                last_updated = NOW()
            WHERE agent_id = p_agent_id AND user_id = p_user_id AND state_hash = p_state_hash;
        ELSE
            -- Update existing action Q-value with Bayesian inference
            visit_count := COALESCE((action_data -> 'visits')::INTEGER, 1);
            new_mean := ((action_data ->> 'mean')::FLOAT * visit_count + p_reward) / (visit_count + 1);
            new_variance := 0.9 * (action_data ->> 'variance')::FLOAT + 0.1 * ABS(p_reward - new_mean);

            current_q := jsonb_set(
                jsonb_set(current_q, ARRAY[p_action, 'mean'], to_jsonb(new_mean)),
                ARRAY[p_action, 'variance'], to_jsonb(new_variance)
            );

            UPDATE rl_policies
            SET q_values = current_q,
                action_counts = jsonb_set(action_counts, ARRAY[p_action], to_jsonb(visit_count + 1)),
                total_visits = total_visits + 1,
                average_reward = (average_reward * total_visits + p_reward) / (total_visits + 1),
                cumulative_reward = cumulative_reward + p_reward,
                last_updated = NOW()
            WHERE agent_id = p_agent_id AND user_id = p_user_id AND state_hash = p_state_hash;
        END IF;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust for production)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO agentdb_app;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO agentdb_app;

-- Vacuum and analyze for optimal query performance
VACUUM ANALYZE agent_episodes;
VACUUM ANALYZE rl_policies;
VACUUM ANALYZE learned_patterns;
VACUUM ANALYZE user_states;

-- Comments for documentation
COMMENT ON TABLE agent_episodes IS 'Episodic memory: state-action-reward trajectories for RL training';
COMMENT ON TABLE rl_policies IS 'Procedural memory: learned RL policies (Q-values, contextual bandit parameters)';
COMMENT ON TABLE learned_patterns IS 'Semantic memory: generalized knowledge patterns extracted from episodes';
COMMENT ON TABLE user_states IS 'Current user representations for fast cold-start and personalization';
COMMENT ON TABLE reward_signals IS 'Raw feedback signals before aggregation into episodes';
COMMENT ON TABLE performance_metrics IS 'Hourly rollup of agent performance metrics';
