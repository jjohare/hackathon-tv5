/// User profile and psychographic modeling
///
/// Represents users with:
/// - Behavioral history (watch patterns)
/// - Psychographic states (mood, preferences)
/// - Taste clusters (collaborative grouping)
/// - Contextual information (device, time, location)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique user identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UserId(pub Uuid);

impl UserId {
    /// Create new random user ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Parse from string
    pub fn from_string(s: &str) -> Result<Self, uuid::Error> {
        Ok(Self(Uuid::parse_str(s)?))
    }

    /// Get as string
    pub fn as_str(&self) -> String {
        self.0.to_string()
    }
}

impl Default for UserId {
    fn default() -> Self {
        Self::new()
    }
}

/// User profile with learned preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    /// Unique user identifier
    pub user_id: UserId,

    /// User embedding (1024-dim, learned from interactions)
    pub user_embedding: Vec<f32>,

    /// Watch history (recent 100 interactions)
    pub watch_history: Vec<Interaction>,

    /// Explicit preferences
    pub preferences: UserPreferences,

    /// Current psychographic state
    pub current_state: Option<PsychographicState>,

    /// Assigned taste cluster
    pub taste_cluster: Option<TasteCluster>,

    /// Tolerance levels for various attributes
    pub tolerances: ToleranceLevels,

    /// Profile metadata
    pub metadata: UserMetadata,

    /// Last updated timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl UserProfile {
    /// Create new user profile with defaults
    pub fn new(user_id: UserId) -> Self {
        Self {
            user_id,
            user_embedding: vec![0.0; 1024],
            watch_history: Vec::new(),
            preferences: UserPreferences::default(),
            current_state: None,
            taste_cluster: None,
            tolerances: ToleranceLevels::default(),
            metadata: UserMetadata::default(),
            last_updated: chrono::Utc::now(),
        }
    }

    /// Update user embedding from recent interactions
    pub fn update_embedding(&mut self, content_embeddings: &[Vec<f32>]) {
        if content_embeddings.is_empty() {
            return;
        }

        // Weighted average of watched content (recent = higher weight)
        let mut new_embedding = vec![0.0; 1024];
        let mut total_weight = 0.0;

        for (i, embedding) in content_embeddings.iter().enumerate() {
            let weight = ((i + 1) as f32 / content_embeddings.len() as f32).powi(2);
            total_weight += weight;

            for (j, value) in embedding.iter().enumerate() {
                if j < new_embedding.len() {
                    new_embedding[j] += value * weight;
                }
            }
        }

        for value in &mut new_embedding {
            *value /= total_weight;
        }

        self.user_embedding = new_embedding;
        self.last_updated = chrono::Utc::now();
    }

    /// Add interaction to history
    pub fn add_interaction(&mut self, interaction: Interaction) {
        self.watch_history.push(interaction);

        // Keep only recent 100 interactions
        if self.watch_history.len() > 100 {
            self.watch_history.remove(0);
        }

        self.last_updated = chrono::Utc::now();
    }

    /// Get average watch completion rate
    pub fn avg_completion_rate(&self) -> f32 {
        if self.watch_history.is_empty() {
            return 0.0;
        }

        let sum: f32 = self.watch_history
            .iter()
            .filter_map(|i| i.watch_completion_rate)
            .sum();

        sum / self.watch_history.len() as f32
    }
}

/// User interaction record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interaction {
    /// Content ID
    pub content_id: String,

    /// Interaction type
    pub interaction_type: InteractionType,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Watch duration (seconds)
    pub watch_duration: Option<u32>,

    /// Total content duration (seconds)
    pub content_duration: Option<u32>,

    /// Watch completion rate (0.0-1.0)
    pub watch_completion_rate: Option<f32>,

    /// Explicit rating (1-5 stars)
    pub rating: Option<u8>,

    /// Device type
    pub device: DeviceType,

    /// Viewing context
    pub context: Option<ViewingContext>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractionType {
    /// User clicked on content
    Click,
    /// User started watching
    Watch,
    /// User skipped/abandoned
    Skip,
    /// User completed watching
    Complete,
    /// User provided explicit rating
    Rate,
    /// User added to watchlist
    Watchlist,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    Mobile,
    Tablet,
    Desktop,
    TV,
    Console,
}

/// User preferences (explicit)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    /// Preferred genres
    pub favorite_genres: Vec<String>,

    /// Disliked genres
    pub disliked_genres: Vec<String>,

    /// Preferred languages
    pub languages: Vec<String>,

    /// Subtitle preferences
    pub subtitle_preference: SubtitlePreference,

    /// Content rating restrictions
    pub max_content_rating: String,

    /// Custom preferences
    pub custom: HashMap<String, String>,
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            favorite_genres: Vec::new(),
            disliked_genres: Vec::new(),
            languages: vec!["en".to_string()],
            subtitle_preference: SubtitlePreference::Optional,
            max_content_rating: "R".to_string(),
            custom: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubtitlePreference {
    /// Never show subtitles
    Never,
    /// Optional subtitles
    Optional,
    /// Always require subtitles
    Always,
}

/// Psychographic state (current mood/mindset)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsychographicState {
    /// State label
    pub state: StateType,

    /// State intensity (0.0-1.0)
    pub intensity: f32,

    /// How long in this state
    pub duration: chrono::Duration,

    /// State embedding (128-dim)
    pub state_embedding: Option<Vec<f32>>,

    /// Detected timestamp
    pub detected_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateType {
    /// Seeking comfort/familiarity
    SeekingComfort,
    /// Seeking mental challenge
    SeekingChallenge,
    /// Nostalgic mood
    Nostalgic,
    /// Energetic/excited
    Energetic,
    /// Relaxed/winding down
    Relaxed,
    /// Stressed/anxious
    Stressed,
    /// Social viewing mode
    Social,
    /// Solo focused viewing
    Focused,
}

impl PsychographicState {
    /// Create new state
    pub fn new(state: StateType, intensity: f32) -> Self {
        Self {
            state,
            intensity: intensity.clamp(0.0, 1.0),
            duration: chrono::Duration::zero(),
            state_embedding: None,
            detected_at: chrono::Utc::now(),
        }
    }

    /// Check if state is still valid (not too old)
    pub fn is_current(&self) -> bool {
        let elapsed = chrono::Utc::now() - self.detected_at;
        elapsed < chrono::Duration::hours(2)
    }
}

/// Taste cluster (collaborative filtering group)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TasteCluster {
    /// Cluster ID
    pub cluster_id: u32,

    /// Cluster centroid (1024-dim)
    pub centroid: Vec<f32>,

    /// Number of users in cluster
    pub size: usize,

    /// Cluster characteristics (tags)
    pub characteristics: Vec<String>,

    /// Typical genres for this cluster
    pub typical_genres: Vec<String>,
}

impl TasteCluster {
    /// Create new cluster
    pub fn new(cluster_id: u32, centroid: Vec<f32>) -> Self {
        Self {
            cluster_id,
            centroid,
            size: 0,
            characteristics: Vec::new(),
            typical_genres: Vec::new(),
        }
    }

    /// Compute distance from cluster centroid
    pub fn distance_from_centroid(&self, user_embedding: &[f32]) -> f32 {
        if self.centroid.len() != user_embedding.len() {
            return f32::INFINITY;
        }

        self.centroid
            .iter()
            .zip(user_embedding)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// Tolerance levels for content attributes
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ToleranceLevels {
    /// Tolerance for violence (0.0 = none, 1.0 = high)
    pub violence: f32,

    /// Tolerance for sexual content
    pub sexual_content: f32,

    /// Tolerance for profanity
    pub profanity: f32,

    /// Tolerance for complexity (reading level)
    pub complexity: f32,

    /// Tolerance for slow pacing
    pub slow_pacing: f32,

    /// Tolerance for subtitles
    pub subtitles: f32,
}

impl Default for ToleranceLevels {
    fn default() -> Self {
        Self {
            violence: 0.5,
            sexual_content: 0.5,
            profanity: 0.5,
            complexity: 0.5,
            slow_pacing: 0.5,
            subtitles: 0.5,
        }
    }
}

/// Viewing context (current situation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewingContext {
    /// Time of day
    pub time_of_day: TimeOfDay,

    /// Day of week
    pub day_of_week: chrono::Weekday,

    /// Device being used
    pub device: DeviceType,

    /// Network speed (mbps)
    pub network_speed: Option<f32>,

    /// Location (city, region)
    pub location: Option<String>,

    /// Social setting
    pub social_setting: SocialSetting,

    /// Ambient conditions
    pub ambient: AmbientConditions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeOfDay {
    Morning,    // 6am-12pm
    Afternoon,  // 12pm-6pm
    Evening,    // 6pm-10pm
    Night,      // 10pm-6am
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SocialSetting {
    /// Watching alone
    Solo,
    /// Date night
    DateNight,
    /// Family viewing
    Family,
    /// Friends gathering
    Friends,
    /// Party/large group
    Party,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientConditions {
    /// Ambient light level (0.0 = dark, 1.0 = bright)
    pub light_level: f32,

    /// Ambient noise level (0.0 = quiet, 1.0 = loud)
    pub noise_level: f32,
}

/// User metadata (non-preference information)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMetadata {
    /// Account creation date
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Total watch time (hours)
    pub total_watch_hours: f32,

    /// Number of ratings provided
    pub ratings_count: u32,

    /// Account tier (free, premium, etc.)
    pub account_tier: String,

    /// Geographic region
    pub region: Option<String>,
}

impl Default for UserMetadata {
    fn default() -> Self {
        Self {
            created_at: chrono::Utc::now(),
            total_watch_hours: 0.0,
            ratings_count: 0,
            account_tier: "free".to_string(),
            region: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_id_creation() {
        let id1 = UserId::new();
        let id2 = UserId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_user_profile_interaction() {
        let mut profile = UserProfile::new(UserId::new());

        let interaction = Interaction {
            content_id: "film123".to_string(),
            interaction_type: InteractionType::Watch,
            timestamp: chrono::Utc::now(),
            watch_duration: Some(3600),
            content_duration: Some(7200),
            watch_completion_rate: Some(0.5),
            rating: None,
            device: DeviceType::TV,
            context: None,
        };

        profile.add_interaction(interaction);
        assert_eq!(profile.watch_history.len(), 1);
    }

    #[test]
    fn test_psychographic_state_expiry() {
        let state = PsychographicState::new(StateType::Relaxed, 0.8);
        assert!(state.is_current());
    }

    #[test]
    fn test_taste_cluster_distance() {
        let cluster = TasteCluster::new(1, vec![0.5, 0.5, 0.5]);
        let user_embedding = vec![0.6, 0.6, 0.6];
        let distance = cluster.distance_from_centroid(&user_embedding);
        assert!(distance > 0.0 && distance < 1.0);
    }
}
