use rand::Rng;
use std::collections::HashMap;

pub struct Policy {
    pub agent_id: String,
    pub preferences: HashMap<String, f32>,
}

const GENRE_PREFERENCES: &[&str] = &[
    "genre_scifi",
    "genre_action",
    "genre_romance",
    "genre_comedy",
    "genre_drama",
    "genre_horror",
    "genre_thriller",
];

pub fn generate_random_policy(user_id: &str) -> Policy {
    let mut rng = rand::thread_rng();

    let mut preferences = HashMap::new();

    // Generate random preferences for each genre
    for genre_pref in GENRE_PREFERENCES {
        let score = rng.gen_range(0.0..1.0);
        preferences.insert(genre_pref.to_string(), score);
    }

    Policy {
        agent_id: user_id.to_string(),
        preferences,
    }
}

pub fn generate_scifi_lover_policy(user_id: &str) -> Policy {
    let mut preferences = HashMap::new();

    preferences.insert("genre_scifi".to_string(), 0.95);
    preferences.insert("genre_action".to_string(), 0.75);
    preferences.insert("genre_thriller".to_string(), 0.60);
    preferences.insert("genre_romance".to_string(), 0.15);
    preferences.insert("genre_comedy".to_string(), 0.30);

    Policy {
        agent_id: user_id.to_string(),
        preferences,
    }
}

pub fn generate_romance_lover_policy(user_id: &str) -> Policy {
    let mut preferences = HashMap::new();

    preferences.insert("genre_romance".to_string(), 0.95);
    preferences.insert("genre_comedy".to_string(), 0.80);
    preferences.insert("genre_drama".to_string(), 0.65);
    preferences.insert("genre_scifi".to_string(), 0.10);
    preferences.insert("genre_horror".to_string(), 0.05);

    Policy {
        agent_id: user_id.to_string(),
        preferences,
    }
}

pub fn generate_diverse_policies(count: usize) -> Vec<Policy> {
    (0..count)
        .map(|i| generate_random_policy(&format!("user_{}", i)))
        .collect()
}
