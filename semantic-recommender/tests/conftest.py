#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for test suite
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def system():
    """Shared GPU hyper-personalization system for all tests"""
    try:
        from scripts.utils.gpu_hyper_personalization import GPUHyperPersonalization
        return GPUHyperPersonalization(use_tensorrt=True)
    except ImportError:
        pytest.skip("GPU Hyper-Personalization not available")


@pytest.fixture
def sample_queries():
    """Sample queries for testing"""
    return [
        "action movies with explosions",
        "romantic comedy",
        "sci-fi thriller",
        "psychological horror",
        "animated family film"
    ]


@pytest.fixture
def sample_contexts():
    """Sample contexts for personalization testing"""
    return {
        'evening_solo': {
            'time_of_day': [0.1, 0.2, 0.7],
            'genre_prefs': [0.6, 0.3, 0.1],
            'social_signal': [1.0, 0.0]
        },
        'afternoon_group': {
            'time_of_day': [0.2, 0.7, 0.1],
            'genre_prefs': [0.3, 0.5, 0.2],
            'social_signal': [0.2, 0.8]
        }
    }
