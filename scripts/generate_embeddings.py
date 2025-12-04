#!/usr/bin/env python3
"""
Generate multimodal embeddings for synthetic media dataset.
Produces 1024-dimensional embeddings combining text, visual, and audio features.
"""

import torch
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
import multiprocessing as mp
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate multimodal embeddings from media metadata."""

    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")

        # Visual embeddings (CLIP)
        logger.info("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        # Text embeddings (Sentence-BERT)
        logger.info("Loading Sentence-BERT model...")
        self.text_model = SentenceTransformer('all-mpnet-base-v2')

        # Move to GPU
        if self.device == 'cuda':
            self.clip_model = self.clip_model.cuda()
            self.text_model = self.text_model.cuda()

    def generate_multimodal_embeddings(self, batch):
        """
        Generate 1024-dim embeddings from metadata.

        Structure:
        - Text (384 dims): Title + Description
        - Visual (512 dims): Synthetic visual features from CLIP
        - Audio (128 dims): Mood-based audio features
        """
        embeddings = []

        for item in batch:
            # Text embedding from title + description (384 dims)
            text = f"{item['title']} {item['description']}"
            text_emb = self.text_model.encode(text, convert_to_numpy=True)

            # Generate synthetic visual embedding from metadata (512 dims)
            visual_prompt = self.generate_visual_prompt(item)
            inputs = self.clip_processor(text=[visual_prompt], return_tensors="pt")
            if self.device == 'cuda':
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                visual_emb = self.clip_model.get_text_features(**inputs).cpu().numpy()[0]

            # Generate synthetic audio embedding (128 dims)
            audio_emb = self.generate_audio_embedding(item)

            # Concatenate to 1024 dims
            full_embedding = np.concatenate([
                text_emb[:384],
                visual_emb[:512],
                audio_emb[:128]
            ])

            embeddings.append(full_embedding.astype(np.float16))

        return np.array(embeddings)

    def generate_visual_prompt(self, item):
        """Create CLIP prompt from metadata."""
        mood = item.get('mood', 'neutral')
        genre = item.get('genre', 'general')

        # Extract cultural context
        cultural_context = item.get('cultural_context', {})
        if isinstance(cultural_context, str):
            import json
            cultural_context = json.loads(cultural_context)

        region = cultural_context.get('region', 'international')

        return f"A {mood} {genre} scene with {region} aesthetics"

    def generate_audio_embedding(self, item):
        """Synthetic audio embedding based on mood/genre."""
        mood = item.get('mood', 'neutral')

        # Predetermined mood vectors with characteristic features
        mood_vectors = {
            "Uplifting": np.random.normal(0.7, 0.1, 128),
            "Melancholic": np.random.normal(-0.3, 0.2, 128),
            "Tense": np.random.normal(0.2, 0.3, 128),
            "Romantic": np.random.normal(0.1, 0.1, 128),
            "Comedic": np.random.normal(0.5, 0.2, 128)
        }

        base = mood_vectors.get(mood, np.random.normal(0, 0.2, 128))
        noise = np.random.normal(0, 0.05, 128)

        return (base + noise).astype(np.float16)


def process_parquet_file(file_path, output_path, device='cuda'):
    """Process parquet file and add embeddings."""
    logger.info(f"Processing {file_path}")

    try:
        table = pq.read_table(file_path)
        df = table.to_pandas()

        generator = EmbeddingGenerator(device=device)
        batch_size = 1000

        embeddings = []
        for i in tqdm(range(0, len(df), batch_size), desc=f"Generating embeddings"):
            batch = df.iloc[i:i+batch_size].to_dict('records')
            batch_embeddings = generator.generate_multimodal_embeddings(batch)
            embeddings.extend(batch_embeddings)

        # Add embeddings to dataframe
        df['embedding'] = embeddings

        # Save with embeddings
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(output_path), compression='snappy')

        logger.info(f"Saved embeddings to {output_path}")

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        raise


def process_file_wrapper(args):
    """Wrapper for multiprocessing."""
    file_path, output_path, device = args
    process_parquet_file(Path(file_path), Path(output_path), device)


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for synthetic media dataset")
    parser.add_argument("--input-dir", type=str, default="/data/synthetic/tv5_media/media",
                        help="Input directory with parquet files")
    parser.add_argument("--output-dir", type=str, default="/home/devuser/workspace/hackathon-tv5/data/embedded",
                        help="Output directory for embedded data")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--device", type=str, default='cuda',
                        choices=['cuda', 'cpu'], help="Device to use")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all parquet files
    parquet_files = sorted(input_dir.glob("**/*.parquet"))
    logger.info(f"Found {len(parquet_files)} parquet files")

    if not parquet_files:
        logger.warning("No parquet files found. Creating sample data for testing...")
        # Create sample data
        sample_data = {
            'content_id': list(range(100)),
            'title': [f"Sample Title {i}" for i in range(100)],
            'description': [f"This is a sample description for item {i}" for i in range(100)],
            'genre': ['Action'] * 50 + ['Drama'] * 50,
            'mood': ['Uplifting'] * 25 + ['Tense'] * 25 + ['Romantic'] * 25 + ['Comedic'] * 25,
            'cultural_context': ['{"region": "Europe", "language": "French"}'] * 100,
            'popularity': np.random.zipf(1.5, 100) / 100,
            'release_year': np.random.randint(1990, 2024, 100)
        }

        import pandas as pd
        df = pd.DataFrame(sample_data)
        sample_file = input_dir / "sample_data.parquet"
        sample_file.parent.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(sample_file), compression='snappy')

        parquet_files = [sample_file]
        logger.info(f"Created sample data at {sample_file}")

    # Prepare tasks
    tasks = [
        (str(f), str(output_dir / f.relative_to(input_dir)), args.device)
        for f in parquet_files
    ]

    # Process in parallel
    if args.num_workers > 1:
        with mp.Pool(processes=args.num_workers) as pool:
            pool.map(process_file_wrapper, tasks)
    else:
        for task in tasks:
            process_file_wrapper(task)

    logger.info("Embedding generation complete!")


if __name__ == "__main__":
    main()
