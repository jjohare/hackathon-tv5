#!/usr/bin/env python3
"""
Document Scoring System for Archival Decisions
Scans all .md files and scores them on novelty, relevance, quality, and integration.
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

class DocumentScorer:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.exclude_patterns = ['node_modules', '.git', 'venv', '__pycache__']
        self.main_readme = None
        self.main_docs = set()
        self.scores = []

        # Working doc indicators
        self.working_indicators = [
            'DRAFT', 'WIP', 'WORKING', 'TEMP', 'TODO', 'FIXME',
            'temp-', 'working-', 'scratch', 'notes'
        ]

        # Main documentation files (keep regardless)
        self.main_doc_files = {
            'README.md', 'ARCHITECTURE.md', 'CONTRIBUTING.md',
            'LICENSE.md', 'CHANGELOG.md', 'API.md'
        }

    def should_exclude(self, path: str) -> bool:
        """Check if path should be excluded"""
        for pattern in self.exclude_patterns:
            if pattern in path:
                return True
        return False

    def is_working_doc(self, path: str, content: str) -> bool:
        """Detect if this is a working/temporary document"""
        path_lower = path.lower()
        content_sample = content[:500].upper() if content else ""

        for indicator in self.working_indicators:
            if indicator.lower() in path_lower:
                return True
            if indicator in content_sample:
                return True

        return False

    def scan_files(self) -> List[Dict]:
        """Scan all markdown files"""
        files = []
        for md_file in self.root_path.rglob('*.md'):
            if self.should_exclude(str(md_file)):
                continue

            try:
                stat = md_file.stat()
                rel_path = md_file.relative_to(self.root_path)

                try:
                    with open(md_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                except:
                    content = ""

                files.append({
                    'path': str(rel_path),
                    'abs_path': str(md_file),
                    'size': stat.st_size,
                    'modified': stat.st_mtime,
                    'content': content,
                    'is_working': self.is_working_doc(str(rel_path), content)
                })
            except Exception as e:
                print(f"Error scanning {md_file}: {e}")

        return files

    def score_novelty(self, doc: Dict) -> Tuple[int, str]:
        """Score document novelty (0-10)"""
        content = doc['content']
        path = doc['path']

        # Main README gets baseline
        if path == 'README.md':
            self.main_readme = content
            return 10, "Main README - reference document"

        # No content
        if not content or len(content) < 100:
            return 0, "Empty or minimal content"

        # Check for unique technical content
        unique_markers = [
            r'```\w+\n',  # Code blocks
            r'##\s+\w+',  # Headers
            r'\|\s+\w+\s+\|',  # Tables
            r'https?://',  # URLs
        ]

        unique_count = sum(len(re.findall(pattern, content)) for pattern in unique_markers)

        # Working docs get low novelty
        if doc['is_working']:
            return min(2, unique_count // 20), "Working document"

        # Check for duplicated content from main README
        if self.main_readme:
            overlap_ratio = self._calculate_overlap(content, self.main_readme)
            if overlap_ratio > 0.8:
                return 2, f"High overlap with main README ({overlap_ratio:.0%})"
            elif overlap_ratio > 0.5:
                return 4, f"Moderate overlap with main README ({overlap_ratio:.0%})"

        # Score based on unique content indicators
        if unique_count > 100:
            return 9, f"Rich unique content ({unique_count} technical elements)"
        elif unique_count > 50:
            return 7, f"Good unique content ({unique_count} technical elements)"
        elif unique_count > 20:
            return 5, f"Moderate unique content ({unique_count} technical elements)"
        else:
            return 3, f"Limited unique content ({unique_count} technical elements)"

    def score_relevance(self, doc: Dict) -> Tuple[int, str]:
        """Score document relevance to current project (0-10)"""
        content = doc['content'].lower()
        path = doc['path'].lower()

        # Main documentation always relevant
        if any(main in doc['path'] for main in self.main_doc_files):
            return 10, "Main project documentation"

        # Check for project-specific keywords
        relevant_keywords = [
            'tv5', 'media gateway', 'hackathon', 'cuda', 'gpu',
            'semantic search', 'ontology', 'neo4j', 'milvus',
            'tensor core', 'sssp', 'dijkstra', 'agentdb'
        ]

        keyword_matches = sum(1 for kw in relevant_keywords if kw in content)

        # Obsolete indicators
        obsolete_keywords = [
            'deprecated', 'obsolete', 'old version', 'archived',
            'no longer used', 'superseded'
        ]

        if any(kw in content for kw in obsolete_keywords):
            return 1, "Marked as deprecated/obsolete"

        # Working docs less relevant
        if doc['is_working']:
            return 3, "Working document - transitional"

        # temp- directories
        if 'temp-' in path:
            return 2, "Temporary directory content"

        # Archive directories
        if 'archive' in path or 'old' in path:
            return 1, "Already in archive location"

        # Score based on keyword matches
        if keyword_matches >= 5:
            return 9, f"Highly relevant ({keyword_matches} key topics)"
        elif keyword_matches >= 3:
            return 7, f"Relevant ({keyword_matches} key topics)"
        elif keyword_matches >= 1:
            return 5, f"Moderately relevant ({keyword_matches} key topics)"
        else:
            return 2, "Minimal project relevance"

    def score_quality(self, doc: Dict) -> Tuple[int, str]:
        """Score document quality (0-10)"""
        content = doc['content']

        # Empty documents
        if not content or len(content) < 100:
            return 0, "Empty or minimal content"

        # Check for professional markers
        has_title = bool(re.search(r'^#\s+\w+', content, re.MULTILINE))
        has_sections = len(re.findall(r'^##\s+', content, re.MULTILINE))
        has_code = len(re.findall(r'```', content))
        has_lists = len(re.findall(r'^\s*[-*]\s+', content, re.MULTILINE))

        # Working doc markers (rough quality)
        if doc['is_working']:
            return 3, "Working document - rough quality"

        # Calculate quality score
        quality_score = 0
        quality_reasons = []

        if has_title:
            quality_score += 2
            quality_reasons.append("proper title")

        if has_sections >= 5:
            quality_score += 3
            quality_reasons.append(f"{has_sections} sections")
        elif has_sections >= 2:
            quality_score += 2
            quality_reasons.append(f"{has_sections} sections")

        if has_code >= 3:
            quality_score += 2
            quality_reasons.append(f"{has_code} code blocks")
        elif has_code >= 1:
            quality_score += 1
            quality_reasons.append(f"{has_code} code blocks")

        if has_lists >= 5:
            quality_score += 2
            quality_reasons.append("well-structured lists")
        elif has_lists >= 2:
            quality_score += 1

        if len(content) > 5000:
            quality_score += 1
            quality_reasons.append("comprehensive")

        return min(10, quality_score), f"Professional: {', '.join(quality_reasons)}"

    def score_integration(self, doc: Dict) -> Tuple[int, str]:
        """Score integration status (0=integrated, 10=standalone)"""
        path = doc['path']

        # Check if content appears in main docs
        # Higher score = more standalone (less integrated)

        # Summary/report documents are usually standalone
        if any(x in path.lower() for x in ['summary', 'report', 'status', 'complete']):
            return 9, "Summary/report document - standalone"

        # Implementation details usually standalone
        if 'implementation' in path.lower():
            return 8, "Implementation details - standalone"

        # Guides are usually well integrated
        if 'guide' in path.lower() or 'tutorial' in path.lower():
            return 3, "Guide/tutorial - likely referenced"

        # API docs are critical
        if 'api' in path.lower():
            return 2, "API documentation - critical reference"

        # Archive reports
        if '.doc-alignment-reports' in path:
            return 10, "Archive report - fully standalone"

        # Reference docs
        if 'reference' in path.lower() or 'quick' in path.lower():
            return 5, "Reference document - moderate integration"

        return 6, "Standard document - unclear integration"

    def calculate_total_score(self, novelty: int, relevance: int, quality: int, integration: int) -> int:
        """Calculate weighted total score"""
        # Weights: novelty (30%), relevance (30%), quality (25%), integration (15%)
        return int(
            novelty * 0.30 +
            relevance * 0.30 +
            quality * 0.25 +
            integration * 0.15
        )

    def categorize_document(self, total_score: int, doc: Dict) -> str:
        """Categorize document based on score"""
        if doc['is_working']:
            if total_score < 3:
                return 'DELETE'
            else:
                return 'ARCHIVE'

        if total_score >= 7:
            return 'KEEP'
        elif total_score >= 4:
            return 'EXTRACT'
        elif total_score >= 2:
            return 'ARCHIVE'
        else:
            return 'DELETE'

    def _calculate_overlap(self, content1: str, content2: str) -> float:
        """Calculate content overlap ratio"""
        if not content1 or not content2:
            return 0.0

        # Simple word-based overlap
        words1 = set(re.findall(r'\w+', content1.lower()))
        words2 = set(re.findall(r'\w+', content2.lower()))

        if not words1:
            return 0.0

        overlap = len(words1 & words2)
        return overlap / len(words1)

    def score_document(self, doc: Dict) -> Dict:
        """Score a single document"""
        novelty, novelty_reason = self.score_novelty(doc)
        relevance, relevance_reason = self.score_relevance(doc)
        quality, quality_reason = self.score_quality(doc)
        integration, integration_reason = self.score_integration(doc)

        total = self.calculate_total_score(novelty, relevance, quality, integration)
        category = self.categorize_document(total, doc)

        return {
            'path': doc['path'],
            'size_kb': round(doc['size'] / 1024, 1),
            'modified': datetime.fromtimestamp(doc['modified']).isoformat(),
            'is_working': doc['is_working'],
            'scores': {
                'novelty': novelty,
                'relevance': relevance,
                'quality': quality,
                'integration': integration,
                'total': total
            },
            'reasons': {
                'novelty': novelty_reason,
                'relevance': relevance_reason,
                'quality': quality_reason,
                'integration': integration_reason
            },
            'category': category
        }

    def run_analysis(self) -> Dict:
        """Run complete analysis"""
        print("Scanning markdown files...")
        files = self.scan_files()
        print(f"Found {len(files)} markdown files")

        print("Scoring documents...")
        scored_docs = []
        for i, doc in enumerate(files, 1):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(files)} files...")
            scored = self.score_document(doc)
            scored_docs.append(scored)

        # Generate statistics
        stats = self._generate_statistics(scored_docs)

        return {
            'scan_date': datetime.now().isoformat(),
            'total_files': len(scored_docs),
            'statistics': stats,
            'documents': scored_docs
        }

    def _generate_statistics(self, docs: List[Dict]) -> Dict:
        """Generate summary statistics"""
        categories = defaultdict(int)
        total_size_kb = 0
        score_distribution = defaultdict(int)

        for doc in docs:
            categories[doc['category']] += 1
            total_size_kb += doc['size_kb']
            score_range = f"{(doc['scores']['total'] // 2) * 2}-{(doc['scores']['total'] // 2) * 2 + 1}"
            score_distribution[score_range] += 1

        return {
            'total_size_kb': round(total_size_kb, 1),
            'categories': dict(categories),
            'score_distribution': dict(sorted(score_distribution.items())),
            'working_docs': sum(1 for d in docs if d['is_working']),
            'avg_score': round(sum(d['scores']['total'] for d in docs) / len(docs), 1) if docs else 0
        }

def main():
    root = Path('/home/devuser/workspace/hackathon-tv5')
    scorer = DocumentScorer(root)

    results = scorer.run_analysis()

    # Save JSON results
    output_dir = root / 'docs' / '.doc-alignment-reports'
    output_dir.mkdir(parents=True, exist_ok=True)

    json_file = output_dir / 'document-scores.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Analysis complete!")
    print(f"   Total files: {results['total_files']}")
    print(f"   Total size: {results['statistics']['total_size_kb']} KB")
    print(f"\nCategories:")
    for cat, count in sorted(results['statistics']['categories'].items()):
        print(f"   {cat}: {count} files")
    print(f"\n   JSON saved to: {json_file}")

    return results

if __name__ == '__main__':
    main()
