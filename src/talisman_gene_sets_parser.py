#!/usr/bin/env python3
"""
Talisman Gene Sets Parser

Comprehensive parser for gene set data from talisman-paper/genesets/human/
Handles HALLMARK pathways, bicluster sets, custom pathways, and other curated gene sets.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import re

# Configure logging
logger = logging.getLogger(__name__)

class TalismanGeneSetsParser:
    """Parser for talisman paper gene sets with comprehensive data type handling"""
    
    def __init__(self, data_dir: str = None):
        # Set default data directory
        if data_dir is None:
            # Try both possible locations
            possible_dirs = [
                "talisman-paper/genesets/human",
                "llm_evaluation_for_gene_set_interpretation/data/talisman-paper/genesets/human"
            ]
            for possible_dir in possible_dirs:
                if os.path.exists(possible_dir):
                    data_dir = possible_dir
                    break
            
            if data_dir is None:
                raise FileNotFoundError("Could not find talisman gene sets directory")
        
        self.data_dir = Path(data_dir)
        self.parsed_data = {
            'hallmark_sets': {},
            'bicluster_sets': {},  
            'pathway_sets': {},
            'go_custom_sets': {},
            'disease_sets': {},
            'other_sets': {}
        }
        self.parsing_stats = defaultdict(int)
        
        logger.info(f"Initialized TalismanGeneSetsParser with data directory: {self.data_dir}")
        
    def parse_all_gene_sets(self) -> Dict[str, Dict]:
        """Parse all talisman gene sets with comprehensive data type handling"""
        logger.info("Starting comprehensive parsing of talisman gene sets...")
        
        try:
            # Get all YAML and JSON files
            all_files = list(self.data_dir.glob("*.yaml")) + list(self.data_dir.glob("*.json"))
            logger.info(f"Found {len(all_files)} gene set files")
            
            # Track processed files to avoid duplicates
            processed_base_names = set()
            
            # Process each file
            for file_path in sorted(all_files):
                try:
                    base_name = file_path.stem  # filename without extension
                    
                    # Skip if we already processed this base name (prefer YAML over JSON)
                    if base_name in processed_base_names:
                        continue
                    
                    # Check if both YAML and JSON exist for this base name
                    yaml_path = file_path.parent / f"{base_name}.yaml"
                    json_path = file_path.parent / f"{base_name}.json"
                    
                    # Prefer YAML format when both exist
                    if yaml_path.exists() and json_path.exists():
                        chosen_path = yaml_path
                        logger.debug(f"Both formats exist for {base_name}, choosing YAML")
                    else:
                        chosen_path = file_path
                    
                    # Parse the file
                    self._parse_single_gene_set(chosen_path)
                    processed_base_names.add(base_name)
                    self.parsing_stats['files_processed'] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to parse {file_path.name}: {str(e)}")
                    self.parsing_stats['files_failed'] += 1
            
            # Generate parsing statistics
            self._generate_parsing_statistics()
            
            logger.info(f"Talisman gene sets parsing completed: {self.parsing_stats['files_processed']} files processed")
            return self.parsed_data
            
        except Exception as e:
            logger.error(f"Failed to parse talisman gene sets: {str(e)}")
            raise
    
    def _parse_single_gene_set(self, file_path: Path) -> None:
        """Parse a single gene set file"""
        content = self._load_file_content(file_path)
        if content is None:
            return
        
        # Extract metadata and genes
        gene_set_data = self._extract_gene_set_data(file_path.name, content)
        if gene_set_data is None:
            return
        
        # Classify and store by data type
        data_type = self._classify_gene_set_type(file_path.name, gene_set_data)
        
        # Store in appropriate category
        if data_type == 'hallmark':
            self.parsed_data['hallmark_sets'][gene_set_data['id']] = gene_set_data
            self.parsing_stats['hallmark_sets'] += 1
        elif data_type == 'bicluster':
            self.parsed_data['bicluster_sets'][gene_set_data['id']] = gene_set_data
            self.parsing_stats['bicluster_sets'] += 1
        elif data_type == 'pathway':
            self.parsed_data['pathway_sets'][gene_set_data['id']] = gene_set_data
            self.parsing_stats['pathway_sets'] += 1
        elif data_type == 'go_custom':
            self.parsed_data['go_custom_sets'][gene_set_data['id']] = gene_set_data
            self.parsing_stats['go_custom_sets'] += 1
        elif data_type == 'disease':
            self.parsed_data['disease_sets'][gene_set_data['id']] = gene_set_data
            self.parsing_stats['disease_sets'] += 1
        else:
            self.parsed_data['other_sets'][gene_set_data['id']] = gene_set_data
            self.parsing_stats['other_sets'] += 1
        
        self.parsing_stats['total_gene_sets'] += 1
        self.parsing_stats['total_genes'] += len(gene_set_data['genes'])
    
    def _load_file_content(self, file_path: Path) -> Any:
        """Load YAML or JSON file content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.yaml':
                    return yaml.safe_load(f)
                elif file_path.suffix == '.json':
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {file_path.name}: {str(e)}")
            return None
    
    def _extract_gene_set_data(self, filename: str, content: Any) -> Optional[Dict[str, Any]]:
        """Extract standardized gene set data from file content"""
        if not isinstance(content, dict):
            logger.warning(f"Invalid content format in {filename}")
            return None
        
        gene_set_data = {
            'id': Path(filename).stem,  # filename without extension
            'filename': filename,
            'name': None,
            'description': None,
            'genes': [],
            'gene_count': 0,
            'metadata': {}
        }
        
        # Handle different data formats
        if 'gene_symbols' in content:
            # Standard YAML format with gene symbols
            gene_set_data['name'] = content.get('name', gene_set_data['id'])
            gene_set_data['description'] = content.get('description')
            gene_set_data['genes'] = content.get('gene_symbols', [])
            gene_set_data['metadata'] = {
                'taxon': content.get('taxon'),
                'source_format': 'yaml_standard'
            }
            
            # Handle gene_ids if present
            if 'gene_ids' in content and content['gene_ids']:
                gene_set_data['metadata']['gene_ids'] = content['gene_ids'][:10]  # Sample first 10
                gene_set_data['metadata']['id_type'] = self._extract_id_type(content['gene_ids'])
        
        elif 'gene_ids' in content and content['gene_ids']:
            # YAML format with only gene IDs (need to use IDs as placeholder for genes)
            gene_set_data['name'] = content.get('name', gene_set_data['id'])
            gene_set_data['description'] = content.get('description')
            # Use gene_ids as genes (will be HGNC IDs like HGNC:1234)
            gene_set_data['genes'] = content['gene_ids']
            gene_set_data['metadata'] = {
                'taxon': content.get('taxon'),
                'source_format': 'yaml_ids_only',
                'id_type': self._extract_id_type(content['gene_ids']),
                'note': 'Using gene IDs as gene identifiers'
            }
        
        elif len(content) == 1:
            # MSigDB JSON format
            set_name = list(content.keys())[0]
            set_data = content[set_name]
            
            gene_set_data['name'] = set_name
            gene_set_data['description'] = set_data.get('exactSource')
            gene_set_data['genes'] = set_data.get('geneSymbols', [])
            gene_set_data['metadata'] = {
                'systematic_name': set_data.get('systematicName'),
                'pmid': set_data.get('pmid'),
                'collection': set_data.get('collection'),
                'msigdb_url': set_data.get('msigdbURL'),
                'source_format': 'msigdb_json'
            }
        
        else:
            logger.warning(f"Unrecognized format in {filename}")
            return None
        
        # Clean up genes list and count
        if gene_set_data['genes']:
            gene_set_data['genes'] = [gene.strip() for gene in gene_set_data['genes'] if gene and gene.strip()]
            gene_set_data['gene_count'] = len(gene_set_data['genes'])
        
        # Skip if no genes found
        if gene_set_data['gene_count'] == 0:
            logger.warning(f"No genes found in {filename}")
            return None
        
        return gene_set_data
    
    def _extract_id_type(self, gene_ids: List[str]) -> str:
        """Extract the predominant gene ID type from a list"""
        if not gene_ids:
            return 'unknown'
        
        id_types = defaultdict(int)
        for gene_id in gene_ids[:20]:  # Sample first 20
            if ':' in str(gene_id):
                id_type = str(gene_id).split(':')[0]
                id_types[id_type] += 1
        
        if id_types:
            return max(id_types.items(), key=lambda x: x[1])[0]
        return 'unknown'
    
    def _classify_gene_set_type(self, filename: str, gene_set_data: Dict) -> str:
        """Classify gene set type based on filename and metadata"""
        filename_lower = filename.lower()
        
        # HALLMARK gene sets
        if filename_lower.startswith('hallmark_'):
            return 'hallmark'
        
        # Bicluster sets
        if filename_lower.startswith('bicluster_'):
            return 'bicluster'
        
        # GO-related custom sets
        if filename_lower.startswith('go-') or 'go:' in filename_lower:
            return 'go_custom'
        
        # Disease-specific
        disease_terms = ['eds', 'fa', 'progeria', 'ataxia']
        if any(term in filename_lower for term in disease_terms):
            return 'disease'
        
        # Pathway-specific
        pathway_terms = ['dopamine', 'canonical-', 'mtorc1', 'glycolysis']
        if any(term in filename_lower for term in pathway_terms):
            return 'pathway'
        
        return 'other'
    
    def _generate_parsing_statistics(self) -> None:
        """Generate comprehensive parsing statistics"""
        stats = {
            'overall_summary': {
                'files_processed': self.parsing_stats['files_processed'],
                'files_failed': self.parsing_stats['files_failed'],
                'total_gene_sets': self.parsing_stats['total_gene_sets'],
                'total_unique_genes': len(self._get_all_unique_genes()),
                'avg_genes_per_set': self.parsing_stats['total_genes'] / max(1, self.parsing_stats['total_gene_sets'])
            },
            'by_data_type': {
                'hallmark_sets': self.parsing_stats['hallmark_sets'],
                'bicluster_sets': self.parsing_stats['bicluster_sets'],
                'pathway_sets': self.parsing_stats['pathway_sets'],
                'go_custom_sets': self.parsing_stats['go_custom_sets'],
                'disease_sets': self.parsing_stats['disease_sets'],
                'other_sets': self.parsing_stats['other_sets']
            }
        }
        
        # Add detailed statistics for each data type
        for data_type in ['hallmark_sets', 'bicluster_sets', 'pathway_sets', 'go_custom_sets', 'disease_sets', 'other_sets']:
            data = self.parsed_data[data_type]
            if data:
                gene_counts = [gene_set['gene_count'] for gene_set in data.values()]
                stats[data_type] = {
                    'count': len(data),
                    'total_genes': sum(gene_counts),
                    'avg_genes_per_set': sum(gene_counts) / len(gene_counts),
                    'min_genes': min(gene_counts),
                    'max_genes': max(gene_counts)
                }
        
        self.parsing_stats.update(stats)
    
    def _get_all_unique_genes(self) -> set:
        """Get all unique genes across all gene sets"""
        all_genes = set()
        
        for data_type in self.parsed_data.values():
            for gene_set in data_type.values():
                all_genes.update(gene_set['genes'])
        
        return all_genes
    
    def get_parsing_statistics(self) -> Dict[str, Any]:
        """Return comprehensive parsing statistics"""
        return dict(self.parsing_stats)
    
    def get_gene_set_summary(self, data_type: str = None) -> Dict[str, Any]:
        """Get summary of gene sets by data type"""
        if data_type and data_type in self.parsed_data:
            return {
                'data_type': data_type,
                'gene_sets': list(self.parsed_data[data_type].keys()),
                'count': len(self.parsed_data[data_type])
            }
        
        return {
            'all_data_types': {
                data_type: {
                    'count': len(gene_sets),
                    'gene_sets': list(gene_sets.keys())
                }
                for data_type, gene_sets in self.parsed_data.items()
            }
        }
    
    def validate_parsing_quality(self) -> Dict[str, Any]:
        """Validate parsing quality and data integrity"""
        validation = {
            'total_gene_sets': 0,
            'gene_sets_with_names': 0,
            'gene_sets_with_descriptions': 0,
            'gene_sets_with_metadata': 0,
            'min_gene_count': float('inf'),
            'max_gene_count': 0,
            'quality_issues': []
        }
        
        for data_type, gene_sets in self.parsed_data.items():
            for gene_set_id, gene_set in gene_sets.items():
                validation['total_gene_sets'] += 1
                
                if gene_set['name']:
                    validation['gene_sets_with_names'] += 1
                
                if gene_set['description']:
                    validation['gene_sets_with_descriptions'] += 1
                
                if gene_set['metadata']:
                    validation['gene_sets_with_metadata'] += 1
                
                gene_count = gene_set['gene_count']
                validation['min_gene_count'] = min(validation['min_gene_count'], gene_count)
                validation['max_gene_count'] = max(validation['max_gene_count'], gene_count)
                
                # Quality checks
                if gene_count < 5:
                    validation['quality_issues'].append(f"{gene_set_id}: Very few genes ({gene_count})")
                if not gene_set['genes']:
                    validation['quality_issues'].append(f"{gene_set_id}: No genes found")
                if gene_count > 500:
                    validation['quality_issues'].append(f"{gene_set_id}: Very large gene set ({gene_count})")
        
        # Calculate quality metrics
        if validation['total_gene_sets'] > 0:
            validation['name_coverage'] = validation['gene_sets_with_names'] / validation['total_gene_sets']
            validation['description_coverage'] = validation['gene_sets_with_descriptions'] / validation['total_gene_sets']
            validation['metadata_coverage'] = validation['gene_sets_with_metadata'] / validation['total_gene_sets']
        
        return validation

def main():
    """Test the parser"""
    logging.basicConfig(level=logging.INFO)
    
    parser = TalismanGeneSetsParser()
    parsed_data = parser.parse_all_gene_sets()
    stats = parser.get_parsing_statistics()
    validation = parser.validate_parsing_quality()
    
    print(f"Parsed {stats['overall_summary']['total_gene_sets']} gene sets")
    print(f"Total unique genes: {stats['overall_summary']['total_unique_genes']}")
    print(f"Data types: {list(stats['by_data_type'].keys())}")
    print(f"Quality issues: {len(validation['quality_issues'])}")

if __name__ == "__main__":
    main()