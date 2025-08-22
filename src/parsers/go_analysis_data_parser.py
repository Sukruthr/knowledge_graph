#!/usr/bin/env python3
"""
GO Analysis Data Parser

Comprehensive parser for GO_term_analysis/data_files folder containing:
- Core GO term datasets (1000 and 100 selected terms)
- Contamination analysis datasets
- Enrichment analysis results 
- Human evaluation with confidence scoring
- GO hierarchy data

This data is distinct from the LLM_processed directory data.
"""

import os
import csv
import pandas as pd
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GOAnalysisDataParser:
    """
    Parser for GO analysis data files including GO term sets, contamination studies,
    enrichment analysis, confidence evaluations, and hierarchy data.
    """
    
    def __init__(self, data_dir: str = "llm_evaluation_for_gene_set_interpretation/data/GO_term_analysis/data_files"):
        """Initialize the GO analysis data parser."""
        self.data_dir = Path(data_dir)
        self.core_go_terms = {}
        self.contamination_datasets = {}
        self.enrichment_results = {}
        self.confidence_evaluations = {}
        self.hierarchy_data = {}
        self.similarity_scores = {}
        
        self.processing_stats = {
            'files_processed': 0,
            'total_go_terms': 0,
            'total_genes': 0,
            'contamination_datasets': 0,
            'enrichment_analyses': 0,
            'confidence_evaluations': 0,
            'hierarchy_relationships': 0,
            'errors': []
        }
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def parse_all_go_analysis_data(self) -> Dict[str, Dict]:
        """Parse all GO analysis data files and return comprehensive results."""
        logger.info("🔬 Starting GO analysis data parsing...")
        
        try:
            # Parse core GO term datasets
            self._parse_core_go_terms()
            
            # Parse contamination datasets
            self._parse_contamination_datasets()
            
            # Enrichment results are handled in core GO terms parsing
            
            # Parse confidence evaluations
            self._parse_confidence_evaluations()
            
            # Parse hierarchy data
            self._parse_hierarchy_data()
            
            # Parse similarity scores
            self._parse_similarity_scores()
            
            self._calculate_final_stats()
            
            logger.info(f"✅ GO analysis data parsing completed successfully!")
            logger.info(f"   📊 Processed {self.processing_stats['files_processed']} files")
            logger.info(f"   🔬 {self.processing_stats['total_go_terms']} GO terms analyzed")
            logger.info(f"   🧬 {self.processing_stats['total_genes']} unique genes")
            
            return {
                'core_go_terms': self.core_go_terms,
                'contamination_datasets': self.contamination_datasets,
                'enrichment_results': self.enrichment_results,
                'confidence_evaluations': self.confidence_evaluations,
                'hierarchy_data': self.hierarchy_data,
                'similarity_scores': self.similarity_scores,
                'processing_stats': self.processing_stats
            }
            
        except Exception as e:
            logger.error(f"Error in GO analysis data parsing: {e}")
            self.processing_stats['errors'].append(str(e))
            raise
    
    def _parse_core_go_terms(self):
        """Parse core GO term datasets (1000 and 100 selected terms)."""
        logger.info("📋 Parsing core GO term datasets...")
        
        # Parse 1000 selected GO terms
        file_1000 = self.data_dir / "1000_selected_go_terms.csv"
        if file_1000.exists():
            try:
                df = pd.read_csv(file_1000)
                dataset_1000 = {}
                
                for _, row in df.iterrows():
                    go_id = row['GO']
                    genes = str(row['Genes']).split() if pd.notna(row['Genes']) else []
                    
                    dataset_1000[go_id] = {
                        'go_id': go_id,
                        'genes': genes,
                        'gene_count': int(row['Gene_Count']) if pd.notna(row['Gene_Count']) else 0,
                        'term_description': str(row['Term_Description']) if pd.notna(row['Term_Description']) else '',
                        'dataset': '1000_selected',
                        'dataset_type': 'core_terms'
                    }
                
                self.core_go_terms['1000_selected'] = dataset_1000
                self.processing_stats['files_processed'] += 1
                logger.info(f"   ✅ Parsed 1000 selected GO terms: {len(dataset_1000)} terms")
                
            except Exception as e:
                error_msg = f"Error parsing 1000 selected GO terms: {e}"
                logger.error(error_msg)
                self.processing_stats['errors'].append(error_msg)
        
        # Parse 100 GO terms enricher results
        file_100 = self.data_dir / "100_GO_terms_enricher_res.tsv"
        if file_100.exists():
            try:
                df = pd.read_csv(file_100, sep='\t')
                dataset_100 = {}
                
                for _, row in df.iterrows():
                    go_id = row['GO']
                    genes = str(row['Genes']).split() if pd.notna(row['Genes']) else []
                    
                    # Parse enrichment data for each contamination level
                    enrichment_data = {}
                    for prefix in ['Genes', '50perc_contaminated_Genes', '100perc_contaminated_Genes']:
                        enrichment_data[prefix] = {
                            'enriched_term_id': row.get(f'{prefix} enriched term id', ''),
                            'enriched_term_description': row.get(f'{prefix} enriched term description', ''),
                            'adj_p_value': row.get(f'{prefix} adj p-value', ''),
                            'overlap_over_enriched': row.get(f'{prefix} overlap over enriched', ''),
                            'overlapping_genes': str(row.get(f'{prefix} overlaping genes', '')).split() if pd.notna(row.get(f'{prefix} overlaping genes', '')) else []
                        }
                    
                    dataset_100[go_id] = {
                        'go_id': go_id,
                        'genes': genes,
                        'gene_count': int(row['Gene_Count']) if pd.notna(row['Gene_Count']) else 0,
                        'term_description': str(row['Term_Description']) if pd.notna(row['Term_Description']) else '',
                        'contaminated_50perc': str(row['50perc_contaminated_Genes']).split() if pd.notna(row['50perc_contaminated_Genes']) else [],
                        'contaminated_100perc': str(row['100perc_contaminated_Genes']).split() if pd.notna(row['100perc_contaminated_Genes']) else [],
                        'enrichment_analysis': enrichment_data,
                        'dataset': '100_enricher_results',
                        'dataset_type': 'enrichment_analysis'
                    }
                
                self.core_go_terms['100_enricher_results'] = dataset_100
                self.processing_stats['files_processed'] += 1
                self.processing_stats['enrichment_analyses'] += len(dataset_100)
                logger.info(f"   ✅ Parsed 100 GO terms enricher results: {len(dataset_100)} terms")
                
            except Exception as e:
                error_msg = f"Error parsing 100 GO terms enricher results: {e}"
                logger.error(error_msg)
                self.processing_stats['errors'].append(error_msg)
    
    def _parse_contamination_datasets(self):
        """Parse contamination analysis datasets."""
        logger.info("🧪 Parsing contamination datasets...")
        
        # Parse 1000 selected GO contaminated
        file_1000_cont = self.data_dir / "1000_selected_go_contaminated.csv"
        if file_1000_cont.exists():
            try:
                df = pd.read_csv(file_1000_cont)
                dataset_1000_cont = {}
                
                for _, row in df.iterrows():
                    go_id = row['GO']
                    genes = str(row['Genes']).split() if pd.notna(row['Genes']) else []
                    
                    dataset_1000_cont[go_id] = {
                        'go_id': go_id,
                        'genes': genes,
                        'gene_count': int(row['Gene_Count']) if pd.notna(row['Gene_Count']) else 0,
                        'term_description': str(row['Term_Description']) if pd.notna(row['Term_Description']) else '',
                        'contaminated_50perc': str(row['50perc_contaminated_Genes']).split() if pd.notna(row['50perc_contaminated_Genes']) else [],
                        'contaminated_100perc': str(row['100perc_contaminated_Genes']).split() if pd.notna(row['100perc_contaminated_Genes']) else [],
                        'dataset': '1000_selected_contaminated',
                        'dataset_type': 'contamination_analysis'
                    }
                
                self.contamination_datasets['1000_selected_contaminated'] = dataset_1000_cont
                self.processing_stats['files_processed'] += 1
                self.processing_stats['contamination_datasets'] += 1
                logger.info(f"   ✅ Parsed 1000 selected GO contaminated: {len(dataset_1000_cont)} terms")
                
            except Exception as e:
                error_msg = f"Error parsing 1000 selected GO contaminated: {e}"
                logger.error(error_msg)
                self.processing_stats['errors'].append(error_msg)
        
        # Parse 100 selected GO contaminated
        file_100_cont = self.data_dir / "100_selected_go_contaminated.csv"
        if file_100_cont.exists():
            try:
                df = pd.read_csv(file_100_cont)
                dataset_100_cont = {}
                
                for _, row in df.iterrows():
                    go_id = row['GO']
                    genes = str(row['Genes']).split() if pd.notna(row['Genes']) else []
                    
                    dataset_100_cont[go_id] = {
                        'go_id': go_id,
                        'genes': genes,
                        'gene_count': int(row['Gene_Count']) if pd.notna(row['Gene_Count']) else 0,
                        'term_description': str(row['Term_Description']) if pd.notna(row['Term_Description']) else '',
                        'contaminated_50perc': str(row['50perc_contaminated_Genes']).split() if pd.notna(row['50perc_contaminated_Genes']) else [],
                        'contaminated_100perc': str(row['100perc_contaminated_Genes']).split() if pd.notna(row['100perc_contaminated_Genes']) else [],
                        'dataset': '100_selected_contaminated',
                        'dataset_type': 'contamination_analysis'
                    }
                
                self.contamination_datasets['100_selected_contaminated'] = dataset_100_cont
                self.processing_stats['files_processed'] += 1
                self.processing_stats['contamination_datasets'] += 1
                logger.info(f"   ✅ Parsed 100 selected GO contaminated: {len(dataset_100_cont)} terms")
                
            except Exception as e:
                error_msg = f"Error parsing 100 selected GO contaminated: {e}"
                logger.error(error_msg)
                self.processing_stats['errors'].append(error_msg)
    
    def _parse_confidence_evaluations(self):
        """Parse confidence evaluation data with human review."""
        logger.info("🎯 Parsing confidence evaluations...")
        
        file_conf = self.data_dir / "confidence_eval_25_sample_with_human_review.tsv"
        if file_conf.exists():
            try:
                df = pd.read_csv(file_conf, sep='\t')
                confidence_data = {}
                
                for _, row in df.iterrows():
                    go_id = row['GO']
                    genes = str(row['Genes']).split() if pd.notna(row['Genes']) else []
                    
                    confidence_data[go_id] = {
                        'go_id': go_id,
                        'genes': genes,
                        'gene_count': int(row['Gene_Count']) if pd.notna(row['Gene_Count']) else 0,
                        'llm_name': str(row['LLM Name']) if pd.notna(row['LLM Name']) else '',
                        'llm_analysis': str(row['LLM Analysis (removing confidence)']) if pd.notna(row['LLM Analysis (removing confidence)']) else '',
                        'reviewer_score_bin': str(row["Reviewer's score bin (High, Medium)"]) if pd.notna(row["Reviewer's score bin (High, Medium)"]) else '',
                        'raw_score': row['Raw score'] if pd.notna(row['Raw score']) else 0,
                        'notes': str(row['NOTES']) if pd.notna(row['NOTES']) else '',
                        'reviewer_score_bin_final': str(row["Reviewer's score bin"]) if pd.notna(row["Reviewer's score bin"]) else '',
                        'dataset': 'confidence_evaluation',
                        'dataset_type': 'human_evaluation'
                    }
                
                self.confidence_evaluations['confidence_eval_25_sample'] = confidence_data
                self.processing_stats['files_processed'] += 1
                self.processing_stats['confidence_evaluations'] += len(confidence_data)
                logger.info(f"   ✅ Parsed confidence evaluations: {len(confidence_data)} evaluations")
                
            except Exception as e:
                error_msg = f"Error parsing confidence evaluations: {e}"
                logger.error(error_msg)
                self.processing_stats['errors'].append(error_msg)
    
    def _parse_hierarchy_data(self):
        """Parse GO hierarchy data."""
        logger.info("🌳 Parsing GO hierarchy data...")
        
        # Parse hierarchy relationships
        file_hierarchy = self.data_dir / "GO_0010897_subhierarchy.txt"
        if file_hierarchy.exists():
            try:
                df = pd.read_csv(file_hierarchy, sep='\t')
                relationships = []
                
                for _, row in df.iterrows():
                    relationships.append({
                        'child': row['child'],
                        'parent': row['parent'],
                        'relationship_type': 'parent_child'
                    })
                
                self.hierarchy_data['relationships'] = relationships
                self.processing_stats['files_processed'] += 1
                self.processing_stats['hierarchy_relationships'] += len(relationships)
                logger.info(f"   ✅ Parsed hierarchy relationships: {len(relationships)} relationships")
                
            except Exception as e:
                error_msg = f"Error parsing hierarchy relationships: {e}"
                logger.error(error_msg)
                self.processing_stats['errors'].append(error_msg)
        
        # Parse hierarchy nodes (if available - this seems to be a large file)
        file_nodes = self.data_dir / "GO_0010897_subhierarchy_nodes.txt"
        if file_nodes.exists():
            try:
                # Read just the first few lines to understand structure
                with open(file_nodes, 'r') as f:
                    lines = f.readlines()
                    
                # This file appears to be very large (43 records but 959KB)
                # It likely contains detailed node information
                self.hierarchy_data['nodes_file'] = {
                    'file_path': str(file_nodes),
                    'line_count': len(lines),
                    'file_size': file_nodes.stat().st_size,
                    'note': 'Large file with detailed node information - processed separately if needed'
                }
                
                logger.info(f"   ℹ️  Noted hierarchy nodes file: {len(lines)} lines, {file_nodes.stat().st_size:,} bytes")
                
            except Exception as e:
                error_msg = f"Error noting hierarchy nodes file: {e}"
                logger.error(error_msg)
                self.processing_stats['errors'].append(error_msg)
    
    def _parse_similarity_scores(self):
        """Parse similarity scores data."""
        logger.info("📊 Parsing similarity scores...")
        
        file_sim = self.data_dir / "all_go_sim_scores_toy.txt"
        if file_sim.exists():
            try:
                # This is a large file (131K+ lines), so we'll sample it
                scores = []
                with open(file_sim, 'r') as f:
                    for i, line in enumerate(f):
                        if i >= 1000:  # Read first 1000 lines for structure analysis
                            break
                        try:
                            score = float(line.strip())
                            scores.append(score)
                        except ValueError:
                            pass
                
                self.similarity_scores['toy_scores'] = {
                    'sample_scores': scores[:100],  # Store first 100 as sample
                    'sample_count': len(scores),
                    'file_path': str(file_sim),
                    'total_lines': 131362,  # From our earlier analysis
                    'file_size': file_sim.stat().st_size,
                    'score_range': {
                        'min': min(scores) if scores else 0,
                        'max': max(scores) if scores else 0,
                        'mean': sum(scores) / len(scores) if scores else 0
                    }
                }
                
                logger.info(f"   ✅ Analyzed similarity scores: sampled {len(scores)} scores from 131K+ total")
                
            except Exception as e:
                error_msg = f"Error parsing similarity scores: {e}"
                logger.error(error_msg)
                self.processing_stats['errors'].append(error_msg)
    
    def _calculate_final_stats(self):
        """Calculate final processing statistics."""
        unique_genes = set()
        total_go_terms = 0
        
        # Count from core GO terms
        for dataset in self.core_go_terms.values():
            total_go_terms += len(dataset)
            for term_data in dataset.values():
                unique_genes.update(term_data['genes'])
        
        # Count from contamination datasets
        for dataset in self.contamination_datasets.values():
            total_go_terms += len(dataset)
            for term_data in dataset.values():
                unique_genes.update(term_data['genes'])
                unique_genes.update(term_data['contaminated_50perc'])
                unique_genes.update(term_data['contaminated_100perc'])
        
        # Count from confidence evaluations
        for dataset in self.confidence_evaluations.values():
            total_go_terms += len(dataset)
            for term_data in dataset.values():
                unique_genes.update(term_data['genes'])
        
        self.processing_stats['total_go_terms'] = total_go_terms
        self.processing_stats['total_genes'] = len(unique_genes)
    
    # Query methods
    def get_core_go_terms(self, dataset: str = None, go_id: str = None) -> Dict[str, Dict]:
        """Get core GO terms with optional filtering."""
        if dataset and dataset in self.core_go_terms:
            data = self.core_go_terms[dataset]
        else:
            data = {}
            for dataset_data in self.core_go_terms.values():
                data.update(dataset_data)
        
        if go_id and go_id in data:
            return {go_id: data[go_id]}
        
        return data
    
    def get_contamination_datasets(self, dataset: str = None, go_id: str = None) -> Dict[str, Dict]:
        """Get contamination datasets with optional filtering."""
        if dataset and dataset in self.contamination_datasets:
            data = self.contamination_datasets[dataset]
        else:
            data = {}
            for dataset_data in self.contamination_datasets.values():
                data.update(dataset_data)
        
        if go_id and go_id in data:
            return {go_id: data[go_id]}
        
        return data
    
    def get_confidence_evaluations(self, go_id: str = None) -> Dict[str, Dict]:
        """Get confidence evaluations with optional filtering."""
        data = {}
        for dataset_data in self.confidence_evaluations.values():
            data.update(dataset_data)
        
        if go_id and go_id in data:
            return {go_id: data[go_id]}
        
        return data
    
    def get_hierarchy_data(self) -> Dict[str, Any]:
        """Get hierarchy data."""
        return self.hierarchy_data
    
    def get_similarity_scores(self) -> Dict[str, Any]:
        """Get similarity scores data."""
        return self.similarity_scores
    
    def query_go_term_analysis_profile(self, go_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive analysis profile for a GO term."""
        profile = {
            'go_id': go_id,
            'core_terms': {},
            'contamination_analysis': {},
            'confidence_evaluation': {},
            'enrichment_data': {}
        }
        
        # Get core term data
        for dataset, data in self.core_go_terms.items():
            if go_id in data:
                profile['core_terms'][dataset] = data[go_id]
        
        # Get contamination data
        for dataset, data in self.contamination_datasets.items():
            if go_id in data:
                profile['contamination_analysis'][dataset] = data[go_id]
        
        # Get confidence evaluation
        for dataset, data in self.confidence_evaluations.items():
            if go_id in data:
                profile['confidence_evaluation'][dataset] = data[go_id]
        
        # Return None if no data found
        if not any([profile['core_terms'], profile['contamination_analysis'], 
                   profile['confidence_evaluation']]):
            return None
        
        return profile
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats.copy()

def main():
    """Test the parser functionality."""
    parser = GOAnalysisDataParser()
    
    try:
        # Parse all data
        results = parser.parse_all_go_analysis_data()
        
        # Print summary
        stats = parser.get_processing_stats()
        print("\n" + "="*60)
        print("GO ANALYSIS DATA PARSER - TEST RESULTS")
        print("="*60)
        print(f"Files processed: {stats['files_processed']}")
        print(f"Total GO terms: {stats['total_go_terms']}")
        print(f"Unique genes: {stats['total_genes']}")
        print(f"Contamination datasets: {stats['contamination_datasets']}")
        print(f"Enrichment analyses: {stats['enrichment_analyses']}")
        print(f"Confidence evaluations: {stats['confidence_evaluations']}")
        print(f"Hierarchy relationships: {stats['hierarchy_relationships']}")
        
        if stats['errors']:
            print(f"\nErrors: {len(stats['errors'])}")
            for error in stats['errors']:
                print(f"  - {error}")
        
        # Test query functionality
        print(f"\n--- TESTING QUERY FUNCTIONALITY ---")
        
        # Test core GO terms query
        core_terms = parser.get_core_go_terms()
        print(f"Total core GO terms available: {len(core_terms)}")
        
        if core_terms:
            # Test specific GO term profile
            first_go_id = list(core_terms.keys())[0]
            profile = parser.query_go_term_analysis_profile(first_go_id)
            if profile:
                print(f"Profile for {first_go_id}: {len(profile['core_terms'])} core datasets, {len(profile['contamination_analysis'])} contamination datasets")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error testing parser: {e}")
        raise

if __name__ == "__main__":
    main()