#!/usr/bin/env python3
"""
LLM Processed Data Parser for comprehensive multi-model LLM analysis integration.

This parser handles the LLM_processed data files containing:
- Multi-model LLM interpretations (GPT-4, GPT-3.5, Gemini Pro, Llama2, Mistral, Mixtral)
- Contamination robustness analysis
- Similarity rankings and p-value statistics
- Model comparison and evaluation metrics

Author: Claude Code Assistant
Date: 2025-08-21
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProcessedParser:
    """
    Comprehensive parser for LLM_processed data integration.
    
    Handles multi-model LLM analysis, contamination studies, similarity rankings,
    and model comparison data for biomedical knowledge graph integration.
    """
    
    def __init__(self, data_dir: str = "llm_evaluation_for_gene_set_interpretation/data/GO_term_analysis/LLM_processed"):
        """Initialize the LLM processed data parser."""
        self.data_dir = Path(data_dir)
        
        # Core data containers
        self.main_llm_interpretations = {}
        self.model_comparison_data = {}
        self.contamination_analysis = {}
        self.similarity_rankings = {}
        self.similarity_pvalues = {}
        
        # Model mappings
        self.supported_models = [
            'gpt_4', 'gpt_35', 'gemini_pro', 'llama2_70b', 
            'llama2_7b', 'mistral_7b', 'mixtral_instruct', 'mixtral_latest'
        ]
        
        # Processing statistics
        self.processing_stats = {
            'files_processed': 0,
            'total_interpretations': 0,
            'total_go_terms': 0,
            'models_analyzed': 0,
            'unique_genes': set()
        }
        
        logger.info(f"Initialized LLMProcessedParser for directory: {self.data_dir}")
    
    def parse_all_llm_processed_data(self) -> Dict[str, Dict]:
        """Parse all LLM processed data files and return comprehensive results."""
        logger.info("🔬 Starting comprehensive LLM processed data parsing...")
        
        try:
            # Parse main LLM interpretation datasets
            self._parse_main_llm_datasets()
            
            # Parse model comparison files
            self._parse_model_comparison_data()
            
            # Parse contamination analysis files
            self._parse_contamination_analysis()
            
            # Parse similarity rankings
            self._parse_similarity_rankings()
            
            # Parse similarity p-values
            self._parse_similarity_pvalues()
            
            # Update processing statistics
            self._update_processing_stats()
            
            logger.info(f"✅ Successfully parsed {self.processing_stats['files_processed']} LLM processed files")
            
            return {
                'main_interpretations': self.main_llm_interpretations,
                'model_comparison': self.model_comparison_data,
                'contamination_analysis': self.contamination_analysis,
                'similarity_rankings': self.similarity_rankings,
                'similarity_pvalues': self.similarity_pvalues,
                'processing_stats': self.processing_stats
            }
            
        except Exception as e:
            logger.error(f"Error during LLM processed data parsing: {e}")
            raise
    
    def _parse_main_llm_datasets(self):
        """Parse main LLM interpretation datasets."""
        logger.info("📚 Parsing main LLM interpretation datasets...")
        
        main_files = [
            'LLM_processed_selected_1000_go_terms.tsv',
            'LLM_processed_GO_representative_top_bottom_5.tsv'
        ]
        
        for filename in main_files:
            file_path = self.data_dir / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, sep='\t', dtype=str)
                    
                    dataset_key = filename.replace('LLM_processed_', '').replace('.tsv', '')
                    interpretations = {}
                    
                    for _, row in df.iterrows():
                        go_id = row['GO']
                        
                        interpretation_data = {
                            'go_id': go_id,
                            'genes': row['Genes'].split() if pd.notna(row['Genes']) else [],
                            'gene_count': int(row['Gene_Count']) if pd.notna(row['Gene_Count']) else 0,
                            'term_description': row['Term_Description'] if pd.notna(row['Term_Description']) else '',
                            'contaminated_genes_50perc': row['50perc_contaminated_Genes'].split() if pd.notna(row['50perc_contaminated_Genes']) and '50perc_contaminated_Genes' in row else [],
                            'contaminated_genes_100perc': row['100perc_contaminated_Genes'].split() if pd.notna(row['100perc_contaminated_Genes']) and '100perc_contaminated_Genes' in row else [],
                            'llm_name': row['gpt_4_default Name'] if pd.notna(row['gpt_4_default Name']) else '',
                            'llm_analysis': row['gpt_4_default Analysis'] if pd.notna(row['gpt_4_default Analysis']) else '',
                            'llm_score': float(row['gpt_4_default Score']) if pd.notna(row['gpt_4_default Score']) else 0.0,
                            'dataset_source': dataset_key,
                            'model': 'gpt_4'
                        }
                        
                        # Update unique genes tracking
                        self.processing_stats['unique_genes'].update(interpretation_data['genes'])
                        
                        interpretations[go_id] = interpretation_data
                    
                    self.main_llm_interpretations[dataset_key] = interpretations
                    self.processing_stats['files_processed'] += 1
                    
                    logger.info(f"   ✅ Parsed {len(interpretations)} interpretations from {filename}")
                    
                except Exception as e:
                    logger.error(f"Error parsing {filename}: {e}")
    
    def _parse_model_comparison_data(self):
        """Parse model comparison datasets."""
        logger.info("🤖 Parsing model comparison data...")
        
        comparison_file = self.data_dir / 'model_comparison_terms.csv'
        if comparison_file.exists():
            try:
                df = pd.read_csv(comparison_file, dtype=str)
                
                comparison_data = {}
                for _, row in df.iterrows():
                    go_id = row['GO']
                    
                    comparison_entry = {
                        'go_id': go_id,
                        'genes': row['Genes'].split() if pd.notna(row['Genes']) else [],
                        'gene_count': int(row['Gene_Count']) if pd.notna(row['Gene_Count']) else 0,
                        'term_description': row['Term_Description'] if pd.notna(row['Term_Description']) else '',
                        'contaminated_genes_50perc': row['50perc_contaminated_Genes'].split() if pd.notna(row['50perc_contaminated_Genes']) else [],
                        'contaminated_genes_100perc': row['100perc_contaminated_Genes'].split() if pd.notna(row['100perc_contaminated_Genes']) else []
                    }
                    
                    comparison_data[go_id] = comparison_entry
                
                self.model_comparison_data = comparison_data
                self.processing_stats['files_processed'] += 1
                
                logger.info(f"   ✅ Parsed {len(comparison_data)} terms for model comparison")
                
            except Exception as e:
                logger.error(f"Error parsing model comparison data: {e}")
    
    def _parse_contamination_analysis(self):
        """Parse contamination analysis files for each model."""
        logger.info("🧪 Parsing contamination analysis files...")
        
        contamination_pattern = re.compile(r'LLM_processed_toy_example_w_contamination_(.+)\.tsv')
        
        for file_path in self.data_dir.glob('LLM_processed_toy_example_w_contamination_*.tsv'):
            match = contamination_pattern.match(file_path.name)
            if match:
                model_name = match.group(1)
                
                try:
                    df = pd.read_csv(file_path, sep='\t', dtype=str)
                    
                    model_analysis = {}
                    contamination_scenarios = ['default', '50perc_contaminated', '100perc_contaminated']
                    
                    for _, row in df.iterrows():
                        go_id = row['GO']
                        
                        go_analysis = {
                            'go_id': go_id,
                            'genes': row['Genes'].split() if pd.notna(row['Genes']) else [],
                            'gene_count': int(row['Gene_Count']) if pd.notna(row['Gene_Count']) else 0,
                            'term_description': row['Term_Description'] if pd.notna(row['Term_Description']) else '',
                            'contaminated_genes_50perc': row['50perc_contaminated_Genes'].split() if pd.notna(row['50perc_contaminated_Genes']) else [],
                            'contaminated_genes_100perc': row['100perc_contaminated_Genes'].split() if pd.notna(row['100perc_contaminated_Genes']) else [],
                            'model': model_name,
                            'scenarios': {}
                        }
                        
                        # Parse each contamination scenario
                        for scenario in contamination_scenarios:
                            name_col = f'{model_name}_{scenario} Name'
                            analysis_col = f'{model_name}_{scenario} Analysis'
                            score_col = f'{model_name}_{scenario} Score'
                            
                            if name_col in df.columns:
                                try:
                                    score_val = 0.0
                                    if pd.notna(row[score_col]):
                                        # Clean score value - remove trailing punctuation
                                        score_str = str(row[score_col]).rstrip(').').strip()
                                        score_val = float(score_str)
                                except (ValueError, TypeError):
                                    score_val = 0.0
                                    
                                go_analysis['scenarios'][scenario] = {
                                    'name': row[name_col] if pd.notna(row[name_col]) else '',
                                    'analysis': row[analysis_col] if pd.notna(row[analysis_col]) else '',
                                    'score': score_val
                                }
                        
                        model_analysis[go_id] = go_analysis
                    
                    self.contamination_analysis[model_name] = model_analysis
                    self.processing_stats['files_processed'] += 1
                    
                    logger.info(f"   ✅ Parsed contamination analysis for {model_name}: {len(model_analysis)} terms")
                    
                except Exception as e:
                    logger.error(f"Error parsing contamination analysis for {model_name}: {e}")
    
    def _parse_similarity_rankings(self):
        """Parse similarity ranking files."""
        logger.info("📈 Parsing similarity ranking files...")
        
        ranking_files = [
            'simrank_LLM_processed_selected_1000_go_terms.tsv',
            'simrank_LLM_processed_toy_example.tsv'
        ]
        
        for filename in ranking_files:
            file_path = self.data_dir / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, sep='\t', dtype=str)
                    
                    dataset_key = filename.replace('simrank_LLM_processed_', '').replace('.tsv', '')
                    rankings = {}
                    
                    for _, row in df.iterrows():
                        go_id = row['GO']
                        
                        ranking_data = {
                            'go_id': go_id,
                            'genes': row['Genes'].split() if pd.notna(row['Genes']) else [],
                            'gene_count': int(row['Gene_Count']) if pd.notna(row['Gene_Count']) else 0,
                            'term_description': row['Term_Description'] if pd.notna(row['Term_Description']) else '',
                            'llm_name': row['gpt_4_default Name'] if pd.notna(row['gpt_4_default Name']) else '',
                            'llm_analysis': row['gpt_4_default Analysis'] if pd.notna(row['gpt_4_default Analysis']) else '',
                            'llm_score': float(row['gpt_4_default Score']) if pd.notna(row['gpt_4_default Score']) else 0.0,
                            'score_bin': row['gpt_4_default Score Bin'] if 'gpt_4_default Score Bin' in df.columns and pd.notna(row['gpt_4_default Score Bin']) else '',
                            'llm_go_similarity': float(row['LLM_name_GO_term_sim']) if pd.notna(row['LLM_name_GO_term_sim']) else 0.0,
                            'similarity_rank': int(row['sim_rank']) if pd.notna(row['sim_rank']) else 0,
                            'similarity_percentile': float(row['true_GO_term_sim_percentile']) if pd.notna(row['true_GO_term_sim_percentile']) else 0.0,
                            'random_go_name': row['random_GO_name'] if pd.notna(row['random_GO_name']) else '',
                            'random_go_similarity': float(row['random_go_llm_sim']) if pd.notna(row['random_go_llm_sim']) else 0.0,
                            'random_similarity_rank': int(row['random_sim_rank']) if pd.notna(row['random_sim_rank']) else 0,
                            'random_similarity_percentile': float(row['random_sim_percentile']) if pd.notna(row['random_sim_percentile']) else 0.0,
                            'top_3_hits': row['top_3_hits'].split('|') if pd.notna(row['top_3_hits']) else [],
                            'top_3_similarities': [float(x) for x in row['top_3_sim'].split('|')] if pd.notna(row['top_3_sim']) else [],
                            'dataset_source': dataset_key
                        }
                        
                        rankings[go_id] = ranking_data
                    
                    self.similarity_rankings[dataset_key] = rankings
                    self.processing_stats['files_processed'] += 1
                    
                    logger.info(f"   ✅ Parsed {len(rankings)} similarity rankings from {filename}")
                    
                except Exception as e:
                    logger.error(f"Error parsing {filename}: {e}")
    
    def _parse_similarity_pvalues(self):
        """Parse similarity p-value files."""
        logger.info("📊 Parsing similarity p-value files...")
        
        pval_file = self.data_dir / 'simrank_pval_LLM_processed_selected_1000_go_terms.tsv'
        if pval_file.exists():
            try:
                df = pd.read_csv(pval_file, sep='\t', dtype=str)
                
                pval_data = {}
                for _, row in df.iterrows():
                    go_id = row['GO']
                    
                    # Extract p-value columns dynamically
                    pval_columns = [col for col in df.columns if 'pval' in col.lower()]
                    pval_entry = {
                        'go_id': go_id,
                        'genes': row['Genes'].split() if pd.notna(row['Genes']) else [],
                        'gene_count': int(row['Gene_Count']) if pd.notna(row['Gene_Count']) else 0,
                        'term_description': row['Term_Description'] if pd.notna(row['Term_Description']) else '',
                        'pvalues': {}
                    }
                    
                    # Add all p-value data
                    for col in pval_columns:
                        if pd.notna(row[col]):
                            try:
                                pval_entry['pvalues'][col] = float(row[col])
                            except (ValueError, TypeError):
                                pval_entry['pvalues'][col] = row[col]  # Keep as string if not numeric
                    
                    pval_data[go_id] = pval_entry
                
                self.similarity_pvalues = pval_data
                self.processing_stats['files_processed'] += 1
                
                logger.info(f"   ✅ Parsed {len(pval_data)} p-value entries")
                
            except Exception as e:
                logger.error(f"Error parsing similarity p-values: {e}")
    
    def _update_processing_stats(self):
        """Update processing statistics."""
        # Count total interpretations
        total_interp = 0
        for dataset in self.main_llm_interpretations.values():
            total_interp += len(dataset)
        
        for model_data in self.contamination_analysis.values():
            total_interp += len(model_data)
        
        # Count unique GO terms
        go_terms = set()
        for dataset in self.main_llm_interpretations.values():
            go_terms.update(dataset.keys())
        for dataset in self.similarity_rankings.values():
            go_terms.update(dataset.keys())
        go_terms.update(self.similarity_pvalues.keys())
        go_terms.update(self.model_comparison_data.keys())
        
        # Count models
        models_analyzed = len(self.contamination_analysis)
        
        self.processing_stats.update({
            'total_interpretations': total_interp,
            'total_go_terms': len(go_terms),
            'models_analyzed': models_analyzed,
            'unique_genes': len(self.processing_stats['unique_genes'])
        })
    
    def get_llm_interpretations(self, dataset: str = None, go_id: str = None) -> Dict[str, Dict]:
        """Get LLM interpretations with optional filtering."""
        if dataset and dataset in self.main_llm_interpretations:
            data = self.main_llm_interpretations[dataset]
        else:
            # Combine all datasets
            data = {}
            for dataset_data in self.main_llm_interpretations.values():
                data.update(dataset_data)
        
        if go_id:
            return {go_id: data[go_id]} if go_id in data else {}
        
        return data
    
    def get_contamination_analysis(self, model: str = None, go_id: str = None) -> Dict[str, Dict]:
        """Get contamination analysis with optional filtering."""
        if model and model in self.contamination_analysis:
            data = self.contamination_analysis[model]
        else:
            # Combine all models
            data = {}
            for model_data in self.contamination_analysis.values():
                data.update(model_data)
        
        if go_id:
            return {go_id: data[go_id]} if go_id in data else {}
        
        return data
    
    def get_similarity_rankings(self, dataset: str = None, go_id: str = None) -> Dict[str, Dict]:
        """Get similarity rankings with optional filtering."""
        if dataset and dataset in self.similarity_rankings:
            data = self.similarity_rankings[dataset]
        else:
            # Combine all datasets
            data = {}
            for dataset_data in self.similarity_rankings.values():
                data.update(dataset_data)
        
        if go_id:
            return {go_id: data[go_id]} if go_id in data else {}
        
        return data
    
    def get_model_comparison_data(self, go_id: str = None) -> Dict[str, Dict]:
        """Get model comparison data with optional filtering."""
        if go_id:
            return {go_id: self.model_comparison_data[go_id]} if go_id in self.model_comparison_data else {}
        
        return self.model_comparison_data
    
    def get_similarity_pvalues(self, go_id: str = None) -> Dict[str, Dict]:
        """Get similarity p-values with optional filtering."""
        if go_id:
            return {go_id: self.similarity_pvalues[go_id]} if go_id in self.similarity_pvalues else {}
        
        return self.similarity_pvalues
    
    def query_go_term_llm_profile(self, go_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive LLM profile for a specific GO term."""
        profile = {
            'go_id': go_id,
            'main_interpretations': {},
            'contamination_analysis': {},
            'similarity_rankings': {},
            'similarity_pvalues': {},
            'model_comparison': {}
        }
        
        # Get main interpretations
        for dataset, data in self.main_llm_interpretations.items():
            if go_id in data:
                profile['main_interpretations'][dataset] = data[go_id]
        
        # Get contamination analysis
        for model, data in self.contamination_analysis.items():
            if go_id in data:
                profile['contamination_analysis'][model] = data[go_id]
        
        # Get similarity rankings
        for dataset, data in self.similarity_rankings.items():
            if go_id in data:
                profile['similarity_rankings'][dataset] = data[go_id]
        
        # Get similarity p-values
        if go_id in self.similarity_pvalues:
            profile['similarity_pvalues'] = self.similarity_pvalues[go_id]
        
        # Get model comparison data
        if go_id in self.model_comparison_data:
            profile['model_comparison'] = self.model_comparison_data[go_id]
        
        # Return None if no data found
        if not any([profile['main_interpretations'], profile['contamination_analysis'], 
                   profile['similarity_rankings'], profile['similarity_pvalues'], 
                   profile['model_comparison']]):
            return None
        
        return profile
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        return self.processing_stats.copy()

def main():
    """Test the LLM processed parser."""
    print("🧪 Testing LLMProcessedParser...")
    
    parser = LLMProcessedParser()
    results = parser.parse_all_llm_processed_data()
    
    print(f"\n📊 Processing Statistics:")
    stats = parser.get_processing_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n🔍 Sample GO term profile:")
    # Get a sample GO term
    if parser.main_llm_interpretations:
        sample_dataset = list(parser.main_llm_interpretations.keys())[0]
        sample_go = list(parser.main_llm_interpretations[sample_dataset].keys())[0]
        profile = parser.query_go_term_llm_profile(sample_go)
        print(f"   GO term: {sample_go}")
        print(f"   Profile sections: {list(profile.keys())}")
    
    return results

if __name__ == "__main__":
    main()