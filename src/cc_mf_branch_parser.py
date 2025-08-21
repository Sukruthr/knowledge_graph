#!/usr/bin/env python3
"""
CC_MF_Branch Parser for biomedical knowledge graph.

Parses CC (Cellular Component) and MF (Molecular Function) GO terms with:
- Gene associations
- LLM interpretations and confidence scores
- Similarity rankings and comparative analysis
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Set, Tuple, Optional
import json
import logging

class CCMFBranchParser:
    """Parser for CC and MF GO terms with LLM analysis and similarity rankings."""
    
    def __init__(self, data_path: str):
        """Initialize the CC_MF_Branch parser.
        
        Args:
            data_path: Path to the data directory containing CC_MF_branch files
        """
        self.data_path = data_path
        self.cc_mf_data_path = os.path.join(data_path, "GO_term_analysis", "CC_MF_branch")
        
        # Data containers
        self.cc_go_terms = {}
        self.mf_go_terms = {}
        self.cc_llm_interpretations = {}
        self.mf_llm_interpretations = {}
        self.cc_similarity_rankings = {}
        self.mf_similarity_rankings = {}
        
        # Processing stats
        self.processing_stats = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def parse_all_cc_mf_data(self) -> Dict[str, Dict]:
        """Parse all CC_MF_branch data files.
        
        Returns:
            Dict containing all parsed data
        """
        self.logger.info("Starting CC_MF_branch data parsing...")
        
        # Parse GO terms
        self.cc_go_terms = self._parse_go_terms('CC')
        self.mf_go_terms = self._parse_go_terms('MF')
        
        # Parse LLM interpretations
        self.cc_llm_interpretations = self._parse_llm_interpretations('CC')
        self.mf_llm_interpretations = self._parse_llm_interpretations('MF')
        
        # Parse similarity rankings
        self.cc_similarity_rankings = self._parse_similarity_rankings('CC')
        self.mf_similarity_rankings = self._parse_similarity_rankings('MF')
        
        # Generate processing statistics
        self._generate_processing_stats()
        
        self.logger.info("CC_MF_branch data parsing completed successfully")
        
        return {
            'cc_go_terms': self.cc_go_terms,
            'mf_go_terms': self.mf_go_terms,
            'cc_llm_interpretations': self.cc_llm_interpretations,
            'mf_llm_interpretations': self.mf_llm_interpretations,
            'cc_similarity_rankings': self.cc_similarity_rankings,
            'mf_similarity_rankings': self.mf_similarity_rankings,
            'processing_stats': self.processing_stats
        }
    
    def _parse_go_terms(self, namespace: str) -> Dict[str, Dict]:
        """Parse GO terms for CC or MF namespace.
        
        Args:
            namespace: Either 'CC' or 'MF'
            
        Returns:
            Dict mapping GO IDs to term data
        """
        filename = f"{namespace}_go_terms.csv"
        filepath = os.path.join(self.cc_mf_data_path, filename)
        
        if not os.path.exists(filepath):
            self.logger.warning(f"GO terms file not found: {filepath}")
            return {}
        
        try:
            df = pd.read_csv(filepath)
            
            go_terms = {}
            for _, row in df.iterrows():
                go_id = row['GO']
                
                # Parse genes
                genes = []
                if pd.notna(row['Genes']):
                    genes = [g.strip() for g in str(row['Genes']).split()]
                
                go_terms[go_id] = {
                    'go_id': go_id,
                    'namespace': namespace,
                    'term_description': row['Term_Description'],
                    'genes': genes,
                    'gene_count': int(row['Gene_Count']),
                    'source_file': filename
                }
            
            self.logger.info(f"Parsed {len(go_terms)} {namespace} GO terms")
            return go_terms
            
        except Exception as e:
            self.logger.error(f"Error parsing {namespace} GO terms: {e}")
            return {}
    
    def _parse_llm_interpretations(self, namespace: str) -> Dict[str, Dict]:
        """Parse LLM interpretations for CC or MF namespace.
        
        Args:
            namespace: Either 'CC' or 'MF'
            
        Returns:
            Dict mapping GO IDs to LLM interpretation data
        """
        filename = f"LLM_processed_selected_1000_go_{namespace}terms.tsv"
        filepath = os.path.join(self.cc_mf_data_path, filename)
        
        if not os.path.exists(filepath):
            self.logger.warning(f"LLM interpretations file not found: {filepath}")
            return {}
        
        try:
            df = pd.read_csv(filepath, delimiter='\t')
            
            interpretations = {}
            for _, row in df.iterrows():
                go_id = row['GO']
                
                interpretations[go_id] = {
                    'go_id': go_id,
                    'namespace': namespace,
                    'llm_name': row.get('gpt_4_default Name', ''),
                    'llm_analysis': row.get('gpt_4_default Analysis', ''),
                    'llm_score': float(row.get('gpt_4_default Score', 0.0)),
                    'term_description': row.get('Term_Description', ''),
                    'gene_count': int(row.get('Gene_Count', 0)),
                    'source_file': filename
                }
            
            self.logger.info(f"Parsed {len(interpretations)} {namespace} LLM interpretations")
            return interpretations
            
        except Exception as e:
            self.logger.error(f"Error parsing {namespace} LLM interpretations: {e}")
            return {}
    
    def _parse_similarity_rankings(self, namespace: str) -> Dict[str, Dict]:
        """Parse similarity rankings for CC or MF namespace.
        
        Args:
            namespace: Either 'CC' or 'MF'
            
        Returns:
            Dict mapping GO IDs to similarity ranking data
        """
        filename = f"sim_rank_LLM_processed_selected_1000_go_{namespace}terms.tsv"
        filepath = os.path.join(self.cc_mf_data_path, filename)
        
        if not os.path.exists(filepath):
            self.logger.warning(f"Similarity rankings file not found: {filepath}")
            return {}
        
        try:
            df = pd.read_csv(filepath, delimiter='\t')
            
            rankings = {}
            for _, row in df.iterrows():
                go_id = row['GO']
                
                # Parse top 3 hits and similarities
                top_3_hits = []
                top_3_sims = []
                
                if pd.notna(row.get('top_3_hits')):
                    top_3_hits = str(row['top_3_hits']).split('|')
                
                if pd.notna(row.get('top_3_sim')):
                    try:
                        top_3_sims = [float(x) for x in str(row['top_3_sim']).split('|')]
                    except:
                        top_3_sims = []
                
                rankings[go_id] = {
                    'go_id': go_id,
                    'namespace': namespace,
                    'llm_name_go_term_sim': float(row.get('LLM_name_GO_term_sim', 0.0)),
                    'sim_rank': int(row.get('sim_rank', 0)),
                    'true_go_term_sim_percentile': float(row.get('true_GO_term_sim_percentile', 0.0)),
                    'random_go_name': row.get('random_GO_name', ''),
                    'random_go_llm_sim': float(row.get('random_go_llm_sim', 0.0)),
                    'random_sim_rank': int(row.get('random_sim_rank', 0)),
                    'random_sim_percentile': float(row.get('random_sim_percentile', 0.0)),
                    'top_3_hits': top_3_hits,
                    'top_3_similarities': top_3_sims,
                    'source_file': filename
                }
            
            self.logger.info(f"Parsed {len(rankings)} {namespace} similarity rankings")
            return rankings
            
        except Exception as e:
            self.logger.error(f"Error parsing {namespace} similarity rankings: {e}")
            return {}
    
    def _generate_processing_stats(self):
        """Generate comprehensive processing statistics."""
        
        # Basic counts
        cc_terms = len(self.cc_go_terms)
        mf_terms = len(self.mf_go_terms)
        cc_interpretations = len(self.cc_llm_interpretations)
        mf_interpretations = len(self.mf_llm_interpretations)
        cc_rankings = len(self.cc_similarity_rankings)
        mf_rankings = len(self.mf_similarity_rankings)
        
        # Gene analysis
        cc_genes = set()
        mf_genes = set()
        
        for term_data in self.cc_go_terms.values():
            cc_genes.update(term_data.get('genes', []))
        
        for term_data in self.mf_go_terms.values():
            mf_genes.update(term_data.get('genes', []))
        
        gene_overlap = cc_genes.intersection(mf_genes)
        total_unique_genes = cc_genes.union(mf_genes)
        
        # LLM score analysis
        cc_scores = [data.get('llm_score', 0) for data in self.cc_llm_interpretations.values()]
        mf_scores = [data.get('llm_score', 0) for data in self.mf_llm_interpretations.values()]
        
        # Similarity ranking analysis
        cc_percentiles = [data.get('true_go_term_sim_percentile', 0) 
                         for data in self.cc_similarity_rankings.values()]
        mf_percentiles = [data.get('true_go_term_sim_percentile', 0) 
                         for data in self.mf_similarity_rankings.values()]
        
        self.processing_stats = {
            'data_counts': {
                'cc_go_terms': cc_terms,
                'mf_go_terms': mf_terms,
                'total_go_terms': cc_terms + mf_terms,
                'cc_llm_interpretations': cc_interpretations,
                'mf_llm_interpretations': mf_interpretations,
                'total_llm_interpretations': cc_interpretations + mf_interpretations,
                'cc_similarity_rankings': cc_rankings,
                'mf_similarity_rankings': mf_rankings,
                'total_similarity_rankings': cc_rankings + mf_rankings
            },
            'gene_analysis': {
                'cc_unique_genes': len(cc_genes),
                'mf_unique_genes': len(mf_genes),
                'overlapping_genes': len(gene_overlap),
                'total_unique_genes': len(total_unique_genes),
                'overlap_ratio': len(gene_overlap) / len(total_unique_genes) if total_unique_genes else 0
            },
            'llm_score_stats': {
                'cc_scores': {
                    'mean': np.mean(cc_scores) if cc_scores else 0,
                    'std': np.std(cc_scores) if cc_scores else 0,
                    'min': min(cc_scores) if cc_scores else 0,
                    'max': max(cc_scores) if cc_scores else 0,
                    'count': len(cc_scores)
                },
                'mf_scores': {
                    'mean': np.mean(mf_scores) if mf_scores else 0,
                    'std': np.std(mf_scores) if mf_scores else 0,
                    'min': min(mf_scores) if mf_scores else 0,
                    'max': max(mf_scores) if mf_scores else 0,
                    'count': len(mf_scores)
                }
            },
            'similarity_stats': {
                'cc_percentiles': {
                    'mean': np.mean(cc_percentiles) if cc_percentiles else 0,
                    'std': np.std(cc_percentiles) if cc_percentiles else 0,
                    'min': min(cc_percentiles) if cc_percentiles else 0,
                    'max': max(cc_percentiles) if cc_percentiles else 0,
                    'count': len(cc_percentiles)
                },
                'mf_percentiles': {
                    'mean': np.mean(mf_percentiles) if mf_percentiles else 0,
                    'std': np.std(mf_percentiles) if mf_percentiles else 0,
                    'min': min(mf_percentiles) if mf_percentiles else 0,
                    'max': max(mf_percentiles) if mf_percentiles else 0,
                    'count': len(mf_percentiles)
                }
            }
        }
    
    def get_cc_terms(self) -> Dict[str, Dict]:
        """Get all CC GO terms."""
        return self.cc_go_terms
    
    def get_mf_terms(self) -> Dict[str, Dict]:
        """Get all MF GO terms."""
        return self.mf_go_terms
    
    def get_cc_mf_terms(self) -> Dict[str, Dict]:
        """Get combined CC and MF GO terms."""
        combined = {}
        combined.update(self.cc_go_terms)
        combined.update(self.mf_go_terms)
        return combined
    
    def get_llm_interpretations(self, namespace: Optional[str] = None) -> Dict[str, Dict]:
        """Get LLM interpretations for specified namespace or all.
        
        Args:
            namespace: 'CC', 'MF', or None for both
            
        Returns:
            Dict of LLM interpretations
        """
        if namespace == 'CC':
            return self.cc_llm_interpretations
        elif namespace == 'MF':
            return self.mf_llm_interpretations
        else:
            combined = {}
            combined.update(self.cc_llm_interpretations)
            combined.update(self.mf_llm_interpretations)
            return combined
    
    def get_similarity_rankings(self, namespace: Optional[str] = None) -> Dict[str, Dict]:
        """Get similarity rankings for specified namespace or all.
        
        Args:
            namespace: 'CC', 'MF', or None for both
            
        Returns:
            Dict of similarity rankings
        """
        if namespace == 'CC':
            return self.cc_similarity_rankings
        elif namespace == 'MF':
            return self.mf_similarity_rankings
        else:
            combined = {}
            combined.update(self.cc_similarity_rankings)
            combined.update(self.mf_similarity_rankings)
            return combined
    
    def get_genes_for_namespace(self, namespace: str) -> Set[str]:
        """Get all unique genes for a namespace.
        
        Args:
            namespace: 'CC' or 'MF'
            
        Returns:
            Set of gene symbols
        """
        if namespace == 'CC':
            genes = set()
            for term_data in self.cc_go_terms.values():
                genes.update(term_data.get('genes', []))
            return genes
        elif namespace == 'MF':
            genes = set()
            for term_data in self.mf_go_terms.values():
                genes.update(term_data.get('genes', []))
            return genes
        else:
            return set()
    
    def get_all_unique_genes(self) -> Set[str]:
        """Get all unique genes across CC and MF namespaces."""
        cc_genes = self.get_genes_for_namespace('CC')
        mf_genes = self.get_genes_for_namespace('MF')
        return cc_genes.union(mf_genes)
    
    def query_go_term(self, go_id: str) -> Optional[Dict[str, Any]]:
        """Query comprehensive data for a GO term.
        
        Args:
            go_id: GO term identifier
            
        Returns:
            Dict containing all available data for the GO term
        """
        result = {}
        
        # Check CC terms
        if go_id in self.cc_go_terms:
            result.update(self.cc_go_terms[go_id])
            
            # Add LLM interpretation if available
            if go_id in self.cc_llm_interpretations:
                result['llm_interpretation'] = self.cc_llm_interpretations[go_id]
            
            # Add similarity ranking if available
            if go_id in self.cc_similarity_rankings:
                result['similarity_ranking'] = self.cc_similarity_rankings[go_id]
        
        # Check MF terms
        elif go_id in self.mf_go_terms:
            result.update(self.mf_go_terms[go_id])
            
            # Add LLM interpretation if available
            if go_id in self.mf_llm_interpretations:
                result['llm_interpretation'] = self.mf_llm_interpretations[go_id]
            
            # Add similarity ranking if available
            if go_id in self.mf_similarity_rankings:
                result['similarity_ranking'] = self.mf_similarity_rankings[go_id]
        
        return result if result else None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats

def main():
    """Test the CC_MF_Branch parser."""
    import sys
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the parser
    data_path = "llm_evaluation_for_gene_set_interpretation/data"
    parser = CCMFBranchParser(data_path)
    
    try:
        # Parse all data
        results = parser.parse_all_cc_mf_data()
        
        # Print summary
        stats = parser.get_stats()
        print("CC_MF_BRANCH PARSER TEST RESULTS")
        print("=" * 50)
        print(f"Total GO terms: {stats['data_counts']['total_go_terms']}")
        print(f"Total LLM interpretations: {stats['data_counts']['total_llm_interpretations']}")
        print(f"Total similarity rankings: {stats['data_counts']['total_similarity_rankings']}")
        print(f"Total unique genes: {stats['gene_analysis']['total_unique_genes']}")
        print(f"Gene overlap ratio: {stats['gene_analysis']['overlap_ratio']:.3f}")
        
        # Test specific query
        sample_go_ids = list(parser.get_cc_mf_terms().keys())[:3]
        for go_id in sample_go_ids:
            term_data = parser.query_go_term(go_id)
            if term_data:
                print(f"\nSample query - {go_id}: {term_data['term_description']}")
                print(f"  Genes: {len(term_data.get('genes', []))}")
                print(f"  LLM Score: {term_data.get('llm_interpretation', {}).get('llm_score', 'N/A')}")
        
        print("\n✅ CC_MF_Branch parser test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ CC_MF_Branch parser test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)