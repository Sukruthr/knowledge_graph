#!/usr/bin/env python3
"""
Check for duplication risk between new files and existing KG data

This script compares the remaining data files with existing data to identify:
1. Potential duplications
2. New/complementary information
3. Integration strategy based on overlap analysis
"""

import os
import pandas as pd
import json
import gzip
from pathlib import Path
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DuplicationChecker:
    """Check for potential duplication between new and existing data"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.existing_go_bp_data = None
        self.new_gmt_data = None
        
    def load_existing_go_bp_data(self):
        """Load existing GO_BP data from the knowledge graph"""
        
        go_bp_dir = self.data_dir / "GO_BP"
        
        # Load GO terms and genes from existing data
        go_tab_file = go_bp_dir / "go.tab"
        collapsed_file = go_bp_dir / "collapsed_go.symbol"
        
        existing_data = {
            'go_terms': set(),
            'genes': set(),
            'term_gene_pairs': set()
        }
        
        try:
            # Load GO terms from go.tab
            if go_tab_file.exists():
                with open(go_tab_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        if line_num == 0:  # Skip header
                            continue
                        parts = line.strip().split('\t')
                        if len(parts) >= 1:
                            go_term = parts[0]
                            existing_data['go_terms'].add(go_term)
            
            # Load gene-GO associations from collapsed file
            if collapsed_file.exists():
                with open(collapsed_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            gene = parts[0]
                            go_terms = parts[1].split(';') if len(parts) > 1 else []
                            
                            existing_data['genes'].add(gene)
                            for go_term in go_terms:
                                existing_data['term_gene_pairs'].add((go_term, gene))
            
            logger.info(f"Loaded existing GO_BP data: {len(existing_data['go_terms'])} GO terms, {len(existing_data['genes'])} genes")
            self.existing_go_bp_data = existing_data
            
        except Exception as e:
            logger.error(f"Error loading existing GO_BP data: {str(e)}")
            self.existing_go_bp_data = {'go_terms': set(), 'genes': set(), 'term_gene_pairs': set()}
    
    def load_new_gmt_data(self):
        """Load the new GMT file data"""
        
        gmt_file = self.data_dir / "remaining_data_files" / "GO_BP_20231115.gmt"
        
        new_data = {
            'go_terms': set(),
            'genes': set(),
            'term_gene_pairs': set(),
            'gene_sets': []
        }
        
        try:
            with open(gmt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        go_info = parts[0]  # This contains both description and GO ID
                        genes = parts[2:] if len(parts) > 2 else []
                        
                        # Extract GO ID from the go_info string
                        go_id = None
                        if '(GO:' in go_info:
                            start_idx = go_info.find('(GO:')
                            end_idx = go_info.find(')', start_idx)
                            if end_idx > start_idx:
                                go_id = go_info[start_idx+1:end_idx]
                        
                        if go_id:
                            new_data['go_terms'].add(go_id)
                            new_data['gene_sets'].append({
                                'go_id': go_id,
                                'description': go_info,
                                'genes': genes,
                                'gene_count': len(genes)
                            })
                            
                            for gene in genes:
                                new_data['genes'].add(gene)
                                new_data['term_gene_pairs'].add((go_id, gene))
            
            logger.info(f"Loaded new GMT data: {len(new_data['go_terms'])} GO terms, {len(new_data['genes'])} genes")
            self.new_gmt_data = new_data
            
        except Exception as e:
            logger.error(f"Error loading new GMT data: {str(e)}")
            self.new_gmt_data = {'go_terms': set(), 'genes': set(), 'term_gene_pairs': set(), 'gene_sets': []}
    
    def analyze_overlap(self) -> Dict[str, Any]:
        """Analyze overlap between existing and new data"""
        
        if not self.existing_go_bp_data or not self.new_gmt_data:
            return {'error': 'Data not loaded properly'}
        
        existing = self.existing_go_bp_data
        new = self.new_gmt_data
        
        # Calculate overlaps
        go_terms_overlap = existing['go_terms'].intersection(new['go_terms'])
        genes_overlap = existing['genes'].intersection(new['genes'])
        pairs_overlap = existing['term_gene_pairs'].intersection(new['term_gene_pairs'])
        
        # Calculate new additions
        new_go_terms = new['go_terms'] - existing['go_terms']
        new_genes = new['genes'] - existing['genes']
        new_pairs = new['term_gene_pairs'] - existing['term_gene_pairs']
        
        # Calculate statistics
        analysis = {
            'existing_data': {
                'go_terms': len(existing['go_terms']),
                'genes': len(existing['genes']),
                'term_gene_pairs': len(existing['term_gene_pairs'])
            },
            'new_data': {
                'go_terms': len(new['go_terms']),
                'genes': len(new['genes']),
                'term_gene_pairs': len(new['term_gene_pairs'])
            },
            'overlap': {
                'go_terms': len(go_terms_overlap),
                'genes': len(genes_overlap),
                'term_gene_pairs': len(pairs_overlap)
            },
            'new_additions': {
                'go_terms': len(new_go_terms),
                'genes': len(new_genes),
                'term_gene_pairs': len(new_pairs)
            },
            'overlap_percentages': {
                'go_terms_in_existing': (len(go_terms_overlap) / len(new['go_terms'])) * 100 if new['go_terms'] else 0,
                'genes_in_existing': (len(genes_overlap) / len(new['genes'])) * 100 if new['genes'] else 0,
                'pairs_in_existing': (len(pairs_overlap) / len(new['term_gene_pairs'])) * 100 if new['term_gene_pairs'] else 0
            },
            'new_value_percentages': {
                'new_go_terms': (len(new_go_terms) / len(new['go_terms'])) * 100 if new['go_terms'] else 0,
                'new_genes': (len(new_genes) / len(new['genes'])) * 100 if new['genes'] else 0,
                'new_pairs': (len(new_pairs) / len(new['term_gene_pairs'])) * 100 if new['term_gene_pairs'] else 0
            }
        }
        
        # Add sample data for inspection
        analysis['samples'] = {
            'overlapping_go_terms': list(go_terms_overlap)[:10],
            'new_go_terms': list(new_go_terms)[:10],
            'overlapping_genes': list(genes_overlap)[:10],
            'new_genes': list(new_genes)[:10]
        }
        
        return analysis
    
    def check_existing_data_coverage(self):
        """Check what types of data we already have in the KG"""
        
        coverage = {
            'go_namespaces': [],
            'omics_data_types': [],
            'model_comparison_data': False,
            'llm_processed_data': False,
            'go_analysis_data': False,
            'expression_data_types': []
        }
        
        # Check GO namespaces
        for namespace in ['GO_BP', 'GO_CC', 'GO_MF']:
            namespace_dir = self.data_dir / namespace
            if namespace_dir.exists() and any(namespace_dir.iterdir()):
                coverage['go_namespaces'].append(namespace)
        
        # Check Omics data
        omics_dir = self.data_dir / "Omics_data"
        if omics_dir.exists():
            for file in omics_dir.iterdir():
                if file.is_file():
                    coverage['omics_data_types'].append(file.name)
        
        # Check model comparison data
        model_compare_dir = self.data_dir / "model_compare"
        if model_compare_dir.exists() and any(model_compare_dir.iterdir()):
            coverage['model_comparison_data'] = True
        
        # Check LLM processed data  
        llm_dir = self.data_dir / "LLM_processed"
        if llm_dir.exists() and any(llm_dir.iterdir()):
            coverage['llm_processed_data'] = True
        
        # Check GO analysis data
        go_analysis_dir = self.data_dir / "GO_term_analysis"
        if go_analysis_dir.exists() and any(go_analysis_dir.iterdir()):
            coverage['go_analysis_data'] = True
        
        return coverage
    
    def generate_duplication_report(self) -> str:
        """Generate comprehensive duplication analysis report"""
        
        # Load data
        self.load_existing_go_bp_data()
        self.load_new_gmt_data()
        
        # Analyze overlaps
        overlap_analysis = self.analyze_overlap()
        coverage = self.check_existing_data_coverage()
        
        report = []
        report.append("# DUPLICATION RISK ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Current KG coverage
        report.append("## EXISTING KNOWLEDGE GRAPH COVERAGE")
        report.append(f"GO Namespaces: {', '.join(coverage['go_namespaces'])}")
        report.append(f"Omics Data Types: {len(coverage['omics_data_types'])} files")
        report.append(f"Model Comparison Data: {'Yes' if coverage['model_comparison_data'] else 'No'}")
        report.append(f"LLM Processed Data: {'Yes' if coverage['llm_processed_data'] else 'No'}")
        report.append(f"GO Analysis Data: {'Yes' if coverage['go_analysis_data'] else 'No'}")
        report.append("")
        
        # Overlap analysis for GO_BP GMT file
        if 'error' not in overlap_analysis:
            report.append("## GO_BP GMT FILE OVERLAP ANALYSIS")
            report.append("### Data Sizes")
            report.append(f"Existing GO_BP data: {overlap_analysis['existing_data']['go_terms']} terms, {overlap_analysis['existing_data']['genes']} genes")
            report.append(f"New GMT data: {overlap_analysis['new_data']['go_terms']} terms, {overlap_analysis['new_data']['genes']} genes")
            report.append("")
            
            report.append("### Overlap Statistics")
            report.append(f"Overlapping GO terms: {overlap_analysis['overlap']['go_terms']} ({overlap_analysis['overlap_percentages']['go_terms_in_existing']:.1f}% of new data)")
            report.append(f"Overlapping genes: {overlap_analysis['overlap']['genes']} ({overlap_analysis['overlap_percentages']['genes_in_existing']:.1f}% of new data)")
            report.append(f"Overlapping term-gene pairs: {overlap_analysis['overlap']['term_gene_pairs']} ({overlap_analysis['overlap_percentages']['pairs_in_existing']:.1f}% of new data)")
            report.append("")
            
            report.append("### New Value Assessment")
            report.append(f"New GO terms: {overlap_analysis['new_additions']['go_terms']} ({overlap_analysis['new_value_percentages']['new_go_terms']:.1f}% of new data)")
            report.append(f"New genes: {overlap_analysis['new_additions']['genes']} ({overlap_analysis['new_value_percentages']['new_genes']:.1f}% of new data)")
            report.append(f"New term-gene pairs: {overlap_analysis['new_additions']['term_gene_pairs']} ({overlap_analysis['new_value_percentages']['new_pairs']:.1f}% of new data)")
            report.append("")
            
            # Integration recommendation
            if overlap_analysis['new_value_percentages']['new_pairs'] > 10:
                report.append("### RECOMMENDATION: INTEGRATE")
                report.append(f"The GMT file provides {overlap_analysis['new_value_percentages']['new_pairs']:.1f}% new gene-term associations.")
                report.append("This represents significant new value and should be integrated.")
            elif overlap_analysis['new_value_percentages']['new_pairs'] > 5:
                report.append("### RECOMMENDATION: SELECTIVE INTEGRATION")
                report.append(f"The GMT file provides {overlap_analysis['new_value_percentages']['new_pairs']:.1f}% new gene-term associations.")
                report.append("Consider integrating only the new associations to avoid duplication.")
            else:
                report.append("### RECOMMENDATION: SKIP INTEGRATION") 
                report.append(f"The GMT file provides only {overlap_analysis['new_value_percentages']['new_pairs']:.1f}% new gene-term associations.")
                report.append("The duplication risk is high with minimal new value.")
            report.append("")
        
        # File-by-file recommendations for remaining files
        report.append("## INTEGRATION RECOMMENDATIONS FOR REMAINING FILES")
        
        remaining_files_recommendations = [
            {
                'filename': 'reference_evaluation.tsv',
                'risk': 'LOW',
                'reason': 'Literature evaluation data - complements existing gene-GO associations with citation context',
                'recommendation': 'INTEGRATE - Adds new dimension of literature support'
            },
            {
                'filename': 'L1000_sep_count_DF.txt',
                'risk': 'LOW',
                'reason': 'Expression perturbation data - different from existing GO associations',
                'recommendation': 'INTEGRATE - Adds experimental perturbation context'
            },
            {
                'filename': 'all_go_terms_embeddings_dict.pkl',
                'risk': 'LOW',
                'reason': 'Embedding vectors - computational representation of GO terms',
                'recommendation': 'INTEGRATE - Enables similarity computations and ML applications'
            },
            {
                'filename': 'SupplementTable3_0715.tsv',
                'risk': 'MEDIUM',
                'reason': 'Large supplementary table - need to examine content for overlap',
                'recommendation': 'EXAMINE FIRST - Check for overlap with existing LLM evaluation data'
            },
            {
                'filename': 'go_terms.csv',
                'risk': 'HIGH',
                'reason': 'GO terms data - likely overlaps with existing GO ontology data',
                'recommendation': 'COMPARE FIRST - May duplicate existing GO term information'
            }
        ]
        
        for rec in remaining_files_recommendations:
            report.append(f"### {rec['filename']}")
            report.append(f"**Risk Level**: {rec['risk']}")
            report.append(f"**Reason**: {rec['reason']}")
            report.append(f"**Recommendation**: {rec['recommendation']}")
            report.append("")
        
        return "\n".join(report)

def main():
    """Main execution function"""
    
    data_dir = "llm_evaluation_for_gene_set_interpretation/data"
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Initialize checker
    checker = DuplicationChecker(data_dir)
    
    # Generate report
    logger.info("Analyzing duplication risks...")
    report = checker.generate_duplication_report()
    
    # Save report
    output_file = "duplication_risk_analysis_report.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Duplication analysis complete! Report saved to: {output_file}")
    print(report)

if __name__ == "__main__":
    main()