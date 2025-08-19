#!/usr/bin/env python3
"""
Focused test for viral expression matrix integration validation.
"""

import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from kg_builder import ComprehensiveBiomedicalKnowledgeGraph

def test_viral_expression_integration():
    """Test viral expression matrix integration specifically."""
    
    print("=== VIRAL EXPRESSION MATRIX INTEGRATION TEST ===")
    
    # Initialize and load system
    base_data_dir = str(project_root / "llm_evaluation_for_gene_set_interpretation" / "data")
    kg = ComprehensiveBiomedicalKnowledgeGraph(use_neo4j=False)
    kg.load_data(base_data_dir)
    kg.build_comprehensive_graph()
    
    # Get comprehensive statistics
    stats = kg.get_comprehensive_stats()
    
    print(f"\nGraph Statistics with Viral Expression:")
    print(f"  Total edges: {stats['total_edges']:,}")
    print(f"  Viral expression edges: {stats['edge_counts'].get('gene_viral_expression', 0):,}")
    print(f"  Viral response edges: {stats['edge_counts'].get('gene_viral_response', 0):,}")
    
    # Test specific gene queries for viral expression data
    test_genes = ['TP53', 'BRCA1', 'EGFR']
    
    for gene in test_genes:
        profile = kg.query_gene_comprehensive(gene)
        if profile:
            # Count expression vs response types
            expression_responses = [vr for vr in profile['viral_responses'] if vr.get('type') == 'expression']
            response_responses = [vr for vr in profile['viral_responses'] if vr.get('type') == 'response']
            
            print(f"\n{gene} Viral Data:")
            print(f"  Total viral responses: {len(profile['viral_responses'])}")
            print(f"  Expression type: {len(expression_responses)}")
            print(f"  Response type: {len(response_responses)}")
            
            # Show sample expression data
            if expression_responses:
                sample_expr = expression_responses[0]
                print(f"  Sample expression: {sample_expr['expression_direction']} "
                      f"({sample_expr['expression_value']:.3f}) in {sample_expr['condition']}")
    
    # Test viral expression matrix parsing directly
    if hasattr(kg.parser, 'omics_parser') and kg.parser.omics_parser:
        viral_matrix = kg.parser.omics_parser.parse_viral_expression_matrix(expression_threshold=0.5)
        
        print(f"\nDirect Viral Expression Matrix Parsing:")
        print(f"  Genes with significant expression: {len(viral_matrix)}")
        
        # Show sample data
        sample_gene = 'TP53' if 'TP53' in viral_matrix else list(viral_matrix.keys())[0]
        if sample_gene in viral_matrix:
            sample_data = viral_matrix[sample_gene]
            print(f"  Sample gene ({sample_gene}): {sample_data['num_significant_conditions']} conditions")
            if sample_data['expressions']:
                sample_expr = sample_data['expressions'][0]
                print(f"    Sample: {sample_expr['expression_direction']} "
                      f"({sample_expr['expression_value']:.3f}) in {sample_expr['viral_perturbation']}")
    
    print("\n=== TEST COMPLETE ===")
    
    return True

if __name__ == "__main__":
    success = test_viral_expression_integration()
    sys.exit(0 if success else 1)