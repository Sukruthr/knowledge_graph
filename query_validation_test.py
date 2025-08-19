#!/usr/bin/env python3
"""
Comprehensive query validation test for the biomedical knowledge graph.
"""

import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from kg_builder import ComprehensiveBiomedicalKnowledgeGraph

def test_query_correctness():
    """Test query outputs for correctness and completeness."""
    
    print("=== COMPREHENSIVE QUERY VALIDATION TEST ===")
    
    # Initialize system
    base_data_dir = str(project_root / "llm_evaluation_for_gene_set_interpretation" / "data")
    kg = ComprehensiveBiomedicalKnowledgeGraph(use_neo4j=False)
    kg.load_data(base_data_dir)
    kg.build_comprehensive_graph()
    
    # Test comprehensive gene queries
    test_genes = ['TP53', 'BRCA1', 'EGFR', 'MYC', 'GAPDH']
    
    validation_results = []
    
    for gene in test_genes:
        print(f"\n--- Testing {gene} ---")
        
        profile = kg.query_gene_comprehensive(gene)
        
        if not profile:
            print(f"❌ {gene}: No profile found")
            validation_results.append(False)
            continue
        
        # Validate structure
        expected_keys = ['gene_symbol', 'go_annotations', 'disease_associations', 
                        'drug_perturbations', 'viral_responses', 'cluster_memberships']
        
        missing_keys = [key for key in expected_keys if key not in profile]
        if missing_keys:
            print(f"❌ {gene}: Missing keys {missing_keys}")
            validation_results.append(False)
            continue
        
        # Validate content
        go_count = len(profile['go_annotations'])
        disease_count = len(profile['disease_associations'])
        drug_count = len(profile['drug_perturbations'])
        viral_count = len(profile['viral_responses'])
        
        print(f"✓ {gene} structure valid")
        print(f"  GO annotations: {go_count}")
        print(f"  Disease associations: {disease_count}")
        print(f"  Drug perturbations: {drug_count}")
        print(f"  Viral responses: {viral_count}")
        
        # Validate GO annotations have required fields
        if go_count > 0:
            sample_go = profile['go_annotations'][0]
            go_fields = ['go_id', 'go_name', 'namespace', 'evidence_code']
            go_valid = all(field in sample_go for field in go_fields)
            print(f"  GO annotation structure: {'✓' if go_valid else '❌'}")
        
        # Validate viral responses include both types
        viral_expression_count = sum(1 for vr in profile['viral_responses'] if vr.get('type') == 'expression')
        viral_response_count = sum(1 for vr in profile['viral_responses'] if vr.get('type') == 'response')
        
        print(f"  Viral expression events: {viral_expression_count}")
        print(f"  Viral response events: {viral_response_count}")
        
        # Check for expression data quality
        if viral_expression_count > 0:
            sample_expr = next(vr for vr in profile['viral_responses'] if vr.get('type') == 'expression')
            expression_fields = ['expression_value', 'expression_direction', 'expression_magnitude']
            expr_valid = all(field in sample_expr for field in expression_fields)
            print(f"  Viral expression structure: {'✓' if expr_valid else '❌'}")
            
            if expr_valid:
                # Validate expression values are consistent
                expr_val = sample_expr['expression_value']
                expr_dir = sample_expr['expression_direction']
                expr_mag = sample_expr['expression_magnitude']
                
                direction_valid = (expr_dir == 'up' and expr_val > 0) or (expr_dir == 'down' and expr_val < 0)
                magnitude_valid = abs(expr_val) == expr_mag
                
                print(f"  Expression consistency: {'✓' if direction_valid and magnitude_valid else '❌'}")
        
        # Overall validation for this gene
        gene_valid = (go_count > 0 and (disease_count > 0 or drug_count > 0 or viral_count > 0))
        validation_results.append(gene_valid)
        print(f"  Overall {gene}: {'✅ PASS' if gene_valid else '❌ FAIL'}")
    
    # Summary
    passed = sum(validation_results)
    total = len(validation_results)
    
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Genes tested: {total}")
    print(f"Genes passed: {passed}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL QUERY VALIDATION TESTS PASSED!")
    else:
        print("⚠️ Some validation tests failed")
    
    # Test edge cases
    print(f"\n--- Edge Case Testing ---")
    
    # Test non-existent gene
    fake_profile = kg.query_gene_comprehensive('NONEXISTENTGENE123')
    print(f"Non-existent gene query: {'✓ Empty dict' if fake_profile == {} else '❌ Unexpected result'}")
    
    # Test case sensitivity
    lower_profile = kg.query_gene_comprehensive('tp53')
    print(f"Case sensitivity test: {'✓ Empty' if lower_profile == {} else '❌ Found match'}")
    
    print("\n=== QUERY VALIDATION COMPLETE ===")
    
    return passed == total

if __name__ == "__main__":
    success = test_query_correctness()
    sys.exit(0 if success else 1)