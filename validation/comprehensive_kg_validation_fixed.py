#!/usr/bin/env python3
"""
Comprehensive validation of the updated KG builder to ensure all 
parser enhancements are correctly integrated.
"""

import sys
sys.path.append('/home/mreddy1/knowledge_graph/src')

from data_parsers import GOBPDataParser
from kg_builder import GOBPKnowledgeGraph

def comprehensive_kg_validation():
    """Comprehensive validation of the updated KG builder."""
    
    data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data/GO_BP"
    
    print("=" * 80)
    print("COMPREHENSIVE KG BUILDER VALIDATION")
    print("=" * 80)
    
    # Build KG
    kg = GOBPKnowledgeGraph(use_neo4j=False)
    kg.load_data(data_dir)
    kg.build_graph()
    
    stats = kg.get_stats()
    
    print("COMPREHENSIVE KNOWLEDGE GRAPH STATISTICS:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    
    # Validation tests
    print(f"\n" + "=" * 80)
    print("COMPREHENSIVE VALIDATION TESTS")
    print("=" * 80)
    
    errors = []
    successes = []
    
    # Test 1: GO clustering relationships
    print("\n1. VALIDATING GO CLUSTERING RELATIONSHIPS")
    print("-" * 50)
    
    expected_clusters = 7580  # From parser analysis
    actual_clusters = stats['go_clusters']
    
    if actual_clusters >= expected_clusters * 0.9:  # Allow 10% variance for missing nodes
        successes.append(f"GO clusters: {actual_clusters:,} (expected ~{expected_clusters:,})")
        print(f"✓ GO clusters: {actual_clusters:,}")
    else:
        errors.append(f"GO clusters: {actual_clusters:,} (expected ~{expected_clusters:,})")
        print(f"❌ GO clusters: {actual_clusters:,}")
    
    # Test 2: Comprehensive gene associations
    print("\n2. VALIDATING COMPREHENSIVE GENE ASSOCIATIONS")
    print("-" * 50)
    
    gaf_assoc = stats['gaf_associations']
    collapsed_assoc = stats['collapsed_associations']
    total_assoc = stats['gene_associations']
    
    if gaf_assoc > 160000 and collapsed_assoc > 200000:
        successes.append(f"Gene associations: GAF={gaf_assoc:,}, Collapsed={collapsed_assoc:,}")
        print(f"✓ GAF associations: {gaf_assoc:,}")
        print(f"✓ Collapsed associations: {collapsed_assoc:,}")
        print(f"✓ Total associations: {total_assoc:,}")
    else:
        errors.append(f"Insufficient gene associations: GAF={gaf_assoc:,}, Collapsed={collapsed_assoc:,}")
    
    # Test 3: Gene cross-references
    print("\n3. VALIDATING GENE CROSS-REFERENCES")
    print("-" * 50)
    
    cross_refs = stats['gene_cross_references']
    gene_ids = stats['gene_identifiers']
    
    if cross_refs > 15000 and gene_ids > 15000:
        successes.append(f"Gene cross-references: {cross_refs:,}, Gene identifiers: {gene_ids:,}")
        print(f"✓ Gene cross-references: {cross_refs:,}")
        print(f"✓ Gene identifier nodes: {gene_ids:,}")
    else:
        errors.append(f"Insufficient cross-references: {cross_refs:,}")
    
    # Test 4: Enhanced gene nodes
    print("\n4. VALIDATING ENHANCED GENE NODES")
    print("-" * 50)
    
    # Check if gene nodes have enhanced attributes
    sample_genes = ['TP53', 'BRCA1', 'MYC']
    enhanced_genes = 0
    
    for gene in sample_genes:
        if gene in kg.graph:
            node_data = kg.graph.nodes[gene]
            if ('entrez_id' in node_data or 'cross_ref_uniprot' in node_data or
                'sources' in node_data):
                enhanced_genes += 1
    
    if enhanced_genes >= len(sample_genes) * 0.5:
        successes.append(f"Enhanced gene nodes: {enhanced_genes}/{len(sample_genes)} sample genes")
        print(f"✓ Enhanced gene nodes: {enhanced_genes}/{len(sample_genes)} sample genes")
    else:
        errors.append(f"Gene nodes not enhanced: {enhanced_genes}/{len(sample_genes)}")
    
    # Test 5: OBO enhancement
    print("\n5. VALIDATING OBO ENHANCEMENT")
    print("-" * 50)
    
    enhanced_go = stats['enhanced_go_terms']
    total_go = stats['go_terms']
    enhancement_ratio = enhanced_go / total_go if total_go > 0 else 0
    
    if enhancement_ratio > 0.9:
        successes.append(f"OBO enhancement: {enhanced_go:,}/{total_go:,} ({enhancement_ratio:.1%})")
        print(f"✓ OBO enhancement: {enhanced_go:,}/{total_go:,} ({enhancement_ratio:.1%})")
    else:
        errors.append(f"Low OBO enhancement: {enhanced_go:,}/{total_go:,} ({enhancement_ratio:.1%})")
    
    # Test 6: Graph integrity
    print("\n6. VALIDATING GRAPH INTEGRITY")
    print("-" * 50)
    
    validation = kg.validate_graph_integrity()
    
    if validation['overall_valid']:
        successes.append("Graph integrity validation passed")
        print("✓ Graph integrity validation passed")
    else:
        errors.append(f"Graph integrity issues: {validation}")
        print(f"❌ Graph integrity issues: {validation}")
    
    # Test 7: Query functionality
    print("\n7. VALIDATING QUERY FUNCTIONALITY")
    print("-" * 50)
    
    query_tests = 0
    
    # Test gene function query
    if 'TP53' in kg.graph:
        functions = kg.query_gene_functions('TP53')
        if len(functions) > 0:
            query_tests += 1
            print(f"✓ Gene function query: {len(functions)} GO terms for TP53")
    
    # Test GO term search
    dna_terms = kg.search_go_terms_by_definition('DNA damage')
    if len(dna_terms) > 0:
        query_tests += 1
        print(f"✓ GO term search: {len(dna_terms)} terms for 'DNA damage'")
    
    # Test cross-reference lookup
    if 'TP53' in kg.graph:
        cross_refs = kg.get_gene_cross_references('TP53')
        if len(cross_refs) > 0:
            query_tests += 1
            print(f"✓ Cross-reference lookup: {len(cross_refs)} refs for TP53")
    
    if query_tests >= 2:
        successes.append(f"Query functionality: {query_tests}/3 tests passed")
    else:
        errors.append(f"Query functionality issues: {query_tests}/3 tests passed")
    
    # Test 8: Data coverage analysis
    print("\n8. VALIDATING DATA COVERAGE")
    print("-" * 50)
    
    parser = GOBPDataParser(data_dir)
    
    # Parse original data for comparison
    go_terms = parser.parse_go_terms()
    relationships = parser.parse_go_relationships()
    alt_ids = parser.parse_go_alternative_ids()
    
    coverage_tests = 0
    
    # GO terms coverage
    if stats['go_terms'] >= len(go_terms) * 0.95:
        coverage_tests += 1
        print(f"✓ GO terms coverage: {stats['go_terms']:,}/{len(go_terms):,}")
    
    # Relationships coverage
    if stats['go_relationships'] >= len(relationships) * 0.95:
        coverage_tests += 1
        print(f"✓ GO relationships coverage: {stats['go_relationships']:,}/{len(relationships):,}")
    
    # Alternative IDs coverage
    if stats['alternative_go_ids'] >= len(alt_ids) * 0.95:
        coverage_tests += 1
        print(f"✓ Alternative IDs coverage: {stats['alternative_go_ids']:,}/{len(alt_ids):,}")
    
    if coverage_tests >= 2:
        successes.append(f"Data coverage: {coverage_tests}/3 tests passed")
    else:
        errors.append(f"Data coverage issues: {coverage_tests}/3 tests passed")
    
    # Summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE VALIDATION SUMMARY")
    print("=" * 80)
    
    if errors:
        print(f"❌ FOUND {len(errors)} ISSUES:")
        for i, error in enumerate(errors, 1):
            print(f"{i}. {error}")
    
    if successes:
        print(f"\n✅ SUCCESSFUL VALIDATIONS ({len(successes)}):")
        for i, success in enumerate(successes, 1):
            print(f"{i}. {success}")
    
    success_rate = len(successes) / (len(successes) + len(errors)) if (successes or errors) else 0
    
    if success_rate >= 0.9:
        print(f"\n🎉 VALIDATION PASSED ({success_rate:.1%} success rate)")
        print("The updated KG builder successfully incorporates all parser enhancements!")
        return True
    else:
        print(f"\n⚠️ VALIDATION ISSUES ({success_rate:.1%} success rate)")
        print("The KG builder needs further improvements.")
        return False

if __name__ == "__main__":
    success = comprehensive_kg_validation()
    exit(0 if success else 1)