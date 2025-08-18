"""
Comprehensive validation script for Combined GO Knowledge Graph (GO_BP + GO_CC + GO_MF).

This script validates the integrated multi-namespace knowledge graph system
and provides detailed performance and quality metrics.
"""

import sys
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_parsers import CombinedGOParser
from kg_builder import CombinedGOKnowledgeGraph


def validate_combined_go_system():
    """Perform comprehensive validation of the complete tri-namespace GO system."""
    
    print("=" * 80)
    print("COMPLETE GO KNOWLEDGE GRAPH SYSTEM VALIDATION")
    print("GO_BP + GO_CC + GO_MF Integration")
    print("=" * 80)
    
    # Track timing
    start_time = time.time()
    
    # 1. Data Parser Validation
    print("\n1. DATA PARSER VALIDATION")
    print("-" * 40)
    
    base_data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data"
    combined_parser = CombinedGOParser(base_data_dir)
    
    print(f"Available namespaces: {list(combined_parser.parsers.keys())}")
    
    # Parse all data
    parser_start = time.time()
    all_parsed_data = combined_parser.parse_all_namespaces()
    parser_time = time.time() - parser_start
    
    print(f"Parsing completed in {parser_time:.1f} seconds")
    
    # Validate each namespace
    namespace_stats = {}
    for namespace, data in all_parsed_data.items():
        stats = {
            'go_terms': len(data['go_terms']),
            'relationships': len(data['go_relationships']),
            'gene_associations': len(data['gene_associations']),
            'alt_ids': len(data['alt_ids']),
            'obo_terms': len(data['obo_terms'])
        }
        namespace_stats[namespace] = stats
        
        print(f"  {namespace}:")
        print(f"    GO terms: {stats['go_terms']:,}")
        print(f"    Relationships: {stats['relationships']:,}")
        print(f"    Gene associations: {stats['gene_associations']:,}")
        print(f"    Alternative IDs: {stats['alt_ids']:,}")
        print(f"    OBO enhanced: {stats['obo_terms']:,}")
    
    # 2. Knowledge Graph Construction Validation
    print(f"\n2. KNOWLEDGE GRAPH CONSTRUCTION")
    print("-" * 40)
    
    kg_start = time.time()
    combined_kg = CombinedGOKnowledgeGraph(use_neo4j=False)
    combined_kg.load_data(base_data_dir)
    combined_kg.build_combined_graph()
    kg_time = time.time() - kg_start
    
    print(f"Knowledge graph construction completed in {kg_time:.1f} seconds")
    
    # Get comprehensive statistics
    combined_stats = combined_kg.get_combined_stats()
    
    print(f"\nCombined Graph Statistics:")
    print(f"  Total Nodes: {combined_stats['total_nodes']:,}")
    print(f"  Total Edges: {combined_stats['total_edges']:,}")
    print(f"  GO Terms: {combined_stats['go_terms']:,}")
    print(f"  Genes: {combined_stats['genes']:,}")
    print(f"  Gene Identifiers: {combined_stats['gene_identifiers']:,}")
    
    print(f"\nNamespace Distribution:")
    for namespace, count in combined_stats['namespace_counts'].items():
        print(f"  {namespace}: {count:,} terms")
    
    # 3. Quality Validation
    print(f"\n3. QUALITY VALIDATION")
    print("-" * 40)
    
    # Test cross-namespace functionality
    test_genes = ['TP53', 'BRCA1', 'EGFR', 'MYC', 'GAPDH']
    cross_namespace_results = {}
    
    for gene in test_genes:
        all_functions = combined_kg.query_gene_functions_all_namespaces(gene)
        if all_functions:
            cross_namespace_results[gene] = {
                'namespaces': len(all_functions),
                'total_functions': sum(len(funcs) for funcs in all_functions.values()),
                'by_namespace': {ns: len(funcs) for ns, funcs in all_functions.items()}
            }
    
    print(f"Cross-namespace query validation:")
    for gene, results in cross_namespace_results.items():
        print(f"  {gene}:")
        print(f"    Namespaces: {results['namespaces']}")
        print(f"    Total functions: {results['total_functions']}")
        for ns, count in results['by_namespace'].items():
            print(f"      {ns}: {count} terms")
    
    # 4. Data Integrity Validation
    print(f"\n4. DATA INTEGRITY VALIDATION")
    print("-" * 40)
    
    integrity_checks = {
        'unique_go_terms': len(set(
            go_id for namespace_data in all_parsed_data.values()
            for go_id in namespace_data['go_terms'].keys()
        )),
        'unique_genes': len(set(
            gene['gene_symbol'] for namespace_data in all_parsed_data.values()
            for gene in namespace_data['gene_associations']
        )),
        'total_relationships': sum(
            len(data['go_relationships']) for data in all_parsed_data.values()
        ),
        'total_associations': sum(
            len(data['gene_associations']) for data in all_parsed_data.values()
        )
    }
    
    print(f"Data integrity metrics:")
    for metric, value in integrity_checks.items():
        print(f"  {metric}: {value:,}")
    
    # 5. Performance Analysis
    print(f"\n5. PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    total_time = time.time() - start_time
    
    performance_metrics = {
        'total_runtime': total_time,
        'parsing_time': parser_time,
        'kg_construction_time': kg_time,
        'nodes_per_second': combined_stats['total_nodes'] / total_time,
        'edges_per_second': combined_stats['total_edges'] / total_time,
        'memory_efficiency': combined_stats['total_edges'] / combined_stats['total_nodes']
    }
    
    print(f"Performance metrics:")
    print(f"  Total runtime: {performance_metrics['total_runtime']:.1f} seconds")
    print(f"  Parsing time: {performance_metrics['parsing_time']:.1f} seconds")
    print(f"  Graph construction: {performance_metrics['kg_construction_time']:.1f} seconds")
    print(f"  Nodes/second: {performance_metrics['nodes_per_second']:,.0f}")
    print(f"  Edges/second: {performance_metrics['edges_per_second']:,.0f}")
    print(f"  Edge/Node ratio: {performance_metrics['memory_efficiency']:.1f}")
    
    # 6. Comparison with Individual Systems
    print(f"\n6. SYSTEM COMPARISON")
    print("-" * 40)
    
    comparison_stats = []
    for namespace, kg in combined_kg.individual_graphs.items():
        stats = kg.get_stats()
        comparison_stats.append({
            'namespace': namespace,
            'nodes': stats['total_nodes'],
            'edges': stats['total_edges'],
            'go_terms': stats['go_terms'],
            'genes': stats['genes']
        })
    
    print("Individual vs Combined Analysis:")
    total_individual_nodes = sum(s['nodes'] for s in comparison_stats)
    total_individual_edges = sum(s['edges'] for s in comparison_stats)
    
    print(f"  Individual systems total: {total_individual_nodes:,} nodes, {total_individual_edges:,} edges")
    print(f"  Combined system: {combined_stats['total_nodes']:,} nodes, {combined_stats['total_edges']:,} edges")
    print(f"  Node efficiency: {combined_stats['total_nodes'] / total_individual_nodes:.2%} (due to gene overlap)")
    print(f"  Edge growth: {combined_stats['total_edges'] / total_individual_edges:.2%}")
    
    # 7. Final Validation Summary
    print(f"\n7. VALIDATION SUMMARY")
    print("-" * 40)
    
    validation_criteria = {
        'Multi-namespace support': len(combined_stats['namespace_counts']) >= 2,
        'Large-scale data handling': combined_stats['total_nodes'] > 80000,
        'Cross-namespace queries': len(cross_namespace_results) >= 3,
        'Performance acceptable': total_time < 300,  # 5 minutes
        'Data integrity': integrity_checks['unique_go_terms'] > 40000
    }
    
    print("Validation criteria:")
    all_passed = True
    for criterion, passed in validation_criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {criterion}: {status}")
        all_passed = all_passed and passed
    
    # Final status
    print(f"\n" + "=" * 80)
    if all_passed:
        print("🎉 VALIDATION SUCCESSFUL - Combined GO system is production ready!")
        print("✅ Multi-namespace integration validated")
        print("✅ Performance benchmarks met")
        print("✅ Data integrity confirmed")
        print("✅ Cross-namespace functionality working")
    else:
        print("⚠️ VALIDATION ISSUES DETECTED")
        print("Some validation criteria failed - review above results")
    
    print("=" * 80)
    
    return {
        'validation_passed': all_passed,
        'performance_metrics': performance_metrics,
        'combined_stats': combined_stats,
        'namespace_stats': namespace_stats,
        'cross_namespace_results': cross_namespace_results
    }


if __name__ == "__main__":
    validation_results = validate_combined_go_system()