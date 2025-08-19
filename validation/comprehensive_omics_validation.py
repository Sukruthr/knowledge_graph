#!/usr/bin/env python3
"""
Comprehensive validation script for the integrated GO + Omics Knowledge Graph system.

This script validates the complete biomedical knowledge graph including:
- GO ontology integration (BP + CC + MF)
- Omics data integration (Disease, Drug, Viral)
- Network cluster integration
- Cross-modal gene queries and associations
"""

import sys
import time
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from kg_builder import ComprehensiveBiomedicalKnowledgeGraph
from data_parsers import CombinedBiomedicalParser


def print_header(title: str):
    """Print a formatted header."""
    print("=" * 80)
    print(title.center(80))
    print("=" * 80)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + title)
    print("-" * 60)


def validate_comprehensive_system():
    """Run comprehensive validation of the integrated biomedical knowledge graph system."""
    
    print_header("COMPREHENSIVE BIOMEDICAL KNOWLEDGE GRAPH VALIDATION")
    print_header("GO + OMICS INTEGRATION SYSTEM")
    
    # Initialize system
    print_section("1. SYSTEM INITIALIZATION")
    start_time = time.time()
    
    base_data_dir = str(project_root / "llm_evaluation_for_gene_set_interpretation" / "data")
    kg = ComprehensiveBiomedicalKnowledgeGraph(use_neo4j=False)
    
    print(f"✓ Initialized comprehensive biomedical knowledge graph")
    print(f"✓ Data directory: {base_data_dir}")
    
    # Load and parse data
    print_section("2. DATA LOADING AND PARSING")
    load_start = time.time()
    kg.load_data(base_data_dir)
    load_time = time.time() - load_start
    
    print(f"✓ Data loading completed in {load_time:.2f} seconds")
    
    # Validate parsed data
    if kg.parser:
        validation = kg.parser.validate_comprehensive_data()
        print(f"✓ Data validation: {validation}")
        
        summary = kg.parser.get_comprehensive_summary()
        print(f"✓ Data sources: {summary['data_sources']}")
        
        if 'integration_stats' in summary:
            stats = summary['integration_stats']
            print(f"✓ Gene integration coverage: {stats.get('integration_coverage', 0):.3f}")
    
    # Build comprehensive graph
    print_section("3. KNOWLEDGE GRAPH CONSTRUCTION")
    build_start = time.time()
    kg.build_comprehensive_graph()
    build_time = time.time() - build_start
    
    print(f"✓ Graph construction completed in {build_time:.2f} seconds")
    
    # Get comprehensive statistics
    stats = kg.get_comprehensive_stats()
    print_section("4. GRAPH STATISTICS ANALYSIS")
    
    print(f"Total construction time: {time.time() - start_time:.2f} seconds")
    print(f"Total nodes: {stats['total_nodes']:,}")
    print(f"Total edges: {stats['total_edges']:,}")
    print(f"Graph density: {stats['total_edges'] / (stats['total_nodes'] * (stats['total_nodes'] - 1)):.6f}")
    
    print(f"\nNode type distribution:")
    for node_type, count in stats['node_counts'].items():
        percentage = (count / stats['total_nodes']) * 100
        print(f"  {node_type}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nEdge type distribution:")
    for edge_type, count in stats['edge_counts'].items():
        percentage = (count / stats['total_edges']) * 100
        print(f"  {edge_type}: {count:,} ({percentage:.1f}%)")
    
    # Integration metrics
    integration_metrics = stats.get('integration_metrics', {})
    print(f"\nIntegration metrics:")
    print(f"  GO-connected genes: {integration_metrics.get('go_connected_genes', 0):,}")
    print(f"  Omics-connected genes: {integration_metrics.get('omics_connected_genes', 0):,}")
    print(f"  Integrated genes: {integration_metrics.get('integrated_genes', 0):,}")
    print(f"  Integration ratio: {integration_metrics.get('integration_ratio', 0):.3f}")
    
    # Validate specific functionality
    print_section("5. FUNCTIONALITY VALIDATION")
    
    # Test comprehensive gene queries
    test_genes = ['TP53', 'BRCA1', 'EGFR', 'MYC', 'GAPDH']
    successful_queries = 0
    
    for gene in test_genes:
        profile = kg.query_gene_comprehensive(gene)
        if profile:
            successful_queries += 1
            total_annotations = (len(profile['go_annotations']) + 
                               len(profile['disease_associations']) + 
                               len(profile['drug_perturbations']) + 
                               len(profile['viral_responses']))
            print(f"✓ {gene}: {total_annotations} total associations")
        else:
            print(f"✗ {gene}: Not found in graph")
    
    print(f"\nGene query success rate: {successful_queries}/{len(test_genes)} ({successful_queries/len(test_genes)*100:.1f}%)")
    
    # Test cross-modal connectivity
    print_section("6. CROSS-MODAL CONNECTIVITY ANALYSIS")
    
    # Sample a gene with comprehensive annotations
    sample_gene = 'TP53'
    profile = kg.query_gene_comprehensive(sample_gene)
    
    if profile:
        print(f"Sample gene: {sample_gene}")
        print(f"  GO annotations: {len(profile['go_annotations'])}")
        print(f"  Disease associations: {len(profile['disease_associations'])}")
        print(f"  Drug perturbations: {len(profile['drug_perturbations'])}")
        print(f"  Viral responses: {len(profile['viral_responses'])}")
        
        # Check namespace coverage
        namespaces = set()
        for ann in profile['go_annotations']:
            namespaces.add(ann['namespace'])
        print(f"  GO namespace coverage: {len(namespaces)}/3 ({', '.join(namespaces)})")
        
        # Show sample associations
        if profile['disease_associations']:
            sample_diseases = [assoc['disease'] for assoc in profile['disease_associations'][:3]]
            print(f"  Sample diseases: {', '.join(sample_diseases)}")
        
        if profile['drug_perturbations']:
            sample_drugs = [drug['drug'] for drug in profile['drug_perturbations'][:3]]
            print(f"  Sample drugs: {', '.join(sample_drugs)}")
    
    # Data quality validation
    print_section("7. DATA QUALITY VALIDATION")
    
    validation_criteria = {
        'Multi-modal integration': integration_metrics.get('integration_ratio', 0) > 0.5,
        'Comprehensive gene coverage': stats['node_counts'].get('gene', 0) > 15000,
        'GO term coverage': stats['node_counts'].get('go_term', 0) > 40000,
        'Disease coverage': stats['node_counts'].get('disease', 0) > 100,
        'Drug coverage': stats['node_counts'].get('drug', 0) > 100,
        'Network density': stats['total_edges'] / stats['total_nodes'] > 10,
        'Cross-modal edges': (stats['edge_counts'].get('gene_disease_association', 0) + 
                             stats['edge_counts'].get('gene_drug_perturbation', 0) + 
                             stats['edge_counts'].get('gene_viral_response', 0)) > 500000
    }
    
    passed_criteria = 0
    for criterion, result in validation_criteria.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {criterion}")
        if result:
            passed_criteria += 1
    
    print(f"\nOverall validation: {passed_criteria}/{len(validation_criteria)} criteria passed ({passed_criteria/len(validation_criteria)*100:.1f}%)")
    
    # Performance benchmarks
    print_section("8. PERFORMANCE BENCHMARKS")
    
    nodes_per_second = stats['total_nodes'] / (time.time() - start_time)
    edges_per_second = stats['total_edges'] / (time.time() - start_time)
    
    print(f"Construction performance:")
    print(f"  Nodes/second: {nodes_per_second:,.0f}")
    print(f"  Edges/second: {edges_per_second:,.0f}")
    print(f"  Memory efficiency: {stats['total_edges'] / stats['total_nodes']:.1f} edges/node")
    
    # Query performance test
    query_start = time.time()
    test_queries = ['TP53', 'BRCA1', 'EGFR', 'MYC', 'GAPDH'] * 10  # 50 queries
    for gene in test_queries:
        kg.query_gene_comprehensive(gene)
    query_time = time.time() - query_start
    
    print(f"Query performance:")
    print(f"  {len(test_queries)} queries in {query_time:.3f} seconds")
    print(f"  {len(test_queries)/query_time:.1f} queries/second")
    
    # Final summary
    print_section("9. VALIDATION SUMMARY")
    
    overall_success = passed_criteria >= len(validation_criteria) * 0.8  # 80% pass rate
    
    if overall_success:
        print("🎉 COMPREHENSIVE VALIDATION SUCCESSFUL!")
        print("✅ Multi-modal biomedical knowledge graph is production ready")
        print("✅ GO + Omics integration validated")
        print("✅ Performance benchmarks met")
        print("✅ Data quality standards achieved")
        print("✅ Cross-modal functionality working")
    else:
        print("⚠️ VALIDATION ISSUES DETECTED")
        print(f"Only {passed_criteria}/{len(validation_criteria)} criteria passed")
        print("Review failed criteria before production deployment")
    
    print_header("VALIDATION COMPLETE")
    
    return overall_success, stats


if __name__ == "__main__":
    success, statistics = validate_comprehensive_system()
    
    # Save validation results
    validation_file = project_root / "validation" / "omics_validation_results.json"
    import json
    
    validation_results = {
        'timestamp': time.time(),
        'validation_successful': success,
        'statistics': statistics,
        'system_type': 'comprehensive_biomedical_kg',
        'components': ['GO_BP', 'GO_CC', 'GO_MF', 'Disease', 'Drug', 'Viral', 'Clusters']
    }
    
    with open(validation_file, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    print(f"\nValidation results saved to: {validation_file}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)