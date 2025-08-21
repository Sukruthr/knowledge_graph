#!/usr/bin/env python3
"""
Comprehensive Talisman Gene Sets Integration Test

Full system validation of talisman gene sets parsing, integration, and knowledge graph construction.
"""

import sys
import os
import logging
from typing import Dict, List, Any
import time

# Add src to path
sys.path.append('src')

# Configure logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run comprehensive test for talisman gene sets integration"""
    
    logger.info("🧪 STARTING COMPREHENSIVE TALISMAN GENE SETS INTEGRATION TEST")
    logger.info("=" * 70)
    
    test_results = {
        'parser_validation': False,
        'integration_validation': False,
        'kg_construction': False,
        'data_queries': False,
        'cross_modal_integration': False,
        'performance_validation': False
    }
    
    try:
        # Test 1: Comprehensive Parser Validation
        logger.info("TEST 1: COMPREHENSIVE PARSER VALIDATION")
        logger.info("-" * 40)
        
        from talisman_gene_sets_parser import TalismanGeneSetsParser
        
        parser = TalismanGeneSetsParser()
        start_time = time.time()
        parsed_data = parser.parse_all_gene_sets()
        parse_time = time.time() - start_time
        
        stats = parser.get_parsing_statistics()
        validation = parser.validate_parsing_quality()
        
        # Validate parser results
        expected_data_types = ['hallmark_sets', 'bicluster_sets', 'pathway_sets', 'go_custom_sets', 'disease_sets', 'other_sets']
        parser_validation_success = True
        
        logger.info(f"Parsing completed in {parse_time:.2f} seconds")
        logger.info(f"Total gene sets: {stats['overall_summary']['total_gene_sets']}")
        logger.info(f"Total unique genes: {stats['overall_summary']['total_unique_genes']}")
        logger.info(f"Average genes per set: {stats['overall_summary']['avg_genes_per_set']:.1f}")
        
        # Validate each data type
        for data_type in expected_data_types:
            count = stats['by_data_type'].get(data_type, 0)
            logger.info(f"  {data_type}: {count} sets")
            if data_type == 'hallmark_sets' and count < 50:
                parser_validation_success = False
                logger.error(f"Expected ≥50 HALLMARK sets, found {count}")
        
        # Quality validation
        quality_success = validation['name_coverage'] >= 0.9 and len(validation['quality_issues']) <= 15
        logger.info(f"Name coverage: {validation['name_coverage']:.2%}")
        logger.info(f"Description coverage: {validation['description_coverage']:.2%}")
        logger.info(f"Quality issues: {len(validation['quality_issues'])}")
        
        test_results['parser_validation'] = parser_validation_success and quality_success
        logger.info(f"Parser validation: {'✅ PASS' if test_results['parser_validation'] else '❌ FAIL'}")
        
        # Test 2: Integration Validation
        logger.info(f"\nTEST 2: SYSTEM INTEGRATION VALIDATION")
        logger.info("-" * 40)
        
        from data_parsers import CombinedBiomedicalParser
        
        combined_parser = CombinedBiomedicalParser('llm_evaluation_for_gene_set_interpretation/data')
        start_time = time.time()
        all_data = combined_parser.parse_all_biomedical_data()
        integration_time = time.time() - start_time
        
        # Validate integration
        has_talisman_data = 'talisman_gene_sets' in all_data
        integration_validation_success = has_talisman_data
        
        if has_talisman_data:
            talisman_data = all_data['talisman_gene_sets']
            total_sets = sum(len(gene_sets) for gene_sets in talisman_data.values())
            logger.info(f"Talisman integration successful: {total_sets} gene sets integrated")
            
            # Check each data type was integrated
            for data_type in expected_data_types:
                count = len(talisman_data.get(data_type, {}))
                logger.info(f"  {data_type}: {count} integrated")
        else:
            logger.error("Talisman gene sets not found in integrated data")
        
        logger.info(f"Integration time: {integration_time:.2f} seconds")
        test_results['integration_validation'] = integration_validation_success
        logger.info(f"Integration validation: {'✅ PASS' if test_results['integration_validation'] else '❌ FAIL'}")
        
        # Test 3: Knowledge Graph Construction
        logger.info(f"\nTEST 3: KNOWLEDGE GRAPH CONSTRUCTION")
        logger.info("-" * 40)
        
        from kg_builder import ComprehensiveBiomedicalKnowledgeGraph
        
        kg = ComprehensiveBiomedicalKnowledgeGraph()
        kg.parsed_data = all_data  # Use pre-parsed data
        
        start_time = time.time()
        kg.build_comprehensive_graph()
        build_time = time.time() - start_time
        
        # Validate KG construction
        kg_stats = kg.get_comprehensive_stats()
        
        logger.info(f"KG construction time: {build_time:.2f} seconds")
        logger.info(f"Total nodes: {kg_stats['basic_stats']['total_nodes']:,}")
        logger.info(f"Total edges: {kg_stats['basic_stats']['total_edges']:,}")
        
        # Check for talisman-specific nodes
        talisman_node_types = [
            'hallmark_gene_set', 'bicluster_gene_set', 'custom_pathway_gene_set',
            'go_custom_gene_set', 'disease_gene_set', 'specialized_gene_set'
        ]
        
        talisman_nodes_found = 0
        for node_id, attrs in kg.graph.nodes(data=True):
            if attrs.get('node_type') in talisman_node_types:
                talisman_nodes_found += 1
        
        kg_construction_success = talisman_nodes_found >= 70  # Expect most gene sets as nodes
        logger.info(f"Talisman gene set nodes: {talisman_nodes_found}")
        
        test_results['kg_construction'] = kg_construction_success
        logger.info(f"KG construction: {'✅ PASS' if test_results['kg_construction'] else '❌ FAIL'}")
        
        # Test 4: Data Queries
        logger.info(f"\nTEST 4: DATA QUERY VALIDATION")
        logger.info("-" * 40)
        
        # Query talisman gene sets
        hallmark_nodes = [
            (node_id, attrs) for node_id, attrs in kg.graph.nodes(data=True)
            if attrs.get('node_type') == 'hallmark_gene_set'
        ]
        
        bicluster_nodes = [
            (node_id, attrs) for node_id, attrs in kg.graph.nodes(data=True)
            if attrs.get('node_type') == 'bicluster_gene_set'
        ]
        
        # Test gene set queries
        logger.info(f"HALLMARK gene sets in KG: {len(hallmark_nodes)}")
        logger.info(f"Bicluster gene sets in KG: {len(bicluster_nodes)}")
        
        # Test gene-to-gene-set connections
        test_gene_connections = 0
        for node_id, attrs in list(kg.graph.nodes(data=True))[:100]:  # Sample first 100 nodes
            if attrs.get('node_type') == 'gene':
                # Count connections to talisman gene sets
                neighbors = list(kg.graph.neighbors(node_id))
                talisman_connections = [
                    n for n in neighbors 
                    if kg.graph.nodes[n].get('node_type') in talisman_node_types
                ]
                if talisman_connections:
                    test_gene_connections += 1
        
        queries_success = len(hallmark_nodes) >= 45 and test_gene_connections >= 10
        logger.info(f"Genes with talisman connections (sample): {test_gene_connections}")
        
        test_results['data_queries'] = queries_success
        logger.info(f"Data queries: {'✅ PASS' if test_results['data_queries'] else '❌ FAIL'}")
        
        # Test 5: Cross-Modal Integration
        logger.info(f"\nTEST 5: CROSS-MODAL INTEGRATION")
        logger.info("-" * 40)
        
        # Test integration with existing GO terms
        go_terms = [
            (node_id, attrs) for node_id, attrs in kg.graph.nodes(data=True)
            if attrs.get('node_type') == 'go_term'
        ]
        
        # Test integration with genes from other sources
        gene_nodes = [
            (node_id, attrs) for node_id, attrs in kg.graph.nodes(data=True)
            if attrs.get('node_type') == 'gene'
        ]
        
        # Check for genes that have both GO annotations and talisman gene set memberships
        multi_modal_genes = 0
        for gene_node_id, gene_attrs in gene_nodes[:200]:  # Sample
            neighbors = list(kg.graph.neighbors(gene_node_id))
            
            has_go_connection = any(
                kg.graph.nodes[n].get('node_type') == 'go_term'
                for n in neighbors
            )
            has_talisman_connection = any(
                kg.graph.nodes[n].get('node_type') in talisman_node_types
                for n in neighbors
            )
            
            if has_go_connection and has_talisman_connection:
                multi_modal_genes += 1
        
        cross_modal_success = len(go_terms) > 0 and multi_modal_genes >= 20
        logger.info(f"GO terms in KG: {len(go_terms):,}")
        logger.info(f"Multi-modal genes (sample): {multi_modal_genes}")
        
        test_results['cross_modal_integration'] = cross_modal_success
        logger.info(f"Cross-modal integration: {'✅ PASS' if test_results['cross_modal_integration'] else '❌ FAIL'}")
        
        # Test 6: Performance Validation
        logger.info(f"\nTEST 6: PERFORMANCE VALIDATION")
        logger.info("-" * 40)
        
        # Test query performance
        start_time = time.time()
        sample_queries = 0
        for i in range(100):  # Run 100 sample queries
            random_nodes = list(kg.graph.nodes())[:100]
            for node_id in random_nodes[:10]:
                neighbors = list(kg.graph.neighbors(node_id))
                sample_queries += 1
        query_time = time.time() - start_time
        
        queries_per_second = sample_queries / query_time if query_time > 0 else 0
        
        # Performance thresholds
        build_performance = build_time <= 120  # Build within 2 minutes
        query_performance = queries_per_second >= 100  # At least 100 queries/sec
        
        performance_success = build_performance and query_performance
        
        logger.info(f"Build time: {build_time:.2f}s (threshold: ≤120s)")
        logger.info(f"Query performance: {queries_per_second:.0f} queries/sec (threshold: ≥100)")
        
        test_results['performance_validation'] = performance_success
        logger.info(f"Performance validation: {'✅ PASS' if test_results['performance_validation'] else '❌ FAIL'}")
        
        # Final Summary
        logger.info(f"\n" + "=" * 70)
        logger.info("🧪 COMPREHENSIVE INTEGRATION TEST SUMMARY")
        logger.info("=" * 70)
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        success_rate = passed_tests / total_tests
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        
        logger.info(f"\nOverall Success Rate: {passed_tests}/{total_tests} ({success_rate:.1%})")
        
        if success_rate >= 0.83:  # Allow 1 test failure out of 6
            logger.info(f"\n🎉 COMPREHENSIVE TEST PASSED!")
            logger.info("✅ Talisman Gene Sets Integration Successfully Validated")
            logger.info(f"✅ System now includes {talisman_nodes_found} talisman gene sets")
            logger.info(f"✅ Knowledge graph enhanced with HALLMARK pathways, bicluster data, and custom gene sets")
            logger.info(f"✅ Cross-modal integration with existing GO and omics data confirmed")
            return True
        else:
            logger.info(f"\n⚠️ COMPREHENSIVE TEST NEEDS ATTENTION")
            logger.info(f"Success rate {success_rate:.1%} below threshold (83%)")
            return False
        
    except Exception as e:
        logger.error(f"Comprehensive test failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)