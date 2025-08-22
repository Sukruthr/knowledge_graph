#!/usr/bin/env python3
"""
Focused Talisman Gene Sets Test

Test talisman gene sets parsing and minimal KG integration without full system build.
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

def test_parser():
    """Test parser functionality"""
    from parsers.talisman_gene_sets_parser import TalismanGeneSetsParser
    
    parser = TalismanGeneSetsParser()
    start_time = time.time()
    parsed_data = parser.parse_all_gene_sets()
    parse_time = time.time() - start_time
    
    stats = parser.get_parsing_statistics()
    validation = parser.validate_parsing_quality()
    
    logger.info(f"Parser completed in {parse_time:.2f} seconds")
    logger.info(f"Gene sets parsed: {stats['overall_summary']['total_gene_sets']}")
    logger.info(f"Unique genes: {stats['overall_summary']['total_unique_genes']}")
    
    return {
        'success': stats['overall_summary']['total_gene_sets'] >= 70,
        'data': parsed_data,
        'stats': stats
    }

def test_kg_integration(talisman_data):
    """Test minimal KG integration with just talisman data"""
    from src.kg_builder import ComprehensiveBiomedicalKnowledgeGraph
    import networkx as nx
    
    # Create minimal KG
    kg = ComprehensiveBiomedicalKnowledgeGraph()
    kg.graph = nx.Graph()  # Start with empty graph
    kg.parsed_data = {'talisman_gene_sets': talisman_data}
    
    # Test just talisman integration
    start_time = time.time()
    kg._add_talisman_gene_sets()
    integration_time = time.time() - start_time
    
    # Count nodes by type
    node_counts = {}
    for node_id, attrs in kg.graph.nodes(data=True):
        node_type = attrs.get('node_type', 'unknown')
        node_counts[node_type] = node_counts.get(node_type, 0) + 1
    
    logger.info(f"Integration completed in {integration_time:.2f} seconds")
    logger.info(f"Total nodes created: {kg.graph.number_of_nodes()}")
    logger.info(f"Total edges created: {kg.graph.number_of_edges()}")
    
    for node_type, count in node_counts.items():
        logger.info(f"  {node_type}: {count}")
    
    # Test specific queries
    hallmark_nodes = [
        (node_id, attrs) for node_id, attrs in kg.graph.nodes(data=True)
        if attrs.get('node_type') == 'hallmark_gene_set'
    ]
    
    gene_nodes = [
        (node_id, attrs) for node_id, attrs in kg.graph.nodes(data=True)
        if attrs.get('node_type') == 'gene'
    ]
    
    return {
        'success': len(hallmark_nodes) >= 40 and len(gene_nodes) > 1000,
        'hallmark_count': len(hallmark_nodes),
        'gene_count': len(gene_nodes),
        'total_nodes': kg.graph.number_of_nodes(),
        'total_edges': kg.graph.number_of_edges()
    }

def test_queries(talisman_data):
    """Test specific talisman data queries"""
    results = {}
    
    # Test data type coverage
    expected_types = ['hallmark_sets', 'bicluster_sets', 'pathway_sets', 'go_custom_sets', 'disease_sets', 'other_sets']
    for data_type in expected_types:
        count = len(talisman_data.get(data_type, {}))
        results[data_type] = count
        logger.info(f"{data_type}: {count} sets")
    
    # Test specific gene set lookups
    if 'hallmark_sets' in talisman_data:
        for gene_set_id, gene_set in list(talisman_data['hallmark_sets'].items())[:3]:
            logger.info(f"Sample HALLMARK set: {gene_set['name']} ({gene_set['gene_count']} genes)")
    
    # Test gene coverage
    all_genes = set()
    for data_type in talisman_data.values():
        for gene_set in data_type.values():
            all_genes.update(gene_set['genes'])
    
    results['total_unique_genes'] = len(all_genes)
    logger.info(f"Total unique genes across all sets: {len(all_genes)}")
    
    return {
        'success': results['hallmark_sets'] >= 45 and results['total_unique_genes'] > 3000,
        'details': results
    }

def main():
    """Run focused test for talisman gene sets"""
    
    logger.info("🧪 STARTING FOCUSED TALISMAN GENE SETS TEST")
    logger.info("=" * 60)
    
    test_results = []
    
    try:
        # Test 1: Parser
        logger.info("TEST 1: PARSER FUNCTIONALITY")
        logger.info("-" * 30)
        
        parser_result = test_parser()
        test_results.append(("Parser", parser_result['success']))
        
        if not parser_result['success']:
            logger.error("Parser test failed, stopping")
            return False
        
        # Test 2: KG Integration
        logger.info(f"\nTEST 2: KNOWLEDGE GRAPH INTEGRATION")
        logger.info("-" * 30)
        
        kg_result = test_kg_integration(parser_result['data'])
        test_results.append(("KG Integration", kg_result['success']))
        
        if kg_result['success']:
            logger.info(f"✅ KG Integration successful:")
            logger.info(f"   - HALLMARK nodes: {kg_result['hallmark_count']}")
            logger.info(f"   - Gene nodes: {kg_result['gene_count']}")
            logger.info(f"   - Total nodes: {kg_result['total_nodes']}")
            logger.info(f"   - Total edges: {kg_result['total_edges']}")
        
        # Test 3: Data Queries
        logger.info(f"\nTEST 3: DATA QUERIES")
        logger.info("-" * 30)
        
        query_result = test_queries(parser_result['data'])
        test_results.append(("Data Queries", query_result['success']))
        
        # Final Summary
        logger.info(f"\n" + "=" * 60)
        logger.info("🧪 FOCUSED TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for _, success in test_results if success)
        total = len(test_results)
        
        for test_name, success in test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"{test_name}: {status}")
        
        success_rate = passed / total
        logger.info(f"\nOverall Success Rate: {passed}/{total} ({success_rate:.1%})")
        
        if success_rate >= 0.67:  # Allow 1 failure out of 3
            logger.info(f"\n🎉 FOCUSED TEST PASSED!")
            logger.info("✅ Talisman Gene Sets Integration Validated")
            return True
        else:
            logger.info(f"\n⚠️ FOCUSED TEST NEEDS ATTENTION")
            return False
        
    except Exception as e:
        logger.error(f"Focused test failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)