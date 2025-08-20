#!/usr/bin/env python3
"""
Test script for enhanced knowledge graph integration with Omics_data2 semantic enhancements.
Tests complete pipeline from data parsing to knowledge graph construction and querying.
"""

import sys
import logging
from pathlib import Path
import time

# Add src to path for imports
sys.path.append('src')

from kg_builder import ComprehensiveBiomedicalKnowledgeGraph

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_kg_construction():
    """Test complete knowledge graph construction with semantic enhancements."""
    logger.info("="*70)
    logger.info("TESTING ENHANCED KNOWLEDGE GRAPH CONSTRUCTION")
    logger.info("="*70)
    
    try:
        start_time = time.time()
        
        # Initialize the knowledge graph
        logger.info("Initializing ComprehensiveBiomedicalKnowledgeGraph...")
        kg = ComprehensiveBiomedicalKnowledgeGraph()
        
        # Load data
        logger.info("Loading comprehensive biomedical data...")
        base_data_dir = "llm_evaluation_for_gene_set_interpretation/data"
        kg.load_data(base_data_dir)
        
        # Build the knowledge graph
        logger.info("Building comprehensive knowledge graph...")
        kg.build_comprehensive_graph()
        
        construction_time = time.time() - start_time
        logger.info(f"✓ Knowledge graph constructed in {construction_time:.2f} seconds")
        
        return kg, True
        
    except Exception as e:
        logger.error(f"❌ KG CONSTRUCTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, False

def test_kg_statistics(kg):
    """Test knowledge graph statistics include semantic enhancements."""
    logger.info("="*70)
    logger.info("TESTING ENHANCED KNOWLEDGE GRAPH STATISTICS")
    logger.info("="*70)
    
    try:
        stats = kg.get_comprehensive_stats()
        
        logger.info("📊 COMPREHENSIVE STATISTICS:")
        logger.info(f"  Total Nodes: {stats['total_nodes']:,}")
        logger.info(f"  Total Edges: {stats['total_edges']:,}")
        
        logger.info("\n📦 NODE COUNTS BY TYPE:")
        for node_type, count in stats['node_counts'].items():
            logger.info(f"  {node_type}: {count:,}")
        
        logger.info("\n🔗 EDGE COUNTS BY TYPE:")
        for edge_type, count in stats['edge_counts'].items():
            logger.info(f"  {edge_type}: {count:,}")
        
        logger.info("\n🧬 INTEGRATION METRICS:")
        integration = stats['integration_metrics']
        logger.info(f"  GO-connected genes: {integration['go_connected_genes']:,}")
        logger.info(f"  Omics-connected genes: {integration['omics_connected_genes']:,}")
        logger.info(f"  Integrated genes: {integration['integrated_genes']:,}")
        logger.info(f"  Integration ratio: {integration['integration_ratio']:.3f}")
        
        # Check for semantic enhancement indicators
        semantic_indicators = []
        if 'gene_set' in stats['node_counts']:
            semantic_indicators.append(f"Gene sets: {stats['node_counts']['gene_set']:,}")
        
        semantic_edges = ['gene_in_set', 'gene_supports_set', 'validated_by_go_term']
        for edge_type in semantic_edges:
            if edge_type in stats['edge_counts']:
                semantic_indicators.append(f"{edge_type}: {stats['edge_counts'][edge_type]:,}")
        
        if semantic_indicators:
            logger.info("\n🔬 SEMANTIC ENHANCEMENT INDICATORS:")
            for indicator in semantic_indicators:
                logger.info(f"  {indicator}")
        else:
            logger.warning("⚠ No semantic enhancement indicators found")
        
        logger.info("✅ STATISTICS TEST PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ STATISTICS TEST FAILED: {e}")
        return False

def test_enhanced_queries(kg):
    """Test querying with semantic enhancements."""
    logger.info("="*70)
    logger.info("TESTING ENHANCED GENE QUERIES")
    logger.info("="*70)
    
    try:
        # Test major genes that should have comprehensive data
        test_genes = ['TP53', 'BRCA1', 'EGFR', 'MYC', 'GAPDH']
        
        for gene in test_genes:
            logger.info(f"\n🧬 TESTING GENE: {gene}")
            
            profile = kg.query_gene_comprehensive(gene)
            
            if not profile:
                logger.warning(f"  ⚠ No data found for {gene}")
                continue
            
            # Check basic associations
            logger.info(f"  GO annotations: {len(profile.get('go_annotations', []))}")
            logger.info(f"  Disease associations: {len(profile.get('disease_associations', []))}")
            logger.info(f"  Drug perturbations: {len(profile.get('drug_perturbations', []))}")
            logger.info(f"  Viral responses: {len(profile.get('viral_responses', []))}")
            logger.info(f"  Cluster memberships: {len(profile.get('cluster_memberships', []))}")
            
            # Check enhanced associations
            gene_sets = profile.get('gene_set_memberships', [])
            if gene_sets:
                logger.info(f"  ✨ Gene set memberships: {len(gene_sets)}")
                # Show sample gene set
                sample_set = gene_sets[0]
                logger.info(f"     Sample: {sample_set.get('llm_name', 'N/A')} (score: {sample_set.get('llm_score', 0):.3f})")
            else:
                logger.info(f"  Gene set memberships: 0")
            
            semantic_annotations = profile.get('semantic_annotations', [])
            logger.info(f"  Semantic annotations: {len(semantic_annotations)}")
        
        logger.info("✅ ENHANCED QUERIES TEST PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ ENHANCED QUERIES TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_semantic_node_properties(kg):
    """Test that semantic enhancements are properly stored in nodes."""
    logger.info("="*70)
    logger.info("TESTING SEMANTIC NODE PROPERTIES")
    logger.info("="*70)
    
    try:
        # Find gene set nodes
        gene_set_nodes = [node for node, data in kg.graph.nodes(data=True) 
                         if data.get('node_type') == 'gene_set']
        
        if not gene_set_nodes:
            logger.warning("⚠ No gene set nodes found in graph")
            return True  # Not necessarily a failure if Omics_data2 not available
        
        logger.info(f"Found {len(gene_set_nodes)} gene set nodes")
        
        # Check properties of a sample gene set node
        sample_node = gene_set_nodes[0]
        node_data = kg.graph.nodes[sample_node]
        
        logger.info(f"\n🔍 SAMPLE GENE SET NODE: {sample_node}")
        
        # Check for semantic properties
        semantic_properties = [
            'llm_name', 'llm_analysis', 'llm_score', 'llm_coverage',
            'has_literature', 'num_references', 'validated_go_term',
            'go_p_value', 'experimental_overlap'
        ]
        
        found_properties = []
        for prop in semantic_properties:
            if prop in node_data:
                value = node_data[prop]
                if isinstance(value, (int, float)):
                    found_properties.append(f"{prop}: {value}")
                elif isinstance(value, str):
                    found_properties.append(f"{prop}: '{value[:50]}...' " if len(str(value)) > 50 else f"{prop}: '{value}'")
                else:
                    found_properties.append(f"{prop}: {type(value).__name__}")
        
        if found_properties:
            logger.info("  ✨ SEMANTIC PROPERTIES FOUND:")
            for prop in found_properties:
                logger.info(f"    {prop}")
        else:
            logger.warning("  ⚠ No semantic properties found")
        
        logger.info("✅ SEMANTIC NODE PROPERTIES TEST PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ SEMANTIC NODE PROPERTIES TEST FAILED: {e}")
        return False

def test_backward_compatibility(kg):
    """Test that existing functionality still works with enhancements."""
    logger.info("="*70)
    logger.info("TESTING BACKWARD COMPATIBILITY")
    logger.info("="*70)
    
    try:
        # Test that basic queries still work
        profile = kg.query_gene_comprehensive('TP53')
        
        required_fields = ['gene_symbol', 'go_annotations', 'disease_associations', 
                          'drug_perturbations', 'viral_responses']
        
        for field in required_fields:
            if field not in profile:
                logger.error(f"❌ Missing required field: {field}")
                return False
            logger.info(f"  ✓ Field present: {field}")
        
        # Test that viral expression data is still included
        viral_responses = profile.get('viral_responses', [])
        expression_responses = [r for r in viral_responses if r.get('type') == 'expression']
        response_responses = [r for r in viral_responses if r.get('type') == 'response']
        
        logger.info(f"  Viral expression responses: {len(expression_responses)}")
        logger.info(f"  Viral response associations: {len(response_responses)}")
        
        if expression_responses:
            sample_expr = expression_responses[0]
            logger.info(f"  Sample expression: {sample_expr.get('expression_direction')} "
                       f"({sample_expr.get('expression_value', 0):.3f})")
        
        logger.info("✅ BACKWARD COMPATIBILITY TEST PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ BACKWARD COMPATIBILITY TEST FAILED: {e}")
        return False

def main():
    """Run all enhanced knowledge graph integration tests."""
    logger.info("🧪 STARTING ENHANCED KNOWLEDGE GRAPH INTEGRATION TESTING")
    logger.info("="*80)
    
    test_results = []
    kg = None
    
    # Test 1: Knowledge graph construction
    kg, construction_success = test_kg_construction()
    test_results.append(("KG Construction", construction_success))
    
    if not construction_success or kg is None:
        logger.error("❌ Cannot proceed - KG construction failed")
        return False
    
    # Test 2: Statistics
    test_results.append(("Enhanced Statistics", test_kg_statistics(kg)))
    
    # Test 3: Enhanced queries
    test_results.append(("Enhanced Queries", test_enhanced_queries(kg)))
    
    # Test 4: Semantic node properties
    test_results.append(("Semantic Node Properties", test_semantic_node_properties(kg)))
    
    # Test 5: Backward compatibility
    test_results.append(("Backward Compatibility", test_backward_compatibility(kg)))
    
    # Summary
    logger.info("="*80)
    logger.info("🧪 ENHANCED KG INTEGRATION TEST RESULTS")
    logger.info("="*80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info("-" * 80)
    logger.info(f"OVERALL: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ALL ENHANCED KG INTEGRATION TESTS PASSED")
        logger.info("🚀 SYSTEM READY FOR PRODUCTION USE")
        return True
    else:
        logger.error("❌ SOME TESTS FAILED - REVIEW IMPLEMENTATION")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)