#!/usr/bin/env python3
"""
Quick integration test for model comparison and existing components.
"""

import sys
import time
import logging

# Add src to path
sys.path.append('src')

from kg_builder import ComprehensiveBiomedicalKnowledgeGraph

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_integration_test():
    """Quick test of all major components."""
    logger.info("="*80)
    logger.info("QUICK INTEGRATION TEST")
    logger.info("="*80)
    
    try:
        # Build system
        logger.info("1. Building comprehensive knowledge graph...")
        start_time = time.time()
        
        kg = ComprehensiveBiomedicalKnowledgeGraph()
        kg.load_data('llm_evaluation_for_gene_set_interpretation/data')
        kg.build_comprehensive_graph()
        
        build_time = time.time() - start_time
        logger.info(f"✓ System built in {build_time:.2f} seconds")
        
        # Get stats
        stats = kg.get_comprehensive_stats()
        logger.info(f"✓ Nodes: {stats['total_nodes']:,}, Edges: {stats['total_edges']:,}")
        
        # Test GO component
        go_terms = stats['node_counts'].get('go_term', 0)
        if go_terms < 40000:
            logger.error(f"Insufficient GO terms: {go_terms}")
            return False
        logger.info(f"✓ GO terms: {go_terms:,}")
        
        # Test Omics component
        diseases = stats['node_counts'].get('disease', 0)
        drugs = stats['node_counts'].get('drug', 0)
        viral = stats['node_counts'].get('viral_condition', 0)
        
        if diseases == 0 or drugs == 0 or viral == 0:
            logger.error(f"Missing Omics data: diseases={diseases}, drugs={drugs}, viral={viral}")
            return False
        logger.info(f"✓ Omics: {diseases} diseases, {drugs} drugs, {viral} viral")
        
        # Test Model Comparison component
        models = stats['node_counts'].get('llm_model', 0)
        predictions = stats['node_counts'].get('model_prediction', 0)
        rankings = stats['node_counts'].get('similarity_ranking', 0)
        
        if models < 3 or predictions < 1000 or rankings < 100:
            logger.error(f"Insufficient model data: models={models}, predictions={predictions}, rankings={rankings}")
            return False
        logger.info(f"✓ Models: {models} models, {predictions} predictions, {rankings} rankings")
        
        # Test integration
        integration_ratio = stats['integration_metrics']['integration_ratio']
        if integration_ratio < 0.8:
            logger.error(f"Low integration ratio: {integration_ratio:.3f}")
            return False
        logger.info(f"✓ Integration ratio: {integration_ratio:.3f}")
        
        # Test queries
        logger.info("2. Testing query functionality...")
        
        # Gene query
        profile = kg.query_gene_comprehensive('TP53')
        if not profile or profile['gene_symbol'] != 'TP53':
            logger.error("Gene query failed")
            return False
        
        required_fields = ['go_annotations', 'disease_associations', 'model_predictions']
        for field in required_fields:
            if field not in profile:
                logger.error(f"Missing query field: {field}")
                return False
        
        logger.info(f"✓ TP53 query: {len(profile['go_annotations'])} GO, {len(profile['disease_associations'])} diseases, {len(profile['model_predictions'])} predictions")
        
        # Model query
        model_summary = kg.query_model_comparison_summary()
        if not model_summary or model_summary['total_models'] < 3:
            logger.error("Model summary query failed")
            return False
        
        logger.info(f"✓ Model summary: {model_summary['total_models']} models")
        
        # Model predictions query
        predictions = kg.query_model_predictions()
        if not predictions or len(predictions) < 500:
            logger.error("Model predictions query failed")
            return False
        
        logger.info(f"✓ Model predictions: {len(predictions)} total")
        
        # Test edge types
        logger.info("3. Testing edge types...")
        
        required_edges = [
            'go_hierarchy', 'gene_annotation', 'gene_disease_association',
            'gene_drug_perturbation', 'gene_viral_expression', 
            'model_predicts', 'predicts_go_term'
        ]
        
        missing_edges = []
        for edge_type in required_edges:
            if edge_type not in stats['edge_counts']:
                missing_edges.append(edge_type)
            else:
                count = stats['edge_counts'][edge_type]
                logger.info(f"✓ {edge_type}: {count:,}")
        
        if missing_edges:
            logger.error(f"Missing edge types: {missing_edges}")
            return False
        
        logger.info("="*80)
        logger.info("✅ QUICK INTEGRATION TEST PASSED")
        logger.info("All major components working correctly!")
        logger.info("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Quick integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_integration_test()
    sys.exit(0 if success else 1)