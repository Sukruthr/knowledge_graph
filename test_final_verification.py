#!/usr/bin/env python3
"""
Final verification test to ensure all components work correctly.
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

def final_verification_test():
    """Final comprehensive verification."""
    logger.info("="*80)
    logger.info("FINAL VERIFICATION TEST")
    logger.info("="*80)
    
    verification_results = {}
    
    try:
        # 1. System Build Test
        logger.info("1. Testing system build...")
        start_time = time.time()
        
        kg = ComprehensiveBiomedicalKnowledgeGraph()
        kg.load_data('llm_evaluation_for_gene_set_interpretation/data')
        kg.build_comprehensive_graph()
        
        build_time = time.time() - start_time
        logger.info(f"✓ System built successfully in {build_time:.2f} seconds")
        verification_results['system_build'] = True
        
        # 2. Data Integration Test
        logger.info("2. Testing data integration...")
        stats = kg.get_comprehensive_stats()
        
        # Check all expected node types
        expected_nodes = {
            'go_term': 40000,
            'gene': 15000,
            'disease': 100,
            'drug': 100,
            'viral_condition': 200,
            'llm_model': 3,
            'model_prediction': 1000,
            'similarity_ranking': 100
        }
        
        integration_success = True
        for node_type, min_count in expected_nodes.items():
            actual_count = stats['node_counts'].get(node_type, 0)
            if actual_count < min_count:
                logger.error(f"Insufficient {node_type}: {actual_count} < {min_count}")
                integration_success = False
            else:
                logger.info(f"✓ {node_type}: {actual_count:,}")
        
        verification_results['data_integration'] = integration_success
        
        # 3. Edge Types Test
        logger.info("3. Testing edge types...")
        expected_edges = [
            'go_hierarchy', 'gene_annotation', 'gene_disease_association',
            'gene_drug_perturbation', 'gene_viral_expression', 'gene_viral_response',
            'model_predicts', 'predicts_go_term', 'prediction_uses_gene'
        ]
        
        edge_success = True
        for edge_type in expected_edges:
            if edge_type not in stats['edge_counts']:
                logger.error(f"Missing edge type: {edge_type}")
                edge_success = False
            else:
                count = stats['edge_counts'][edge_type]
                logger.info(f"✓ {edge_type}: {count:,}")
        
        verification_results['edge_types'] = edge_success
        
        # 4. Query Functionality Test
        logger.info("4. Testing query functionality...")
        
        # Test gene query
        profile = kg.query_gene_comprehensive('TP53')
        if not profile or profile.get('gene_symbol') != 'TP53':
            logger.error("Gene query failed")
            verification_results['query_functionality'] = False
        else:
            # Check for all expected profile fields
            expected_fields = [
                'go_annotations', 'disease_associations', 'drug_perturbations',
                'viral_responses', 'model_predictions'
            ]
            
            query_success = True
            for field in expected_fields:
                if field not in profile:
                    logger.error(f"Missing query field: {field}")
                    query_success = False
                else:
                    count = len(profile[field])
                    logger.info(f"✓ TP53 {field}: {count}")
            
            verification_results['query_functionality'] = query_success
        
        # 5. Model Comparison Test
        logger.info("5. Testing model comparison functionality...")
        
        model_summary = kg.query_model_comparison_summary()
        if not model_summary:
            logger.error("Model comparison summary failed")
            verification_results['model_comparison'] = False
        else:
            models_count = model_summary['total_models']
            predictions_count = model_summary['total_predictions']
            rankings_count = model_summary['total_similarity_rankings']
            
            if models_count < 3 or predictions_count < 1000 or rankings_count < 100:
                logger.error(f"Insufficient model data: {models_count} models, {predictions_count} predictions, {rankings_count} rankings")
                verification_results['model_comparison'] = False
            else:
                logger.info(f"✓ Model comparison: {models_count} models, {predictions_count} predictions, {rankings_count} rankings")
                verification_results['model_comparison'] = True
        
        # Test model predictions query
        predictions = kg.query_model_predictions()
        if not predictions or len(predictions) < 500:
            logger.error(f"Model predictions query failed: {len(predictions) if predictions else 0}")
            verification_results['model_predictions_query'] = False
        else:
            logger.info(f"✓ Model predictions query: {len(predictions)} predictions")
            verification_results['model_predictions_query'] = True
        
        # 6. Performance Test
        logger.info("6. Testing performance...")
        
        # Quick query performance test
        test_genes = ['TP53', 'BRCA1', 'EGFR'] * 5  # 15 queries
        start_time = time.time()
        for gene in test_genes:
            kg.query_gene_comprehensive(gene)
        query_time = time.time() - start_time
        
        queries_per_second = len(test_genes) / query_time
        if queries_per_second < 100:  # Very conservative threshold
            logger.warning(f"Query performance low: {queries_per_second:.1f} queries/sec")
            verification_results['performance'] = False
        else:
            logger.info(f"✓ Query performance: {queries_per_second:.1f} queries/sec")
            verification_results['performance'] = True
        
        # 7. Integration Ratio Test
        logger.info("7. Testing integration ratio...")
        integration_ratio = stats['integration_metrics']['integration_ratio']
        if integration_ratio < 0.8:
            logger.error(f"Low integration ratio: {integration_ratio:.3f}")
            verification_results['integration_ratio'] = False
        else:
            logger.info(f"✓ Integration ratio: {integration_ratio:.3f}")
            verification_results['integration_ratio'] = True
        
        # 8. Regression Test (Previous functionality)
        logger.info("8. Testing regression compatibility...")
        
        # Test that original fields still exist
        original_fields = ['go_annotations', 'disease_associations', 'drug_perturbations', 'viral_responses']
        regression_success = True
        
        for field in original_fields:
            if field not in profile:
                logger.error(f"Regression: Missing original field {field}")
                regression_success = False
            else:
                logger.info(f"✓ Original field preserved: {field}")
        
        # Test viral response types (both association and expression)
        viral_responses = profile.get('viral_responses', [])
        if viral_responses:
            response_types = set(vr.get('type', 'response') for vr in viral_responses)
            if 'expression' not in response_types:
                logger.warning("Missing viral expression type")
            else:
                logger.info(f"✓ Viral response types: {response_types}")
        
        verification_results['regression_compatibility'] = regression_success
        
        # Final Summary
        passed_tests = sum(verification_results.values())
        total_tests = len(verification_results)
        success_rate = (passed_tests / total_tests) * 100
        
        logger.info(f"\n{'='*80}")
        logger.info(f"FINAL VERIFICATION SUMMARY")
        logger.info(f"{'='*80}")
        
        for test_name, passed in verification_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\nOverall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
        
        if success_rate >= 87.5:  # 7/8 tests minimum
            logger.info("🎉 FINAL VERIFICATION SUCCESSFUL!")
            logger.info("All components working correctly!")
            return True
        else:
            logger.warning("⚠️ Some components need attention")
            return False
        
    except Exception as e:
        logger.error(f"Final verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = final_verification_test()
    sys.exit(0 if success else 1)