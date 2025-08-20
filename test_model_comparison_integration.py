#!/usr/bin/env python3
"""
Test the model comparison integration with the comprehensive biomedical knowledge graph.
"""

import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

from kg_builder import ComprehensiveBiomedicalKnowledgeGraph

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_comparison_integration():
    """Test the complete model comparison integration."""
    logger.info("="*80)
    logger.info("TESTING MODEL COMPARISON INTEGRATION")
    logger.info("="*80)
    
    try:
        # Initialize comprehensive knowledge graph
        logger.info("1. Initializing ComprehensiveBiomedicalKnowledgeGraph...")
        kg = ComprehensiveBiomedicalKnowledgeGraph()
        
        # Load data (should include model comparison data now)
        logger.info("2. Loading comprehensive biomedical data...")
        start_time = time.time()
        kg.load_data('llm_evaluation_for_gene_set_interpretation/data')
        load_time = time.time() - start_time
        logger.info(f"Data loading completed in {load_time:.2f} seconds")
        
        # Build comprehensive graph (should include model comparison integration)
        logger.info("3. Building comprehensive biomedical knowledge graph...")
        start_time = time.time()
        kg.build_comprehensive_graph()
        build_time = time.time() - start_time
        logger.info(f"Graph construction completed in {build_time:.2f} seconds")
        
        # Get comprehensive statistics
        logger.info("4. Analyzing comprehensive graph statistics...")
        stats = kg.get_comprehensive_stats()
        
        logger.info("="*60)
        logger.info("COMPREHENSIVE GRAPH STATISTICS")
        logger.info("="*60)
        logger.info(f"Total Nodes: {stats['total_nodes']:,}")
        logger.info(f"Total Edges: {stats['total_edges']:,}")
        
        logger.info("\nNode Types:")
        for node_type, count in stats['node_counts'].items():
            logger.info(f"  {node_type}: {count:,}")
        
        logger.info("\nEdge Types:")
        for edge_type, count in stats['edge_counts'].items():
            logger.info(f"  {edge_type}: {count:,}")
        
        # Test model comparison specific queries
        logger.info("\n5. Testing model comparison queries...")
        
        # Test model comparison summary
        model_summary = kg.query_model_comparison_summary()
        logger.info("="*60)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("="*60)
        logger.info(f"Total LLM Models: {model_summary['total_models']}")
        logger.info(f"Total Predictions: {model_summary['total_predictions']}")
        logger.info(f"Total Similarity Rankings: {model_summary['total_similarity_rankings']}")
        logger.info(f"GO Terms with Predictions: {model_summary['go_term_coverage']}")
        
        if model_summary['model_performance']:
            logger.info("\nModel Performance Metrics:")
            for model_name, metrics in model_summary['model_performance'].items():
                logger.info(f"  {model_name}:")
                logger.info(f"    Mean Confidence: {metrics['mean_confidence']:.3f}")
                logger.info(f"    Mean Similarity: {metrics['mean_similarity']:.3f}")
                logger.info(f"    Mean Percentile: {metrics['mean_percentile']:.3f}")
                logger.info(f"    Robustness Score: {metrics['contamination_robustness']:.3f}")
        
        if model_summary['scenario_coverage']:
            logger.info("\nScenario Coverage:")
            for scenario, count in model_summary['scenario_coverage'].items():
                logger.info(f"  {scenario}: {count} predictions")
        
        # Test model predictions query
        logger.info("\n6. Testing model predictions query...")
        all_predictions = kg.query_model_predictions()
        logger.info(f"Total model predictions available: {len(all_predictions)}")
        
        if all_predictions:
            # Show top 3 predictions by confidence
            logger.info("\nTop 3 Predictions by Confidence:")
            for i, pred in enumerate(all_predictions[:3], 1):
                logger.info(f"  {i}. {pred['model_name']} - {pred['go_id']}")
                logger.info(f"     Scenario: {pred['scenario']}")
                logger.info(f"     Predicted: {pred['predicted_name']}")
                logger.info(f"     Confidence: {pred['confidence_score']:.3f}")
                logger.info(f"     True Description: {pred['true_description']}")
        
        # Test filtering by specific model
        if model_summary['model_performance']:
            test_model = list(model_summary['model_performance'].keys())[0]
            model_predictions = kg.query_model_predictions(model_name=test_model)
            logger.info(f"\nPredictions for {test_model}: {len(model_predictions)}")
        
        # Test gene query with model predictions
        logger.info("\n7. Testing comprehensive gene query with model predictions...")
        test_gene = "TP53"
        gene_profile = kg.query_gene_comprehensive(test_gene)
        
        if gene_profile:
            logger.info(f"\n{test_gene} Comprehensive Profile:")
            logger.info(f"  GO Annotations: {len(gene_profile.get('go_annotations', []))}")
            logger.info(f"  Disease Associations: {len(gene_profile.get('disease_associations', []))}")
            logger.info(f"  Drug Perturbations: {len(gene_profile.get('drug_perturbations', []))}")
            logger.info(f"  Viral Responses: {len(gene_profile.get('viral_responses', []))}")
            logger.info(f"  Model Predictions: {len(gene_profile.get('model_predictions', []))}")
            
            # Show model predictions for this gene
            model_preds = gene_profile.get('model_predictions', [])
            if model_preds:
                logger.info(f"\n  Model Predictions for {test_gene}:")
                for pred in model_preds[:3]:  # Show first 3
                    logger.info(f"    {pred['model_name']} - {pred['scenario']}: {pred['predicted_name']} (confidence: {pred['confidence_score']:.3f})")
        
        # Validate integration success
        logger.info("\n8. Validating integration success...")
        
        # Check if we have all expected components
        validation_criteria = {
            'has_go_terms': stats['node_counts'].get('go_term', 0) > 0,
            'has_genes': stats['node_counts'].get('gene', 0) > 0,
            'has_omics_data': any(stats['node_counts'].get(t, 0) > 0 for t in ['disease', 'drug', 'viral_condition']),
            'has_model_data': stats['node_counts'].get('llm_model', 0) > 0,
            'has_model_predictions': stats['node_counts'].get('model_prediction', 0) > 0,
            'has_similarity_rankings': stats['node_counts'].get('similarity_ranking', 0) > 0,
            'has_model_edges': any('model' in edge_type for edge_type in stats['edge_counts'].keys()),
            'integration_ratio_good': stats['integration_metrics']['integration_ratio'] > 0.5
        }
        
        logger.info("="*60)
        logger.info("INTEGRATION VALIDATION")
        logger.info("="*60)
        
        passed_count = 0
        for criterion, passed in validation_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"{criterion}: {status}")
            if passed:
                passed_count += 1
        
        success_rate = (passed_count / len(validation_criteria)) * 100
        logger.info(f"\nValidation Success Rate: {success_rate:.1f}% ({passed_count}/{len(validation_criteria)})")
        
        if success_rate >= 87.5:  # 7/8 criteria
            logger.info("🎉 MODEL COMPARISON INTEGRATION SUCCESSFUL!")
            return True
        else:
            logger.warning("⚠️ Model comparison integration needs attention")
            return False
            
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    logger.info("🧪 Starting Model Comparison Integration Test")
    
    success = test_model_comparison_integration()
    
    logger.info("\n" + "="*80)
    if success:
        logger.info("✅ MODEL COMPARISON INTEGRATION TEST PASSED")
        logger.info("The knowledge graph successfully integrated model comparison data!")
    else:
        logger.info("❌ MODEL COMPARISON INTEGRATION TEST FAILED") 
        logger.info("Please check the integration and try again.")
    logger.info("="*80)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)