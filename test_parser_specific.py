#!/usr/bin/env python3
"""
Specific parser testing for model comparison and other components.
"""

import sys
import time
import logging

# Add src to path
sys.path.append('src')

from data_parsers import CombinedBiomedicalParser
from model_compare_parser import ModelCompareParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_compare_parser():
    """Test model comparison parser independently."""
    logger.info("="*60)
    logger.info("TESTING MODEL COMPARISON PARSER")
    logger.info("="*60)
    
    try:
        # Test parser initialization
        model_dir = 'llm_evaluation_for_gene_set_interpretation/data/GO_term_analysis/model_compare'
        parser = ModelCompareParser(model_dir)
        logger.info("✓ ModelCompareParser initialized")
        
        # Test data parsing
        start_time = time.time()
        model_data = parser.parse_all_model_data()
        parse_time = time.time() - start_time
        logger.info(f"✓ Data parsing completed in {parse_time:.2f} seconds")
        
        # Validate structure
        required_keys = ['model_predictions', 'similarity_rankings', 'evaluation_metrics', 'contamination_results', 'available_models']
        for key in required_keys:
            if key not in model_data:
                logger.error(f"Missing key: {key}")
                return False
            logger.info(f"✓ {key} present")
        
        # Check models
        models = model_data['available_models']
        if len(models) < 3:
            logger.error(f"Insufficient models: {len(models)}")
            return False
        logger.info(f"✓ Found {len(models)} models: {models}")
        
        # Check predictions
        predictions = model_data['model_predictions']
        if len(predictions) < 3:
            logger.error(f"Insufficient prediction sets: {len(predictions)}")
            return False
        
        total_predictions = sum(len(pred['go_predictions']) for pred in predictions.values())
        logger.info(f"✓ Found {total_predictions} total GO predictions")
        
        # Check similarity rankings
        rankings = model_data['similarity_rankings']
        if len(rankings) < 3:
            logger.error(f"Insufficient ranking sets: {len(rankings)}")
            return False
        
        total_rankings = sum(len(rank['similarity_metrics']) for rank in rankings.values())
        logger.info(f"✓ Found {total_rankings} similarity rankings")
        
        # Check evaluation metrics
        metrics = model_data['evaluation_metrics']
        for model_name in models:
            if model_name not in metrics:
                logger.error(f"Missing metrics for {model_name}")
                return False
            
            model_metrics = metrics[model_name]
            required_metric_keys = ['confidence_stats', 'similarity_stats']
            for metric_key in required_metric_keys:
                if metric_key in model_metrics and model_metrics[metric_key]:
                    logger.info(f"✓ {model_name} {metric_key} present")
        
        # Check contamination analysis
        contamination = model_data['contamination_results']
        for model_name in models:
            if model_name in contamination:
                robustness = contamination[model_name].get('robustness_score', 0)
                logger.info(f"✓ {model_name} robustness score: {robustness:.3f}")
        
        logger.info("✅ Model comparison parser test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Model comparison parser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_combined_parser():
    """Test combined biomedical parser."""
    logger.info("="*60)
    logger.info("TESTING COMBINED BIOMEDICAL PARSER")
    logger.info("="*60)
    
    try:
        # Test parser initialization
        data_dir = 'llm_evaluation_for_gene_set_interpretation/data'
        parser = CombinedBiomedicalParser(data_dir)
        logger.info("✓ CombinedBiomedicalParser initialized")
        
        # Test data parsing (quick subset)
        start_time = time.time()
        
        # Test full parsing
        logger.info("Testing full data parsing...")
        parsed_data = parser.parse_all_biomedical_data()
        if not parsed_data:
            logger.error("Data parsing failed")
            return False
        
        # Check GO data
        if 'go_data' not in parsed_data:
            logger.error("GO data missing")
            return False
        
        go_data = parsed_data['go_data']
        if len(go_data) < 3:
            logger.error("Insufficient GO namespaces")
            return False
        logger.info(f"✓ GO data parsed: {list(go_data.keys())}")
        
        # Check Omics data
        if 'omics_data' not in parsed_data:
            logger.error("Omics data missing")
            return False
        
        omics_data = parsed_data['omics_data']
        
        required_omics = ['disease_associations', 'drug_associations', 'viral_associations']
        for omics_type in required_omics:
            if omics_type not in omics_data:
                logger.error(f"Missing {omics_type}")
                return False
            logger.info(f"✓ {omics_type}: {len(omics_data[omics_type])} entries")
        
        # Check Model Comparison data
        if 'model_compare_data' not in parsed_data:
            logger.warning("Model comparison data missing")
        else:
            model_data = parsed_data['model_compare_data']
            if 'available_models' in model_data:
                logger.info(f"✓ Model comparison data parsed: {len(model_data['available_models'])} models")
            else:
                logger.warning("Model comparison data incomplete")
        
        parse_time = time.time() - start_time
        logger.info(f"✓ Combined parsing completed in {parse_time:.2f} seconds")
        
        logger.info("✅ Combined biomedical parser test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Combined parser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_specific_components():
    """Test specific parser components individually."""
    logger.info("="*60)
    logger.info("TESTING SPECIFIC PARSER COMPONENTS")
    logger.info("="*60)
    
    tests = [
        ("Model Comparison Parser", test_model_compare_parser),
        ("Combined Biomedical Parser", test_combined_parser)
    ]
    
    passed_tests = 0
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        try:
            if test_func():
                logger.info(f"✅ {test_name} PASSED")
                passed_tests += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"💥 {test_name} ERROR: {e}")
    
    success_rate = (passed_tests / len(tests)) * 100
    logger.info(f"\n{'='*60}")
    logger.info(f"PARSER TESTING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Tests Passed: {passed_tests}/{len(tests)} ({success_rate:.1f}%)")
    
    if success_rate >= 50:  # At least half
        logger.info("🎉 PARSER TESTING SUCCESSFUL!")
        return True
    else:
        logger.warning("⚠️ Parser testing needs attention")
        return False

if __name__ == "__main__":
    success = test_specific_components()
    sys.exit(0 if success else 1)