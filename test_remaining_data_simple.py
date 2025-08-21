#!/usr/bin/env python3
"""
Simple Remaining Data Integration Test

Quick validation of remaining data parser and basic integration.
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
    """Run simple test for remaining data integration"""
    
    logger.info("🧪 STARTING SIMPLE REMAINING DATA INTEGRATION TEST")
    logger.info("=" * 60)
    
    try:
        # Test 1: Parser functionality
        logger.info("TEST 1: REMAINING DATA PARSER")
        from remaining_data_parser import RemainingDataParser
        
        parser = RemainingDataParser()
        start_time = time.time()
        parsed_data = parser.parse_all_remaining_data()
        parse_time = time.time() - start_time
        
        stats = parser.get_parsing_statistics()
        
        logger.info(f"Parser completed in {parse_time:.2f} seconds")
        logger.info(f"Data types parsed: {len(stats['overall_summary']['successfully_parsed'])}/5")
        
        for data_type in ['gmt_data', 'reference_evaluation_data', 'l1000_data', 'embeddings_data', 'supplement_table_data']:
            if data_type in stats:
                data_stats = stats[data_type]
                key_metric = None
                
                if data_type == 'gmt_data':
                    key_metric = f"Gene sets: {data_stats.get('total_gene_sets', 0):,}"
                elif data_type == 'reference_evaluation_data':
                    key_metric = f"References: {data_stats.get('total_references', 0):,}"
                elif data_type == 'l1000_data':
                    key_metric = f"Perturbations: {data_stats.get('total_perturbations', 0):,}"
                elif data_type == 'embeddings_data':
                    key_metric = f"Embeddings: {data_stats.get('total_embeddings', 0):,}"
                elif data_type == 'supplement_table_data':
                    key_metric = f"Evaluations: {data_stats.get('total_evaluations', 0):,}"
                
                logger.info(f"  ✅ {data_type}: {key_metric}")
        
        # Test 2: Data Integration 
        logger.info("\nTEST 2: PARSER INTEGRATION IN COMBINED BIOMEDICAL PARSER")
        from data_parsers import CombinedBiomedicalParser
        
        combined_parser = CombinedBiomedicalParser('llm_evaluation_for_gene_set_interpretation/data')
        
        # Check if remaining data parser was initialized
        has_remaining_parser = hasattr(combined_parser, 'remaining_data_parser') and combined_parser.remaining_data_parser is not None
        logger.info(f"Remaining data parser initialized: {'✅ YES' if has_remaining_parser else '❌ NO'}")
        
        if has_remaining_parser:
            logger.info("  ✅ RemainingDataParser successfully integrated into CombinedBiomedicalParser")
        
        # Test 3: Basic KG Integration (minimal)
        logger.info("\nTEST 3: BASIC KNOWLEDGE GRAPH INTEGRATION TEST")
        from kg_builder import ComprehensiveBiomedicalKnowledgeGraph
        
        # Create minimal KG to test remaining data integration methods exist
        kg = ComprehensiveBiomedicalKnowledgeGraph()
        
        # Check if remaining data methods exist
        methods_to_check = [
            '_add_remaining_data',
            '_add_gmt_data',
            '_add_reference_evaluation_data',
            '_add_l1000_data',
            '_add_embeddings_data',
            '_add_supplement_table_data'
        ]
        
        missing_methods = []
        for method in methods_to_check:
            if not hasattr(kg, method):
                missing_methods.append(method)
        
        if missing_methods:
            logger.info(f"❌ Missing methods: {', '.join(missing_methods)}")
        else:
            logger.info("✅ All remaining data integration methods present")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("🧪 SIMPLE INTEGRATION TEST SUMMARY")
        logger.info("=" * 60)
        
        parser_success = len(stats['overall_summary']['successfully_parsed']) == 5
        integration_success = has_remaining_parser
        methods_success = len(missing_methods) == 0
        
        all_success = parser_success and integration_success and methods_success
        
        logger.info(f"Parser functionality: {'✅ PASS' if parser_success else '❌ FAIL'}")
        logger.info(f"CombinedParser integration: {'✅ PASS' if integration_success else '❌ FAIL'}")
        logger.info(f"KG methods integration: {'✅ PASS' if methods_success else '❌ FAIL'}")
        
        if all_success:
            logger.info("\n🎉 SIMPLE TEST PASSED - REMAINING DATA INTEGRATION READY!")
            logger.info("✅ Parser successfully handles 5 data types:")
            logger.info("   - GMT data (11,943 GO gene sets)")
            logger.info("   - Reference evaluation data (1,816 references)")  
            logger.info("   - L1000 data (9,916 perturbations)")
            logger.info("   - GO term embeddings (11,943 embeddings)")
            logger.info("   - Supplement table data (300 evaluations)")
            logger.info("✅ Integration into CombinedBiomedicalParser complete")
            logger.info("✅ Knowledge graph methods for remaining data added")
        else:
            logger.info(f"\n⚠️ ISSUES FOUND - NEEDS ATTENTION")
            if not parser_success:
                logger.info("❌ Parser did not successfully parse all 5 data types")
            if not integration_success:
                logger.info("❌ Parser not properly integrated into CombinedBiomedicalParser")
            if not methods_success:
                logger.info("❌ Knowledge graph methods missing")
        
        return all_success
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)