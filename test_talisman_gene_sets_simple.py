#!/usr/bin/env python3
"""
Simple Talisman Gene Sets Integration Test

Quick validation of talisman gene sets parser and basic integration.
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
    """Run simple test for talisman gene sets integration"""
    
    logger.info("🧪 STARTING SIMPLE TALISMAN GENE SETS INTEGRATION TEST")
    logger.info("=" * 60)
    
    try:
        # Test 1: Parser functionality
        logger.info("TEST 1: TALISMAN GENE SETS PARSER")
        from talisman_gene_sets_parser import TalismanGeneSetsParser
        
        parser = TalismanGeneSetsParser()
        start_time = time.time()
        parsed_data = parser.parse_all_gene_sets()
        parse_time = time.time() - start_time
        
        stats = parser.get_parsing_statistics()
        validation = parser.validate_parsing_quality()
        
        logger.info(f"Parser completed in {parse_time:.2f} seconds")
        logger.info(f"Gene sets parsed: {stats['overall_summary']['total_gene_sets']}")
        logger.info(f"Unique genes: {stats['overall_summary']['total_unique_genes']}")
        logger.info(f"Avg genes per set: {stats['overall_summary']['avg_genes_per_set']:.1f}")
        
        # Check each data type
        data_types = ['hallmark_sets', 'bicluster_sets', 'pathway_sets', 'go_custom_sets', 'disease_sets', 'other_sets']
        for data_type in data_types:
            count = stats['by_data_type'].get(data_type, 0)
            if count > 0:
                logger.info(f"  ✅ {data_type}: {count} sets")
        
        # Quality validation
        quality_issues = len(validation['quality_issues'])
        logger.info(f"Quality issues found: {quality_issues}")
        
        # Test 2: Data Integration 
        logger.info(f"\nTEST 2: PARSER INTEGRATION IN COMBINED BIOMEDICAL PARSER")
        from data_parsers import CombinedBiomedicalParser
        
        combined_parser = CombinedBiomedicalParser('llm_evaluation_for_gene_set_interpretation/data')
        
        # Check if talisman gene sets parser was initialized
        has_talisman_parser = hasattr(combined_parser, 'talisman_gene_sets_parser') and combined_parser.talisman_gene_sets_parser is not None
        logger.info(f"Talisman gene sets parser initialized: {'✅ YES' if has_talisman_parser else '❌ NO'}")
        
        if has_talisman_parser:
            logger.info("  ✅ TalismanGeneSetsParser successfully integrated into CombinedBiomedicalParser")
        
        # Test 3: Basic KG Integration (minimal)
        logger.info(f"\nTEST 3: BASIC KNOWLEDGE GRAPH INTEGRATION TEST")
        from kg_builder import ComprehensiveBiomedicalKnowledgeGraph
        
        # Create minimal KG to test talisman gene sets integration methods exist
        kg = ComprehensiveBiomedicalKnowledgeGraph()
        
        # Check if talisman gene sets methods exist
        methods_to_check = [
            '_add_talisman_gene_sets',
            '_add_hallmark_gene_sets',
            '_add_bicluster_gene_sets',
            '_add_pathway_gene_sets',
            '_add_go_custom_gene_sets',
            '_add_disease_gene_sets',
            '_add_other_gene_sets',
            '_add_gene_node_if_missing'
        ]
        
        missing_methods = []
        for method in methods_to_check:
            if not hasattr(kg, method):
                missing_methods.append(method)
        
        if missing_methods:
            logger.info(f"❌ Missing methods: {', '.join(missing_methods)}")
        else:
            logger.info("✅ All talisman gene sets integration methods present")
        
        # Summary
        logger.info(f"\n" + "=" * 60)
        logger.info("🧪 SIMPLE INTEGRATION TEST SUMMARY")
        logger.info("=" * 60)
        
        parser_success = stats['overall_summary']['total_gene_sets'] >= 70  # Expecting ~77 files
        integration_success = has_talisman_parser
        methods_success = len(missing_methods) == 0
        quality_success = quality_issues <= 10  # Allow some minor quality issues
        
        all_success = parser_success and integration_success and methods_success and quality_success
        
        logger.info(f"Parser functionality: {'✅ PASS' if parser_success else '❌ FAIL'}")
        logger.info(f"CombinedParser integration: {'✅ PASS' if integration_success else '❌ FAIL'}")
        logger.info(f"KG methods integration: {'✅ PASS' if methods_success else '❌ FAIL'}")
        logger.info(f"Data quality validation: {'✅ PASS' if quality_success else '❌ FAIL'}")
        
        if all_success:
            logger.info(f"\n🎉 SIMPLE TEST PASSED - TALISMAN GENE SETS INTEGRATION READY!")
            logger.info("✅ Parser successfully handles all talisman gene set types:")
            logger.info(f"   - HALLMARK sets: {stats['by_data_type'].get('hallmark_sets', 0)}")
            logger.info(f"   - Bicluster sets: {stats['by_data_type'].get('bicluster_sets', 0)}")
            logger.info(f"   - Pathway sets: {stats['by_data_type'].get('pathway_sets', 0)}")
            logger.info(f"   - GO custom sets: {stats['by_data_type'].get('go_custom_sets', 0)}")
            logger.info(f"   - Disease sets: {stats['by_data_type'].get('disease_sets', 0)}")
            logger.info(f"   - Other sets: {stats['by_data_type'].get('other_sets', 0)}")
            logger.info("✅ Integration into CombinedBiomedicalParser complete")
            logger.info("✅ Knowledge graph methods for talisman gene sets added")
            logger.info(f"✅ Data quality validated ({validation['total_gene_sets']} gene sets, {quality_issues} minor issues)")
        else:
            logger.info(f"\n⚠️ ISSUES FOUND - NEEDS ATTENTION")
            if not parser_success:
                logger.info(f"❌ Parser did not find expected number of gene sets (found {stats['overall_summary']['total_gene_sets']}, expected ≥70)")
            if not integration_success:
                logger.info("❌ Parser not properly integrated into CombinedBiomedicalParser")
            if not methods_success:
                logger.info("❌ Knowledge graph methods missing")
            if not quality_success:
                logger.info(f"❌ Data quality issues exceed threshold ({quality_issues} > 10)")
        
        return all_success
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)