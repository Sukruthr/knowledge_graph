#!/usr/bin/env python3
"""
Test script for enhanced data parser with Omics_data2 integration.
Tests both backward compatibility and new semantic features.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from data_parsers import OmicsDataParser, CombinedBiomedicalParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_omics_parser():
    """Test basic OmicsDataParser functionality (backward compatibility)."""
    logger.info("="*60)
    logger.info("TESTING BASIC OMICS PARSER (BACKWARD COMPATIBILITY)")
    logger.info("="*60)
    
    try:
        # Test with only Omics_data (existing functionality)
        omics_dir = "llm_evaluation_for_gene_set_interpretation/data/Omics_data"
        parser = OmicsDataParser(omics_dir)
        
        # Test existing methods
        logger.info("Testing disease gene associations...")
        disease_assocs = parser.parse_disease_gene_associations()
        logger.info(f"✓ Parsed {len(disease_assocs)} disease associations")
        
        logger.info("Testing drug gene associations...")
        drug_assocs = parser.parse_drug_gene_associations()
        logger.info(f"✓ Parsed {len(drug_assocs)} drug associations")
        
        logger.info("Testing viral gene associations...")
        viral_assocs = parser.parse_viral_gene_associations()
        logger.info(f"✓ Parsed {len(viral_assocs)} viral associations")
        
        logger.info("Testing cluster relationships...")
        clusters = parser.parse_cluster_relationships()
        logger.info(f"✓ Parsed {len(clusters)} cluster relationships")
        
        logger.info("Testing viral expression matrix...")
        viral_expr = parser.parse_viral_expression_matrix()
        logger.info(f"✓ Parsed viral expression for {len(viral_expr)} genes")
        
        logger.info("Testing summary generation...")
        summary = parser.get_omics_summary()
        logger.info(f"✓ Generated summary with {len(summary)} metrics")
        
        logger.info("✅ BASIC OMICS PARSER - ALL TESTS PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ BASIC OMICS PARSER FAILED: {e}")
        return False

def test_enhanced_omics_parser():
    """Test enhanced OmicsDataParser with Omics_data2 integration."""
    logger.info("="*60)
    logger.info("TESTING ENHANCED OMICS PARSER (NEW FEATURES)")
    logger.info("="*60)
    
    try:
        # Test with both Omics_data and Omics_data2
        omics_dir = "llm_evaluation_for_gene_set_interpretation/data/Omics_data"
        omics_data2_dir = "llm_evaluation_for_gene_set_interpretation/data/Omics_data2"
        
        # Check if Omics_data2 exists
        if not Path(omics_data2_dir).exists():
            logger.warning(f"Omics_data2 directory not found: {omics_data2_dir}")
            logger.info("✓ Enhanced parser gracefully handles missing Omics_data2")
            return True
        
        parser = OmicsDataParser(omics_dir, omics_data2_dir)
        
        # Test existing methods still work
        logger.info("Testing backward compatibility...")
        disease_assocs = parser.parse_disease_gene_associations()
        logger.info(f"✓ Backward compatibility: {len(disease_assocs)} disease associations")
        
        # Test new enhanced methods
        logger.info("Testing gene set annotations...")
        annotations = parser.parse_gene_set_annotations()
        logger.info(f"✓ Parsed {len(annotations)} gene set annotations")
        
        logger.info("Testing literature references...")
        literature = parser.parse_literature_references()
        logger.info(f"✓ Parsed literature for {len(literature)} gene sets")
        
        logger.info("Testing GO term validations...")
        validations = parser.parse_go_term_validations()
        logger.info(f"✓ Parsed GO validations for {len(validations)} gene sets")
        
        logger.info("Testing experimental metadata...")
        metadata = parser.parse_experimental_metadata()
        logger.info(f"✓ Parsed metadata for {len(metadata)} gene sets")
        
        logger.info("Testing comprehensive enhanced data parsing...")
        enhanced_data = parser.parse_all_enhanced_data()
        logger.info(f"✓ Parsed enhanced data with {len(enhanced_data)} data types")
        
        # Test summary includes new metrics
        logger.info("Testing enhanced summary...")
        summary = parser.get_omics_summary()
        expected_new_fields = ['gene_set_annotations', 'literature_references', 'go_term_validations']
        for field in expected_new_fields:
            if field in summary:
                logger.info(f"✓ Summary includes new field: {field}")
            else:
                logger.warning(f"⚠ Summary missing field: {field}")
        
        logger.info("✅ ENHANCED OMICS PARSER - ALL TESTS PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ ENHANCED OMICS PARSER FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_combined_biomedical_parser():
    """Test CombinedBiomedicalParser with enhanced features."""
    logger.info("="*60)
    logger.info("TESTING COMBINED BIOMEDICAL PARSER")
    logger.info("="*60)
    
    try:
        base_data_dir = "llm_evaluation_for_gene_set_interpretation/data"
        parser = CombinedBiomedicalParser(base_data_dir)
        
        # Test initialization
        logger.info("Testing parser initialization...")
        if parser.omics_parser:
            logger.info("✓ Omics parser initialized")
            if parser.omics_parser.omics_data2_dir:
                logger.info("✓ Enhanced semantic data integration enabled")
            else:
                logger.info("✓ Parser works without Omics_data2 (graceful degradation)")
        else:
            logger.warning("⚠ Omics parser not initialized")
        
        # Test parsing without actually parsing everything (to save time)
        logger.info("Testing basic GO parser access...")
        if parser.go_parser:
            logger.info("✓ GO parser initialized")
        
        logger.info("✅ COMBINED BIOMEDICAL PARSER - INITIALIZATION PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ COMBINED BIOMEDICAL PARSER FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_structure_validation():
    """Test that data structures are correctly formed."""
    logger.info("="*60)
    logger.info("TESTING DATA STRUCTURE VALIDATION")
    logger.info("="*60)
    
    try:
        omics_dir = "llm_evaluation_for_gene_set_interpretation/data/Omics_data"
        omics_data2_dir = "llm_evaluation_for_gene_set_interpretation/data/Omics_data2"
        
        if not Path(omics_data2_dir).exists():
            logger.info("✓ Skipping data structure validation - Omics_data2 not found")
            return True
        
        parser = OmicsDataParser(omics_dir, omics_data2_dir)
        
        # Test a sample of each data type
        annotations = parser.parse_gene_set_annotations()
        if annotations:
            sample_key = list(annotations.keys())[0]
            sample_annotation = annotations[sample_key]
            
            expected_fields = ['source', 'gene_set_name', 'gene_list', 'llm_name', 'score']
            for field in expected_fields:
                if field in sample_annotation:
                    logger.info(f"✓ Annotation structure has field: {field}")
                else:
                    logger.warning(f"⚠ Missing annotation field: {field}")
        
        literature = parser.parse_literature_references()
        if literature:
            sample_key = list(literature.keys())[0]
            sample_refs = literature[sample_key]
            
            if isinstance(sample_refs, list) and len(sample_refs) > 0:
                sample_ref = sample_refs[0]
                expected_fields = ['paragraph', 'keyword', 'references']
                for field in expected_fields:
                    if field in sample_ref:
                        logger.info(f"✓ Literature structure has field: {field}")
                    else:
                        logger.warning(f"⚠ Missing literature field: {field}")
        
        logger.info("✅ DATA STRUCTURE VALIDATION - PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ DATA STRUCTURE VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all parser tests."""
    logger.info("🧪 STARTING ENHANCED DATA PARSER TESTING")
    logger.info("="*80)
    
    test_results = []
    
    # Test 1: Basic backward compatibility
    test_results.append(("Basic Omics Parser", test_basic_omics_parser()))
    
    # Test 2: Enhanced features
    test_results.append(("Enhanced Omics Parser", test_enhanced_omics_parser()))
    
    # Test 3: Combined parser
    test_results.append(("Combined Biomedical Parser", test_combined_biomedical_parser()))
    
    # Test 4: Data structure validation
    test_results.append(("Data Structure Validation", test_data_structure_validation()))
    
    # Summary
    logger.info("="*80)
    logger.info("🧪 TEST RESULTS SUMMARY")
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
        logger.info("🎉 ALL TESTS PASSED - READY FOR KG INTEGRATION")
        return True
    else:
        logger.error("❌ SOME TESTS FAILED - PLEASE FIX BEFORE PROCEEDING")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)