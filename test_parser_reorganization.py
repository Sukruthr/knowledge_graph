#!/usr/bin/env python3
"""
Test Parser Reorganization

Verify that all parser imports and functionality work correctly after reorganization.
"""

import sys
import os
import logging
import traceback

# Add src to path
sys.path.append('src')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_parser_imports():
    """Test that all parser imports work correctly."""
    logger.info("Testing parser imports...")
    
    try:
        # Test core imports
        from parsers import ParserUtils, GODataParser, OmicsDataParser, CombinedGOParser, CombinedBiomedicalParser
        logger.info("✅ Core parser imports successful")
        
        # Test specialized imports (these might be None if modules not available)
        from parsers import ModelCompareParser, CCMFBranchParser, LLMProcessedParser
        from parsers import GOAnalysisDataParser, RemainingDataParser, TalismanGeneSetsParser
        logger.info("✅ Specialized parser imports successful")
        
        # Test backward compatibility
        from parsers import GOBPDataParser
        assert GOBPDataParser == GODataParser, "Backward compatibility alias failed"
        logger.info("✅ Backward compatibility alias works")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Parser import failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_kg_builder_imports():
    """Test that kg_builder can import parsers correctly."""
    logger.info("Testing kg_builder imports...")
    
    try:
        from kg_builder import ComprehensiveBiomedicalKnowledgeGraph
        logger.info("✅ KG builder import successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ KG builder import failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_parser_instantiation():
    """Test that parsers can be instantiated correctly."""
    logger.info("Testing parser instantiation...")
    
    try:
        from parsers import CombinedBiomedicalParser
        
        # Test with dummy directory (should handle missing directories gracefully)
        data_dir = "/dummy/path"
        parser = CombinedBiomedicalParser(data_dir)
        
        # Test that parser has expected attributes
        assert hasattr(parser, 'go_parser'), "Missing go_parser attribute"
        assert hasattr(parser, 'parse_all_biomedical_data'), "Missing parse_all_biomedical_data method"
        assert hasattr(parser, 'get_comprehensive_summary'), "Missing get_comprehensive_summary method"
        
        logger.info("✅ Parser instantiation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Parser instantiation failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_parser_utils():
    """Test ParserUtils functionality."""
    logger.info("Testing ParserUtils...")
    
    try:
        from parsers import ParserUtils
        
        # Test utility methods
        assert hasattr(ParserUtils, 'load_file_safe'), "Missing load_file_safe method"
        assert hasattr(ParserUtils, 'validate_go_id'), "Missing validate_go_id method"
        assert hasattr(ParserUtils, 'clean_gene_identifiers'), "Missing clean_gene_identifiers method"
        
        # Test GO ID validation
        assert ParserUtils.validate_go_id('GO:0008150'), "Valid GO ID failed validation"
        assert not ParserUtils.validate_go_id('INVALID'), "Invalid GO ID passed validation"
        
        # Test gene cleaning
        cleaned = ParserUtils.clean_gene_identifiers(['  tp53  ', '', 'BRCA1', None])
        assert 'TP53' in cleaned, "Gene symbol cleaning failed"
        assert 'BRCA1' in cleaned, "Gene symbol preservation failed"
        assert len(cleaned) == 2, "Gene list filtering failed"
        
        logger.info("✅ ParserUtils functionality works")
        return True
        
    except Exception as e:
        logger.error(f"❌ ParserUtils test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_directory_structure():
    """Test that the directory structure is correct."""
    logger.info("Testing directory structure...")
    
    expected_files = [
        'src/parsers/__init__.py',
        'src/parsers/parser_utils.py',
        'src/parsers/core_parsers.py',
        'src/parsers/parser_orchestrator.py',
        'src/parsers/model_compare_parser.py',
        'src/parsers/cc_mf_branch_parser.py',
        'src/parsers/llm_processed_parser.py',
        'src/parsers/go_analysis_data_parser.py',
        'src/parsers/remaining_data_parser.py',
        'src/parsers/talisman_gene_sets_parser.py'
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"❌ Missing files: {missing_files}")
        return False
    
    # Check that old data_parsers.py is backed up or removed
    old_file = 'src/data_parsers.py'
    if os.path.exists(old_file):
        logger.warning(f"⚠️ Old data_parsers.py still exists: {old_file}")
    
    logger.info("✅ Directory structure is correct")
    return True

def main():
    """Run all reorganization tests."""
    logger.info("🧪 STARTING PARSER REORGANIZATION TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Parser Imports", test_parser_imports),
        ("KG Builder Imports", test_kg_builder_imports),
        ("Parser Instantiation", test_parser_instantiation),
        ("Parser Utils", test_parser_utils)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\nTEST: {test_name}")
        logger.info("-" * 30)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"❌ Test {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n" + "=" * 60)
    logger.info("🧪 PARSER REORGANIZATION TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    success_rate = passed / total
    logger.info(f"\nOverall Success Rate: {passed}/{total} ({success_rate:.1%})")
    
    if success_rate == 1.0:
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("✅ Parser reorganization completed successfully")
        return True
    else:
        logger.info(f"\n⚠️ {total - passed} TESTS FAILED")
        logger.info("❌ Parser reorganization needs attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)