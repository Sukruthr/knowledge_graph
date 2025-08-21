#!/usr/bin/env python3
"""
Quick test for GO Analysis Data integration in knowledge graph
Tests just the integration without full graph construction
"""

import sys
sys.path.append('src')

import logging
from data_parsers import CombinedBiomedicalParser

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_go_analysis_parser_integration():
    """Test that GO Analysis Data parser integrates correctly with the main parser."""
    
    print("=" * 60)
    print("GO ANALYSIS DATA PARSER INTEGRATION TEST")
    print("=" * 60)
    
    try:
        # Test parser integration
        logger.info("Testing CombinedBiomedicalParser with GO Analysis Data...")
        
        parser = CombinedBiomedicalParser("llm_evaluation_for_gene_set_interpretation/data")
        
        # Check if GO analysis parser was initialized
        has_go_analysis_parser = hasattr(parser, 'go_analysis_data_parser') and parser.go_analysis_data_parser is not None
        
        print(f"✅ Parser Integration Test:")
        print(f"   GO Analysis Data parser initialized: {has_go_analysis_parser}")
        
        if has_go_analysis_parser:
            print(f"   Parser type: {type(parser.go_analysis_data_parser).__name__}")
            
            # Test parsing
            logger.info("Testing data parsing...")
            parsed_data = parser.parse_all_biomedical_data()
            
            # Check if GO analysis data was parsed
            has_go_analysis_data = 'go_analysis_data' in parsed_data
            print(f"   GO Analysis Data parsed: {has_go_analysis_data}")
            
            if has_go_analysis_data:
                go_data = parsed_data['go_analysis_data']
                print(f"   Core GO terms: {len(go_data.get('core_go_terms', {}))}")
                print(f"   Contamination datasets: {len(go_data.get('contamination_datasets', {}))}")
                print(f"   Confidence evaluations: {len(go_data.get('confidence_evaluations', {}))}")
                print(f"   Processing stats: {go_data.get('processing_stats', {})}")
                
                # Check data content
                stats = go_data.get('processing_stats', {})
                print(f"\n✅ Data Content Validation:")
                print(f"   Files processed: {stats.get('files_processed', 0)}")
                print(f"   Total GO terms: {stats.get('total_go_terms', 0)}")
                print(f"   Unique genes: {stats.get('total_genes', 0)}")
                print(f"   Contamination datasets: {stats.get('contamination_datasets', 0)}")
                print(f"   Confidence evaluations: {stats.get('confidence_evaluations', 0)}")
                print(f"   Errors: {len(stats.get('errors', []))}")
                
                # Success criteria for parser integration
                success_criteria = [
                    has_go_analysis_parser,
                    has_go_analysis_data,
                    stats.get('files_processed', 0) >= 6,
                    stats.get('total_go_terms', 0) > 2000,
                    stats.get('total_genes', 0) > 10000,
                    len(stats.get('errors', [])) == 0
                ]
                
                passed_criteria = sum(success_criteria)
                
                print(f"\n🏆 PARSER INTEGRATION SUCCESS: {passed_criteria}/{len(success_criteria)} PASSED")
                
                if passed_criteria == len(success_criteria):
                    print("🎉 GO ANALYSIS DATA PARSER INTEGRATION SUCCESSFUL!")
                    return True
                else:
                    print("⚠️  SOME INTEGRATION TESTS FAILED")
                    return False
            else:
                print("❌ GO Analysis Data was not parsed")
                return False
        else:
            print("❌ GO Analysis Data parser was not initialized")
            return False
            
    except Exception as e:
        print(f"❌ ERROR during testing: {e}")
        logger.error(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_go_analysis_parser_integration()
    sys.exit(0 if success else 1)