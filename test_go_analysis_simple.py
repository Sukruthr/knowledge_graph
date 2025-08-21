#!/usr/bin/env python3
"""
Simple test for GO Analysis Data parser functionality
"""

import sys
sys.path.append('src')

from go_analysis_data_parser import GOAnalysisDataParser

def test_go_analysis_parser():
    """Test just the GO Analysis Data parser."""
    
    print("=" * 60)
    print("GO ANALYSIS DATA PARSER TEST")
    print("=" * 60)
    
    try:
        # Test parser directly
        parser = GOAnalysisDataParser()
        
        # Parse all data
        results = parser.parse_all_go_analysis_data()
        
        # Get stats
        stats = parser.get_processing_stats()
        
        print(f"✅ Parser Test Results:")
        print(f"   Files processed: {stats['files_processed']}")
        print(f"   Total GO terms: {stats['total_go_terms']}")
        print(f"   Unique genes: {stats['total_genes']}")
        print(f"   Contamination datasets: {stats['contamination_datasets']}")
        print(f"   Enrichment analyses: {stats['enrichment_analyses']}")
        print(f"   Confidence evaluations: {stats['confidence_evaluations']}")
        print(f"   Hierarchy relationships: {stats['hierarchy_relationships']}")
        
        if stats['errors']:
            print(f"   Errors: {len(stats['errors'])}")
            for error in stats['errors'][:3]:  # Show first 3 errors
                print(f"     - {error}")
        
        # Test query functionality
        print(f"\n✅ Query Functionality Test:")
        
        # Test core terms query
        core_terms = parser.get_core_go_terms()
        print(f"   Core GO terms available: {len(core_terms)}")
        
        # Test contamination datasets query  
        contamination_data = parser.get_contamination_datasets()
        print(f"   Contamination datasets: {len(contamination_data)}")
        
        # Test confidence evaluations query
        confidence_data = parser.get_confidence_evaluations()
        print(f"   Confidence evaluations: {len(confidence_data)}")
        
        # Test hierarchy data query
        hierarchy_data = parser.get_hierarchy_data()
        print(f"   Hierarchy relationships: {len(hierarchy_data.get('relationships', []))}")
        
        # Test specific GO term profile
        if core_terms:
            first_go_id = list(core_terms.keys())[0]
            profile = parser.query_go_term_analysis_profile(first_go_id)
            if profile:
                print(f"   Sample GO term profile for {first_go_id}:")
                print(f"     Core datasets: {len(profile['core_terms'])}")
                print(f"     Contamination datasets: {len(profile['contamination_analysis'])}")
                print(f"     Confidence evaluations: {len(profile['confidence_evaluation'])}")
        
        print(f"\n🎉 GO ANALYSIS DATA PARSER TEST SUCCESSFUL!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_go_analysis_parser()
    sys.exit(0 if success else 1)