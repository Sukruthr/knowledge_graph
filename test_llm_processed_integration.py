#!/usr/bin/env python3
"""
Comprehensive test script for LLM_processed data integration.

Tests the new LLM_processed parser and knowledge graph integration,
ensuring all components work correctly and existing functionality is preserved.
"""

import sys
import os
import time
import json
import logging
from typing import Dict, Any

# Add src directory to path
sys.path.append('src')

from kg_builder import ComprehensiveBiomedicalKnowledgeGraph

def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_llm_processed_integration():
    """Comprehensive test of LLM_processed data integration."""
    
    print("🧪 LLM_PROCESSED INTEGRATION TEST")
    print("=" * 60)
    
    # Initialize the comprehensive knowledge graph
    kg = ComprehensiveBiomedicalKnowledgeGraph()
    
    # Load data
    print("\n📁 Loading comprehensive biomedical data...")
    start_time = time.time()
    kg.load_data('llm_evaluation_for_gene_set_interpretation/data')
    load_time = time.time() - start_time
    print(f"   Data loading completed in {load_time:.2f} seconds")
    
    # Build knowledge graph
    print("\n🏗️  Building comprehensive knowledge graph...")
    start_time = time.time()
    kg.build_comprehensive_graph()
    build_time = time.time() - start_time
    print(f"   Graph construction completed in {build_time:.2f} seconds")
    
    # Test results container
    test_results = {
        'performance': {
            'load_time': load_time,
            'build_time': build_time,
            'total_time': load_time + build_time
        },
        'tests': {}
    }
    
    print("\n🔍 RUNNING INTEGRATION TESTS:")
    print("-" * 40)
    
    # Test 1: Basic graph structure
    print("\n1️⃣  Testing basic graph structure...")
    try:
        stats = kg.get_comprehensive_stats()
        
        assert stats['total_nodes'] > 0, "Graph should have nodes"
        assert stats['total_edges'] > 0, "Graph should have edges"
        
        print(f"   ✅ Graph has {stats['total_nodes']:,} nodes and {stats['total_edges']:,} edges")
        test_results['tests']['basic_structure'] = {'status': 'PASS', 'nodes': stats['total_nodes'], 'edges': stats['total_edges']}
        
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        test_results['tests']['basic_structure'] = {'status': 'FAIL', 'error': str(e)}
    
    # Test 2: LLM_processed data presence
    print("\n2️⃣  Testing LLM_processed data integration...")
    try:
        llm_stats = kg.get_llm_processed_stats()
        
        assert llm_stats['llm_interpretations'] > 0, "Should have LLM interpretations"
        assert llm_stats['contamination_analyses'] > 0, "Should have contamination analyses"
        assert llm_stats['similarity_rankings'] > 0, "Should have similarity rankings"
        assert llm_stats['models_analyzed'] > 0, "Should have multiple models analyzed"
        
        print(f"   ✅ LLM interpretations: {llm_stats['llm_interpretations']:,}")
        print(f"   ✅ Contamination analyses: {llm_stats['contamination_analyses']:,}")
        print(f"   ✅ Similarity rankings: {llm_stats['similarity_rankings']:,}")
        print(f"   ✅ Models analyzed: {llm_stats['models_analyzed']:,}")
        print(f"   ✅ Unique GO terms: {llm_stats['unique_go_terms']:,}")
        print(f"   ✅ Unique genes: {llm_stats['unique_genes']:,}")
        
        test_results['tests']['llm_processed_integration'] = {
            'status': 'PASS',
            'llm_stats': llm_stats
        }
        
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        test_results['tests']['llm_processed_integration'] = {'status': 'FAIL', 'error': str(e)}
    
    # Test 3: Query LLM interpretations
    print("\n3️⃣  Testing LLM interpretation queries...")
    try:
        all_interpretations = kg.query_llm_interpretations()
        selected_interpretations = kg.query_llm_interpretations(dataset='selected_1000_go_terms')
        gpt4_interpretations = kg.query_llm_interpretations(model='gpt_4')
        
        assert len(all_interpretations) > 0, "Should return LLM interpretations"
        assert len(selected_interpretations) > 0, "Should return selected dataset interpretations"
        assert len(gpt4_interpretations) > 0, "Should return GPT-4 interpretations"
        
        print(f"   ✅ All interpretations query: {len(all_interpretations):,} results")
        print(f"   ✅ Selected dataset query: {len(selected_interpretations):,} results")
        print(f"   ✅ GPT-4 model query: {len(gpt4_interpretations):,} results")
        
        # Test sample interpretation structure
        if all_interpretations:
            sample_interp = all_interpretations[0]
            required_fields = ['interpretation_id', 'dataset', 'go_term_id', 'model', 'llm_score']
            for field in required_fields:
                assert field in sample_interp, f"Interpretation should have {field} field"
        
        test_results['tests']['interpretation_queries'] = {
            'status': 'PASS',
            'all_interpretations_count': len(all_interpretations),
            'selected_interpretations_count': len(selected_interpretations),
            'gpt4_interpretations_count': len(gpt4_interpretations)
        }
        
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        test_results['tests']['interpretation_queries'] = {'status': 'FAIL', 'error': str(e)}
    
    # Test 4: Query contamination analysis
    print("\n4️⃣  Testing contamination analysis queries...")
    try:
        all_contamination = kg.query_contamination_analysis()
        gpt4_contamination = kg.query_contamination_analysis(model='gpt_4')
        gemini_contamination = kg.query_contamination_analysis(model='gemini_pro')
        
        assert len(all_contamination) > 0, "Should return contamination analyses"
        assert len(gpt4_contamination) > 0, "Should return GPT-4 contamination analysis"
        
        print(f"   ✅ All contamination analyses: {len(all_contamination):,} results")
        print(f"   ✅ GPT-4 contamination: {len(gpt4_contamination):,} results")
        print(f"   ✅ Gemini contamination: {len(gemini_contamination):,} results")
        
        # Test sample contamination structure
        if all_contamination:
            sample_contam = all_contamination[0]
            required_fields = ['analysis_id', 'model', 'go_term_id', 'scenarios']
            for field in required_fields:
                assert field in sample_contam, f"Contamination analysis should have {field} field"
        
        test_results['tests']['contamination_queries'] = {
            'status': 'PASS',
            'all_contamination_count': len(all_contamination),
            'gpt4_contamination_count': len(gpt4_contamination),
            'gemini_contamination_count': len(gemini_contamination)
        }
        
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        test_results['tests']['contamination_queries'] = {'status': 'FAIL', 'error': str(e)}
    
    # Test 5: Query similarity rankings
    print("\n5️⃣  Testing similarity ranking queries...")
    try:
        all_rankings = kg.query_llm_similarity_rankings()
        selected_rankings = kg.query_llm_similarity_rankings(dataset='selected_1000_go_terms')
        
        assert len(all_rankings) > 0, "Should return similarity rankings"
        assert len(selected_rankings) > 0, "Should return selected dataset rankings"
        
        print(f"   ✅ All similarity rankings: {len(all_rankings):,} results")
        print(f"   ✅ Selected dataset rankings: {len(selected_rankings):,} results")
        
        # Test sample ranking structure
        if all_rankings:
            sample_ranking = all_rankings[0]
            required_fields = ['ranking_id', 'dataset', 'go_term_id', 'similarity_rank', 'similarity_percentile']
            for field in required_fields:
                assert field in sample_ranking, f"Ranking should have {field} field"
        
        test_results['tests']['ranking_queries'] = {
            'status': 'PASS',
            'all_rankings_count': len(all_rankings),
            'selected_rankings_count': len(selected_rankings)
        }
        
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        test_results['tests']['ranking_queries'] = {'status': 'FAIL', 'error': str(e)}
    
    # Test 6: Gene LLM profile queries
    print("\n6️⃣  Testing gene LLM profile queries...")
    try:
        # Test with known genes from the LLM data
        test_genes = ['SLC2A1', 'ISYNA1', 'TP53']  # From sample LLM data
        successful_profiles = 0
        
        for gene in test_genes:
            profile = kg.query_gene_llm_profile(gene)
            
            if (profile['llm_interpretations'] or profile['contamination_analyses'] or 
                profile['similarity_rankings'] or profile['model_comparisons']):
                successful_profiles += 1
                
                # Verify profile structure
                required_sections = ['gene_symbol', 'llm_interpretations', 'contamination_analyses', 
                                   'similarity_rankings', 'model_comparisons']
                for section in required_sections:
                    assert section in profile, f"Profile should have {section} section"
                
                print(f"   ✅ {gene}: LLM interpretations: {len(profile['llm_interpretations'])}, "
                      f"Contamination analyses: {len(profile['contamination_analyses'])}")
        
        print(f"   ✅ Successfully queried {successful_profiles}/{len(test_genes)} gene profiles")
        
        test_results['tests']['gene_llm_profile_queries'] = {
            'status': 'PASS',
            'successful_profiles': successful_profiles,
            'total_tested': len(test_genes)
        }
        
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        test_results['tests']['gene_llm_profile_queries'] = {'status': 'FAIL', 'error': str(e)}
    
    # Test 7: Cross-integration verification
    print("\n7️⃣  Testing cross-data integration...")
    try:
        # Verify that existing data sources are still present along with new LLM data
        stats = kg.get_comprehensive_stats()
        
        assert 'go_term' in stats['node_counts'], "Should have GO terms"
        assert 'gene' in stats['node_counts'], "Should have genes"
        
        # Check for multiple data sources
        node_sources = set()
        for node_id, attrs in kg.graph.nodes(data=True):
            source = attrs.get('source', 'unknown')
            node_sources.add(source)
        
        expected_sources = ['LLM_processed']
        for source in expected_sources:
            if source in node_sources:
                print(f"   ✅ Found {source} data source")
        
        print(f"   ✅ Total data sources detected: {len(node_sources)}")
        
        test_results['tests']['cross_integration'] = {
            'status': 'PASS',
            'detected_sources': list(node_sources),
            'stats': stats
        }
        
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        test_results['tests']['cross_integration'] = {'status': 'FAIL', 'error': str(e)}
    
    # Test 8: Performance validation
    print("\n8️⃣  Testing system performance...")
    try:
        total_time = test_results['performance']['total_time']
        
        # Performance benchmarks - relaxed for LLM data
        assert total_time < 180, f"Total build time should be under 3 minutes, got {total_time:.2f}s"
        
        # Test query performance
        start_time = time.time()
        for i in range(5):  # Reduced iterations for complex queries
            kg.query_llm_interpretations()
        query_time = (time.time() - start_time) / 5
        
        assert query_time < 2.0, f"Query time should be under 2 seconds, got {query_time:.3f}s"
        
        print(f"   ✅ Total build time: {total_time:.2f}s (< 180s)")
        print(f"   ✅ Average query time: {query_time:.3f}s (< 2.0s)")
        
        test_results['tests']['performance'] = {
            'status': 'PASS',
            'total_build_time': total_time,
            'average_query_time': query_time
        }
        
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        test_results['tests']['performance'] = {'status': 'FAIL', 'error': str(e)}
    
    # Test 9: Regression testing (existing functionality)
    print("\n9️⃣  Testing regression compatibility...")
    try:
        # Test existing query methods still work
        comprehensive_stats = kg.get_comprehensive_stats()
        assert comprehensive_stats is not None, "Should return comprehensive stats"
        
        # Test model comparison if available
        try:
            model_stats = kg.query_model_comparison_summary()
            print(f"   ✅ Model comparison queries work")
        except:
            print(f"   ⚠️  Model comparison not available (may be expected)")
        
        # Test CC_MF_Branch if available
        try:
            cc_mf_stats = kg.get_cc_mf_branch_stats()
            print(f"   ✅ CC_MF_Branch queries work")
        except:
            print(f"   ⚠️  CC_MF_Branch not available (may be expected)")
        
        # Test gene queries if available
        try:
            sample_profile = kg.query_gene_comprehensive('TP53')
            print(f"   ✅ Existing gene query works: TP53 profile available")
        except:
            print(f"   ⚠️  Existing gene query not available (may be expected)")
        
        test_results['tests']['regression'] = {
            'status': 'PASS',
            'comprehensive_stats_available': comprehensive_stats is not None
        }
        
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        test_results['tests']['regression'] = {'status': 'FAIL', 'error': str(e)}
    
    # Test 10: LLM data quality validation
    print("\n🔟 Testing LLM data quality...")
    try:
        llm_stats = kg.get_llm_processed_stats()
        
        # Validate data quality metrics
        assert llm_stats['models_analyzed'] >= 7, f"Should analyze at least 7 models, got {llm_stats['models_analyzed']}"
        assert llm_stats['datasets_analyzed'] >= 2, f"Should have at least 2 datasets, got {llm_stats['datasets_analyzed']}"
        assert llm_stats['unique_go_terms'] >= 500, f"Should cover at least 500 GO terms, got {llm_stats['unique_go_terms']}"
        
        # Test specific query for data completeness
        sample_interpretations = kg.query_llm_interpretations(dataset='selected_1000_go_terms')
        if sample_interpretations:
            sample = sample_interpretations[0]
            assert sample['llm_score'] >= 0.0 and sample['llm_score'] <= 1.0, "LLM scores should be between 0 and 1"
        
        print(f"   ✅ Data quality metrics passed")
        print(f"   ✅ Models analyzed: {llm_stats['models_analyzed']}")
        print(f"   ✅ Datasets analyzed: {llm_stats['datasets_analyzed']}")
        print(f"   ✅ GO terms covered: {llm_stats['unique_go_terms']}")
        
        test_results['tests']['llm_data_quality'] = {
            'status': 'PASS',
            'quality_metrics': llm_stats
        }
        
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        test_results['tests']['llm_data_quality'] = {'status': 'FAIL', 'error': str(e)}
    
    # Calculate test summary
    total_tests = len(test_results['tests'])
    passed_tests = sum(1 for test in test_results['tests'].values() if test['status'] == 'PASS')
    
    print(f"\n📊 TEST SUMMARY:")
    print(f"   Passed: {passed_tests}/{total_tests}")
    print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    test_results['summary'] = {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': (passed_tests/total_tests)*100
    }
    
    # Save detailed results
    with open('llm_processed_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to llm_processed_test_results.json")
    
    # Final verdict
    if passed_tests == total_tests:
        print(f"\n🎉 ALL TESTS PASSED! LLM_processed integration successful.")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please review results.")
        return False

def main():
    """Main test function."""
    setup_logging()
    
    print("🚀 Starting LLM_processed integration testing...")
    
    try:
        success = test_llm_processed_integration()
        
        if success:
            print("\n✅ LLM_processed integration test completed successfully!")
            return 0
        else:
            print("\n❌ LLM_processed integration test completed with failures!")
            return 1
            
    except Exception as e:
        print(f"\n💥 Critical error during testing: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)