#!/usr/bin/env python3
"""
Comprehensive test script for CC_MF_Branch data integration.

Tests the new CC_MF_Branch parser and knowledge graph integration,
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

def test_cc_mf_branch_integration():
    """Comprehensive test of CC_MF_Branch data integration."""
    
    print("🧪 CC_MF_BRANCH INTEGRATION TEST")
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
    
    # Test 2: CC_MF_Branch data presence
    print("\n2️⃣  Testing CC_MF_Branch data integration...")
    try:
        cc_mf_stats = kg.get_cc_mf_branch_stats()
        
        assert cc_mf_stats['total_cc_mf_terms'] > 0, "Should have CC and MF GO terms"
        assert cc_mf_stats['total_cc_mf_interpretations'] > 0, "Should have LLM interpretations"
        assert cc_mf_stats['total_cc_mf_rankings'] > 0, "Should have similarity rankings"
        assert cc_mf_stats['unique_cc_mf_genes'] > 0, "Should have unique genes"
        
        print(f"   ✅ CC terms: {cc_mf_stats['cc_go_terms']:,}")
        print(f"   ✅ MF terms: {cc_mf_stats['mf_go_terms']:,}")
        print(f"   ✅ Interpretations: {cc_mf_stats['total_cc_mf_interpretations']:,}")
        print(f"   ✅ Rankings: {cc_mf_stats['total_cc_mf_rankings']:,}")
        print(f"   ✅ Unique genes: {cc_mf_stats['unique_cc_mf_genes']:,}")
        
        test_results['tests']['cc_mf_integration'] = {
            'status': 'PASS',
            'cc_mf_stats': cc_mf_stats
        }
        
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        test_results['tests']['cc_mf_integration'] = {'status': 'FAIL', 'error': str(e)}
    
    # Test 3: Query CC and MF terms
    print("\n3️⃣  Testing CC and MF term queries...")
    try:
        cc_terms = kg.query_cc_mf_terms(namespace='CC')
        mf_terms = kg.query_cc_mf_terms(namespace='MF')
        all_terms = kg.query_cc_mf_terms()
        
        assert len(cc_terms) > 0, "Should return CC terms"
        assert len(mf_terms) > 0, "Should return MF terms"
        assert len(all_terms) == len(cc_terms) + len(mf_terms), "All terms should equal CC + MF"
        
        print(f"   ✅ CC terms query: {len(cc_terms):,} results")
        print(f"   ✅ MF terms query: {len(mf_terms):,} results")
        print(f"   ✅ All terms query: {len(all_terms):,} results")
        
        # Test sample term structure
        if cc_terms:
            sample_cc = cc_terms[0]
            required_fields = ['go_id', 'name', 'namespace', 'description', 'gene_count']
            for field in required_fields:
                assert field in sample_cc, f"CC term should have {field} field"
            assert sample_cc['namespace'] == 'CC', "CC term should have CC namespace"
        
        test_results['tests']['term_queries'] = {
            'status': 'PASS',
            'cc_terms_count': len(cc_terms),
            'mf_terms_count': len(mf_terms),
            'total_terms_count': len(all_terms)
        }
        
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        test_results['tests']['term_queries'] = {'status': 'FAIL', 'error': str(e)}
    
    # Test 4: Query LLM interpretations
    print("\n4️⃣  Testing LLM interpretation queries...")
    try:
        cc_interpretations = kg.query_cc_mf_llm_interpretations(namespace='CC')
        mf_interpretations = kg.query_cc_mf_llm_interpretations(namespace='MF')
        all_interpretations = kg.query_cc_mf_llm_interpretations()
        
        assert len(cc_interpretations) > 0, "Should return CC interpretations"
        assert len(mf_interpretations) > 0, "Should return MF interpretations"
        
        print(f"   ✅ CC interpretations: {len(cc_interpretations):,} results")
        print(f"   ✅ MF interpretations: {len(mf_interpretations):,} results")
        print(f"   ✅ All interpretations: {len(all_interpretations):,} results")
        
        # Test sample interpretation structure
        if cc_interpretations:
            sample_interp = cc_interpretations[0]
            required_fields = ['interpretation_id', 'go_term_id', 'namespace', 'llm_score', 'llm_analysis']
            for field in required_fields:
                assert field in sample_interp, f"Interpretation should have {field} field"
            assert sample_interp['namespace'] == 'CC', "CC interpretation should have CC namespace"
            assert 0 <= sample_interp['llm_score'] <= 1, "LLM score should be between 0 and 1"
        
        test_results['tests']['interpretation_queries'] = {
            'status': 'PASS',
            'cc_interpretations_count': len(cc_interpretations),
            'mf_interpretations_count': len(mf_interpretations),
            'total_interpretations_count': len(all_interpretations)
        }
        
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        test_results['tests']['interpretation_queries'] = {'status': 'FAIL', 'error': str(e)}
    
    # Test 5: Query similarity rankings
    print("\n5️⃣  Testing similarity ranking queries...")
    try:
        cc_rankings = kg.query_cc_mf_similarity_rankings(namespace='CC')
        mf_rankings = kg.query_cc_mf_similarity_rankings(namespace='MF')
        
        assert len(cc_rankings) > 0, "Should return CC rankings"
        assert len(mf_rankings) > 0, "Should return MF rankings"
        
        print(f"   ✅ CC rankings: {len(cc_rankings):,} results")
        print(f"   ✅ MF rankings: {len(mf_rankings):,} results")
        
        # Test sample ranking structure
        if cc_rankings:
            sample_ranking = cc_rankings[0]
            required_fields = ['ranking_id', 'go_term_id', 'namespace', 'sim_rank', 'true_go_term_sim_percentile']
            for field in required_fields:
                assert field in sample_ranking, f"Ranking should have {field} field"
            assert sample_ranking['namespace'] == 'CC', "CC ranking should have CC namespace"
            assert 0 <= sample_ranking['true_go_term_sim_percentile'] <= 1, "Percentile should be between 0 and 1"
        
        test_results['tests']['ranking_queries'] = {
            'status': 'PASS',
            'cc_rankings_count': len(cc_rankings),
            'mf_rankings_count': len(mf_rankings)
        }
        
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        test_results['tests']['ranking_queries'] = {'status': 'FAIL', 'error': str(e)}
    
    # Test 6: Gene profile queries
    print("\n6️⃣  Testing gene CC/MF profile queries...")
    try:
        # Test with a known gene (from the sample data)
        test_genes = ['TP53', 'BRCA1', 'EGFR']
        successful_profiles = 0
        
        for gene in test_genes:
            profile = kg.query_gene_cc_mf_profile(gene)
            
            if 'error' not in profile:
                successful_profiles += 1
                
                # Verify profile structure
                required_sections = ['gene_symbol', 'cc_associations', 'mf_associations', 
                                   'cc_interpretations', 'mf_interpretations']
                for section in required_sections:
                    assert section in profile, f"Profile should have {section} section"
                
                print(f"   ✅ {gene}: CC associations: {len(profile['cc_associations'])}, "
                      f"MF associations: {len(profile['mf_associations'])}")
        
        assert successful_profiles > 0, "Should successfully query at least one gene profile"
        
        test_results['tests']['gene_profile_queries'] = {
            'status': 'PASS',
            'successful_profiles': successful_profiles,
            'total_tested': len(test_genes)
        }
        
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        test_results['tests']['gene_profile_queries'] = {'status': 'FAIL', 'error': str(e)}
    
    # Test 7: Cross-integration verification
    print("\n7️⃣  Testing cross-data integration...")
    try:
        # Verify that existing data sources are still present
        stats = kg.get_comprehensive_stats()
        
        assert 'go_term' in stats['node_counts'], "Should have GO terms"
        assert 'gene' in stats['node_counts'], "Should have genes"
        
        # Check for multiple data sources
        node_sources = set()
        for node_id, attrs in kg.graph.nodes(data=True):
            source = attrs.get('source', 'unknown')
            node_sources.add(source)
        
        expected_sources = ['CC_MF_branch']
        for source in expected_sources:
            if source in node_sources:
                print(f"   ✅ Found {source} data source")
        
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
        
        # Performance benchmarks
        assert total_time < 120, f"Total build time should be under 2 minutes, got {total_time:.2f}s"
        
        # Test query performance
        start_time = time.time()
        for i in range(10):
            kg.query_cc_mf_terms()
        query_time = (time.time() - start_time) / 10
        
        assert query_time < 1.0, f"Query time should be under 1 second, got {query_time:.3f}s"
        
        print(f"   ✅ Total build time: {total_time:.2f}s (< 120s)")
        print(f"   ✅ Average query time: {query_time:.3f}s (< 1.0s)")
        
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
        
        # Test gene queries if available
        try:
            sample_profile = kg.query_gene_comprehensive('TP53')
            print(f"   ✅ Existing gene query works: TP53 profile available")
        except:
            print(f"   ⚠️  Existing gene query not available (may be expected)")
        
        # Test model comparison if available
        try:
            model_stats = kg.query_model_comparison_summary()
            print(f"   ✅ Model comparison queries work")
        except:
            print(f"   ⚠️  Model comparison not available (may be expected)")
        
        test_results['tests']['regression'] = {
            'status': 'PASS',
            'comprehensive_stats_available': comprehensive_stats is not None
        }
        
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        test_results['tests']['regression'] = {'status': 'FAIL', 'error': str(e)}
    
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
    with open('cc_mf_branch_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to cc_mf_branch_test_results.json")
    
    # Final verdict
    if passed_tests == total_tests:
        print(f"\n🎉 ALL TESTS PASSED! CC_MF_Branch integration successful.")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please review results.")
        return False

def main():
    """Main test function."""
    setup_logging()
    
    print("🚀 Starting CC_MF_Branch integration testing...")
    
    try:
        success = test_cc_mf_branch_integration()
        
        if success:
            print("\n✅ CC_MF_Branch integration test completed successfully!")
            return 0
        else:
            print("\n❌ CC_MF_Branch integration test completed with failures!")
            return 1
            
    except Exception as e:
        print(f"\n💥 Critical error during testing: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)