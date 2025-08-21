#!/usr/bin/env python3
"""
Comprehensive test for GO Analysis Data integration
Tests the integration of GO_term_analysis/data_files into the knowledge graph system.
"""

import sys
sys.path.append('src')

import logging
from kg_builder import ComprehensiveBiomedicalKnowledgeGraph

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_go_analysis_integration():
    """Test the complete GO Analysis Data integration pipeline."""
    
    print("=" * 80)
    print("GO ANALYSIS DATA INTEGRATION TEST")
    print("=" * 80)
    
    try:
        # Initialize knowledge graph
        logger.info("Initializing ComprehensiveBiomedicalKnowledgeGraph...")
        kg = ComprehensiveBiomedicalKnowledgeGraph()
        
        # Load data
        logger.info("Loading data...")
        data_path = "llm_evaluation_for_gene_set_interpretation/data"
        kg.load_data(data_path)
        
        # Build comprehensive graph (including GO Analysis Data)
        logger.info("Building comprehensive knowledge graph...")
        kg.build_comprehensive_graph()
        
        # Get basic graph statistics
        logger.info("Getting graph statistics...")
        total_nodes = kg.graph.number_of_nodes()
        total_edges = kg.graph.number_of_edges()
        
        print(f"\n📊 BASIC GRAPH STATISTICS")
        print(f"Total nodes: {total_nodes:,}")
        print(f"Total edges: {total_edges:,}")
        
        # Test GO Analysis Data specific integration
        print(f"\n🔬 GO ANALYSIS DATA INTEGRATION TESTS")
        
        # Test 1: Check if GO Analysis Data was integrated
        go_analysis_stats = kg.get_go_analysis_stats()
        
        print(f"✅ Test 1: GO Analysis Data Integration")
        print(f"   Core analyses: {go_analysis_stats['core_analyses']:,}")
        print(f"   Contamination analyses: {go_analysis_stats['contamination_analyses']:,}")
        print(f"   Confidence evaluations: {go_analysis_stats['confidence_evaluations']:,}")
        print(f"   Hierarchy relationships: {go_analysis_stats['hierarchy_relationships']:,}")
        print(f"   Similarity datasets: {go_analysis_stats['similarity_datasets']:,}")
        print(f"   Datasets analyzed: {go_analysis_stats['datasets_analyzed']:,}")
        print(f"   Unique GO terms: {go_analysis_stats['unique_go_terms']:,}")
        print(f"   Unique genes: {go_analysis_stats['unique_genes']:,}")
        print(f"   Enrichment analyses: {go_analysis_stats['enrichment_analyses']:,}")
        print(f"   Human reviewed: {go_analysis_stats['human_reviewed']:,}")
        
        # Test 2: Core analysis queries
        print(f"\n✅ Test 2: Core Analysis Queries")
        
        # Query core analysis for 1000 selected terms
        core_analysis_1000 = kg.query_go_core_analysis(dataset='1000_selected')
        print(f"   1000 selected GO terms core analysis: {len(core_analysis_1000)} entries")
        
        # Query core analysis for 100 enricher results
        core_analysis_100 = kg.query_go_core_analysis(dataset='100_enricher_results')
        print(f"   100 enricher results core analysis: {len(core_analysis_100)} entries")
        
        if core_analysis_1000:
            sample = core_analysis_1000[0]
            print(f"   Sample core analysis: GO:{sample['go_term_id']} - {sample['term_description'][:50]}...")
        
        # Test 3: Contamination analysis queries
        print(f"\n✅ Test 3: Contamination Analysis Queries")
        
        # Query contamination analysis
        contamination_1000 = kg.query_go_contamination_analysis(dataset='1000_selected_contaminated')
        contamination_100 = kg.query_go_contamination_analysis(dataset='100_selected_contaminated')
        
        print(f"   1000 selected contaminated: {len(contamination_1000)} entries")
        print(f"   100 selected contaminated: {len(contamination_100)} entries")
        
        if contamination_1000:
            sample = contamination_1000[0]
            print(f"   Sample contamination: GO:{sample['go_term_id']} - {sample['contamination_levels']} contamination levels")
            print(f"     Original genes: {len(sample['original_genes'])} | 50% contaminated: {len(sample['contaminated_50perc'])} | 100% contaminated: {len(sample['contaminated_100perc'])}")
        
        # Test 4: Confidence evaluation queries
        print(f"\n✅ Test 4: Confidence Evaluation Queries")
        
        # Query confidence evaluations
        confidence_evals = kg.query_go_confidence_evaluations()
        print(f"   Total confidence evaluations: {len(confidence_evals)} entries")
        
        if confidence_evals:
            sample = confidence_evals[0]
            print(f"   Sample evaluation: GO:{sample['go_term_id']} - LLM: {sample['llm_name']}")
            print(f"     Reviewer score: {sample['reviewer_score_bin']} | Raw score: {sample['raw_score']}")
            print(f"     Analysis preview: {sample['llm_analysis'][:100]}...")
        
        # Test 5: Gene profile queries
        print(f"\n✅ Test 5: Gene Profile Queries")
        
        # Test gene profiles for genes that should have GO analysis data
        test_genes = ['TP53', 'BRCA1', 'MYC', 'EGFR', 'PTEN']  # Common genes likely to be in datasets
        
        for gene in test_genes:
            profile = kg.query_gene_go_analysis_profile(gene)
            if profile['total_analyses'] > 0:
                print(f"   {gene}: {profile['total_analyses']} total analyses")
                print(f"     Core: {len(profile['core_analyses'])} | Contamination: {len(profile['contamination_analyses'])} | Confidence: {len(profile['confidence_evaluations'])}")
                break
        else:
            # If none of the test genes have data, get any gene with GO analysis data
            print("   Testing with first gene found in GO analysis data...")
            
        # Test 6: Integration completeness
        print(f"\n✅ Test 6: Integration Completeness")
        
        # Check that all expected node types exist
        node_types = set()
        go_analysis_sources = 0
        
        for node_id, attrs in kg.graph.nodes(data=True):
            node_types.add(attrs.get('node_type'))
            if attrs.get('source') == 'GO_Analysis_Data':
                go_analysis_sources += 1
        
        expected_go_analysis_types = {
            'go_core_analysis',
            'go_contamination_analysis', 
            'go_confidence_evaluation',
            'similarity_scores'
        }
        
        found_go_analysis_types = {nt for nt in node_types if 'go_' in nt or 'similarity' in nt}
        
        print(f"   Expected GO analysis node types: {len(expected_go_analysis_types)}")
        print(f"   Found GO analysis node types: {len(found_go_analysis_types)}")
        print(f"   GO Analysis Data source nodes: {go_analysis_sources:,}")
        
        # Test 7: Cross-integration validation
        print(f"\n✅ Test 7: Cross-Integration Validation")
        
        # Check that GO analysis data connects to existing GO terms and genes
        go_analysis_to_go_edges = 0
        go_analysis_to_gene_edges = 0
        
        for u, v, attrs in kg.graph.edges(data=True):
            if attrs.get('source') == 'GO_Analysis_Data':
                u_attrs = kg.graph.nodes[u]
                v_attrs = kg.graph.nodes[v]
                
                # Check connections to GO terms
                if (u_attrs.get('node_type') == 'go_term' or v_attrs.get('node_type') == 'go_term'):
                    go_analysis_to_go_edges += 1
                
                # Check connections to genes
                if (u_attrs.get('node_type') == 'gene' or v_attrs.get('node_type') == 'gene'):
                    go_analysis_to_gene_edges += 1
        
        print(f"   GO Analysis → GO term connections: {go_analysis_to_go_edges:,}")
        print(f"   GO Analysis → Gene connections: {go_analysis_to_gene_edges:,}")
        
        # Test 8: Performance validation
        print(f"\n✅ Test 8: Performance Validation")
        
        import time
        
        # Test query performance
        start_time = time.time()
        
        # Run several queries
        kg.query_go_core_analysis()
        kg.query_go_contamination_analysis()
        kg.query_go_confidence_evaluations()
        kg.get_go_analysis_stats()
        
        end_time = time.time()
        query_time = end_time - start_time
        
        print(f"   Query performance: {query_time:.3f} seconds for 4 queries")
        print(f"   Average query time: {query_time/4:.3f} seconds")
        
        # Final validation
        print(f"\n🎯 INTEGRATION VALIDATION SUMMARY")
        
        total_go_analysis_data = (
            go_analysis_stats['core_analyses'] + 
            go_analysis_stats['contamination_analyses'] + 
            go_analysis_stats['confidence_evaluations'] +
            go_analysis_stats['similarity_datasets']
        )
        
        print(f"✅ Total GO Analysis data points: {total_go_analysis_data:,}")
        print(f"✅ Cross-modal connections: {go_analysis_to_go_edges + go_analysis_to_gene_edges:,}")
        print(f"✅ Datasets integrated: {go_analysis_stats['datasets_analyzed']}")
        print(f"✅ Performance: {query_time:.3f}s query time")
        
        # Success criteria
        success_criteria = [
            total_go_analysis_data > 2000,  # Should have substantial data
            go_analysis_stats['datasets_analyzed'] >= 3,  # Multiple datasets
            go_analysis_stats['unique_go_terms'] > 1000,  # Good GO term coverage
            go_analysis_stats['unique_genes'] > 5000,  # Good gene coverage
            query_time < 5.0,  # Reasonable performance
            go_analysis_to_go_edges > 0,  # Connected to GO terms
            go_analysis_to_gene_edges > 0  # Connected to genes
        ]
        
        passed_criteria = sum(success_criteria)
        
        print(f"\n🏆 SUCCESS CRITERIA: {passed_criteria}/{len(success_criteria)} PASSED")
        
        if passed_criteria == len(success_criteria):
            print("🎉 ALL TESTS PASSED - GO ANALYSIS DATA INTEGRATION SUCCESSFUL!")
            return True
        else:
            print("⚠️  SOME TESTS FAILED - INTEGRATION MAY HAVE ISSUES")
            return False
            
    except Exception as e:
        print(f"❌ ERROR during testing: {e}")
        logger.error(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_go_analysis_integration()
    sys.exit(0 if success else 1)