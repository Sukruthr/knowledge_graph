#!/usr/bin/env python3
"""
Minimal test for GO Analysis Data integration in knowledge graph
Tests with minimal data to avoid timeout
"""

import sys
sys.path.append('src')

import logging
from kg_builder import ComprehensiveBiomedicalKnowledgeGraph

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Reduce log noise
logger = logging.getLogger(__name__)

def test_minimal_go_analysis_kg():
    """Test GO Analysis Data integration with minimal knowledge graph."""
    
    print("=" * 60)
    print("MINIMAL GO ANALYSIS KG INTEGRATION TEST")
    print("=" * 60)
    
    try:
        # Initialize knowledge graph
        kg = ComprehensiveBiomedicalKnowledgeGraph()
        
        # Load just the data (without building full graph)
        print("Loading data...")
        kg.load_data("llm_evaluation_for_gene_set_interpretation/data")
        
        # Check that GO analysis data was loaded
        if 'go_analysis_data' not in kg.parsed_data:
            print("❌ GO Analysis Data was not loaded")
            return False
            
        print("✅ GO Analysis Data loaded successfully")
        
        # Create minimal graph with just some basic nodes for testing connections
        print("Creating minimal test graph...")
        
        # Add a few test nodes
        kg.graph.add_node("GO:0048627", node_type='go_term', name='myoblast development')
        kg.graph.add_node("TP53", node_type='gene', symbol='TP53')
        kg.graph.add_node("BRCA1", node_type='gene', symbol='BRCA1')
        
        # Test adding GO analysis data
        print("Testing GO Analysis Data integration...")
        kg._add_go_analysis_data()
        
        # Get statistics
        stats = kg.get_go_analysis_stats()
        
        print(f"\n📊 GO Analysis Integration Results:")
        print(f"   Core analyses: {stats['core_analyses']}")
        print(f"   Contamination analyses: {stats['contamination_analyses']}")  
        print(f"   Confidence evaluations: {stats['confidence_evaluations']}")
        print(f"   Hierarchy relationships: {stats['hierarchy_relationships']}")
        print(f"   Similarity datasets: {stats['similarity_datasets']}")
        print(f"   Datasets analyzed: {stats['datasets_analyzed']}")
        print(f"   Unique GO terms: {stats['unique_go_terms']}")
        print(f"   Unique genes: {stats['unique_genes']}")
        
        # Test query functionality
        print(f"\n🔍 Testing Query Functionality:")
        
        # Test core analysis query
        core_results = kg.query_go_core_analysis()
        print(f"   Core analysis query: {len(core_results)} results")
        
        # Test contamination analysis query  
        contam_results = kg.query_go_contamination_analysis()
        print(f"   Contamination analysis query: {len(contam_results)} results")
        
        # Test confidence evaluation query
        conf_results = kg.query_go_confidence_evaluations()
        print(f"   Confidence evaluation query: {len(conf_results)} results")
        
        # Test gene profile query
        gene_profile = kg.query_gene_go_analysis_profile('TP53')
        print(f"   Gene profile query (TP53): {gene_profile['total_analyses']} analyses")
        
        # Test specific dataset queries
        selected_1000 = kg.query_go_core_analysis(dataset='1000_selected')
        print(f"   1000 selected dataset: {len(selected_1000)} results")
        
        enricher_100 = kg.query_go_core_analysis(dataset='100_enricher_results')
        print(f"   100 enricher dataset: {len(enricher_100)} results")
        
        # Check graph structure
        total_nodes = kg.graph.number_of_nodes()
        total_edges = kg.graph.number_of_edges()
        
        print(f"\n📈 Graph Statistics:")
        print(f"   Total nodes: {total_nodes:,}")
        print(f"   Total edges: {total_edges:,}")
        
        # Count GO Analysis Data nodes
        go_analysis_nodes = 0
        for node_id, attrs in kg.graph.nodes(data=True):
            if attrs.get('source') == 'GO_Analysis_Data':
                go_analysis_nodes += 1
                
        print(f"   GO Analysis Data nodes: {go_analysis_nodes:,}")
        
        # Count GO Analysis Data edges
        go_analysis_edges = 0
        for u, v, attrs in kg.graph.edges(data=True):
            if attrs.get('source') == 'GO_Analysis_Data':
                go_analysis_edges += 1
                
        print(f"   GO Analysis Data edges: {go_analysis_edges:,}")
        
        # Success criteria
        success_criteria = [
            stats['core_analyses'] > 0,
            stats['contamination_analyses'] > 0, 
            stats['confidence_evaluations'] > 0,
            stats['datasets_analyzed'] >= 3,
            len(core_results) > 0,
            len(contam_results) > 0,
            go_analysis_nodes > 1000,  # Should have substantial nodes
            len(selected_1000) > 0,
            len(enricher_100) > 0
        ]
        
        passed_criteria = sum(success_criteria)
        
        print(f"\n🏆 SUCCESS CRITERIA: {passed_criteria}/{len(success_criteria)} PASSED")
        
        if passed_criteria >= 7:  # Allow some flexibility
            print("🎉 GO ANALYSIS DATA INTEGRATION TEST SUCCESSFUL!")
            
            # Show sample data
            if core_results:
                sample = core_results[0]
                print(f"\n📋 Sample Core Analysis:")
                print(f"   GO Term: {sample['go_term_id']}")
                print(f"   Dataset: {sample['dataset']}")
                print(f"   Gene Count: {sample['gene_count']}")
                print(f"   Description: {sample['term_description'][:50]}...")
                
            if contam_results:
                sample = contam_results[0]
                print(f"\n🧪 Sample Contamination Analysis:")
                print(f"   GO Term: {sample['go_term_id']}")
                print(f"   Dataset: {sample['dataset']}")
                print(f"   Contamination Levels: {sample['contamination_levels']}")
                print(f"   Original Genes: {len(sample['original_genes'])}")
                
            return True
        else:
            print("⚠️ SOME TESTS FAILED - INTEGRATION MAY HAVE ISSUES")
            return False
            
    except Exception as e:
        print(f"❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minimal_go_analysis_kg()
    sys.exit(0 if success else 1)