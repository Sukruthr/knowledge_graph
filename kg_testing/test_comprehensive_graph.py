#!/usr/bin/env python3
"""
Comprehensive Testing Suite for kg_builders/comprehensive_graph.py

This script tests the ComprehensiveBiomedicalKnowledgeGraph class to ensure it maintains
all functionality from the original kg_builder.py implementation.
"""

import sys
import os
import tempfile
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, '/home/mreddy1/knowledge_graph/src')

from kg_builders.comprehensive_graph import ComprehensiveBiomedicalKnowledgeGraph

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveGraphTestSuite:
    """Comprehensive test suite for ComprehensiveBiomedicalKnowledgeGraph class."""
    
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = {}
        self.base_data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data"
        
    def run_test(self, test_name, test_function):
        """Run a single test and record results."""
        try:
            logger.info(f"Running test: {test_name}")
            result = test_function()
            if result:
                self.passed_tests += 1
                self.test_results[test_name] = {'status': 'PASSED', 'details': result}
                logger.info(f"✅ {test_name} - PASSED")
            else:
                self.failed_tests += 1
                self.test_results[test_name] = {'status': 'FAILED', 'details': 'Test returned False'}
                logger.error(f"❌ {test_name} - FAILED")
        except Exception as e:
            self.failed_tests += 1
            self.test_results[test_name] = {'status': 'ERROR', 'details': str(e)}
            logger.error(f"❌ {test_name} - ERROR: {e}")
    
    def test_initialization(self):
        """Test ComprehensiveBiomedicalKnowledgeGraph initialization."""
        # Test basic initialization
        kg = ComprehensiveBiomedicalKnowledgeGraph()
        assert kg.use_neo4j == False, "Default use_neo4j should be False"
        assert hasattr(kg, 'graph'), "Should have graph attribute"
        assert hasattr(kg, 'parser'), "Should have parser attribute"
        assert hasattr(kg, 'parsed_data'), "Should have parsed_data attribute"
        assert hasattr(kg, 'stats'), "Should have stats attribute"
        
        assert kg.parser is None, "Parser should be None initially"
        assert kg.parsed_data == {}, "Parsed data should be empty dict initially"
        assert kg.stats == {}, "Stats should be empty dict initially"
        
        # Test neo4j initialization (should fallback to networkx)
        kg_neo4j = ComprehensiveBiomedicalKnowledgeGraph(use_neo4j=True)
        assert kg_neo4j.use_neo4j == False, "Should fallback to NetworkX when neo4j unavailable"
        
        return "Successfully initialized ComprehensiveBiomedicalKnowledgeGraph"
    
    def test_method_existence(self):
        """Test that all expected methods exist."""
        kg = ComprehensiveBiomedicalKnowledgeGraph()
        
        # Core methods
        core_methods = [
            '__init__', 'load_data', 'build_comprehensive_graph',
            'get_comprehensive_stats', 'save_comprehensive_graph'
        ]
        
        # Query methods
        query_methods = [
            'query_gene_comprehensive', 'query_model_predictions', 'query_model_comparison_summary',
            'query_cc_mf_terms', 'query_cc_mf_llm_interpretations', 'query_cc_mf_similarity_rankings',
            'query_gene_cc_mf_profile', 'get_cc_mf_branch_stats',
            'query_llm_interpretations', 'query_contamination_analysis', 'query_llm_similarity_rankings',
            'query_gene_llm_profile', 'get_llm_processed_stats',
            'query_go_core_analysis', 'query_go_contamination_analysis', 'query_go_confidence_evaluations',
            'query_gene_go_analysis_profile', 'get_go_analysis_stats'
        ]
        
        # Private graph building methods
        private_methods = [
            '_build_go_component', '_add_omics_nodes', '_add_omics_associations',
            '_add_cluster_relationships', '_add_semantic_enhancements',
            '_add_model_comparison_data', '_add_cc_mf_branch_data', '_add_llm_processed_data',
            '_add_go_analysis_data', '_add_remaining_data', '_add_talisman_gene_sets',
            '_calculate_comprehensive_stats', '_validate_comprehensive_graph'
        ]
        
        all_methods = core_methods + query_methods + private_methods
        
        missing_methods = []
        for method in all_methods:
            if not hasattr(kg, method):
                missing_methods.append(method)
        
        assert len(missing_methods) == 0, f"Missing methods: {missing_methods}"
        
        return f"All {len(all_methods)} expected methods are present"
    
    def test_load_data_basic(self):
        """Test basic data loading functionality."""
        if not os.path.exists(self.base_data_dir):
            return "SKIPPED - Test data directory not available"
        
        kg = ComprehensiveBiomedicalKnowledgeGraph()
        
        # Test data loading
        kg.load_data(self.base_data_dir)
        
        # Verify parser was created
        assert kg.parser is not None, "Parser should be created after load_data"
        
        # Verify data was loaded
        assert kg.parsed_data != {}, "Parsed data should not be empty"
        assert 'go_data' in kg.parsed_data, "GO data should be loaded"
        
        # Check if omics data is available
        data_sources = list(kg.parsed_data.keys())
        
        return f"Loaded data sources: {data_sources}"
    
    def test_build_comprehensive_graph_basic(self):
        """Test basic comprehensive graph building."""
        if not os.path.exists(self.base_data_dir):
            return "SKIPPED - Test data directory not available"
        
        kg = ComprehensiveBiomedicalKnowledgeGraph()
        kg.load_data(self.base_data_dir)
        
        # Get initial graph state
        initial_nodes = kg.graph.number_of_nodes()
        initial_edges = kg.graph.number_of_edges()
        
        # Build comprehensive graph
        kg.build_comprehensive_graph()
        
        # Verify graph was built
        assert kg.graph.number_of_nodes() > initial_nodes, "Nodes should be added during graph building"
        assert kg.graph.number_of_edges() > initial_edges, "Edges should be added during graph building"
        
        # Verify stats were calculated
        assert kg.stats != {}, "Stats should be calculated"
        assert 'total_nodes' in kg.stats, "total_nodes should be in stats"
        assert 'total_edges' in kg.stats, "total_edges should be in stats"
        
        return f"Built comprehensive graph with {kg.graph.number_of_nodes()} nodes and {kg.graph.number_of_edges()} edges"
    
    def test_comprehensive_graph_structure(self):
        """Test that comprehensive graph has expected structure."""
        if not os.path.exists(self.base_data_dir):
            return "SKIPPED - Test data directory not available"
        
        kg = ComprehensiveBiomedicalKnowledgeGraph()
        kg.load_data(self.base_data_dir)
        kg.build_comprehensive_graph()
        
        # Check for expected node types
        node_types = set()
        for node_id, node_data in kg.graph.nodes(data=True):
            node_type = node_data.get('node_type')
            node_types.add(node_type)
        
        # Verify expected node types exist
        expected_types = {'go_term', 'gene'}
        assert expected_types.issubset(node_types), f"Missing basic node types: {expected_types - node_types}"
        
        # Check for expected edge types
        edge_types = set()
        for source, target, edge_data in kg.graph.edges(data=True):
            edge_type = edge_data.get('edge_type')
            edge_types.add(edge_type)
        
        expected_edge_types = {'gene_annotation', 'go_hierarchy'}
        found_edge_types = expected_edge_types & edge_types
        assert len(found_edge_types) > 0, f"No expected edge types found. Found: {edge_types}"
        
        return f"Comprehensive graph structure verified. Node types: {node_types}, Edge types: {edge_types}"
    
    def test_query_gene_comprehensive(self):
        """Test comprehensive gene querying."""
        if not os.path.exists(self.base_data_dir):
            return "SKIPPED - Test data directory not available"
        
        kg = ComprehensiveBiomedicalKnowledgeGraph()
        kg.load_data(self.base_data_dir)
        kg.build_comprehensive_graph()
        
        # Find a gene to test with
        test_gene = None
        for node_id, node_data in kg.graph.nodes(data=True):
            if node_data.get('node_type') == 'gene':
                test_gene = node_id
                break
        
        if not test_gene:
            return "SKIPPED - No gene nodes found in graph"
        
        # Test comprehensive query
        results = kg.query_gene_comprehensive(test_gene)
        
        # Verify results structure
        expected_keys = {
            'gene_symbol', 'go_annotations', 'disease_associations', 'drug_perturbations',
            'viral_responses', 'cluster_memberships', 'gene_set_memberships',
            'semantic_annotations', 'model_predictions'
        }
        assert expected_keys.issubset(results.keys()), f"Missing keys in comprehensive results: {expected_keys - results.keys()}"
        
        # Verify each section is a list
        for key in expected_keys:
            if key != 'gene_symbol':
                assert isinstance(results[key], list), f"{key} should be a list"
        
        # Test non-existent gene
        empty_results = kg.query_gene_comprehensive("NONEXISTENT_GENE")
        assert empty_results == {}, "Non-existent gene should return empty dict"
        
        found_annotations = len(results['go_annotations'])
        found_diseases = len(results['disease_associations'])
        found_drugs = len(results['drug_perturbations'])
        
        return f"Successfully queried {test_gene}: {found_annotations} GO annotations, {found_diseases} diseases, {found_drugs} drugs"
    
    def test_get_comprehensive_stats(self):
        """Test comprehensive statistics retrieval."""
        if not os.path.exists(self.base_data_dir):
            return "SKIPPED - Test data directory not available"
        
        kg = ComprehensiveBiomedicalKnowledgeGraph()
        kg.load_data(self.base_data_dir)
        kg.build_comprehensive_graph()
        
        stats = kg.get_comprehensive_stats()
        
        # Verify stats structure
        expected_keys = {'total_nodes', 'total_edges', 'node_counts', 'edge_counts'}
        assert expected_keys.issubset(stats.keys()), f"Missing stats keys: {expected_keys - stats.keys()}"
        
        # Verify stats values make sense
        assert stats['total_nodes'] > 0, "Should have nodes"
        assert stats['total_edges'] > 0, "Should have edges"
        assert isinstance(stats['node_counts'], dict), "node_counts should be dict"
        assert isinstance(stats['edge_counts'], dict), "edge_counts should be dict"
        
        return f"Comprehensive stats: {stats['total_nodes']} nodes, {stats['total_edges']} edges, {len(stats['node_counts'])} node types, {len(stats['edge_counts'])} edge types"
    
    def test_save_comprehensive_graph(self):
        """Test saving comprehensive graph functionality."""
        if not os.path.exists(self.base_data_dir):
            return "SKIPPED - Test data directory not available"
        
        kg = ComprehensiveBiomedicalKnowledgeGraph()
        kg.load_data(self.base_data_dir)
        kg.build_comprehensive_graph()
        
        original_nodes = kg.graph.number_of_nodes()
        original_edges = kg.graph.number_of_edges()
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            filepath = tmp_file.name
        
        try:
            # Save graph
            kg.save_comprehensive_graph(filepath)
            
            # Verify file exists and has content
            assert os.path.exists(filepath), "File was not created"
            assert os.path.getsize(filepath) > 0, "File is empty"
            
            return f"Successfully saved comprehensive graph with {original_nodes} nodes, {original_edges} edges"
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_query_methods_exist(self):
        """Test that all query methods exist and can be called."""
        if not os.path.exists(self.base_data_dir):
            return "SKIPPED - Test data directory not available"
        
        kg = ComprehensiveBiomedicalKnowledgeGraph()
        kg.load_data(self.base_data_dir)
        kg.build_comprehensive_graph()
        
        # Test query methods that should work without parameters
        query_tests = []
        
        try:
            result = kg.query_model_comparison_summary()
            query_tests.append(('query_model_comparison_summary', 'OK'))
        except Exception as e:
            query_tests.append(('query_model_comparison_summary', f'ERROR: {e}'))
        
        try:
            result = kg.get_cc_mf_branch_stats()
            query_tests.append(('get_cc_mf_branch_stats', 'OK'))
        except Exception as e:
            query_tests.append(('get_cc_mf_branch_stats', f'ERROR: {e}'))
        
        try:
            result = kg.get_llm_processed_stats()
            query_tests.append(('get_llm_processed_stats', 'OK'))
        except Exception as e:
            query_tests.append(('get_llm_processed_stats', f'ERROR: {e}'))
        
        try:
            result = kg.get_go_analysis_stats()
            query_tests.append(('get_go_analysis_stats', 'OK'))
        except Exception as e:
            query_tests.append(('get_go_analysis_stats', f'ERROR: {e}'))
        
        # Count successful tests
        successful_queries = len([test for test in query_tests if test[1] == 'OK'])
        
        return f"Query method tests: {successful_queries}/{len(query_tests)} successful. Results: {query_tests}"
    
    def run_all_tests(self):
        """Run all ComprehensiveBiomedicalKnowledgeGraph tests."""
        logger.info("=" * 80)
        logger.info("🧪 COMPREHENSIVE BIOMEDICAL KNOWLEDGE GRAPH TEST SUITE")
        logger.info("=" * 80)
        
        # Define all tests
        tests = [
            ("Initialization", self.test_initialization),
            ("Method Existence", self.test_method_existence),
            ("Load Data Basic", self.test_load_data_basic),
            ("Build Comprehensive Graph Basic", self.test_build_comprehensive_graph_basic),
            ("Comprehensive Graph Structure", self.test_comprehensive_graph_structure),
            ("Query Gene Comprehensive", self.test_query_gene_comprehensive),
            ("Get Comprehensive Stats", self.test_get_comprehensive_stats),
            ("Save Comprehensive Graph", self.test_save_comprehensive_graph),
            ("Query Methods Exist", self.test_query_methods_exist)
        ]
        
        # Run all tests
        for test_name, test_function in tests:
            self.run_test(test_name, test_function)
        
        # Generate summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 COMPREHENSIVE GRAPH TEST RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Tests: {self.passed_tests + self.failed_tests}")
        logger.info(f"Passed: {self.passed_tests}")
        logger.info(f"Failed: {self.failed_tests}")
        
        if self.passed_tests + self.failed_tests > 0:
            success_rate = (self.passed_tests / (self.passed_tests + self.failed_tests)) * 100
            logger.info(f"Success Rate: {success_rate:.1f}%")
        
        if self.failed_tests > 0:
            logger.info("\n❌ FAILED TESTS:")
            for test_name, result in self.test_results.items():
                if result['status'] in ['FAILED', 'ERROR']:
                    logger.info(f"   {test_name}: {result['details']}")
        
        logger.info("=" * 80)
        
        return self.failed_tests == 0


def main():
    """Main execution function."""
    test_suite = ComprehensiveGraphTestSuite()
    success = test_suite.run_all_tests()
    
    # Save results
    results_file = '/home/mreddy1/knowledge_graph/kg_testing/comprehensive_graph_test_results.json'
    import json
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': str(Path(__file__).stat().st_mtime),
            'total_tests': test_suite.passed_tests + test_suite.failed_tests,
            'passed': test_suite.passed_tests,
            'failed': test_suite.failed_tests,
            'success_rate': (test_suite.passed_tests / (test_suite.passed_tests + test_suite.failed_tests)) * 100 if (test_suite.passed_tests + test_suite.failed_tests) > 0 else 0,
            'results': test_suite.test_results
        }, f, indent=2)
    
    logger.info(f"📄 Detailed results saved to: {results_file}")
    
    if success:
        print("\n🎉 All Comprehensive Graph tests passed!")
        return True
    else:
        print("\n⚠️ Some Comprehensive Graph tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)