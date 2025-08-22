#!/usr/bin/env python3
"""
Comprehensive Testing Suite for kg_builders/combined_go_graph.py

This script tests the CombinedGOKnowledgeGraph class to ensure it maintains
all functionality from the original kg_builder.py implementation.
"""

import sys
import os
import tempfile
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, '/home/mreddy1/knowledge_graph/src')

from kg_builders.combined_go_graph import CombinedGOKnowledgeGraph

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CombinedGOKnowledgeGraphTestSuite:
    """Comprehensive test suite for CombinedGOKnowledgeGraph class."""
    
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
        """Test CombinedGOKnowledgeGraph initialization."""
        # Test basic initialization
        kg = CombinedGOKnowledgeGraph()
        assert kg.use_neo4j == False, "Default use_neo4j should be False"
        assert hasattr(kg, 'graph'), "Should have graph attribute"
        assert hasattr(kg, 'parsers'), "Should have parsers attribute"
        assert hasattr(kg, 'individual_graphs'), "Should have individual_graphs attribute"
        assert hasattr(kg, 'combined_stats'), "Should have combined_stats attribute"
        
        assert kg.parsers == {}, "Parsers should be empty dict initially"
        assert kg.individual_graphs == {}, "Individual graphs should be empty dict initially"
        assert kg.combined_stats == {}, "Combined stats should be empty dict initially"
        
        # Test neo4j initialization (should fallback to networkx)
        kg_neo4j = CombinedGOKnowledgeGraph(use_neo4j=True)
        assert kg_neo4j.use_neo4j == False, "Should fallback to NetworkX when neo4j unavailable"
        
        return "Successfully initialized CombinedGOKnowledgeGraph"
    
    def test_method_existence(self):
        """Test that all expected methods exist."""
        kg = CombinedGOKnowledgeGraph()
        
        expected_methods = [
            # Core methods
            '__init__', 'load_data', 'build_combined_graph',
            # Private methods
            '_calculate_combined_stats', '_validate_combined_graph',
            # Public methods
            'get_combined_stats', 'query_gene_functions_all_namespaces', 'save_combined_graph'
        ]
        
        missing_methods = []
        for method in expected_methods:
            if not hasattr(kg, method):
                missing_methods.append(method)
        
        assert len(missing_methods) == 0, f"Missing methods: {missing_methods}"
        
        return f"All {len(expected_methods)} expected methods are present"
    
    def test_load_data_basic(self):
        """Test basic data loading functionality."""
        if not os.path.exists(self.base_data_dir):
            return "SKIPPED - Test data directory not available"
        
        kg = CombinedGOKnowledgeGraph()
        
        # Test data loading
        kg.load_data(self.base_data_dir)
        
        # Verify individual graphs were created
        assert len(kg.individual_graphs) > 0, "Individual graphs should be created"
        
        # Check expected namespaces
        expected_namespaces = ['biological_process', 'cellular_component', 'molecular_function']
        available_namespaces = list(kg.individual_graphs.keys())
        
        # At least one namespace should be available
        assert len(available_namespaces) > 0, "At least one namespace should be loaded"
        
        # Verify each loaded graph has stats
        for namespace, individual_kg in kg.individual_graphs.items():
            assert hasattr(individual_kg, 'get_stats'), f"{namespace} graph missing get_stats method"
            stats = individual_kg.get_stats()
            assert 'total_nodes' in stats, f"{namespace} stats missing total_nodes"
            assert 'total_edges' in stats, f"{namespace} stats missing total_edges"
            assert stats['total_nodes'] > 0, f"{namespace} should have nodes"
        
        return f"Loaded {len(available_namespaces)} namespaces: {available_namespaces}"
    
    def test_build_combined_graph(self):
        """Test combined graph building functionality."""
        if not os.path.exists(self.base_data_dir):
            return "SKIPPED - Test data directory not available"
        
        kg = CombinedGOKnowledgeGraph()
        kg.load_data(self.base_data_dir)
        
        # Get initial graph state
        initial_nodes = kg.graph.number_of_nodes()
        initial_edges = kg.graph.number_of_edges()
        
        # Build combined graph
        kg.build_combined_graph()
        
        # Verify graph was built
        assert kg.graph.number_of_nodes() > initial_nodes, "Nodes should be added during graph building"
        assert kg.graph.number_of_edges() > initial_edges, "Edges should be added during graph building"
        
        # Verify stats were calculated
        assert kg.combined_stats != {}, "Combined stats should be calculated"
        assert 'total_nodes' in kg.combined_stats, "total_nodes should be in combined_stats"
        assert 'total_edges' in kg.combined_stats, "total_edges should be in combined_stats"
        
        return f"Built combined graph with {kg.graph.number_of_nodes()} nodes and {kg.graph.number_of_edges()} edges"
    
    def test_combined_graph_structure(self):
        """Test that combined graph has expected structure."""
        if not os.path.exists(self.base_data_dir):
            return "SKIPPED - Test data directory not available"
        
        kg = CombinedGOKnowledgeGraph()
        kg.load_data(self.base_data_dir)
        kg.build_combined_graph()
        
        # Check for expected node types
        node_types = set()
        namespace_counts = {}
        
        for node_id, node_data in kg.graph.nodes(data=True):
            node_type = node_data.get('node_type')
            node_types.add(node_type)
            
            if node_type == 'go_term':
                namespace = node_data.get('namespace', 'unknown')
                namespace_counts[namespace] = namespace_counts.get(namespace, 0) + 1
        
        # Verify expected node types exist
        expected_types = {'go_term', 'gene'}
        assert expected_types.issubset(node_types), f"Missing node types: {expected_types - node_types}"
        
        # Should have multiple namespaces if data is available
        if len(kg.individual_graphs) > 1:
            assert len(namespace_counts) > 1, f"Should have multiple namespaces, got: {namespace_counts}"
        
        # Check for expected edge types
        edge_types = set()
        for source, target, edge_data in kg.graph.edges(data=True):
            edge_type = edge_data.get('edge_type')
            edge_types.add(edge_type)
        
        expected_edge_types = {'gene_annotation', 'go_hierarchy'}
        found_edge_types = expected_edge_types & edge_types
        assert len(found_edge_types) > 0, f"No expected edge types found. Found: {edge_types}"
        
        return f"Combined graph structure verified. Node types: {node_types}, Namespaces: {namespace_counts}, Edge types: {edge_types}"
    
    def test_query_gene_functions_all_namespaces(self):
        """Test querying gene functions across all namespaces."""
        if not os.path.exists(self.base_data_dir):
            return "SKIPPED - Test data directory not available"
        
        kg = CombinedGOKnowledgeGraph()
        kg.load_data(self.base_data_dir)
        kg.build_combined_graph()
        
        # Find a gene to test with
        test_gene = None
        for node_id, node_data in kg.graph.nodes(data=True):
            if node_data.get('node_type') == 'gene':
                test_gene = node_id
                break
        
        if not test_gene:
            return "SKIPPED - No gene nodes found in combined graph"
        
        # Test query across all namespaces
        results = kg.query_gene_functions_all_namespaces(test_gene)
        
        # Verify results structure
        assert isinstance(results, dict), "Results should be a dictionary"
        
        # Each namespace in results should have list of functions
        for namespace, functions in results.items():
            assert isinstance(functions, list), f"Functions for {namespace} should be a list"
            if len(functions) > 0:
                function = functions[0]
                expected_keys = {'go_id', 'go_name', 'evidence_code', 'qualifier', 'namespace'}
                assert expected_keys.issubset(function.keys()), f"Missing keys in {namespace} result: {expected_keys - function.keys()}"
        
        # Test non-existent gene
        empty_results = kg.query_gene_functions_all_namespaces("NONEXISTENT_GENE")
        assert empty_results == {}, "Non-existent gene should return empty dict"
        
        return f"Successfully queried {test_gene} across {len(results)} namespaces, found functions in: {list(results.keys())}"
    
    def test_get_combined_stats(self):
        """Test combined statistics retrieval."""
        if not os.path.exists(self.base_data_dir):
            return "SKIPPED - Test data directory not available"
        
        kg = CombinedGOKnowledgeGraph()
        kg.load_data(self.base_data_dir)
        kg.build_combined_graph()
        
        stats = kg.get_combined_stats()
        
        # Verify stats structure
        expected_keys = {
            'total_nodes', 'total_edges', 'go_terms', 'genes', 'gene_identifiers',
            'go_relationships', 'gene_associations', 'namespace_counts', 'individual_stats'
        }
        
        assert expected_keys.issubset(stats.keys()), f"Missing stats keys: {expected_keys - stats.keys()}"
        
        # Verify stats values make sense
        assert stats['total_nodes'] > 0, "Should have nodes"
        assert stats['total_edges'] > 0, "Should have edges"
        assert isinstance(stats['namespace_counts'], dict), "namespace_counts should be dict"
        assert isinstance(stats['individual_stats'], dict), "individual_stats should be dict"
        
        # Verify individual stats are included
        for namespace in kg.individual_graphs.keys():
            assert namespace in stats['individual_stats'], f"Missing individual stats for {namespace}"
        
        return f"Combined stats retrieved: {stats['total_nodes']} nodes, {stats['total_edges']} edges, namespaces: {list(stats['namespace_counts'].keys())}"
    
    def test_save_combined_graph(self):
        """Test saving combined graph functionality."""
        if not os.path.exists(self.base_data_dir):
            return "SKIPPED - Test data directory not available"
        
        kg = CombinedGOKnowledgeGraph()
        kg.load_data(self.base_data_dir)
        kg.build_combined_graph()
        
        original_nodes = kg.graph.number_of_nodes()
        original_edges = kg.graph.number_of_edges()
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            filepath = tmp_file.name
        
        try:
            # Save graph
            kg.save_combined_graph(filepath)
            
            # Verify file exists and has content
            assert os.path.exists(filepath), "File was not created"
            assert os.path.getsize(filepath) > 0, "File is empty"
            
            return f"Successfully saved combined graph with {original_nodes} nodes, {original_edges} edges"
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_validation_methods(self):
        """Test internal validation methods."""
        if not os.path.exists(self.base_data_dir):
            return "SKIPPED - Test data directory not available"
        
        kg = CombinedGOKnowledgeGraph()
        kg.load_data(self.base_data_dir)
        kg.build_combined_graph()
        
        # Test private validation method
        validation = kg._validate_combined_graph()
        
        # Verify validation structure
        expected_keys = {'has_nodes', 'has_edges', 'multiple_namespaces'}
        assert expected_keys.issubset(validation.keys()), f"Missing validation keys: {expected_keys - validation.keys()}"
        
        # Verify basic validation passes
        assert validation['has_nodes'] == True, "Should have nodes"
        assert validation['has_edges'] == True, "Should have edges"
        
        # If multiple namespaces were loaded, should detect them
        if len(kg.individual_graphs) > 1:
            assert validation['multiple_namespaces'] == True, "Should detect multiple namespaces"
        
        return f"Combined graph validation: {validation}"
    
    def run_all_tests(self):
        """Run all CombinedGOKnowledgeGraph tests."""
        logger.info("=" * 80)
        logger.info("🧪 COMBINED GO KNOWLEDGE GRAPH COMPREHENSIVE TEST SUITE")
        logger.info("=" * 80)
        
        # Define all tests
        tests = [
            ("Initialization", self.test_initialization),
            ("Method Existence", self.test_method_existence),
            ("Load Data Basic", self.test_load_data_basic),
            ("Build Combined Graph", self.test_build_combined_graph),
            ("Combined Graph Structure", self.test_combined_graph_structure),
            ("Query Gene Functions All Namespaces", self.test_query_gene_functions_all_namespaces),
            ("Get Combined Stats", self.test_get_combined_stats),
            ("Save Combined Graph", self.test_save_combined_graph),
            ("Validation Methods", self.test_validation_methods)
        ]
        
        # Run all tests
        for test_name, test_function in tests:
            self.run_test(test_name, test_function)
        
        # Generate summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 COMBINED GO KNOWLEDGE GRAPH TEST RESULTS SUMMARY")
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
    test_suite = CombinedGOKnowledgeGraphTestSuite()
    success = test_suite.run_all_tests()
    
    # Save results
    results_file = '/home/mreddy1/knowledge_graph/kg_testing/combined_go_graph_test_results.json'
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
        print("\n🎉 All Combined GO Knowledge Graph tests passed!")
        return True
    else:
        print("\n⚠️ Some Combined GO Knowledge Graph tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)