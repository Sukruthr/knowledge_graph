#!/usr/bin/env python3
"""
Comprehensive Testing Suite for kg_builders/go_knowledge_graph.py

This script tests the GOKnowledgeGraph class to ensure it maintains
all functionality from the original kg_builder.py implementation.
"""

import sys
import os
import tempfile
import networkx as nx
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, '/home/mreddy1/knowledge_graph/src')

from kg_builders.go_knowledge_graph import GOKnowledgeGraph

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GOKnowledgeGraphTestSuite:
    """Comprehensive test suite for GOKnowledgeGraph class."""
    
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = {}
        self.data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data/GO_BP"
        
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
        """Test GOKnowledgeGraph initialization."""
        # Test basic initialization
        kg = GOKnowledgeGraph()
        assert kg.use_neo4j == False, "Default use_neo4j should be False"
        assert kg.namespace == 'biological_process', "Default namespace should be 'biological_process'"
        assert isinstance(kg.graph, nx.MultiDiGraph), "Graph should be MultiDiGraph"
        assert kg.parser is None, "Parser should be None initially"
        assert kg.stats == {}, "Stats should be empty dict initially"
        
        # Test custom initialization
        kg_custom = GOKnowledgeGraph(use_neo4j=False, namespace='molecular_function')
        assert kg_custom.namespace == 'molecular_function', "Custom namespace not set"
        
        # Test neo4j initialization (should fallback to networkx)
        kg_neo4j = GOKnowledgeGraph(use_neo4j=True)
        assert kg_neo4j.use_neo4j == False, "Should fallback to NetworkX when neo4j unavailable"
        
        return f"Initialized GOKnowledgeGraph with namespace: {kg.namespace}"
    
    def test_method_existence(self):
        """Test that all expected methods exist."""
        kg = GOKnowledgeGraph()
        
        expected_methods = [
            # Core methods
            '__init__', 'load_data', 'build_graph',
            # Private graph building methods
            '_add_go_term_nodes', '_add_comprehensive_gene_nodes', '_add_go_relationships',
            '_add_go_clusters', '_add_comprehensive_gene_associations', '_add_gene_cross_references',
            '_add_alternative_id_mappings', '_calculate_stats',
            # Public methods
            'get_stats', 'validate_graph_integrity', 'query_gene_functions', 'query_go_term_genes',
            'save_graph', 'load_graph'
        ]
        
        missing_methods = []
        for method in expected_methods:
            if not hasattr(kg, method):
                missing_methods.append(method)
        
        assert len(missing_methods) == 0, f"Missing methods: {missing_methods}"
        
        return f"All {len(expected_methods)} expected methods are present"
    
    def test_load_data_basic(self):
        """Test basic data loading functionality."""
        if not os.path.exists(self.data_dir):
            return "SKIPPED - Test data directory not available"
        
        kg = GOKnowledgeGraph(namespace='biological_process')
        
        # Test data loading
        kg.load_data(self.data_dir)
        
        # Verify parser was created
        assert kg.parser is not None, "Parser should be created after load_data"
        
        # Verify data was loaded
        assert hasattr(kg, 'go_terms'), "go_terms should be loaded"
        assert hasattr(kg, 'go_relationships'), "go_relationships should be loaded"
        assert hasattr(kg, 'gene_associations'), "gene_associations should be loaded"
        assert hasattr(kg, 'go_alt_ids'), "go_alt_ids should be loaded"
        assert hasattr(kg, 'gene_id_mappings'), "gene_id_mappings should be loaded"
        assert hasattr(kg, 'obo_terms'), "obo_terms should be loaded"
        
        # Check data has content
        assert len(kg.go_terms) > 0, "GO terms should be loaded"
        assert len(kg.gene_associations) > 0, "Gene associations should be loaded"
        
        return f"Loaded {len(kg.go_terms)} GO terms and {len(kg.gene_associations)} gene associations"
    
    def test_build_graph_basic(self):
        """Test basic graph building functionality."""
        if not os.path.exists(self.data_dir):
            return "SKIPPED - Test data directory not available"
        
        kg = GOKnowledgeGraph()
        kg.load_data(self.data_dir)
        
        # Get initial graph state
        initial_nodes = kg.graph.number_of_nodes()
        initial_edges = kg.graph.number_of_edges()
        
        # Build graph
        kg.build_graph()
        
        # Verify graph was built
        assert kg.graph.number_of_nodes() > initial_nodes, "Nodes should be added during graph building"
        assert kg.graph.number_of_edges() > initial_edges, "Edges should be added during graph building"
        
        # Verify stats were calculated
        assert kg.stats != {}, "Stats should be calculated"
        assert 'total_nodes' in kg.stats, "total_nodes should be in stats"
        assert 'total_edges' in kg.stats, "total_edges should be in stats"
        
        return f"Built graph with {kg.graph.number_of_nodes()} nodes and {kg.graph.number_of_edges()} edges"
    
    def test_graph_structure(self):
        """Test that graph has expected structure."""
        if not os.path.exists(self.data_dir):
            return "SKIPPED - Test data directory not available"
        
        kg = GOKnowledgeGraph()
        kg.load_data(self.data_dir)
        kg.build_graph()
        
        # Check for expected node types
        node_types = set()
        go_nodes = 0
        gene_nodes = 0
        
        for node_id, node_data in kg.graph.nodes(data=True):
            node_type = node_data.get('node_type')
            node_types.add(node_type)
            
            if node_type == 'go_term':
                go_nodes += 1
                # Verify GO node structure
                assert 'name' in node_data, f"GO node {node_id} missing name"
                assert 'namespace' in node_data, f"GO node {node_id} missing namespace"
            elif node_type == 'gene':
                gene_nodes += 1
                # Verify gene node structure
                assert 'gene_symbol' in node_data, f"Gene node {node_id} missing gene_symbol"
        
        # Verify expected node types exist
        expected_types = {'go_term', 'gene'}
        assert expected_types.issubset(node_types), f"Missing node types: {expected_types - node_types}"
        
        # Check for expected edge types
        edge_types = set()
        for source, target, edge_data in kg.graph.edges(data=True):
            edge_type = edge_data.get('edge_type')
            edge_types.add(edge_type)
        
        expected_edge_types = {'gene_annotation', 'go_hierarchy'}
        found_edge_types = expected_edge_types & edge_types
        assert len(found_edge_types) > 0, f"No expected edge types found. Found: {edge_types}"
        
        return f"Graph structure verified. Node types: {node_types}, Edge types: {edge_types}"
    
    def test_query_gene_functions(self):
        """Test gene function querying."""
        if not os.path.exists(self.data_dir):
            return "SKIPPED - Test data directory not available"
        
        kg = GOKnowledgeGraph()
        kg.load_data(self.data_dir)
        kg.build_graph()
        
        # Find a gene to test with
        test_gene = None
        for node_id, node_data in kg.graph.nodes(data=True):
            if node_data.get('node_type') == 'gene':
                test_gene = node_id
                break
        
        if not test_gene:
            return "SKIPPED - No gene nodes found in graph"
        
        # Test query
        results = kg.query_gene_functions(test_gene)
        
        # Verify results structure
        assert isinstance(results, list), "Results should be a list"
        
        if len(results) > 0:
            result = results[0]
            expected_keys = {'go_id', 'go_name', 'evidence_code', 'qualifier', 'namespace'}
            assert expected_keys.issubset(result.keys()), f"Missing keys in result: {expected_keys - result.keys()}"
        
        # Test non-existent gene
        empty_results = kg.query_gene_functions("NONEXISTENT_GENE")
        assert empty_results == [], "Non-existent gene should return empty list"
        
        return f"Successfully queried {test_gene}, found {len(results)} GO annotations"
    
    def test_query_go_term_genes(self):
        """Test GO term gene querying."""
        if not os.path.exists(self.data_dir):
            return "SKIPPED - Test data directory not available"
        
        kg = GOKnowledgeGraph()
        kg.load_data(self.data_dir)
        kg.build_graph()
        
        # Find a GO term to test with
        test_go_term = None
        for node_id, node_data in kg.graph.nodes(data=True):
            if node_data.get('node_type') == 'go_term':
                test_go_term = node_id
                break
        
        if not test_go_term:
            return "SKIPPED - No GO term nodes found in graph"
        
        # Test query
        results = kg.query_go_term_genes(test_go_term)
        
        # Verify results structure
        assert isinstance(results, list), "Results should be a list"
        
        if len(results) > 0:
            result = results[0]
            expected_keys = {'gene_symbol', 'gene_name', 'evidence_code', 'qualifier'}
            assert expected_keys.issubset(result.keys()), f"Missing keys in result: {expected_keys - result.keys()}"
        
        # Test non-existent GO term
        empty_results = kg.query_go_term_genes("GO:9999999")
        assert empty_results == [], "Non-existent GO term should return empty list"
        
        return f"Successfully queried {test_go_term}, found {len(results)} associated genes"
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        if not os.path.exists(self.data_dir):
            return "SKIPPED - Test data directory not available"
        
        kg = GOKnowledgeGraph()
        kg.load_data(self.data_dir)
        kg.build_graph()
        
        stats = kg.get_stats()
        
        # Verify stats structure
        expected_keys = {
            'namespace', 'total_nodes', 'total_edges', 'go_terms', 'genes',
            'gene_identifiers', 'enhanced_go_terms', 'alternative_go_ids',
            'go_relationships', 'go_clusters', 'gene_associations',
            'gaf_associations', 'collapsed_associations', 'gene_cross_references',
            'alternative_id_mappings', 'total_gene_id_mappings'
        }
        
        assert expected_keys.issubset(stats.keys()), f"Missing stats keys: {expected_keys - stats.keys()}"
        
        # Verify stats values make sense
        assert stats['total_nodes'] > 0, "Should have nodes"
        assert stats['total_edges'] > 0, "Should have edges"
        assert stats['namespace'] == kg.namespace, "Namespace should match"
        
        return f"Stats retrieved: {stats['total_nodes']} nodes, {stats['total_edges']} edges"
    
    def test_validate_graph_integrity(self):
        """Test graph integrity validation."""
        if not os.path.exists(self.data_dir):
            return "SKIPPED - Test data directory not available"
        
        kg = GOKnowledgeGraph()
        kg.load_data(self.data_dir)
        kg.build_graph()
        
        validation = kg.validate_graph_integrity()
        
        # Verify validation structure
        expected_keys = {
            'has_nodes', 'has_edges', 'go_terms_valid', 'gene_nodes_valid',
            'relationships_valid', 'associations_valid', 'cross_references_valid',
            'overall_valid'
        }
        
        assert expected_keys.issubset(validation.keys()), f"Missing validation keys: {expected_keys - validation.keys()}"
        
        # Verify basic validation passes
        assert validation['has_nodes'] == True, "Should have nodes"
        assert validation['has_edges'] == True, "Should have edges"
        
        return f"Graph validation: {validation}"
    
    def test_save_load_graph(self):
        """Test saving and loading graph functionality."""
        if not os.path.exists(self.data_dir):
            return "SKIPPED - Test data directory not available"
        
        # Create and build original graph
        kg_original = GOKnowledgeGraph()
        kg_original.load_data(self.data_dir)
        kg_original.build_graph()
        
        original_nodes = kg_original.graph.number_of_nodes()
        original_edges = kg_original.graph.number_of_edges()
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            filepath = tmp_file.name
        
        try:
            # Save graph
            kg_original.save_graph(filepath)
            
            # Create new instance and load
            kg_loaded = GOKnowledgeGraph()
            kg_loaded.load_graph(filepath)
            
            # Verify loaded graph
            assert kg_loaded.graph.number_of_nodes() == original_nodes, "Node count mismatch after load"
            assert kg_loaded.graph.number_of_edges() == original_edges, "Edge count mismatch after load"
            
            # Verify stats were recalculated
            assert kg_loaded.stats != {}, "Stats should be recalculated after load"
            
            return f"Successfully saved and loaded graph with {original_nodes} nodes, {original_edges} edges"
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def run_all_tests(self):
        """Run all GOKnowledgeGraph tests."""
        logger.info("=" * 80)
        logger.info("🧪 GO KNOWLEDGE GRAPH COMPREHENSIVE TEST SUITE")
        logger.info("=" * 80)
        
        # Define all tests
        tests = [
            ("Initialization", self.test_initialization),
            ("Method Existence", self.test_method_existence),
            ("Load Data Basic", self.test_load_data_basic),
            ("Build Graph Basic", self.test_build_graph_basic),
            ("Graph Structure", self.test_graph_structure),
            ("Query Gene Functions", self.test_query_gene_functions),
            ("Query GO Term Genes", self.test_query_go_term_genes),
            ("Get Stats", self.test_get_stats),
            ("Validate Graph Integrity", self.test_validate_graph_integrity),
            ("Save Load Graph", self.test_save_load_graph)
        ]
        
        # Run all tests
        for test_name, test_function in tests:
            self.run_test(test_name, test_function)
        
        # Generate summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 GO KNOWLEDGE GRAPH TEST RESULTS SUMMARY")
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
    test_suite = GOKnowledgeGraphTestSuite()
    success = test_suite.run_all_tests()
    
    # Save results
    results_file = '/home/mreddy1/knowledge_graph/kg_testing/go_knowledge_graph_test_results.json'
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
        print("\n🎉 All GO Knowledge Graph tests passed!")
        return True
    else:
        print("\n⚠️ Some GO Knowledge Graph tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)