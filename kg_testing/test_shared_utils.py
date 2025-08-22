#!/usr/bin/env python3
"""
Comprehensive Testing Suite for kg_builders/shared_utils.py

This script tests all utility functions extracted to shared_utils.py
to ensure they work correctly and maintain expected functionality.
"""

import sys
import os
import tempfile
import networkx as nx
import pickle
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, '/home/mreddy1/knowledge_graph/src')

from kg_builders.shared_utils import (
    save_graph_to_file,
    load_graph_from_file,
    initialize_graph_attributes,
    validate_basic_graph_structure,
    count_nodes_by_type,
    count_edges_by_type,
    get_basic_graph_stats
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SharedUtilsTestSuite:
    """Comprehensive test suite for shared utilities."""
    
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = {}
        
    def create_test_graph(self):
        """Create a test graph with known structure."""
        graph = nx.MultiDiGraph()
        
        # Add GO term nodes
        graph.add_node('GO:0008150', node_type='go_term', name='biological_process', namespace='BP')
        graph.add_node('GO:0003674', node_type='go_term', name='molecular_function', namespace='MF')
        
        # Add gene nodes
        graph.add_node('TP53', node_type='gene', gene_symbol='TP53', gene_name='tumor protein p53')
        graph.add_node('BRCA1', node_type='gene', gene_symbol='BRCA1', gene_name='BRCA1 DNA repair associated')
        
        # Add other nodes
        graph.add_node('DISEASE:cancer', node_type='disease', disease_name='cancer')
        graph.add_node('ENTREZ:7157', node_type='gene_identifier', identifier_type='entrez')
        
        # Add edges
        graph.add_edge('TP53', 'GO:0008150', edge_type='gene_annotation', evidence_code='IEA')
        graph.add_edge('BRCA1', 'GO:0003674', edge_type='gene_annotation', evidence_code='IDA')
        graph.add_edge('TP53', 'DISEASE:cancer', edge_type='gene_disease_association', weight=0.95)
        graph.add_edge('GO:0008150', 'GO:0003674', edge_type='go_hierarchy', relationship_type='is_a')
        
        return graph
    
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
    
    def test_save_graph_to_file_pkl(self):
        """Test saving graph to pickle format."""
        graph = self.create_test_graph()
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            filepath = tmp_file.name
        
        try:
            # Test save functionality
            save_graph_to_file(graph, filepath)
            
            # Verify file exists and has content
            assert os.path.exists(filepath), "File was not created"
            assert os.path.getsize(filepath) > 0, "File is empty"
            
            # Test that we can load it back
            with open(filepath, 'rb') as f:
                loaded_graph = pickle.load(f)
            
            # Verify graph structure
            assert loaded_graph.number_of_nodes() == graph.number_of_nodes(), "Node count mismatch"
            assert loaded_graph.number_of_edges() == graph.number_of_edges(), "Edge count mismatch"
            
            return f"Successfully saved and verified pickle format. Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}"
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_save_graph_to_file_graphml(self):
        """Test saving graph to GraphML format."""
        graph = self.create_test_graph()
        
        with tempfile.NamedTemporaryFile(suffix='.graphml', delete=False) as tmp_file:
            filepath = tmp_file.name
        
        try:
            # Test save functionality
            save_graph_to_file(graph, filepath)
            
            # Verify file exists and has content
            assert os.path.exists(filepath), "File was not created"
            assert os.path.getsize(filepath) > 0, "File is empty"
            
            # Test that we can load it back
            loaded_graph = nx.read_graphml(filepath)
            
            # Verify basic structure (GraphML might change some attributes)
            assert loaded_graph.number_of_nodes() == graph.number_of_nodes(), "Node count mismatch"
            assert loaded_graph.number_of_edges() == graph.number_of_edges(), "Edge count mismatch"
            
            return f"Successfully saved and verified GraphML format. Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}"
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_load_graph_from_file_pkl(self):
        """Test loading graph from pickle format."""
        original_graph = self.create_test_graph()
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            filepath = tmp_file.name
        
        try:
            # Save graph first
            with open(filepath, 'wb') as f:
                pickle.dump(original_graph, f)
            
            # Test load functionality
            loaded_graph = load_graph_from_file(filepath)
            
            # Verify graph structure
            assert loaded_graph.number_of_nodes() == original_graph.number_of_nodes(), "Node count mismatch"
            assert loaded_graph.number_of_edges() == original_graph.number_of_edges(), "Edge count mismatch"
            
            # Verify specific nodes exist
            assert 'TP53' in loaded_graph.nodes(), "TP53 node missing"
            assert 'GO:0008150' in loaded_graph.nodes(), "GO term missing"
            
            # Verify node attributes
            tp53_data = loaded_graph.nodes['TP53']
            assert tp53_data.get('node_type') == 'gene', "TP53 node_type incorrect"
            assert tp53_data.get('gene_symbol') == 'TP53', "TP53 gene_symbol incorrect"
            
            return f"Successfully loaded pickle format. Verified {loaded_graph.number_of_nodes()} nodes, {loaded_graph.number_of_edges()} edges"
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_load_graph_from_file_graphml(self):
        """Test loading graph from GraphML format."""
        original_graph = self.create_test_graph()
        
        with tempfile.NamedTemporaryFile(suffix='.graphml', delete=False) as tmp_file:
            filepath = tmp_file.name
        
        try:
            # Save graph first
            nx.write_graphml(original_graph, filepath)
            
            # Test load functionality
            loaded_graph = load_graph_from_file(filepath)
            
            # Verify graph structure
            assert loaded_graph.number_of_nodes() == original_graph.number_of_nodes(), "Node count mismatch"
            assert loaded_graph.number_of_edges() == original_graph.number_of_edges(), "Edge count mismatch"
            
            # Verify specific nodes exist
            assert 'TP53' in loaded_graph.nodes(), "TP53 node missing"
            assert 'GO:0008150' in loaded_graph.nodes(), "GO term missing"
            
            return f"Successfully loaded GraphML format. Verified {loaded_graph.number_of_nodes()} nodes, {loaded_graph.number_of_edges()} edges"
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_initialize_graph_attributes(self):
        """Test initialization of graph attributes."""
        class MockObject:
            pass
        
        obj = MockObject()
        
        # Test initialization
        initialize_graph_attributes(obj)
        
        # Verify attributes were set
        assert hasattr(obj, 'gene_id_mappings'), "gene_id_mappings not set"
        assert hasattr(obj, 'go_terms'), "go_terms not set"
        assert hasattr(obj, 'go_alt_ids'), "go_alt_ids not set"
        
        assert obj.gene_id_mappings == {}, "gene_id_mappings not empty dict"
        assert obj.go_terms == {}, "go_terms not empty dict"
        assert obj.go_alt_ids == {}, "go_alt_ids not empty dict"
        
        # Test that existing attributes are not overwritten
        obj.gene_id_mappings = {'existing': 'data'}
        initialize_graph_attributes(obj)
        assert obj.gene_id_mappings == {'existing': 'data'}, "Existing data was overwritten"
        
        return "Successfully initialized and preserved graph attributes"
    
    def test_validate_basic_graph_structure(self):
        """Test basic graph structure validation."""
        graph = self.create_test_graph()
        
        # Test valid graph
        validation = validate_basic_graph_structure(graph)
        assert validation['has_nodes'] == True, "Should detect nodes"
        assert validation['has_edges'] == True, "Should detect edges"
        assert validation['is_multigraph'] == True, "Should detect MultiDiGraph"
        
        # Test empty graph
        empty_graph = nx.MultiDiGraph()
        empty_validation = validate_basic_graph_structure(empty_graph)
        assert empty_validation['has_nodes'] == False, "Should detect no nodes"
        assert empty_validation['has_edges'] == False, "Should detect no edges"
        assert empty_validation['is_multigraph'] == True, "Should still be MultiDiGraph"
        
        # Test non-multigraph
        simple_graph = nx.DiGraph()
        simple_graph.add_node('test')
        simple_validation = validate_basic_graph_structure(simple_graph)
        assert simple_validation['is_multigraph'] == False, "Should detect non-MultiDiGraph"
        
        return f"Validated graph structure: {validation}"
    
    def test_count_nodes_by_type(self):
        """Test counting nodes by type."""
        graph = self.create_test_graph()
        
        node_counts = count_nodes_by_type(graph)
        
        # Verify expected counts
        assert node_counts.get('go_term', 0) == 2, f"Expected 2 go_terms, got {node_counts.get('go_term', 0)}"
        assert node_counts.get('gene', 0) == 2, f"Expected 2 genes, got {node_counts.get('gene', 0)}"
        assert node_counts.get('disease', 0) == 1, f"Expected 1 disease, got {node_counts.get('disease', 0)}"
        assert node_counts.get('gene_identifier', 0) == 1, f"Expected 1 gene_identifier, got {node_counts.get('gene_identifier', 0)}"
        
        # Test empty graph
        empty_graph = nx.MultiDiGraph()
        empty_counts = count_nodes_by_type(empty_graph)
        assert empty_counts == {}, "Empty graph should return empty dict"
        
        return f"Node counts: {node_counts}"
    
    def test_count_edges_by_type(self):
        """Test counting edges by type."""
        graph = self.create_test_graph()
        
        edge_counts = count_edges_by_type(graph)
        
        # Verify expected counts
        assert edge_counts.get('gene_annotation', 0) == 2, f"Expected 2 gene_annotations, got {edge_counts.get('gene_annotation', 0)}"
        assert edge_counts.get('gene_disease_association', 0) == 1, f"Expected 1 gene_disease_association, got {edge_counts.get('gene_disease_association', 0)}"
        assert edge_counts.get('go_hierarchy', 0) == 1, f"Expected 1 go_hierarchy, got {edge_counts.get('go_hierarchy', 0)}"
        
        # Test empty graph
        empty_graph = nx.MultiDiGraph()
        empty_counts = count_edges_by_type(empty_graph)
        assert empty_counts == {}, "Empty graph should return empty dict"
        
        return f"Edge counts: {edge_counts}"
    
    def test_get_basic_graph_stats(self):
        """Test getting basic graph statistics."""
        graph = self.create_test_graph()
        
        stats = get_basic_graph_stats(graph)
        
        # Verify structure
        assert 'total_nodes' in stats, "total_nodes missing"
        assert 'total_edges' in stats, "total_edges missing"
        assert 'node_counts' in stats, "node_counts missing"
        assert 'edge_counts' in stats, "edge_counts missing"
        
        # Verify values
        assert stats['total_nodes'] == 6, f"Expected 6 total nodes, got {stats['total_nodes']}"
        assert stats['total_edges'] == 4, f"Expected 4 total edges, got {stats['total_edges']}"
        
        # Verify detailed counts
        assert stats['node_counts']['go_term'] == 2, "GO term count incorrect"
        assert stats['node_counts']['gene'] == 2, "Gene count incorrect"
        assert stats['edge_counts']['gene_annotation'] == 2, "Gene annotation count incorrect"
        
        return f"Basic stats: {stats}"
    
    def run_all_tests(self):
        """Run all shared utilities tests."""
        logger.info("=" * 80)
        logger.info("🧪 SHARED UTILITIES COMPREHENSIVE TEST SUITE")
        logger.info("=" * 80)
        
        # Define all tests
        tests = [
            ("Save Graph to Pickle", self.test_save_graph_to_file_pkl),
            ("Save Graph to GraphML", self.test_save_graph_to_file_graphml),
            ("Load Graph from Pickle", self.test_load_graph_from_file_pkl),
            ("Load Graph from GraphML", self.test_load_graph_from_file_graphml),
            ("Initialize Graph Attributes", self.test_initialize_graph_attributes),
            ("Validate Basic Graph Structure", self.test_validate_basic_graph_structure),
            ("Count Nodes by Type", self.test_count_nodes_by_type),
            ("Count Edges by Type", self.test_count_edges_by_type),
            ("Get Basic Graph Stats", self.test_get_basic_graph_stats)
        ]
        
        # Run all tests
        for test_name, test_function in tests:
            self.run_test(test_name, test_function)
        
        # Generate summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 SHARED UTILITIES TEST RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Tests: {self.passed_tests + self.failed_tests}")
        logger.info(f"Passed: {self.passed_tests}")
        logger.info(f"Failed: {self.failed_tests}")
        logger.info(f"Success Rate: {(self.passed_tests / (self.passed_tests + self.failed_tests)) * 100:.1f}%")
        
        if self.failed_tests > 0:
            logger.info("\n❌ FAILED TESTS:")
            for test_name, result in self.test_results.items():
                if result['status'] in ['FAILED', 'ERROR']:
                    logger.info(f"   {test_name}: {result['details']}")
        
        logger.info("=" * 80)
        
        return self.failed_tests == 0


def main():
    """Main execution function."""
    test_suite = SharedUtilsTestSuite()
    success = test_suite.run_all_tests()
    
    # Save results
    results_file = '/home/mreddy1/knowledge_graph/kg_testing/shared_utils_test_results.json'
    import json
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': str(Path(__file__).stat().st_mtime),
            'total_tests': test_suite.passed_tests + test_suite.failed_tests,
            'passed': test_suite.passed_tests,
            'failed': test_suite.failed_tests,
            'success_rate': (test_suite.passed_tests / (test_suite.passed_tests + test_suite.failed_tests)) * 100,
            'results': test_suite.test_results
        }, f, indent=2)
    
    logger.info(f"📄 Detailed results saved to: {results_file}")
    
    if success:
        print("\n🎉 All shared utilities tests passed!")
        return True
    else:
        print("\n⚠️ Some shared utilities tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)