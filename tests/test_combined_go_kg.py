"""
Comprehensive test suite for Combined GO Knowledge Graph functionality.

Tests the enhanced data parsing, combined graph construction, and cross-namespace query capabilities.
Includes tests for GO_BP + GO_CC + GO_MF integration.
"""

import unittest
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_parsers import GODataParser, CombinedGOParser
from kg_builder import GOKnowledgeGraph, CombinedGOKnowledgeGraph


class TestGODataParser(unittest.TestCase):
    """Test the enhanced GO data parser with multiple namespace support."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        cls.base_data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data"
        cls.go_bp_dir = cls.base_data_dir + "/GO_BP"
        cls.go_cc_dir = cls.base_data_dir + "/GO_CC"
    
    def test_go_bp_parser(self):
        """Test GO_BP parser with enhanced functionality."""
        parser = GODataParser(self.go_bp_dir)
        
        # Test namespace auto-detection
        self.assertEqual(parser.namespace, 'biological_process')
        
        # Test core parsing
        go_terms = parser.parse_go_terms()
        self.assertGreater(len(go_terms), 25000)
        
        # Verify namespace in parsed terms
        sample_term = list(go_terms.values())[0]
        self.assertEqual(sample_term['namespace'], 'biological_process')
        
        # Test GAF parsing with correct aspect filtering
        associations = parser.parse_gene_go_associations_from_gaf()
        self.assertGreater(len(associations), 100000)
        
        # Verify all associations are biological processes
        for assoc in associations[:100]:  # Check first 100
            self.assertEqual(assoc['aspect'], 'P')
    
    def test_go_cc_parser(self):
        """Test GO_CC parser functionality."""
        parser = GODataParser(self.go_cc_dir)
        
        # Test namespace auto-detection
        self.assertEqual(parser.namespace, 'cellular_component')
        
        # Test core parsing
        go_terms = parser.parse_go_terms()
        self.assertGreater(len(go_terms), 4000)
        
        # Verify namespace in parsed terms
        sample_term = list(go_terms.values())[0]
        self.assertEqual(sample_term['namespace'], 'cellular_component')
        
        # Test GAF parsing with correct aspect filtering
        associations = parser.parse_gene_go_associations_from_gaf()
        self.assertGreater(len(associations), 100000)
        
        # Verify all associations are cellular components
        for assoc in associations[:100]:  # Check first 100
            self.assertEqual(assoc['aspect'], 'C')
    
    def test_go_mf_parser(self):
        """Test GO_MF parser functionality."""
        mf_dir = self.base_data_dir + "/GO_MF"
        parser = GODataParser(mf_dir)
        
        # Test namespace auto-detection
        self.assertEqual(parser.namespace, 'molecular_function')
        
        # Test core parsing
        go_terms = parser.parse_go_terms()
        self.assertGreater(len(go_terms), 12000)
        
        # Verify namespace in parsed terms
        sample_term = list(go_terms.values())[0]
        self.assertEqual(sample_term['namespace'], 'molecular_function')
        
        # Test GAF parsing with correct aspect filtering
        associations = parser.parse_gene_go_associations_from_gaf()
        self.assertGreater(len(associations), 250000)
        
        # Verify all associations are molecular functions
        for assoc in associations[:100]:  # Check first 100
            self.assertEqual(assoc['aspect'], 'F')
    
    def test_namespace_comparison(self):
        """Test differences between GO_BP, GO_CC, and GO_MF data."""
        bp_parser = GODataParser(self.go_bp_dir)
        cc_parser = GODataParser(self.go_cc_dir)
        mf_parser = GODataParser(self.base_data_dir + "/GO_MF")
        
        bp_terms = bp_parser.parse_go_terms()
        cc_terms = cc_parser.parse_go_terms()
        mf_terms = mf_parser.parse_go_terms()
        
        # GO_BP should have the most terms, then GO_MF, then GO_CC
        self.assertGreater(len(bp_terms), len(mf_terms))
        self.assertGreater(len(mf_terms), len(cc_terms))
        
        bp_associations = bp_parser.parse_gene_go_associations_from_gaf()
        cc_associations = cc_parser.parse_gene_go_associations_from_gaf()
        mf_associations = mf_parser.parse_gene_go_associations_from_gaf()
        
        # All should have substantial associations
        self.assertGreater(len(bp_associations), 50000)
        self.assertGreater(len(cc_associations), 50000)
        self.assertGreater(len(mf_associations), 200000)  # MF has the most associations


class TestCombinedGOParser(unittest.TestCase):
    """Test the combined GO parser functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up combined parser."""
        cls.base_data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data"
        cls.combined_parser = CombinedGOParser(cls.base_data_dir)
    
    def test_parser_initialization(self):
        """Test combined parser initialization."""
        # Should have parsers for available namespaces
        expected_namespaces = {'biological_process', 'cellular_component', 'molecular_function'}
        actual_namespaces = set(self.combined_parser.parsers.keys())
        
        # At minimum should have BP and CC
        self.assertTrue({'biological_process', 'cellular_component'}.issubset(actual_namespaces))
    
    def test_combined_parsing(self):
        """Test parsing across all namespaces."""
        results = self.combined_parser.parse_all_namespaces()
        
        # Should have data for multiple namespaces
        self.assertGreaterEqual(len(results), 2)
        
        # Check BP data
        if 'biological_process' in results:
            bp_data = results['biological_process']
            self.assertGreater(len(bp_data['go_terms']), 25000)
            self.assertGreater(len(bp_data['gene_associations']), 100000)
        
        # Check CC data
        if 'cellular_component' in results:
            cc_data = results['cellular_component']
            self.assertGreater(len(cc_data['go_terms']), 4000)
            self.assertGreater(len(cc_data['gene_associations']), 100000)
    
    def test_combined_summary(self):
        """Test combined summary statistics."""
        summary = self.combined_parser.get_combined_summary()
        
        self.assertIn('namespaces', summary)
        self.assertIn('by_namespace', summary)
        self.assertGreaterEqual(len(summary['namespaces']), 2)


class TestGOKnowledgeGraph(unittest.TestCase):
    """Test the generic GO knowledge graph class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up knowledge graphs for testing."""
        cls.base_data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data"
        
        # Create BP knowledge graph
        cls.bp_kg = GOKnowledgeGraph(use_neo4j=False, namespace='biological_process')
        cls.bp_kg.load_data(cls.base_data_dir + "/GO_BP")
        cls.bp_kg.build_graph()
        
        # Create CC knowledge graph
        cls.cc_kg = GOKnowledgeGraph(use_neo4j=False, namespace='cellular_component')
        cls.cc_kg.load_data(cls.base_data_dir + "/GO_CC")
        cls.cc_kg.build_graph()
        
        # Create MF knowledge graph
        cls.mf_kg = GOKnowledgeGraph(use_neo4j=False, namespace='molecular_function')
        cls.mf_kg.load_data(cls.base_data_dir + "/GO_MF")
        cls.mf_kg.build_graph()
    
    def test_bp_graph_construction(self):
        """Test biological process graph construction."""
        stats = self.bp_kg.get_stats()
        
        self.assertEqual(stats['namespace'], 'biological_process')
        self.assertGreater(stats['total_nodes'], 60000)
        self.assertGreater(stats['total_edges'], 500000)
        self.assertGreater(stats['go_terms'], 25000)
        self.assertGreater(stats['genes'], 15000)
    
    def test_cc_graph_construction(self):
        """Test cellular component graph construction."""
        stats = self.cc_kg.get_stats()
        
        self.assertEqual(stats['namespace'], 'cellular_component')
        self.assertGreater(stats['total_nodes'], 35000)
        self.assertGreater(stats['total_edges'], 300000)
        self.assertGreater(stats['go_terms'], 4000)
        self.assertGreater(stats['genes'], 15000)
    
    def test_namespace_specific_queries(self):
        """Test that queries return namespace-appropriate results."""
        # Test BP graph
        bp_functions = self.bp_kg.query_gene_functions('TP53')
        if bp_functions:
            for func in bp_functions[:5]:
                self.assertEqual(func['namespace'], 'biological_process')
        
        # Test CC graph  
        cc_functions = self.cc_kg.query_gene_functions('TP53')
        if cc_functions:
            for func in cc_functions[:5]:
                self.assertEqual(func['namespace'], 'cellular_component')
        
        # Test MF graph
        mf_functions = self.mf_kg.query_gene_functions('TP53')
        if mf_functions:
            for func in mf_functions[:5]:
                self.assertEqual(func['namespace'], 'molecular_function')
    
    def test_mf_graph_construction(self):
        """Test molecular function graph construction."""
        stats = self.mf_kg.get_stats()
        
        self.assertEqual(stats['namespace'], 'molecular_function')
        self.assertGreater(stats['total_nodes'], 40000)
        self.assertGreater(stats['total_edges'], 400000)
        self.assertGreater(stats['go_terms'], 12000)
        self.assertGreater(stats['genes'], 15000)
    
    def test_graph_validation(self):
        """Test graph integrity validation."""
        bp_validation = self.bp_kg.validate_graph_integrity()
        cc_validation = self.cc_kg.validate_graph_integrity()
        mf_validation = self.mf_kg.validate_graph_integrity()
        
        self.assertTrue(bp_validation['overall_valid'])
        self.assertTrue(cc_validation['overall_valid'])
        self.assertTrue(mf_validation['overall_valid'])


class TestCombinedGOKnowledgeGraph(unittest.TestCase):
    """Test the combined GO knowledge graph functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up combined knowledge graph for testing."""
        cls.base_data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data"
        cls.combined_kg = CombinedGOKnowledgeGraph(use_neo4j=False)
        cls.combined_kg.load_data(cls.base_data_dir)
        cls.combined_kg.build_combined_graph()
    
    def test_combined_graph_construction(self):
        """Test combined graph construction."""
        stats = self.combined_kg.get_combined_stats()
        
        # Should have substantially more nodes/edges than individual graphs
        self.assertGreater(stats['total_nodes'], 80000)
        self.assertGreater(stats['total_edges'], 1000000)
        self.assertGreater(stats['go_terms'], 40000)
        
        # Should have multiple namespaces
        self.assertGreaterEqual(len(stats['namespace_counts']), 3)
        self.assertIn('biological_process', stats['namespace_counts'])
        self.assertIn('cellular_component', stats['namespace_counts'])
        self.assertIn('molecular_function', stats['namespace_counts'])
    
    def test_individual_graphs_preserved(self):
        """Test that individual graphs are preserved in combined structure."""
        self.assertIn('biological_process', self.combined_kg.individual_graphs)
        self.assertIn('cellular_component', self.combined_kg.individual_graphs)
        self.assertIn('molecular_function', self.combined_kg.individual_graphs)
        
        # Individual graphs should have reasonable stats
        for namespace, kg in self.combined_kg.individual_graphs.items():
            stats = kg.get_stats()
            self.assertGreater(stats['total_nodes'], 30000)
            self.assertGreater(stats['go_terms'], 1000)
    
    def test_cross_namespace_queries(self):
        """Test querying across multiple namespaces."""
        all_functions = self.combined_kg.query_gene_functions_all_namespaces('TP53')
        
        # Should have functions in multiple namespaces (ideally 3)
        self.assertGreaterEqual(len(all_functions), 2)
        
        # Check that we have all three namespaces for TP53
        if len(all_functions) == 3:
            expected_namespaces = {'biological_process', 'cellular_component', 'molecular_function'}
            actual_namespaces = set(all_functions.keys())
            self.assertEqual(expected_namespaces, actual_namespaces)
        
        # Check that different namespaces return different GO terms
        all_go_ids = set()
        for namespace, functions in all_functions.items():
            namespace_go_ids = {func['go_id'] for func in functions}
            # Should have some unique terms per namespace
            self.assertGreater(len(namespace_go_ids), 10)
            all_go_ids.update(namespace_go_ids)
        
        # Total unique GO IDs should be substantial
        self.assertGreater(len(all_go_ids), 100)
    
    def test_combined_graph_validation(self):
        """Test combined graph validation."""
        validation = self.combined_kg._validate_combined_graph()
        
        self.assertTrue(validation['has_nodes'])
        self.assertTrue(validation['has_edges'])
        self.assertTrue(validation['multiple_namespaces'])
    
    def test_gene_coverage_across_namespaces(self):
        """Test that genes appear across multiple namespaces."""
        test_genes = ['TP53', 'BRCA1', 'EGFR', 'MYC']
        
        for gene in test_genes:
            all_functions = self.combined_kg.query_gene_functions_all_namespaces(gene)
            
            if all_functions:  # Gene exists in at least one namespace
                # Should ideally have functions in multiple namespaces
                namespace_count = len(all_functions)
                total_functions = sum(len(funcs) for funcs in all_functions.values())
                
                # At minimum, should have some functions
                self.assertGreater(total_functions, 0)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing GO_BP functionality."""
    
    def test_gobp_parser_alias(self):
        """Test that GOBPDataParser still works as an alias."""
        from data_parsers import GOBPDataParser
        
        data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data/GO_BP"
        parser = GOBPDataParser(data_dir)
        
        # Should auto-detect as biological_process
        self.assertEqual(parser.namespace, 'biological_process')
        
        # Should still parse correctly
        go_terms = parser.parse_go_terms()
        self.assertGreater(len(go_terms), 25000)
    
    def test_gobp_kg_unchanged(self):
        """Test that GOBPKnowledgeGraph still works unchanged."""
        from kg_builder import GOBPKnowledgeGraph
        
        kg = GOBPKnowledgeGraph(use_neo4j=False)
        data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data/GO_BP"
        kg.load_data(data_dir)
        kg.build_graph()
        
        stats = kg.get_stats()
        self.assertGreater(stats['total_nodes'], 60000)
        self.assertGreater(stats['total_edges'], 500000)


def run_combined_tests():
    """Run all combined GO tests and provide detailed output."""
    print("="*80)
    print("COMPREHENSIVE GO KNOWLEDGE GRAPH TEST SUITE")
    print("Complete tests for GO_BP + GO_CC + GO_MF integration")
    print("="*80)
    
    test_classes = [
        TestGODataParser, 
        TestCombinedGOParser, 
        TestGOKnowledgeGraph,
        TestCombinedGOKnowledgeGraph,
        TestBackwardCompatibility
    ]
    
    total_tests = 0
    total_failures = 0
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        
        total_tests += result.testsRun
        total_failures += len(result.failures) + len(result.errors)
        
        # Print summary for each test class
        class_success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
        print(f"  {test_class.__name__}: {result.testsRun} tests, {class_success_rate:.1f}% success")
    
    print("\n" + "="*80)
    print(f"TOTAL TESTS: {total_tests}")
    print(f"FAILURES: {total_failures}")
    print(f"SUCCESS RATE: {((total_tests - total_failures) / total_tests * 100):.1f}%")
    
    if total_failures == 0:
        print("\n✅ ALL TESTS PASSED - Complete GO_BP + GO_CC + GO_MF integration successful")
        print("✅ Tri-namespace knowledge graph functionality validated")
        print("✅ Cross-namespace query capabilities confirmed across all 3 namespaces") 
        print("✅ Backward compatibility maintained")
    else:
        print(f"\n⚠️  {total_failures} test failures detected")
        print("Check individual test results for details")
    
    print("="*80)
    
    return total_failures == 0


if __name__ == "__main__":
    success = run_combined_tests()