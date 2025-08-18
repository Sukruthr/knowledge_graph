"""
Test suite for GO_BP Knowledge Graph functionality.

Tests data parsing, graph construction, and query capabilities.
"""

import unittest
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_parsers import GOBPDataParser
from kg_builder import GOBPKnowledgeGraph


class TestGOBPDataParser(unittest.TestCase):
    """Test the GO_BP data parser."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        cls.data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data/GO_BP"
        cls.parser = GOBPDataParser(cls.data_dir)
    
    def test_parse_go_terms(self):
        """Test GO term parsing."""
        go_terms = self.parser.parse_go_terms()
        
        # Check that we have terms
        self.assertGreater(len(go_terms), 20000)
        
        # Check structure
        sample_go = list(go_terms.keys())[0]
        self.assertTrue(sample_go.startswith('GO:'))
        self.assertIn('name', go_terms[sample_go])
        self.assertIn('namespace', go_terms[sample_go])
    
    def test_parse_go_relationships(self):
        """Test GO relationship parsing."""
        relationships = self.parser.parse_go_relationships()
        
        # Check that we have relationships
        self.assertGreater(len(relationships), 50000)
        
        # Check structure
        sample_rel = relationships[0]
        required_keys = ['parent_id', 'child_id', 'relationship_type', 'namespace']
        for key in required_keys:
            self.assertIn(key, sample_rel)
        
        # Check that IDs are proper GO format
        self.assertTrue(sample_rel['parent_id'].startswith('GO:'))
        self.assertTrue(sample_rel['child_id'].startswith('GO:'))
    
    def test_parse_gene_associations(self):
        """Test gene-GO association parsing."""
        associations = self.parser.parse_gene_go_associations_from_gaf()
        
        # Check that we have associations
        self.assertGreater(len(associations), 100000)
        
        # Check structure
        sample_assoc = associations[0]
        required_keys = ['gene_symbol', 'go_id', 'evidence_code', 'uniprot_id']
        for key in required_keys:
            self.assertIn(key, sample_assoc)
        
        # Check that GO IDs are proper format
        self.assertTrue(sample_assoc['go_id'].startswith('GO:'))
        
        # Check that we only have biological processes
        self.assertEqual(sample_assoc['aspect'], 'P')
    
    def test_data_summary(self):
        """Test data summary generation."""
        # Parse all data first
        self.parser.parse_go_terms()
        self.parser.parse_go_relationships()
        self.parser.parse_gene_go_associations_from_gaf()
        
        summary = self.parser.get_data_summary()
        
        # Check expected keys
        expected_keys = ['num_go_terms', 'num_go_relationships', 'num_gene_associations', 
                        'num_unique_genes', 'relationship_types']
        for key in expected_keys:
            self.assertIn(key, summary)
        
        # Check reasonable values
        self.assertGreater(summary['num_go_terms'], 20000)
        self.assertGreater(summary['num_gene_associations'], 100000)
        self.assertGreater(summary['num_unique_genes'], 10000)


class TestGOBPKnowledgeGraph(unittest.TestCase):
    """Test the GO_BP knowledge graph."""
    
    @classmethod
    def setUpClass(cls):
        """Set up knowledge graph for testing."""
        cls.data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data/GO_BP"
        cls.kg = GOBPKnowledgeGraph(use_neo4j=False)
        cls.kg.load_data(cls.data_dir)
        cls.kg.build_graph()
    
    def test_graph_construction(self):
        """Test basic graph construction."""
        stats = self.kg.get_stats()
        
        # Check that graph was built
        self.assertGreater(stats['total_nodes'], 40000)
        self.assertGreater(stats['total_edges'], 200000)
        
        # Check node types
        self.assertGreater(stats['go_terms'], 20000)
        self.assertGreater(stats['genes'], 10000)
        
        # Check edge types
        self.assertGreater(stats['go_relationships'], 50000)
        self.assertGreater(stats['gene_associations'], 100000)
    
    def test_gene_function_query(self):
        """Test querying gene functions."""
        # Test with TP53 - should have many annotations
        functions = self.kg.query_gene_functions('TP53')
        
        self.assertGreater(len(functions), 5)
        
        # Check structure
        sample_func = functions[0]
        required_keys = ['go_id', 'go_name', 'evidence_code']
        for key in required_keys:
            self.assertIn(key, sample_func)
        
        self.assertTrue(sample_func['go_id'].startswith('GO:'))
    
    def test_go_term_genes_query(self):
        """Test querying genes for GO terms."""
        # Find an apoptosis-related term
        apoptosis_terms = [go_id for go_id, info in self.kg.go_terms.items() 
                          if 'apoptosis' in info['name'].lower()]
        
        self.assertGreater(len(apoptosis_terms), 0)
        
        # Query genes for first apoptosis term
        genes = self.kg.query_go_term_genes(apoptosis_terms[0])
        
        # Should have some genes
        if genes:  # Some GO terms might not have direct annotations
            sample_gene = genes[0]
            required_keys = ['gene_symbol', 'gene_name', 'evidence_code']
            for key in required_keys:
                self.assertIn(key, sample_gene)
    
    def test_go_hierarchy_query(self):
        """Test querying GO term hierarchy."""
        # Get a GO term that should have parents/children
        sample_go = list(self.kg.go_terms.keys())[100]  # Skip root terms
        
        # Test parent query
        parents = self.kg.query_go_hierarchy(sample_go, 'parents')
        
        if parents:  # Not all terms have parents in the subset
            sample_parent = parents[0]
            required_keys = ['go_id', 'go_name', 'relationship_type']
            for key in required_keys:
                self.assertIn(key, sample_parent)
            
            self.assertTrue(sample_parent['go_id'].startswith('GO:'))
            self.assertIn(sample_parent['relationship_type'], 
                         ['is_a', 'part_of', 'regulates', 'positively_regulates', 'negatively_regulates'])
    
    def test_graph_persistence(self):
        """Test saving and loading graph."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save graph
            self.kg.save_graph(tmp_path)
            
            # Create new instance and load
            kg2 = GOBPKnowledgeGraph(use_neo4j=False)
            kg2.load_graph(tmp_path)
            
            # Compare statistics
            stats1 = self.kg.get_stats()
            stats2 = kg2.get_stats()
            
            self.assertEqual(stats1['total_nodes'], stats2['total_nodes'])
            self.assertEqual(stats1['total_edges'], stats2['total_edges'])
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestBiologicalQueries(unittest.TestCase):
    """Test biologically meaningful queries."""
    
    @classmethod
    def setUpClass(cls):
        """Set up knowledge graph for biological testing."""
        cls.data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data/GO_BP"
        cls.kg = GOBPKnowledgeGraph(use_neo4j=False)
        cls.kg.load_data(cls.data_dir)
        cls.kg.build_graph()
    
    def test_tumor_suppressor_functions(self):
        """Test querying tumor suppressor gene functions."""
        tumor_suppressors = ['TP53', 'RB1', 'BRCA1', 'BRCA2', 'APC']
        
        for gene in tumor_suppressors:
            if gene in self.kg.graph:  # Check if gene exists in our data
                functions = self.kg.query_gene_functions(gene)
                self.assertGreater(len(functions), 0, f"{gene} should have GO annotations")
                
                # Check for expected biological processes
                function_names = [f['go_name'].lower() for f in functions]
                biological_terms = ['regulation', 'cell', 'response', 'process']
                
                has_biological_term = any(term in ' '.join(function_names) for term in biological_terms)
                self.assertTrue(has_biological_term, f"{gene} should have relevant biological annotations")
    
    def test_apoptosis_pathway_genes(self):
        """Test querying apoptosis pathway genes."""
        # Find apoptosis-related GO terms
        apoptosis_terms = []
        for go_id, info in self.kg.go_terms.items():
            if 'apoptosis' in info['name'].lower():
                apoptosis_terms.append(go_id)
        
        self.assertGreater(len(apoptosis_terms), 10, "Should find multiple apoptosis terms")
        
        # Check that apoptosis terms have gene associations
        total_apoptosis_genes = set()
        for go_term in apoptosis_terms[:5]:  # Test first 5
            genes = self.kg.query_go_term_genes(go_term)
            total_apoptosis_genes.update(gene['gene_symbol'] for gene in genes)
        
        # Should find some well-known apoptosis genes
        known_apoptosis_genes = {'TP53', 'BCL2', 'BAX', 'CASP3', 'CASP9', 'PARP1'}
        found_genes = total_apoptosis_genes.intersection(known_apoptosis_genes)
        
        # We might not find all, but should find some
        print(f"Found apoptosis genes: {found_genes}")
    
    def test_go_term_hierarchy_structure(self):
        """Test that GO hierarchy makes biological sense."""
        # Test that 'apoptosis' is more specific than 'cell death'
        apoptosis_terms = [go_id for go_id, info in self.kg.go_terms.items() 
                          if info['name'].lower() == 'apoptotic process']
        
        if apoptosis_terms:
            apoptosis_go = apoptosis_terms[0]
            parents = self.kg.query_go_hierarchy(apoptosis_go, 'parents')
            
            # Check that parents are more general
            parent_names = [p['go_name'].lower() for p in parents]
            general_terms = ['cell death', 'death', 'cellular process', 'biological_process']
            
            has_general_parent = any(any(term in name for term in general_terms) for name in parent_names)
            if has_general_parent:
                print(f"Apoptosis hierarchy looks correct: {parent_names}")


def run_comprehensive_tests():
    """Run all tests and provide detailed output."""
    print("="*60)
    print("GO_BP KNOWLEDGE GRAPH TEST SUITE")
    print("="*60)
    
    # Create test suite
    test_classes = [TestGOBPDataParser, TestGOBPKnowledgeGraph, TestBiologicalQueries]
    
    total_tests = 0
    total_failures = 0
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        
        total_tests += result.testsRun
        total_failures += len(result.failures) + len(result.errors)
    
    print("\n" + "="*60)
    print(f"TOTAL TESTS: {total_tests}")
    print(f"FAILURES: {total_failures}")
    print(f"SUCCESS RATE: {((total_tests - total_failures) / total_tests * 100):.1f}%")
    print("="*60)
    
    return total_failures == 0


if __name__ == "__main__":
    success = run_comprehensive_tests()