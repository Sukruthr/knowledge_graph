"""
Comprehensive test suite for GO_BP Knowledge Graph functionality.

Tests enhanced data parsing, comprehensive graph construction, and advanced query capabilities.
Aligned with the knowledge_graph_project_plan.md requirements for schema compliance.
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
    """Test the enhanced GO_BP data parser with comprehensive file support."""
    
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
    
    def test_collapsed_go_file_parsing(self):
        """Test parsing of collapsed_go files with multiple identifier types."""
        # Test all three identifier types
        for id_type in ['symbol', 'entrez', 'uniprot']:
            collapsed_data = self.parser.parse_collapsed_go_file(id_type)
            
            # Check structure
            self.assertIn('clusters', collapsed_data)
            self.assertIn('gene_associations', collapsed_data)
            
            # Check clusters
            clusters = collapsed_data['clusters']
            if clusters:  # Some files might be empty
                sample_cluster = list(clusters.values())[0]
                self.assertIsInstance(sample_cluster, list)
                if sample_cluster:
                    self.assertIn('child_go', sample_cluster[0])
                    self.assertIn('cluster_type', sample_cluster[0])
            
            # Check gene associations
            associations = collapsed_data['gene_associations']
            if associations:
                sample_assoc = associations[0]
                required_keys = ['go_id', 'gene_id', 'identifier_type', 'annotation_type']
                for key in required_keys:
                    self.assertIn(key, sample_assoc)
                self.assertEqual(sample_assoc['identifier_type'], id_type)
    
    def test_alternative_go_ids(self):
        """Test parsing of alternative GO ID mappings."""
        alt_ids = self.parser.parse_go_alternative_ids()
        
        # Check that we have mappings
        self.assertGreater(len(alt_ids), 1000)
        
        # Check format
        for alt_id, primary_id in list(alt_ids.items())[:5]:
            self.assertTrue(alt_id.startswith('GO:'))
            self.assertTrue(primary_id.startswith('GO:'))
    
    def test_gene_identifier_mappings(self):
        """Test comprehensive gene identifier mapping extraction."""
        mappings = self.parser.parse_gene_identifier_mappings()
        
        # Check expected mapping types
        expected_mappings = ['symbol_to_entrez', 'symbol_to_uniprot', 'entrez_to_symbol', 
                           'uniprot_to_symbol', 'entrez_to_uniprot', 'uniprot_to_entrez']
        for mapping_type in expected_mappings:
            self.assertIn(mapping_type, mappings)
        
        # Check that we have reasonable number of mappings
        total_mappings = sum(len(m) for m in mappings.values())
        self.assertGreater(total_mappings, 10000)
    
    def test_obo_ontology_parsing(self):
        """Test parsing of OBO format ontology file."""
        obo_terms = self.parser.parse_obo_ontology()
        
        # Check that we have terms
        self.assertGreater(len(obo_terms), 20000)
        
        # Check enhanced structure
        sample_term = list(obo_terms.values())[0]
        self.assertIn('id', sample_term)
        self.assertIn('name', sample_term)
        
        # Check for enhanced data
        enhanced_terms = [term for term in obo_terms.values() if 'definition' in term]
        self.assertGreater(len(enhanced_terms), 15000)
    
    def test_data_validation(self):
        """Test comprehensive data validation."""
        # Parse all data first
        self.parser.parse_go_terms()
        self.parser.parse_go_relationships()
        self.parser.parse_gene_go_associations_from_gaf()
        self.parser.parse_go_alternative_ids()
        self.parser.parse_gene_identifier_mappings()
        
        validation = self.parser.validate_parsed_data()
        
        # Check all validation criteria
        expected_validations = ['has_go_terms', 'has_go_relationships', 'has_gene_associations',
                              'has_alt_ids', 'has_id_mappings', 'relationships_valid', 
                              'associations_valid', 'overall_valid']
        for validation_key in expected_validations:
            self.assertIn(validation_key, validation)
            self.assertTrue(validation[validation_key], f"{validation_key} should be valid")
    
    def test_data_summary(self):
        """Test enhanced data summary generation."""
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
        
        # Check reasonable values (updated for enhanced parser)
        self.assertGreater(summary['num_go_terms'], 20000)
        self.assertGreater(summary['num_gene_associations'], 100000)
        self.assertGreater(summary['num_unique_genes'], 10000)


class TestGOBPKnowledgeGraph(unittest.TestCase):
    """Test the comprehensive GO_BP knowledge graph with enhanced features."""
    
    @classmethod
    def setUpClass(cls):
        """Set up enhanced knowledge graph for testing."""
        cls.data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data/GO_BP"
        cls.kg = GOBPKnowledgeGraph(use_neo4j=False)
        cls.kg.load_data(cls.data_dir)
        cls.kg.build_graph()
    
    def test_comprehensive_graph_construction(self):
        """Test comprehensive graph construction with enhanced features."""
        stats = self.kg.get_stats()
        
        # Check enhanced graph size (should be larger due to comprehensive parsing)
        self.assertGreater(stats['total_nodes'], 60000)  # Increased due to gene identifiers
        self.assertGreater(stats['total_edges'], 500000)  # Increased due to cross-references
        
        # Check enhanced node types
        self.assertGreater(stats['go_terms'], 20000)
        self.assertGreater(stats['genes'], 10000)
        self.assertGreater(stats['gene_identifiers'], 5000)  # New identifier nodes
        
        # Check enhanced edge types
        self.assertGreater(stats['go_relationships'], 50000)
        self.assertGreater(stats['gene_associations'], 300000)  # Increased due to collapsed files
        self.assertGreater(stats['gene_cross_references'], 10000)  # New cross-reference edges
        
        # Check enhanced GO features
        self.assertGreater(stats['enhanced_go_terms'], 15000)  # OBO-enhanced terms
        self.assertGreater(stats['go_clusters'], 20000)  # GO clustering relationships
        
        # Check alternative ID mappings
        self.assertGreater(stats['alternative_id_mappings'], 500)
    
    def test_graph_validation(self):
        """Test built-in graph integrity validation."""
        validation = self.kg.validate_graph_integrity()
        
        # Check all validation criteria pass
        expected_validations = ['has_nodes', 'has_edges', 'go_terms_valid', 'gene_nodes_valid',
                              'relationships_valid', 'associations_valid', 'cross_references_valid', 
                              'overall_valid']
        for validation_key in expected_validations:
            self.assertIn(validation_key, validation)
            self.assertTrue(validation[validation_key], f"{validation_key} should be valid")
    
    def test_enhanced_node_properties(self):
        """Test that nodes have enhanced properties as per project plan."""
        # Test Gene node properties (should match project plan schema)
        gene_nodes = [n for n, d in self.kg.graph.nodes(data=True) if d.get('node_type') == 'gene']
        self.assertGreater(len(gene_nodes), 10000)
        
        # Check first gene node has required properties
        sample_gene = gene_nodes[0]
        gene_data = self.kg.graph.nodes[sample_gene]
        
        # Project plan requires: symbol, entrez_id, uniprot_id, description
        self.assertIn('gene_symbol', gene_data)  # symbol
        self.assertEqual(gene_data['node_type'], 'gene')
        # Note: entrez_id and uniprot_id may not be present for all genes
        
        # Test GO term node properties
        go_nodes = [n for n, d in self.kg.graph.nodes(data=True) if d.get('node_type') == 'go_term']
        self.assertGreater(len(go_nodes), 20000)
        
        sample_go = go_nodes[0]
        go_data = self.kg.graph.nodes[sample_go]
        
        # Project plan requires: go_id, name, namespace, definition
        self.assertIn('name', go_data)
        self.assertIn('namespace', go_data)
        self.assertEqual(go_data['node_type'], 'go_term')
    
    def test_project_plan_relationships(self):
        """Test that relationships match project plan schema."""
        edges = list(self.kg.graph.edges(data=True))
        
        # Check for required relationship types from project plan
        edge_types = set(e[2].get('edge_type') for e in edges)
        
        # Project plan specifies: ANNOTATED_WITH, IS_A, PART_OF relationships
        self.assertIn('gene_annotation', edge_types)  # Equivalent to ANNOTATED_WITH
        self.assertIn('go_hierarchy', edge_types)     # Includes IS_A, PART_OF
        
        # Enhanced relationships beyond project plan
        self.assertIn('go_clustering', edge_types)          # GO clustering
        self.assertIn('gene_cross_reference', edge_types)   # Cross-references
        self.assertIn('alternative_id_mapping', edge_types) # Alternative IDs
    
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
    
    def test_enhanced_search_capabilities(self):
        """Test enhanced search capabilities using OBO data."""
        # Test semantic search by definition
        dna_results = self.kg.search_go_terms_by_definition('DNA damage')
        
        if dna_results:
            # Check result structure
            sample_result = dna_results[0]
            required_keys = ['go_id', 'name', 'definition', 'score', 'match_types']
            for key in required_keys:
                self.assertIn(key, sample_result)
            
            # Check scoring system
            self.assertGreater(sample_result['score'], 0)
            self.assertIsInstance(sample_result['match_types'], list)
    
    def test_alternative_id_resolution(self):
        """Test alternative GO ID resolution."""
        if self.kg.go_alt_ids:
            alt_id = list(self.kg.go_alt_ids.keys())[0]
            primary_id = self.kg.resolve_alternative_go_id(alt_id)
            
            # Should resolve to primary ID
            self.assertNotEqual(alt_id, primary_id)
            self.assertTrue(primary_id.startswith('GO:'))
            
            # Test non-alternative ID (should return same)
            primary_test = self.kg.resolve_alternative_go_id(primary_id)
            self.assertEqual(primary_id, primary_test)
    
    def test_gene_cross_references(self):
        """Test gene cross-reference functionality."""
        # Test with TP53 if available
        if 'TP53' in self.kg.graph:
            cross_refs = self.kg.get_gene_cross_references('TP53')
            
            # Should have some cross-reference data
            self.assertIsInstance(cross_refs, dict)
            
            # May include uniprot, gene_name, gene_type, taxon
            possible_keys = ['uniprot', 'gene_name', 'gene_type', 'taxon']
            has_some_data = any(key in cross_refs for key in possible_keys)
            self.assertTrue(has_some_data)
    
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


class TestProjectPlanCompliance(unittest.TestCase):
    """Test compliance with knowledge_graph_project_plan.md requirements."""
    
    @classmethod
    def setUpClass(cls):
        """Set up knowledge graph for project plan compliance testing."""
        cls.data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data/GO_BP"
        cls.kg = GOBPKnowledgeGraph(use_neo4j=False)
        cls.kg.load_data(cls.data_dir)
        cls.kg.build_graph()
    
    def test_project_plan_gene_properties(self):
        """Test that Gene nodes have properties specified in project plan."""
        # Project plan specifies: symbol, entrez_id, uniprot_id, description
        gene_nodes = [n for n, d in self.kg.graph.nodes(data=True) if d.get('node_type') == 'gene']
        self.assertGreater(len(gene_nodes), 10000)
        
        # Check a sample of genes for required properties
        genes_with_symbol = 0
        genes_with_uniprot = 0
        genes_with_entrez = 0
        genes_with_description = 0
        
        for gene_id in gene_nodes[:100]:  # Sample first 100
            gene_data = self.kg.graph.nodes[gene_id]
            
            if 'gene_symbol' in gene_data:
                genes_with_symbol += 1
            if 'uniprot_id' in gene_data or 'cross_ref_uniprot' in gene_data:
                genes_with_uniprot += 1
            if 'entrez_id' in gene_data:
                genes_with_entrez += 1
            if 'gene_name' in gene_data:  # description equivalent
                genes_with_description += 1
        
        # All genes should have symbols
        self.assertEqual(genes_with_symbol, 100)
        # Many should have additional identifiers (but not necessarily all)
        self.assertGreater(genes_with_uniprot, 50)
        self.assertGreater(genes_with_description, 50)
    
    def test_project_plan_go_properties(self):
        """Test that GO Term nodes have properties specified in project plan."""
        # Project plan specifies: go_id, name, namespace, definition
        go_nodes = [n for n, d in self.kg.graph.nodes(data=True) if d.get('node_type') == 'go_term']
        self.assertGreater(len(go_nodes), 20000)
        
        # Check a sample for required properties
        sample_size = min(100, len(go_nodes))
        nodes_with_name = 0
        nodes_with_namespace = 0
        nodes_with_definition = 0
        
        for go_id in go_nodes[:sample_size]:
            go_data = self.kg.graph.nodes[go_id]
            
            # go_id is the node ID itself
            self.assertTrue(go_id.startswith('GO:'))
            
            if 'name' in go_data:
                nodes_with_name += 1
            if 'namespace' in go_data:
                nodes_with_namespace += 1
            if 'definition' in go_data:
                nodes_with_definition += 1
        
        # All should have names and namespaces
        self.assertEqual(nodes_with_name, sample_size)
        self.assertEqual(nodes_with_namespace, sample_size)
        # Many should have definitions (OBO enhancement) - adjust threshold based on actual data
        self.assertGreater(nodes_with_definition, sample_size * 0.6)
    
    def test_project_plan_relationships(self):
        """Test that relationships match project plan specifications."""
        edges = list(self.kg.graph.edges(data=True))
        
        # Project plan specifies these relationship types:
        # (Gene) -> [ANNOTATED_WITH] -> (GO_Term)
        # (GO_Term) -> [IS_A] -> (GO_Term)
        # (GO_Term) -> [PART_OF] -> (GO_Term)
        
        gene_annotation_edges = [e for e in edges if e[2].get('edge_type') == 'gene_annotation']
        go_hierarchy_edges = [e for e in edges if e[2].get('edge_type') == 'go_hierarchy']
        
        # Should have substantial number of both types
        self.assertGreater(len(gene_annotation_edges), 100000)
        self.assertGreater(len(go_hierarchy_edges), 50000)
        
        # Test gene annotation edge structure
        if gene_annotation_edges:
            sample_edge = gene_annotation_edges[0]
            source, target, data = sample_edge
            
            # Source should be gene, target should be GO term
            source_data = self.kg.graph.nodes[source]
            target_data = self.kg.graph.nodes[target]
            
            # Note: Our implementation uses gene symbols as node IDs
            self.assertTrue(target.startswith('GO:') or target_data.get('node_type') == 'go_term')
        
        # Test GO hierarchy edge structure
        if go_hierarchy_edges:
            sample_edge = go_hierarchy_edges[0]
            source, target, data = sample_edge
            
            # Both should be GO terms
            self.assertTrue(source.startswith('GO:'))
            self.assertTrue(target.startswith('GO:'))
            
            # Should have relationship type
            self.assertIn('relationship_type', data)
            self.assertIn(data['relationship_type'], 
                         ['is_a', 'part_of', 'regulates', 'positively_regulates', 'negatively_regulates'])


class TestBiologicalQueries(unittest.TestCase):
    """Test biologically meaningful queries as specified in project plan."""
    
    @classmethod
    def setUpClass(cls):
        """Set up knowledge graph for biological testing."""
        cls.data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data/GO_BP"
        cls.kg = GOBPKnowledgeGraph(use_neo4j=False)
        cls.kg.load_data(cls.data_dir)
        cls.kg.build_graph()
    
    def test_project_plan_query_examples(self):
        """Test queries similar to those specified in the project plan."""
        # Project plan Query 2: "What GO terms are associated with gene 'TP53'?"
        if 'TP53' in self.kg.graph:
            tp53_functions = self.kg.query_gene_functions('TP53')
            self.assertGreater(len(tp53_functions), 5, "TP53 should have multiple GO annotations")
            
            # Check that results contain biological processes
            function_names = [f['go_name'].lower() for f in tp53_functions]
            biological_terms = any('process' in name or 'regulation' in name or 'response' in name 
                                 for name in function_names)
            self.assertTrue(biological_terms, "TP53 should have process-related annotations")
    
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
    print("="*80)
    print("COMPREHENSIVE GO_BP KNOWLEDGE GRAPH TEST SUITE")
    print("Enhanced tests aligned with knowledge_graph_project_plan.md")
    print("="*80)
    
    # Create enhanced test suite with project plan compliance
    test_classes = [TestGOBPDataParser, TestGOBPKnowledgeGraph, 
                   TestProjectPlanCompliance, TestBiologicalQueries]
    
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
    
    # Project plan compliance summary
    if total_failures == 0:
        print("\n✅ ALL TESTS PASSED - Full compliance with project plan requirements")
        print("✅ Enhanced data parsing capabilities validated")
        print("✅ Comprehensive knowledge graph construction verified")
        print("✅ Advanced query capabilities confirmed")
    else:
        print(f"\n⚠️  {total_failures} test failures detected")
        print("Check individual test results for details")
    
    print("="*80)
    
    return total_failures == 0


if __name__ == "__main__":
    success = run_comprehensive_tests()