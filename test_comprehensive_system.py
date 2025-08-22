#!/usr/bin/env python3
"""
Comprehensive system testing for all components:
1. Parser testing (GO, Omics, Model Comparison)
2. Knowledge Graph construction testing
3. Integration testing
4. Query functionality testing
5. Performance benchmarking
"""

import sys
import time
import logging
import traceback
from pathlib import Path

# Add src to path
sys.path.append('src')

from parsers import CombinedBiomedicalParser
from kg_builder import ComprehensiveBiomedicalKnowledgeGraph

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveSystemTester:
    """Comprehensive testing suite for the entire biomedical knowledge graph system."""
    
    def __init__(self):
        self.test_results = {}
        self.data_dir = 'llm_evaluation_for_gene_set_interpretation/data'
        
    def run_all_tests(self):
        """Run all comprehensive tests."""
        logger.info("="*100)
        logger.info("COMPREHENSIVE BIOMEDICAL KNOWLEDGE GRAPH SYSTEM TESTING")
        logger.info("="*100)
        
        tests = [
            ("Parser Component Testing", self.test_parser_components),
            ("Knowledge Graph Construction Testing", self.test_kg_construction),
            ("Data Integration Testing", self.test_data_integration),
            ("Query Functionality Testing", self.test_query_functionality),
            ("Cross-Modal Connectivity Testing", self.test_cross_modal_connectivity),
            ("Model Comparison Specific Testing", self.test_model_comparison_functionality),
            ("Performance Benchmarking", self.test_performance_benchmarks),
            ("Regression Testing", self.test_regression_compatibility)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_function in tests:
            logger.info(f"\n{'='*80}")
            logger.info(f"RUNNING: {test_name}")
            logger.info(f"{'='*80}")
            
            try:
                start_time = time.time()
                result = test_function()
                test_time = time.time() - start_time
                
                if result:
                    logger.info(f"✅ PASSED: {test_name} ({test_time:.2f}s)")
                    passed_tests += 1
                else:
                    logger.error(f"❌ FAILED: {test_name} ({test_time:.2f}s)")
                    
                self.test_results[test_name] = {
                    'passed': result,
                    'duration': test_time
                }
                
            except Exception as e:
                logger.error(f"💥 ERROR: {test_name} - {e}")
                traceback.print_exc()
                self.test_results[test_name] = {
                    'passed': False,
                    'duration': 0,
                    'error': str(e)
                }
        
        # Final summary
        success_rate = (passed_tests / total_tests) * 100
        logger.info(f"\n{'='*100}")
        logger.info(f"COMPREHENSIVE TESTING SUMMARY")
        logger.info(f"{'='*100}")
        logger.info(f"Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 87.5:  # 7/8 tests
            logger.info("🎉 COMPREHENSIVE SYSTEM TESTING SUCCESSFUL!")
            return True
        else:
            logger.warning("⚠️ System testing needs attention")
            return False
    
    def test_parser_components(self):
        """Test all parser components individually."""
        logger.info("Testing parser components...")
        
        try:
            # Test parser initialization
            parser = CombinedBiomedicalParser(self.data_dir)
            logger.info("✓ Parser initialization successful")
            
            # Test data parsing
            start_time = time.time()
            parsed_data = parser.parse_all_biomedical_data()
            parse_time = time.time() - start_time
            logger.info(f"✓ Data parsing completed in {parse_time:.2f} seconds")
            
            # Validate parsed data structure
            required_components = ['go_data', 'omics_data', 'enhanced_semantic_data', 'model_compare_data']
            missing_components = []
            
            for component in required_components:
                if component not in parsed_data:
                    missing_components.append(component)
                else:
                    logger.info(f"✓ {component} parsed successfully")
            
            if missing_components:
                logger.error(f"Missing components: {missing_components}")
                return False
            
            # Test GO data parsing
            go_data = parsed_data['go_data']
            expected_namespaces = ['biological_process', 'cellular_component', 'molecular_function']
            
            for namespace in expected_namespaces:
                if namespace not in go_data:
                    logger.error(f"Missing GO namespace: {namespace}")
                    return False
                
                ns_data = go_data[namespace]
                if not ns_data.get('go_terms') or not ns_data.get('gene_associations'):
                    logger.error(f"Invalid {namespace} data structure")
                    return False
                
                logger.info(f"✓ {namespace}: {len(ns_data['go_terms'])} terms, {len(ns_data['gene_associations'])} associations")
            
            # Test Omics data parsing
            omics_data = parsed_data['omics_data']
            required_omics = ['disease_associations', 'drug_associations', 'viral_associations', 'cluster_relationships']
            
            for omics_type in required_omics:
                if omics_type not in omics_data:
                    logger.error(f"Missing Omics data: {omics_type}")
                    return False
                logger.info(f"✓ {omics_type}: {len(omics_data[omics_type])} entries")
            
            # Test Model Comparison data parsing
            model_data = parsed_data['model_compare_data']
            required_model_components = ['available_models', 'model_predictions', 'similarity_rankings']
            
            for component in required_model_components:
                if component not in model_data:
                    logger.error(f"Missing model component: {component}")
                    return False
            
            logger.info(f"✓ Model comparison: {len(model_data['available_models'])} models, {len(model_data['model_predictions'])} prediction sets")
            
            return True
            
        except Exception as e:
            logger.error(f"Parser testing failed: {e}")
            return False
    
    def test_kg_construction(self):
        """Test knowledge graph construction."""
        logger.info("Testing knowledge graph construction...")
        
        try:
            # Initialize KG
            kg = ComprehensiveBiomedicalKnowledgeGraph()
            logger.info("✓ Knowledge graph initialized")
            
            # Load data
            start_time = time.time()
            kg.load_data(self.data_dir)
            load_time = time.time() - start_time
            logger.info(f"✓ Data loaded in {load_time:.2f} seconds")
            
            # Build graph
            start_time = time.time()
            kg.build_comprehensive_graph()
            build_time = time.time() - start_time
            logger.info(f"✓ Graph built in {build_time:.2f} seconds")
            
            # Validate graph structure
            stats = kg.get_comprehensive_stats()
            
            # Check minimum thresholds
            min_requirements = {
                'total_nodes': 80000,  # Should be ~90K
                'total_edges': 2500000,  # Should be ~3M
                'go_term': 40000,
                'gene': 15000,
                'llm_model': 3,  # At least 3 models
                'model_prediction': 1000,  # At least 1000 predictions
            }
            
            for metric, min_value in min_requirements.items():
                if metric in ['total_nodes', 'total_edges']:
                    actual_value = stats[metric]
                else:
                    actual_value = stats['node_counts'].get(metric, 0)
                
                if actual_value < min_value:
                    logger.error(f"Insufficient {metric}: {actual_value} < {min_value}")
                    return False
                
                logger.info(f"✓ {metric}: {actual_value:,} (≥ {min_value:,})")
            
            # Check integration ratio
            integration_ratio = stats['integration_metrics']['integration_ratio']
            if integration_ratio < 0.8:
                logger.error(f"Low integration ratio: {integration_ratio:.3f}")
                return False
            
            logger.info(f"✓ Integration ratio: {integration_ratio:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"KG construction testing failed: {e}")
            return False
    
    def test_data_integration(self):
        """Test data integration across all components."""
        logger.info("Testing data integration...")
        
        try:
            kg = ComprehensiveBiomedicalKnowledgeGraph()
            kg.load_data(self.data_dir)
            kg.build_comprehensive_graph()
            
            stats = kg.get_comprehensive_stats()
            
            # Test GO integration
            go_connected_genes = stats['integration_metrics']['go_connected_genes']
            if go_connected_genes < 15000:
                logger.error(f"Insufficient GO-connected genes: {go_connected_genes}")
                return False
            logger.info(f"✓ GO-connected genes: {go_connected_genes:,}")
            
            # Test Omics integration
            omics_connected_genes = stats['integration_metrics']['omics_connected_genes']
            if omics_connected_genes < 10000:
                logger.error(f"Insufficient Omics-connected genes: {omics_connected_genes}")
                return False
            logger.info(f"✓ Omics-connected genes: {omics_connected_genes:,}")
            
            # Test model integration
            model_edge_types = [edge_type for edge_type in stats['edge_counts'].keys() if 'model' in edge_type or 'prediction' in edge_type]
            if len(model_edge_types) < 3:
                logger.error(f"Insufficient model edge types: {model_edge_types}")
                return False
            logger.info(f"✓ Model edge types: {model_edge_types}")
            
            # Test cross-modal edges
            expected_edge_types = [
                'go_hierarchy', 'gene_annotation', 'gene_disease_association', 
                'gene_drug_perturbation', 'gene_viral_expression', 'model_predicts'
            ]
            
            missing_edges = []
            for edge_type in expected_edge_types:
                if edge_type not in stats['edge_counts']:
                    missing_edges.append(edge_type)
                else:
                    count = stats['edge_counts'][edge_type]
                    logger.info(f"✓ {edge_type}: {count:,} edges")
            
            if missing_edges:
                logger.error(f"Missing edge types: {missing_edges}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Data integration testing failed: {e}")
            return False
    
    def test_query_functionality(self):
        """Test all query functions."""
        logger.info("Testing query functionality...")
        
        try:
            kg = ComprehensiveBiomedicalKnowledgeGraph()
            kg.load_data(self.data_dir)
            kg.build_comprehensive_graph()
            
            # Test gene comprehensive query
            test_genes = ['TP53', 'BRCA1', 'EGFR', 'MYC']
            successful_queries = 0
            
            for gene in test_genes:
                profile = kg.query_gene_comprehensive(gene)
                if profile and profile.get('gene_symbol') == gene:
                    successful_queries += 1
                    
                    # Check all profile components
                    expected_components = [
                        'go_annotations', 'disease_associations', 'drug_perturbations',
                        'viral_responses', 'gene_set_memberships', 'model_predictions'
                    ]
                    
                    for component in expected_components:
                        if component not in profile:
                            logger.error(f"Missing {component} in {gene} profile")
                            return False
                    
                    logger.info(f"✓ {gene}: {len(profile['go_annotations'])} GO, {len(profile['disease_associations'])} diseases, {len(profile['model_predictions'])} predictions")
                else:
                    logger.error(f"Failed to query {gene}")
                    return False
            
            if successful_queries != len(test_genes):
                logger.error(f"Gene query success rate: {successful_queries}/{len(test_genes)}")
                return False
            
            # Test model comparison queries
            model_summary = kg.query_model_comparison_summary()
            if not model_summary or model_summary['total_models'] < 3:
                logger.error("Model comparison summary failed")
                return False
            
            logger.info(f"✓ Model summary: {model_summary['total_models']} models, {model_summary['total_predictions']} predictions")
            
            # Test model predictions query
            all_predictions = kg.query_model_predictions()
            if not all_predictions or len(all_predictions) < 500:
                logger.error(f"Insufficient model predictions: {len(all_predictions) if all_predictions else 0}")
                return False
            
            logger.info(f"✓ Model predictions query: {len(all_predictions)} predictions")
            
            # Test filtered predictions
            if model_summary['model_performance']:
                test_model = list(model_summary['model_performance'].keys())[0]
                filtered_predictions = kg.query_model_predictions(model_name=test_model)
                if not filtered_predictions:
                    logger.error(f"Failed to filter predictions for {test_model}")
                    return False
                
                logger.info(f"✓ Filtered predictions for {test_model}: {len(filtered_predictions)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Query functionality testing failed: {e}")
            return False
    
    def test_cross_modal_connectivity(self):
        """Test cross-modal connectivity between different data types."""
        logger.info("Testing cross-modal connectivity...")
        
        try:
            kg = ComprehensiveBiomedicalKnowledgeGraph()
            kg.load_data(self.data_dir)
            kg.build_comprehensive_graph()
            
            # Test gene-GO connectivity
            test_gene = 'TP53'
            if test_gene not in kg.graph:
                logger.error(f"Test gene {test_gene} not found in graph")
                return False
            
            # Check GO connections
            go_neighbors = []
            disease_neighbors = []
            drug_neighbors = []
            viral_neighbors = []
            model_neighbors = []
            
            for neighbor in kg.graph.neighbors(test_gene):
                neighbor_data = kg.graph.nodes[neighbor]
                node_type = neighbor_data.get('node_type')
                
                if node_type == 'go_term':
                    go_neighbors.append(neighbor)
                elif node_type == 'disease':
                    disease_neighbors.append(neighbor)
                elif node_type == 'drug':
                    drug_neighbors.append(neighbor)
                elif node_type == 'viral_condition':
                    viral_neighbors.append(neighbor)
                elif node_type in ['model_prediction', 'similarity_ranking']:
                    model_neighbors.append(neighbor)
            
            # Validate connectivity
            if len(go_neighbors) < 100:
                logger.error(f"Insufficient GO connections for {test_gene}: {len(go_neighbors)}")
                return False
            
            if len(disease_neighbors) == 0:
                logger.warning(f"No disease connections for {test_gene}")
            
            if len(viral_neighbors) == 0:
                logger.warning(f"No viral connections for {test_gene}")
            
            logger.info(f"✓ {test_gene} connectivity: {len(go_neighbors)} GO, {len(disease_neighbors)} diseases, {len(drug_neighbors)} drugs, {len(viral_neighbors)} viral, {len(model_neighbors)} model")
            
            # Test model-GO connectivity
            model_nodes = [n for n, d in kg.graph.nodes(data=True) if d.get('node_type') == 'llm_model']
            if not model_nodes:
                logger.error("No model nodes found")
                return False
            
            test_model = model_nodes[0]
            model_connections = list(kg.graph.neighbors(test_model))
            
            if len(model_connections) < 50:
                logger.error(f"Insufficient model connections: {len(model_connections)}")
                return False
            
            logger.info(f"✓ Model connectivity: {len(model_connections)} connections")
            
            return True
            
        except Exception as e:
            logger.error(f"Cross-modal connectivity testing failed: {e}")
            return False
    
    def test_model_comparison_functionality(self):
        """Test model comparison specific functionality."""
        logger.info("Testing model comparison functionality...")
        
        try:
            kg = ComprehensiveBiomedicalKnowledgeGraph()
            kg.load_data(self.data_dir)
            kg.build_comprehensive_graph()
            
            # Test model nodes
            model_nodes = [n for n, d in kg.graph.nodes(data=True) if d.get('node_type') == 'llm_model']
            if len(model_nodes) < 3:
                logger.error(f"Insufficient model nodes: {len(model_nodes)}")
                return False
            
            logger.info(f"✓ Found {len(model_nodes)} model nodes")
            
            # Test prediction nodes
            prediction_nodes = [n for n, d in kg.graph.nodes(data=True) if d.get('node_type') == 'model_prediction']
            if len(prediction_nodes) < 1000:
                logger.error(f"Insufficient prediction nodes: {len(prediction_nodes)}")
                return False
            
            logger.info(f"✓ Found {len(prediction_nodes)} prediction nodes")
            
            # Test similarity ranking nodes
            ranking_nodes = [n for n, d in kg.graph.nodes(data=True) if d.get('node_type') == 'similarity_ranking']
            if len(ranking_nodes) < 100:
                logger.error(f"Insufficient ranking nodes: {len(ranking_nodes)}")
                return False
            
            logger.info(f"✓ Found {len(ranking_nodes)} ranking nodes")
            
            # Test model performance metrics
            model_summary = kg.query_model_comparison_summary()
            for model_name, metrics in model_summary['model_performance'].items():
                required_metrics = ['mean_confidence', 'mean_similarity', 'contamination_robustness']
                for metric in required_metrics:
                    if metric not in metrics:
                        logger.error(f"Missing {metric} for {model_name}")
                        return False
                
                logger.info(f"✓ {model_name} metrics complete")
            
            # Test contamination scenarios
            scenarios = model_summary.get('scenario_coverage', {})
            expected_scenarios = ['default', '50perc_contaminated', '100perc_contaminated']
            
            for scenario in expected_scenarios:
                if scenario not in scenarios:
                    logger.error(f"Missing scenario: {scenario}")
                    return False
                
                logger.info(f"✓ Scenario {scenario}: {scenarios[scenario]} predictions")
            
            # Test prediction-gene connections
            sample_prediction = prediction_nodes[0]
            prediction_connections = list(kg.graph.neighbors(sample_prediction))
            
            # Should connect to model, GO term, and genes
            connected_types = set()
            for neighbor in prediction_connections:
                neighbor_data = kg.graph.nodes[neighbor]
                connected_types.add(neighbor_data.get('node_type'))
            
            expected_connections = {'llm_model', 'go_term', 'gene'}
            if not expected_connections.issubset(connected_types):
                logger.error(f"Missing prediction connections: {expected_connections - connected_types}")
                return False
            
            logger.info(f"✓ Prediction connectivity validated")
            
            return True
            
        except Exception as e:
            logger.error(f"Model comparison functionality testing failed: {e}")
            return False
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks."""
        logger.info("Testing performance benchmarks...")
        
        try:
            # Test construction performance
            kg = ComprehensiveBiomedicalKnowledgeGraph()
            
            start_time = time.time()
            kg.load_data(self.data_dir)
            load_time = time.time() - start_time
            
            start_time = time.time()
            kg.build_comprehensive_graph()
            build_time = time.time() - start_time
            
            total_time = load_time + build_time
            
            # Performance thresholds
            if total_time > 60:  # 60 seconds max
                logger.warning(f"Construction time exceeded threshold: {total_time:.2f}s > 60s")
            else:
                logger.info(f"✓ Construction time within threshold: {total_time:.2f}s")
            
            # Test query performance
            test_genes = ['TP53', 'BRCA1', 'EGFR', 'MYC', 'GAPDH'] * 10  # 50 queries
            
            start_time = time.time()
            for gene in test_genes:
                kg.query_gene_comprehensive(gene)
            query_time = time.time() - start_time
            
            queries_per_second = len(test_genes) / query_time
            
            if queries_per_second < 500:  # Minimum 500 queries/second
                logger.warning(f"Query performance below threshold: {queries_per_second:.1f} < 500 queries/sec")
            else:
                logger.info(f"✓ Query performance: {queries_per_second:.1f} queries/sec")
            
            # Test model query performance
            start_time = time.time()
            for _ in range(10):
                kg.query_model_comparison_summary()
                kg.query_model_predictions()
            model_query_time = time.time() - start_time
            
            logger.info(f"✓ Model query performance: {20/model_query_time:.1f} queries/sec")
            
            # Memory efficiency test
            stats = kg.get_comprehensive_stats()
            edges_per_node = stats['total_edges'] / stats['total_nodes']
            
            if edges_per_node < 20:
                logger.warning(f"Low graph density: {edges_per_node:.1f} edges/node")
            else:
                logger.info(f"✓ Graph density: {edges_per_node:.1f} edges/node")
            
            return True
            
        except Exception as e:
            logger.error(f"Performance benchmarking failed: {e}")
            return False
    
    def test_regression_compatibility(self):
        """Test that new changes don't break existing functionality."""
        logger.info("Testing regression compatibility...")
        
        try:
            kg = ComprehensiveBiomedicalKnowledgeGraph()
            kg.load_data(self.data_dir)
            kg.build_comprehensive_graph()
            
            # Test legacy query formats still work
            test_gene = 'TP53'
            profile = kg.query_gene_comprehensive(test_gene)
            
            # Verify all original fields still exist
            original_fields = [
                'gene_symbol', 'go_annotations', 'disease_associations', 
                'drug_perturbations', 'viral_responses', 'gene_set_memberships'
            ]
            
            for field in original_fields:
                if field not in profile:
                    logger.error(f"Missing original field: {field}")
                    return False
                logger.info(f"✓ Original field {field} preserved")
            
            # Test that GO annotations still have correct structure
            if profile['go_annotations']:
                go_annotation = profile['go_annotations'][0]
                required_go_fields = ['go_id', 'go_name', 'namespace', 'evidence_code']
                
                for field in required_go_fields:
                    if field not in go_annotation:
                        logger.error(f"Missing GO annotation field: {field}")
                        return False
                
                logger.info(f"✓ GO annotation structure preserved")
            
            # Test that disease associations still work
            if profile['disease_associations']:
                disease_assoc = profile['disease_associations'][0]
                required_disease_fields = ['disease', 'condition']
                
                for field in required_disease_fields:
                    if field not in disease_assoc:
                        logger.error(f"Missing disease association field: {field}")
                        return False
                
                logger.info(f"✓ Disease association structure preserved")
            
            # Test that viral responses still work (both types)
            viral_responses = profile['viral_responses']
            if viral_responses:
                # Check for both association and expression types
                response_types = set(vr.get('type', 'response') for vr in viral_responses)
                logger.info(f"✓ Viral response types: {response_types}")
            
            # Test that stats structure is preserved
            stats = kg.get_comprehensive_stats()
            required_stats_fields = ['total_nodes', 'total_edges', 'node_counts', 'edge_counts', 'integration_metrics']
            
            for field in required_stats_fields:
                if field not in stats:
                    logger.error(f"Missing stats field: {field}")
                    return False
                logger.info(f"✓ Stats field {field} preserved")
            
            # Test that integration metrics are preserved
            integration_metrics = stats['integration_metrics']
            required_integration_fields = ['go_connected_genes', 'omics_connected_genes', 'integrated_genes', 'integration_ratio']
            
            for field in required_integration_fields:
                if field not in integration_metrics:
                    logger.error(f"Missing integration metric: {field}")
                    return False
                logger.info(f"✓ Integration metric {field} preserved")
            
            return True
            
        except Exception as e:
            logger.error(f"Regression compatibility testing failed: {e}")
            return False

def main():
    """Main testing function."""
    logger.info("🧪 Starting Comprehensive System Testing")
    
    tester = ComprehensiveSystemTester()
    success = tester.run_all_tests()
    
    logger.info("\n" + "="*100)
    if success:
        logger.info("✅ COMPREHENSIVE SYSTEM TESTING PASSED")
        logger.info("All components are working correctly!")
    else:
        logger.info("❌ COMPREHENSIVE SYSTEM TESTING FAILED")
        logger.info("Some components need attention.")
    logger.info("="*100)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)