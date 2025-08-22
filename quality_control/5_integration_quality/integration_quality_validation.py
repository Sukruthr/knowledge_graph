#!/usr/bin/env python3
"""
Phase 5: Integration Quality Validation

This script validates the integration quality across all 9 data source phases,
ensuring seamless data integration and cross-modal connectivity.
"""

import sys
import os
import time
import pickle
import logging
import json
import traceback
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.append('/home/mreddy1/knowledge_graph/src')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/mreddy1/knowledge_graph/quality_control/5_integration_quality/integration_quality_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntegrationQualityValidator:
    """Integration quality validation for biomedical knowledge graph."""
    
    def __init__(self, kg_path='/home/mreddy1/knowledge_graph/quality_control/1_build_and_save_kg/saved_graphs/complete_biomedical_kg.pkl'):
        self.kg_path = kg_path
        self.kg = None
        self.graph = None
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'validation_summary': {},
            'detailed_results': {},
            'quality_metrics': {},
            'recommendations': []
        }
    
    def load_knowledge_graph(self):
        """Load the pre-built knowledge graph."""
        logger.info("📊 LOADING KNOWLEDGE GRAPH FOR INTEGRATION QUALITY VALIDATION")
        
        try:
            if not os.path.exists(self.kg_path):
                # Use existing KG if build didn't complete
                from kg_builders import ComprehensiveBiomedicalKnowledgeGraph
                logger.info("Building KG directly for integration testing...")
                self.kg = ComprehensiveBiomedicalKnowledgeGraph()
                self.kg.load_data('/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data')
                self.kg.build_comprehensive_graph()
                self.graph = self.kg.graph
                logger.info("✅ Knowledge graph built directly")
                return True
            
            logger.info(f"Loading knowledge graph from: {self.kg_path}")
            with open(self.kg_path, 'rb') as f:
                self.kg = pickle.load(f)
            self.graph = self.kg.graph
            logger.info("✅ Knowledge graph loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load knowledge graph: {str(e)}")
            return False
    
    def validate_integration_coverage(self):
        """Validate integration coverage across all data sources."""
        logger.info("🔍 VALIDATING INTEGRATION COVERAGE")
        
        try:
            coverage_results = {}
            
            # Expected data source integrations (9 phases)
            expected_integrations = [
                'GO_BP', 'GO_CC', 'GO_MF',  # GO multi-namespace
                'Omics_Data', 'Model_Compare',  # Omics and model comparison
                'CC_MF_Branch', 'LLM_processed',  # Enhanced GO analysis
                'GO_Analysis_Data', 'Remaining_Data'  # Additional datasets
            ]
            
            # Count node types representing different integrations
            node_types = defaultdict(int)
            for node, data in self.graph.nodes(data=True):
                node_type = data.get('type', 'unknown')
                node_types[node_type] += 1
            
            # Check for expected node types from each integration
            integration_indicators = {
                'GO_integration': node_types.get('go_term', 0) > 0,
                'Gene_integration': node_types.get('gene', 0) > 0,
                'Disease_integration': node_types.get('disease', 0) > 0,
                'Drug_integration': node_types.get('drug', 0) > 0,
                'Viral_integration': node_types.get('viral_condition', 0) > 0,
                'Model_integration': node_types.get('model_prediction', 0) > 0,
                'LLM_integration': node_types.get('llm_interpretation', 0) > 0,
                'Cluster_integration': node_types.get('cluster', 0) > 0
            }
            
            integration_count = sum(integration_indicators.values())
            integration_coverage = (integration_count / len(integration_indicators)) * 100
            
            coverage_results = {
                'expected_integrations': expected_integrations,
                'integration_indicators': integration_indicators,
                'total_node_types': len(node_types),
                'integration_coverage_percentage': integration_coverage,
                'node_type_distribution': dict(node_types)
            }
            
            logger.info(f"Integration coverage: {integration_coverage:.1f}%")
            logger.info(f"Total node types: {len(node_types)}")
            
            self.validation_results['detailed_results']['integration_coverage'] = coverage_results
            return True
            
        except Exception as e:
            logger.error(f"❌ Integration coverage validation failed: {str(e)}")
            return False
    
    def validate_cross_modal_connectivity(self):
        """Validate cross-modal connectivity between data sources."""
        logger.info("🔗 VALIDATING CROSS-MODAL CONNECTIVITY")
        
        try:
            connectivity_results = {}
            
            # Sample genes and check their connections to different data types
            gene_nodes = [(n, d) for n, d in self.graph.nodes(data=True) if d.get('type') == 'gene'][:100]
            
            connection_types = defaultdict(int)
            genes_with_connections = defaultdict(int)
            
            for gene_node, gene_data in gene_nodes:
                connected_types = set()
                
                # Check neighbors
                for neighbor in self.graph.neighbors(gene_node):
                    neighbor_data = self.graph.nodes[neighbor]
                    neighbor_type = neighbor_data.get('type', 'unknown')
                    connected_types.add(neighbor_type)
                    connection_types[neighbor_type] += 1
                
                # Count genes with different numbers of connection types
                genes_with_connections[len(connected_types)] += 1
            
            # Calculate connectivity metrics
            avg_connections = sum(k * v for k, v in genes_with_connections.items()) / len(gene_nodes) if gene_nodes else 0
            
            connectivity_results = {
                'genes_analyzed': len(gene_nodes),
                'connection_type_distribution': dict(connection_types),
                'genes_by_connection_count': dict(genes_with_connections),
                'average_connection_types_per_gene': avg_connections,
                'highly_connected_genes': genes_with_connections.get(5, 0) + genes_with_connections.get(6, 0)
            }
            
            logger.info(f"Average connection types per gene: {avg_connections:.2f}")
            logger.info(f"Highly connected genes (≥5 types): {connectivity_results['highly_connected_genes']}")
            
            self.validation_results['detailed_results']['cross_modal_connectivity'] = connectivity_results
            return True
            
        except Exception as e:
            logger.error(f"❌ Cross-modal connectivity validation failed: {str(e)}")
            return False
    
    def generate_quality_metrics(self):
        """Generate integration quality metrics."""
        logger.info("📊 GENERATING INTEGRATION QUALITY METRICS")
        
        try:
            metrics = {}
            
            # Integration coverage score
            coverage = self.validation_results['detailed_results'].get('integration_coverage', {})
            coverage_score = coverage.get('integration_coverage_percentage', 0)
            metrics['integration_coverage_score'] = coverage_score
            
            # Connectivity score
            connectivity = self.validation_results['detailed_results'].get('cross_modal_connectivity', {})
            avg_connections = connectivity.get('average_connection_types_per_gene', 0)
            connectivity_score = min(avg_connections * 25, 100)  # Scale to 0-100
            metrics['connectivity_score'] = connectivity_score
            
            # Overall integration quality
            overall_quality = (coverage_score + connectivity_score) / 2
            metrics['overall_integration_quality'] = overall_quality
            
            # Quality grade
            if overall_quality >= 90:
                grade = 'A'
            elif overall_quality >= 80:
                grade = 'B'
            elif overall_quality >= 70:
                grade = 'C'
            else:
                grade = 'D'
            
            metrics['integration_quality_grade'] = grade
            
            self.validation_results['quality_metrics'] = metrics
            
            logger.info(f"Integration Quality Score: {overall_quality:.1f}/100 (Grade: {grade})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Quality metrics generation failed: {str(e)}")
            return False
    
    def save_results(self):
        """Save validation results."""
        try:
            output_path = '/home/mreddy1/knowledge_graph/quality_control/5_integration_quality/integration_quality_validation_results.json'
            with open(output_path, 'w') as f:
                json.dump(self.validation_results, f, indent=2)
            logger.info(f"📄 Results saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save results: {str(e)}")
            return False
    
    def run_validation(self):
        """Run integration quality validation."""
        logger.info("🔍 INTEGRATION QUALITY VALIDATION")
        
        steps = [
            ('Load Knowledge Graph', self.load_knowledge_graph),
            ('Validate Integration Coverage', self.validate_integration_coverage),
            ('Validate Cross-Modal Connectivity', self.validate_cross_modal_connectivity),
            ('Generate Quality Metrics', self.generate_quality_metrics),
            ('Save Results', self.save_results)
        ]
        
        passed = 0
        for name, func in steps:
            logger.info(f"Executing: {name}")
            if func():
                passed += 1
                logger.info(f"✅ {name} completed")
            else:
                logger.error(f"❌ {name} failed")
        
        success_rate = (passed / len(steps)) * 100
        self.validation_results['validation_summary'] = {
            'total_steps': len(steps),
            'passed_steps': passed,
            'success_rate': success_rate,
            'overall_status': 'PASSED' if passed == len(steps) else 'FAILED'
        }
        
        logger.info(f"Integration Quality Validation: {success_rate:.1f}% success rate")
        return success_rate == 100

def main():
    """Main execution function."""
    try:
        validator = IntegrationQualityValidator()
        success = validator.run_validation()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())