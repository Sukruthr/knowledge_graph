#!/usr/bin/env python3
"""
Phase 2: Structural Integrity Validation

This script performs comprehensive structural validation of the biomedical knowledge graph
including topology analysis, schema adherence, property validation, and relationship integrity.

Validation Categories:
1. Graph Topology Analysis (connectivity, components, cycles)
2. Node Structure Validation (types, properties, completeness)
3. Edge Structure Validation (types, directionality, constraints)
4. Schema Adherence (expected node/edge types present)
5. Data Integrity (referential integrity, orphaned nodes)
"""

import sys
import os
import time
import pickle
import logging
import json
import traceback
from datetime import datetime
from collections import defaultdict, Counter
import networkx as nx
import numpy as np

# Add project root to path
sys.path.append('/home/mreddy1/knowledge_graph/src')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/mreddy1/knowledge_graph/quality_control/2_structural_integrity/structural_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StructuralValidator:
    """Comprehensive structural validation for biomedical knowledge graph."""
    
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
        logger.info("📊 LOADING KNOWLEDGE GRAPH FOR STRUCTURAL VALIDATION")
        logger.info("=" * 70)
        
        try:
            if not os.path.exists(self.kg_path):
                logger.error(f"Knowledge graph file not found: {self.kg_path}")
                return False
                
            logger.info(f"Loading knowledge graph from: {self.kg_path}")
            load_start = time.time()
            
            with open(self.kg_path, 'rb') as f:
                self.kg = pickle.load(f)
                
            self.graph = self.kg.graph
            load_time = time.time() - load_start
            
            logger.info(f"✅ Knowledge graph loaded successfully in {load_time:.2f} seconds")
            logger.info(f"   Nodes: {self.graph.number_of_nodes():,}")
            logger.info(f"   Edges: {self.graph.number_of_edges():,}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load knowledge graph: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def validate_graph_topology(self):
        """Comprehensive topology analysis of the graph structure."""
        logger.info("🔍 VALIDATING GRAPH TOPOLOGY")
        logger.info("-" * 50)
        
        try:
            topology_results = {}
            
            # Basic topology metrics
            num_nodes = self.graph.number_of_nodes()
            num_edges = self.graph.number_of_edges()
            is_directed = self.graph.is_directed()
            
            topology_results['basic_metrics'] = {
                'total_nodes': num_nodes,
                'total_edges': num_edges,
                'is_directed': is_directed,
                'density': nx.density(self.graph),
                'average_degree': sum(dict(self.graph.degree()).values()) / num_nodes if num_nodes > 0 else 0
            }
            
            logger.info(f"Basic metrics: {num_nodes:,} nodes, {num_edges:,} edges")
            logger.info(f"Graph type: {'Directed' if is_directed else 'Undirected'}")
            logger.info(f"Density: {topology_results['basic_metrics']['density']:.6f}")
            
            # Connected components analysis
            if is_directed:
                strongly_connected = list(nx.strongly_connected_components(self.graph))
                weakly_connected = list(nx.weakly_connected_components(self.graph))
                topology_results['connectivity'] = {
                    'strongly_connected_components': len(strongly_connected),
                    'weakly_connected_components': len(weakly_connected),
                    'largest_strongly_connected_size': len(max(strongly_connected, key=len)) if strongly_connected else 0,
                    'largest_weakly_connected_size': len(max(weakly_connected, key=len)) if weakly_connected else 0
                }
                logger.info(f"Strongly connected components: {len(strongly_connected)}")
                logger.info(f"Weakly connected components: {len(weakly_connected)}")
            else:
                connected_components = list(nx.connected_components(self.graph))
                topology_results['connectivity'] = {
                    'connected_components': len(connected_components),
                    'largest_component_size': len(max(connected_components, key=len)) if connected_components else 0,
                    'component_size_distribution': [len(comp) for comp in connected_components[:10]]  # Top 10
                }
                logger.info(f"Connected components: {len(connected_components)}")
            
            # Degree distribution analysis
            degrees = [d for n, d in self.graph.degree()]
            topology_results['degree_distribution'] = {
                'min_degree': min(degrees) if degrees else 0,
                'max_degree': max(degrees) if degrees else 0,
                'mean_degree': np.mean(degrees) if degrees else 0,
                'median_degree': np.median(degrees) if degrees else 0,
                'std_degree': np.std(degrees) if degrees else 0
            }
            
            logger.info(f"Degree distribution: min={topology_results['degree_distribution']['min_degree']}, "
                       f"max={topology_results['degree_distribution']['max_degree']}, "
                       f"mean={topology_results['degree_distribution']['mean_degree']:.2f}")
            
            # Isolated nodes check
            isolated_nodes = list(nx.isolates(self.graph))
            topology_results['isolated_nodes'] = {
                'count': len(isolated_nodes),
                'percentage': (len(isolated_nodes) / num_nodes * 100) if num_nodes > 0 else 0,
                'sample_isolated_nodes': isolated_nodes[:10]  # Sample for inspection
            }
            
            if len(isolated_nodes) > 0:
                logger.warning(f"⚠️ Found {len(isolated_nodes)} isolated nodes ({topology_results['isolated_nodes']['percentage']:.2f}%)")
            else:
                logger.info("✅ No isolated nodes found")
            
            # Cycle detection (for directed graphs)
            if is_directed:
                try:
                    has_cycles = not nx.is_directed_acyclic_graph(self.graph)
                    topology_results['cycles'] = {
                        'has_cycles': has_cycles,
                        'analysis': 'Cycles expected in biological networks (feedback loops, etc.)'
                    }
                    logger.info(f"Cycle detection: {'Contains cycles' if has_cycles else 'Acyclic'}")
                except:
                    topology_results['cycles'] = {'has_cycles': None, 'error': 'Cycle detection failed'}
            
            self.validation_results['detailed_results']['topology'] = topology_results
            logger.info("✅ Graph topology validation completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Graph topology validation failed: {str(e)}")
            self.validation_results['detailed_results']['topology'] = {'error': str(e)}
            return False
    
    def validate_node_structure(self):
        """Validate node types, properties, and data consistency."""
        logger.info("🔍 VALIDATING NODE STRUCTURE")
        logger.info("-" * 50)
        
        try:
            node_results = {}
            
            # Node type distribution
            node_types = defaultdict(int)
            node_properties = defaultdict(set)
            missing_type_nodes = []
            
            for node, data in self.graph.nodes(data=True):
                node_type = data.get('type', 'Unknown')
                node_types[node_type] += 1
                
                if node_type == 'Unknown':
                    missing_type_nodes.append(node)
                
                # Collect property keys for each node type
                for key in data.keys():
                    node_properties[node_type].add(key)
            
            # Convert sets to lists for JSON serialization
            node_properties_serializable = {k: list(v) for k, v in node_properties.items()}
            
            node_results['type_distribution'] = dict(node_types)
            node_results['properties_by_type'] = node_properties_serializable
            node_results['missing_type_count'] = len(missing_type_nodes)
            node_results['missing_type_percentage'] = (len(missing_type_nodes) / self.graph.number_of_nodes() * 100) if self.graph.number_of_nodes() > 0 else 0
            
            logger.info(f"Node type distribution:")
            for node_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                logger.info(f"   {node_type}: {count:,}")
            
            if missing_type_nodes:
                logger.warning(f"⚠️ {len(missing_type_nodes)} nodes missing type information ({node_results['missing_type_percentage']:.2f}%)")
            else:
                logger.info("✅ All nodes have type information")
            
            # Expected biological node types validation
            expected_bio_types = {
                'gene', 'go_term', 'disease', 'drug', 'viral_condition', 
                'cluster', 'model_prediction', 'llm_interpretation', 'gene_set'
            }
            
            found_bio_types = set(node_types.keys()) - {'Unknown'}
            bio_type_coverage = len(found_bio_types & expected_bio_types) / len(expected_bio_types)
            
            node_results['biological_type_coverage'] = {
                'expected_types': list(expected_bio_types),
                'found_types': list(found_bio_types),
                'coverage_percentage': bio_type_coverage * 100,
                'missing_types': list(expected_bio_types - found_bio_types)
            }
            
            logger.info(f"Biological node type coverage: {bio_type_coverage * 100:.1f}%")
            if node_results['biological_type_coverage']['missing_types']:
                logger.warning(f"Missing expected types: {node_results['biological_type_coverage']['missing_types']}")
            
            # Property completeness analysis
            property_completeness = {}
            for node_type, properties in node_properties.items():
                if node_type != 'Unknown':
                    property_completeness[node_type] = len(properties)
            
            node_results['property_completeness'] = property_completeness
            
            # Validate specific node types have required properties
            validation_checks = []
            
            # Check gene nodes
            gene_nodes = [(n, d) for n, d in self.graph.nodes(data=True) if d.get('type') == 'gene']
            gene_with_symbol = sum(1 for n, d in gene_nodes if 'symbol' in d or 'name' in d)
            gene_validation = {
                'total_genes': len(gene_nodes),
                'with_identifier': gene_with_symbol,
                'identifier_coverage': (gene_with_symbol / len(gene_nodes) * 100) if gene_nodes else 0
            }
            validation_checks.append(('Gene identifier coverage', gene_validation['identifier_coverage'], 95.0))
            
            # Check GO term nodes
            go_nodes = [(n, d) for n, d in self.graph.nodes(data=True) if d.get('type') == 'go_term']
            go_with_namespace = sum(1 for n, d in go_nodes if 'namespace' in d)
            go_validation = {
                'total_go_terms': len(go_nodes),
                'with_namespace': go_with_namespace,
                'namespace_coverage': (go_with_namespace / len(go_nodes) * 100) if go_nodes else 0
            }
            validation_checks.append(('GO term namespace coverage', go_validation['namespace_coverage'], 90.0))
            
            node_results['specific_validations'] = {
                'gene_validation': gene_validation,
                'go_validation': go_validation
            }
            
            node_results['quality_checks'] = []
            for check_name, actual, threshold in validation_checks:
                passed = actual >= threshold
                node_results['quality_checks'].append({
                    'check': check_name,
                    'actual': actual,
                    'threshold': threshold,
                    'passed': passed
                })
                status = "✅" if passed else "❌"
                logger.info(f"{status} {check_name}: {actual:.1f}% (threshold: {threshold}%)")
            
            self.validation_results['detailed_results']['nodes'] = node_results
            logger.info("✅ Node structure validation completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Node structure validation failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.validation_results['detailed_results']['nodes'] = {'error': str(e)}
            return False
    
    def validate_edge_structure(self):
        """Validate edge types, directionality, and relationship constraints."""
        logger.info("🔍 VALIDATING EDGE STRUCTURE")
        logger.info("-" * 50)
        
        try:
            edge_results = {}
            
            # Edge type distribution
            edge_types = defaultdict(int)
            edge_properties = defaultdict(set)
            missing_type_edges = []
            
            for source, target, data in self.graph.edges(data=True):
                edge_type = data.get('type', 'Unknown')
                edge_types[edge_type] += 1
                
                if edge_type == 'Unknown':
                    missing_type_edges.append((source, target))
                
                # Collect property keys for each edge type
                for key in data.keys():
                    edge_properties[edge_type].add(key)
            
            # Convert sets to lists for JSON serialization
            edge_properties_serializable = {k: list(v) for k, v in edge_properties.items()}
            
            edge_results['type_distribution'] = dict(edge_types)
            edge_results['properties_by_type'] = edge_properties_serializable
            edge_results['missing_type_count'] = len(missing_type_edges)
            edge_results['missing_type_percentage'] = (len(missing_type_edges) / self.graph.number_of_edges() * 100) if self.graph.number_of_edges() > 0 else 0
            
            logger.info(f"Edge type distribution:")
            for edge_type, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                logger.info(f"   {edge_type}: {count:,}")
            
            if missing_type_edges:
                logger.warning(f"⚠️ {len(missing_type_edges)} edges missing type information ({edge_results['missing_type_percentage']:.2f}%)")
            else:
                logger.info("✅ All edges have type information")
            
            # Expected biological edge types validation
            expected_edge_types = {
                'gene_go_association', 'go_hierarchy', 'gene_disease_association',
                'gene_drug_association', 'gene_viral_association', 'cluster_relationship',
                'model_prediction', 'llm_interpretation', 'similarity_ranking'
            }
            
            found_edge_types = set(edge_types.keys()) - {'Unknown'}
            edge_type_coverage = len(found_edge_types & expected_edge_types) / len(expected_edge_types)
            
            edge_results['biological_type_coverage'] = {
                'expected_types': list(expected_edge_types),
                'found_types': list(found_edge_types),
                'coverage_percentage': edge_type_coverage * 100,
                'missing_types': list(expected_edge_types - found_edge_types)
            }
            
            logger.info(f"Biological edge type coverage: {edge_type_coverage * 100:.1f}%")
            if edge_results['biological_type_coverage']['missing_types']:
                logger.warning(f"Missing expected edge types: {edge_results['biological_type_coverage']['missing_types']}")
            
            # Self-loop analysis
            self_loops = list(nx.selfloop_edges(self.graph))
            edge_results['self_loops'] = {
                'count': len(self_loops),
                'percentage': (len(self_loops) / self.graph.number_of_edges() * 100) if self.graph.number_of_edges() > 0 else 0
            }
            
            if self_loops:
                logger.info(f"Self-loops found: {len(self_loops)} ({edge_results['self_loops']['percentage']:.2f}%)")
            else:
                logger.info("✅ No self-loops found")
            
            # Multi-edge analysis (for MultiGraph types)
            if hasattr(self.graph, 'number_of_edges'):
                simple_edges = len(set((u, v) for u, v, _ in self.graph.edges()))
                total_edges = self.graph.number_of_edges()
                multi_edges = total_edges - simple_edges
                
                edge_results['multi_edges'] = {
                    'count': multi_edges,
                    'percentage': (multi_edges / total_edges * 100) if total_edges > 0 else 0
                }
                
                if multi_edges > 0:
                    logger.info(f"Multi-edges found: {multi_edges} ({edge_results['multi_edges']['percentage']:.2f}%)")
                else:
                    logger.info("✅ No multi-edges found")
            
            self.validation_results['detailed_results']['edges'] = edge_results
            logger.info("✅ Edge structure validation completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Edge structure validation failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.validation_results['detailed_results']['edges'] = {'error': str(e)}
            return False
    
    def validate_referential_integrity(self):
        """Validate referential integrity and cross-references between data sources."""
        logger.info("🔍 VALIDATING REFERENTIAL INTEGRITY")
        logger.info("-" * 50)
        
        try:
            integrity_results = {}
            
            # Gene identifier consistency check
            gene_nodes = [(n, d) for n, d in self.graph.nodes(data=True) if d.get('type') == 'gene']
            
            gene_identifiers = {
                'symbol': set(),
                'entrez': set(),
                'uniprot': set()
            }
            
            for node, data in gene_nodes:
                if 'symbol' in data:
                    gene_identifiers['symbol'].add(data['symbol'])
                if 'entrez' in data:
                    gene_identifiers['entrez'].add(data['entrez'])
                if 'uniprot' in data:
                    gene_identifiers['uniprot'].add(data['uniprot'])
            
            integrity_results['gene_identifier_consistency'] = {
                'total_gene_nodes': len(gene_nodes),
                'unique_symbols': len(gene_identifiers['symbol']),
                'unique_entrez': len(gene_identifiers['entrez']),
                'unique_uniprot': len(gene_identifiers['uniprot']),
                'symbol_coverage': (len(gene_identifiers['symbol']) / len(gene_nodes) * 100) if gene_nodes else 0
            }
            
            logger.info(f"Gene identifier consistency:")
            logger.info(f"   Total gene nodes: {len(gene_nodes):,}")
            logger.info(f"   Unique symbols: {len(gene_identifiers['symbol']):,}")
            logger.info(f"   Symbol coverage: {integrity_results['gene_identifier_consistency']['symbol_coverage']:.1f}%")
            
            # Cross-modal connectivity check
            gene_connections = defaultdict(set)
            
            for node, data in gene_nodes:
                gene_symbol = data.get('symbol', data.get('name', str(node)))
                
                # Check connections to different data modalities
                for neighbor in self.graph.neighbors(node):
                    neighbor_data = self.graph.nodes[neighbor]
                    neighbor_type = neighbor_data.get('type', 'Unknown')
                    gene_connections[gene_symbol].add(neighbor_type)
            
            # Calculate cross-modal connectivity statistics
            connection_stats = defaultdict(int)
            for gene, connections in gene_connections.items():
                connection_stats[len(connections)] += 1
            
            avg_connections = np.mean([len(connections) for connections in gene_connections.values()]) if gene_connections else 0
            
            integrity_results['cross_modal_connectivity'] = {
                'genes_with_connections': len(gene_connections),
                'average_connection_types_per_gene': avg_connections,
                'connection_distribution': dict(connection_stats),
                'highly_connected_genes': len([g for g, c in gene_connections.items() if len(c) >= 5])
            }
            
            logger.info(f"Cross-modal connectivity:")
            logger.info(f"   Genes with connections: {len(gene_connections):,}")
            logger.info(f"   Average connection types per gene: {avg_connections:.2f}")
            logger.info(f"   Highly connected genes (≥5 types): {integrity_results['cross_modal_connectivity']['highly_connected_genes']:,}")
            
            # Orphaned nodes analysis
            orphaned_by_type = defaultdict(int)
            total_orphaned = 0
            
            for node, data in self.graph.nodes(data=True):
                if self.graph.degree(node) == 0:
                    node_type = data.get('type', 'Unknown')
                    orphaned_by_type[node_type] += 1
                    total_orphaned += 1
            
            integrity_results['orphaned_nodes'] = {
                'total_orphaned': total_orphaned,
                'orphaned_percentage': (total_orphaned / self.graph.number_of_nodes() * 100) if self.graph.number_of_nodes() > 0 else 0,
                'orphaned_by_type': dict(orphaned_by_type)
            }
            
            logger.info(f"Orphaned nodes: {total_orphaned} ({integrity_results['orphaned_nodes']['orphaned_percentage']:.2f}%)")
            if orphaned_by_type:
                for node_type, count in sorted(orphaned_by_type.items(), key=lambda x: x[1], reverse=True)[:5]:
                    logger.info(f"   {node_type}: {count}")
            
            self.validation_results['detailed_results']['referential_integrity'] = integrity_results
            logger.info("✅ Referential integrity validation completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Referential integrity validation failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.validation_results['detailed_results']['referential_integrity'] = {'error': str(e)}
            return False
    
    def generate_quality_metrics(self):
        """Generate overall quality metrics and scores."""
        logger.info("📊 GENERATING QUALITY METRICS")
        logger.info("-" * 50)
        
        try:
            metrics = {}
            
            # Topology quality score
            topology = self.validation_results['detailed_results'].get('topology', {})
            connectivity = topology.get('connectivity', {})
            isolated = topology.get('isolated_nodes', {})
            
            topology_score = 100.0
            if isolated.get('percentage', 0) > 5:  # More than 5% isolated is concerning
                topology_score -= min(isolated['percentage'] * 2, 30)  # Cap penalty at 30 points
            
            metrics['topology_quality_score'] = max(topology_score, 0)
            
            # Node structure quality score
            nodes = self.validation_results['detailed_results'].get('nodes', {})
            node_score = 100.0
            
            # Penalty for missing types
            if nodes.get('missing_type_percentage', 0) > 1:
                node_score -= min(nodes['missing_type_percentage'] * 10, 50)
            
            # Bonus for good biological type coverage
            bio_coverage = nodes.get('biological_type_coverage', {}).get('coverage_percentage', 0)
            if bio_coverage < 70:
                node_score -= (70 - bio_coverage) / 2
            
            metrics['node_structure_quality_score'] = max(node_score, 0)
            
            # Edge structure quality score
            edges = self.validation_results['detailed_results'].get('edges', {})
            edge_score = 100.0
            
            # Penalty for missing edge types
            if edges.get('missing_type_percentage', 0) > 1:
                edge_score -= min(edges['missing_type_percentage'] * 10, 50)
            
            metrics['edge_structure_quality_score'] = max(edge_score, 0)
            
            # Referential integrity score
            integrity = self.validation_results['detailed_results'].get('referential_integrity', {})
            integrity_score = 100.0
            
            # Penalty for high orphaned node percentage
            orphaned_pct = integrity.get('orphaned_nodes', {}).get('orphaned_percentage', 0)
            if orphaned_pct > 2:
                integrity_score -= min(orphaned_pct * 5, 40)
            
            # Bonus for good cross-modal connectivity
            avg_connections = integrity.get('cross_modal_connectivity', {}).get('average_connection_types_per_gene', 0)
            if avg_connections < 3:
                integrity_score -= (3 - avg_connections) * 10
            
            metrics['referential_integrity_score'] = max(integrity_score, 0)
            
            # Overall structural quality score
            metrics['overall_structural_quality'] = np.mean([
                metrics['topology_quality_score'],
                metrics['node_structure_quality_score'],
                metrics['edge_structure_quality_score'],
                metrics['referential_integrity_score']
            ])
            
            # Quality grade
            overall_score = metrics['overall_structural_quality']
            if overall_score >= 95:
                quality_grade = 'A+'
            elif overall_score >= 90:
                quality_grade = 'A'
            elif overall_score >= 85:
                quality_grade = 'B+'
            elif overall_score >= 80:
                quality_grade = 'B'
            elif overall_score >= 75:
                quality_grade = 'C+'
            elif overall_score >= 70:
                quality_grade = 'C'
            else:
                quality_grade = 'D'
            
            metrics['structural_quality_grade'] = quality_grade
            
            self.validation_results['quality_metrics'] = metrics
            
            logger.info("Quality Scores:")
            logger.info(f"   Topology Quality: {metrics['topology_quality_score']:.1f}/100")
            logger.info(f"   Node Structure: {metrics['node_structure_quality_score']:.1f}/100")
            logger.info(f"   Edge Structure: {metrics['edge_structure_quality_score']:.1f}/100")
            logger.info(f"   Referential Integrity: {metrics['referential_integrity_score']:.1f}/100")
            logger.info(f"   Overall Structural Quality: {overall_score:.1f}/100 (Grade: {quality_grade})")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Quality metrics generation failed: {str(e)}")
            return False
    
    def generate_recommendations(self):
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Analyze results and generate recommendations
        topology = self.validation_results['detailed_results'].get('topology', {})
        nodes = self.validation_results['detailed_results'].get('nodes', {})
        edges = self.validation_results['detailed_results'].get('edges', {})
        integrity = self.validation_results['detailed_results'].get('referential_integrity', {})
        
        # Topology recommendations
        isolated_pct = topology.get('isolated_nodes', {}).get('percentage', 0)
        if isolated_pct > 5:
            recommendations.append({
                'category': 'Topology',
                'priority': 'HIGH',
                'issue': f'{isolated_pct:.1f}% isolated nodes found',
                'recommendation': 'Review data integration to ensure all nodes are properly connected'
            })
        
        # Node structure recommendations
        missing_type_pct = nodes.get('missing_type_percentage', 0)
        if missing_type_pct > 1:
            recommendations.append({
                'category': 'Node Structure',
                'priority': 'MEDIUM',
                'issue': f'{missing_type_pct:.1f}% nodes missing type information',
                'recommendation': 'Add type information to all nodes during data parsing'
            })
        
        # Edge structure recommendations
        edge_missing_pct = edges.get('missing_type_percentage', 0)
        if edge_missing_pct > 1:
            recommendations.append({
                'category': 'Edge Structure', 
                'priority': 'MEDIUM',
                'issue': f'{edge_missing_pct:.1f}% edges missing type information',
                'recommendation': 'Ensure all edges have appropriate type labels'
            })
        
        # Referential integrity recommendations
        orphaned_pct = integrity.get('orphaned_nodes', {}).get('orphaned_percentage', 0)
        if orphaned_pct > 2:
            recommendations.append({
                'category': 'Referential Integrity',
                'priority': 'HIGH',
                'issue': f'{orphaned_pct:.1f}% orphaned nodes detected',
                'recommendation': 'Review data integration pipeline to ensure proper node connections'
            })
        
        if not recommendations:
            recommendations.append({
                'category': 'Overall',
                'priority': 'INFO',
                'issue': 'No significant structural issues detected',
                'recommendation': 'Graph structure meets quality standards'
            })
        
        self.validation_results['recommendations'] = recommendations
        
        logger.info("Recommendations:")
        for rec in recommendations:
            logger.info(f"   [{rec['priority']}] {rec['category']}: {rec['recommendation']}")
    
    def save_results(self):
        """Save validation results to file."""
        try:
            output_path = '/home/mreddy1/knowledge_graph/quality_control/2_structural_integrity/structural_validation_results.json'
            
            with open(output_path, 'w') as f:
                json.dump(self.validation_results, f, indent=2)
                
            logger.info(f"📄 Results saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {str(e)}")
            return False
    
    def run_comprehensive_validation(self):
        """Run all structural validation checks."""
        logger.info("🔍 COMPREHENSIVE STRUCTURAL INTEGRITY VALIDATION")
        logger.info("=" * 80)
        
        validation_steps = [
            ('Load Knowledge Graph', self.load_knowledge_graph),
            ('Validate Graph Topology', self.validate_graph_topology),
            ('Validate Node Structure', self.validate_node_structure),
            ('Validate Edge Structure', self.validate_edge_structure),
            ('Validate Referential Integrity', self.validate_referential_integrity),
            ('Generate Quality Metrics', self.generate_quality_metrics),
            ('Generate Recommendations', self.generate_recommendations),
            ('Save Results', self.save_results)
        ]
        
        start_time = time.time()
        passed_steps = 0
        
        for step_name, step_function in validation_steps:
            logger.info(f"Executing: {step_name}")
            
            try:
                if step_function():
                    passed_steps += 1
                    logger.info(f"✅ {step_name} completed successfully")
                else:
                    logger.error(f"❌ {step_name} failed")
            except Exception as e:
                logger.error(f"❌ {step_name} failed with exception: {str(e)}")
        
        total_time = time.time() - start_time
        
        # Summary
        self.validation_results['validation_summary'] = {
            'total_steps': len(validation_steps),
            'passed_steps': passed_steps,
            'success_rate': (passed_steps / len(validation_steps)) * 100,
            'execution_time_seconds': total_time,
            'overall_status': 'PASSED' if passed_steps == len(validation_steps) else 'FAILED'
        }
        
        logger.info("=" * 80)
        logger.info("📊 STRUCTURAL VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Steps completed: {passed_steps}/{len(validation_steps)}")
        logger.info(f"Success rate: {self.validation_results['validation_summary']['success_rate']:.1f}%")
        logger.info(f"Execution time: {total_time:.2f} seconds")
        logger.info(f"Overall status: {self.validation_results['validation_summary']['overall_status']}")
        
        quality_metrics = self.validation_results.get('quality_metrics', {})
        if quality_metrics:
            overall_quality = quality_metrics.get('overall_structural_quality', 0)
            quality_grade = quality_metrics.get('structural_quality_grade', 'N/A')
            logger.info(f"Structural Quality Score: {overall_quality:.1f}/100 (Grade: {quality_grade})")
        
        return self.validation_results['validation_summary']['overall_status'] == 'PASSED'

def main():
    """Main execution function."""
    try:
        validator = StructuralValidator()
        success = validator.run_comprehensive_validation()
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in main: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main())