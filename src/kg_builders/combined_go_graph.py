"""
Combined Knowledge Graph for multiple GO namespaces (GO_BP + GO_CC + GO_MF).

Extracted from kg_builder.py for better modularity and maintainability.
"""

import networkx as nx
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

from .go_knowledge_graph import GOKnowledgeGraph
from .shared_utils import save_graph_to_file

logger = logging.getLogger(__name__)


class CombinedGOKnowledgeGraph:
    """Combined Knowledge Graph for multiple GO namespaces (GO_BP + GO_CC + GO_MF)."""
    
    def __init__(self, use_neo4j: bool = False):
        """
        Initialize the combined knowledge graph.
        
        Args:
            use_neo4j: Whether to use Neo4j database or NetworkX
        """
        self.use_neo4j = use_neo4j
        self.graph = nx.MultiDiGraph()
        self.parsers = {}
        self.individual_graphs = {}
        self.combined_stats = {}
        
        if use_neo4j:
            try:
                from neo4j import GraphDatabase
                self.neo4j_driver = None
                logger.info("Neo4j driver available")
            except ImportError:
                logger.warning("Neo4j driver not available, falling back to NetworkX")
                self.use_neo4j = False
    
    def load_data(self, base_data_dir: str):
        """
        Load and parse data from multiple GO namespaces.
        
        Args:
            base_data_dir: Base directory containing GO_BP, GO_CC, GO_MF subdirectories
        """
        logger.info(f"Loading combined GO data from {base_data_dir}")
        
        base_path = Path(base_data_dir)
        namespace_dirs = {
            'biological_process': 'GO_BP',
            'cellular_component': 'GO_CC',
            'molecular_function': 'GO_MF'
        }
        
        for namespace, dir_name in namespace_dirs.items():
            data_dir = base_path / dir_name
            if data_dir.exists():
                logger.info(f"Loading {namespace} data...")
                kg = GOKnowledgeGraph(use_neo4j=False, namespace=namespace)
                kg.load_data(str(data_dir))
                kg.build_graph()
                
                self.individual_graphs[namespace] = kg
                logger.info(f"Loaded {namespace}: {kg.get_stats()['total_nodes']} nodes, {kg.get_stats()['total_edges']} edges")
            else:
                logger.warning(f"Directory not found: {data_dir}")
        
        logger.info("Combined data loading complete")
    
    def build_combined_graph(self):
        """Build a single combined graph from all loaded namespaces."""
        logger.info("Building combined knowledge graph...")
        
        total_nodes_added = 0
        total_edges_added = 0
        
        for namespace, kg in self.individual_graphs.items():
            logger.info(f"Merging {namespace} graph...")
            
            # Add all nodes from this graph
            for node_id, node_data in kg.graph.nodes(data=True):
                if node_id not in self.graph:
                    self.graph.add_node(node_id, **node_data)
                    total_nodes_added += 1
                else:
                    # Merge node attributes if node exists
                    existing_data = self.graph.nodes[node_id]
                    merged_data = {**existing_data, **node_data}
                    # Merge sources if both have them
                    if 'sources' in existing_data and 'sources' in node_data:
                        merged_sources = list(set(existing_data['sources'] + node_data['sources']))
                        merged_data['sources'] = merged_sources
                    self.graph.add_node(node_id, **merged_data)
            
            # Add all edges from this graph
            for source, target, edge_data in kg.graph.edges(data=True):
                self.graph.add_edge(source, target, **edge_data)
                total_edges_added += 1
        
        logger.info(f"Combined graph built: {total_nodes_added} nodes, {total_edges_added} edges added")
        self._calculate_combined_stats()
        self._validate_combined_graph()
    
    def _calculate_combined_stats(self):
        """Calculate statistics for the combined graph."""
        go_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'go_term']
        gene_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'gene']
        gene_id_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'gene_identifier']
        
        # Count by namespace
        namespace_counts = {}
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('node_type') == 'go_term':
                ns = node_data.get('namespace', 'unknown')
                namespace_counts[ns] = namespace_counts.get(ns, 0) + 1
        
        edges = list(self.graph.edges(data=True))
        go_relationships = len([e for e in edges if e[2].get('edge_type') == 'go_hierarchy'])
        gene_associations = len([e for e in edges if e[2].get('edge_type') == 'gene_annotation'])
        
        self.combined_stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'go_terms': len(go_nodes),
            'genes': len(gene_nodes),
            'gene_identifiers': len(gene_id_nodes),
            'go_relationships': go_relationships,
            'gene_associations': gene_associations,
            'namespace_counts': namespace_counts,
            'individual_stats': {ns: kg.get_stats() for ns, kg in self.individual_graphs.items()}
        }
        
        logger.info(f"Combined graph statistics: {self.combined_stats}")
    
    def _validate_combined_graph(self):
        """Validate the combined graph integrity."""
        logger.info("Validating combined graph integrity...")
        
        validation = {
            'has_nodes': self.graph.number_of_nodes() > 0,
            'has_edges': self.graph.number_of_edges() > 0,
            'multiple_namespaces': len(set(d.get('namespace') for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'go_term')) > 1
        }
        
        if all(validation.values()):
            logger.info("✅ Combined graph validation passed")
        else:
            logger.warning(f"⚠️ Combined graph validation issues: {validation}")
        
        return validation
    
    def get_combined_stats(self) -> Dict:
        """Get combined graph statistics."""
        return self.combined_stats.copy()
    
    def query_gene_functions_all_namespaces(self, gene_symbol: str) -> Dict[str, List[Dict]]:
        """Query GO terms across all namespaces for a gene."""
        results = {}
        
        for namespace, kg in self.individual_graphs.items():
            functions = kg.query_gene_functions(gene_symbol)
            if functions:
                results[namespace] = functions
        
        return results
    
    def save_combined_graph(self, filepath: str):
        """Save the combined graph to disk."""
        save_graph_to_file(self.graph, filepath)