"""
Shared utilities for Knowledge Graph builders.

Common functionality extracted from kg_builder.py to reduce code duplication
across different knowledge graph implementations.
"""

import networkx as nx
import pickle
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def save_graph_to_file(graph: nx.MultiDiGraph, filepath: str) -> None:
    """
    Save a NetworkX graph to disk.
    
    Args:
        graph: NetworkX graph to save
        filepath: Path to save the graph (supports .graphml and .pkl formats)
    """
    logger.info(f"Saving graph to {filepath}")
    
    if filepath.endswith('.graphml'):
        nx.write_graphml(graph, filepath)
    elif filepath.endswith('.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(graph, f)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(graph, f)
    
    logger.info("Graph saved successfully")


def load_graph_from_file(filepath: str) -> nx.MultiDiGraph:
    """
    Load a NetworkX graph from disk.
    
    Args:
        filepath: Path to load the graph from
        
    Returns:
        Loaded NetworkX graph
    """
    logger.info(f"Loading graph from {filepath}")
    
    if filepath.endswith('.graphml'):
        graph = nx.read_graphml(filepath)
    elif filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            graph = pickle.load(f)
    else:
        with open(filepath, 'rb') as f:
            graph = pickle.load(f)
    
    logger.info("Graph loaded successfully")
    return graph


def initialize_graph_attributes(obj: Any) -> None:
    """
    Initialize common graph attributes for knowledge graph objects.
    
    Args:
        obj: Knowledge graph object to initialize
    """
    if not hasattr(obj, 'gene_id_mappings'):
        obj.gene_id_mappings = {}
    if not hasattr(obj, 'go_terms'):
        obj.go_terms = {}
    if not hasattr(obj, 'go_alt_ids'):
        obj.go_alt_ids = {}


def validate_basic_graph_structure(graph: nx.MultiDiGraph) -> Dict[str, bool]:
    """
    Perform basic validation on graph structure.
    
    Args:
        graph: NetworkX graph to validate
        
    Returns:
        Dictionary with basic validation results
    """
    validation = {
        'has_nodes': graph.number_of_nodes() > 0,
        'has_edges': graph.number_of_edges() > 0,
        'is_multigraph': isinstance(graph, nx.MultiDiGraph)
    }
    
    return validation


def count_nodes_by_type(graph: nx.MultiDiGraph) -> Dict[str, int]:
    """
    Count nodes by their type attribute.
    
    Args:
        graph: NetworkX graph to analyze
        
    Returns:
        Dictionary mapping node types to counts
    """
    node_counts = {}
    for node_id, node_data in graph.nodes(data=True):
        node_type = node_data.get('node_type', 'unknown')
        node_counts[node_type] = node_counts.get(node_type, 0) + 1
    
    return node_counts


def count_edges_by_type(graph: nx.MultiDiGraph) -> Dict[str, int]:
    """
    Count edges by their type attribute.
    
    Args:
        graph: NetworkX graph to analyze
        
    Returns:
        Dictionary mapping edge types to counts
    """
    edge_counts = {}
    for source, target, edge_data in graph.edges(data=True):
        edge_type = edge_data.get('edge_type', 'unknown')
        edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1
    
    return edge_counts


def get_basic_graph_stats(graph: nx.MultiDiGraph) -> Dict[str, Any]:
    """
    Get basic statistics for a knowledge graph.
    
    Args:
        graph: NetworkX graph to analyze
        
    Returns:
        Dictionary with basic graph statistics
    """
    return {
        'total_nodes': graph.number_of_nodes(),
        'total_edges': graph.number_of_edges(),
        'node_counts': count_nodes_by_type(graph),
        'edge_counts': count_edges_by_type(graph)
    }