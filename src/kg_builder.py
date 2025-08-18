"""
Knowledge Graph Builder for Gene Ontology Biological Process data.

This module builds a knowledge graph using either NetworkX (for local analysis)
or Neo4j (for persistent storage and complex queries).
"""

import networkx as nx
from typing import Dict, List, Optional
import logging
from pathlib import Path
import json
import pickle

from data_parsers import GOBPDataParser

logger = logging.getLogger(__name__)


class GOBPKnowledgeGraph:
    """Knowledge Graph for Gene Ontology Biological Process data."""
    
    def __init__(self, use_neo4j: bool = False):
        """
        Initialize the knowledge graph.
        
        Args:
            use_neo4j: Whether to use Neo4j database or NetworkX
        """
        self.use_neo4j = use_neo4j
        self.graph = nx.MultiDiGraph()  # Directed graph allowing multiple edges
        self.parser = None
        self.stats = {}
        
        if use_neo4j:
            try:
                from neo4j import GraphDatabase
                self.neo4j_driver = None
                logger.info("Neo4j driver available")
            except ImportError:
                logger.warning("Neo4j driver not available, falling back to NetworkX")
                self.use_neo4j = False
    
    def load_data(self, data_dir: str):
        """
        Load and parse GO_BP data.
        
        Args:
            data_dir: Path to GO_BP data directory
        """
        logger.info(f"Loading data from {data_dir}")
        self.parser = GOBPDataParser(data_dir)
        
        # Parse all data components
        self.go_terms = self.parser.parse_go_terms()
        self.go_relationships = self.parser.parse_go_relationships()
        self.gene_associations = self.parser.parse_gene_go_associations_from_gaf()
        self.go_clusters = self.parser.parse_go_term_clustering()
        
        logger.info("Data loading complete")
    
    def build_graph(self):
        """Build the knowledge graph from parsed data."""
        logger.info("Building knowledge graph...")
        
        # Add GO term nodes
        self._add_go_term_nodes()
        
        # Add gene nodes
        self._add_gene_nodes()
        
        # Add GO-GO relationships
        self._add_go_relationships()
        
        # Add gene-GO associations
        self._add_gene_associations()
        
        # Calculate statistics
        self._calculate_stats()
        
        logger.info("Knowledge graph construction complete")
    
    def _add_go_term_nodes(self):
        """Add GO term nodes to the graph."""
        logger.info("Adding GO term nodes...")
        
        for go_id, go_info in self.go_terms.items():
            self.graph.add_node(
                go_id,
                node_type='go_term',
                name=go_info['name'],
                namespace=go_info['namespace'],
                description=go_info.get('description', '')
            )
        
        logger.info(f"Added {len(self.go_terms)} GO term nodes")
    
    def _add_gene_nodes(self):
        """Add gene nodes to the graph."""
        logger.info("Adding gene nodes...")
        
        # Collect unique genes from associations
        genes = {}
        for assoc in self.gene_associations:
            gene_symbol = assoc['gene_symbol']
            if gene_symbol not in genes:
                genes[gene_symbol] = {
                    'uniprot_id': assoc['uniprot_id'],
                    'gene_name': assoc['gene_name'],
                    'gene_type': assoc['gene_type'],
                    'taxon': assoc['taxon']
                }
        
        # Add gene nodes
        for gene_symbol, gene_info in genes.items():
            self.graph.add_node(
                gene_symbol,
                node_type='gene',
                uniprot_id=gene_info['uniprot_id'],
                gene_name=gene_info['gene_name'],
                gene_type=gene_info['gene_type'],
                taxon=gene_info['taxon']
            )
        
        logger.info(f"Added {len(genes)} gene nodes")
    
    def _add_go_relationships(self):
        """Add GO-GO relationships to the graph."""
        logger.info("Adding GO-GO relationships...")
        
        relationship_count = 0
        for rel in self.go_relationships:
            parent_id = rel['parent_id']
            child_id = rel['child_id']
            rel_type = rel['relationship_type']
            
            # Only add if both nodes exist
            if parent_id in self.graph and child_id in self.graph:
                self.graph.add_edge(
                    child_id,
                    parent_id,
                    edge_type='go_hierarchy',
                    relationship_type=rel_type,
                    namespace=rel['namespace']
                )
                relationship_count += 1
        
        logger.info(f"Added {relationship_count} GO-GO relationships")
    
    def _add_gene_associations(self):
        """Add gene-GO associations to the graph."""
        logger.info("Adding gene-GO associations...")
        
        association_count = 0
        for assoc in self.gene_associations:
            gene_symbol = assoc['gene_symbol']
            go_id = assoc['go_id']
            
            # Only add if both nodes exist
            if gene_symbol in self.graph and go_id in self.graph:
                self.graph.add_edge(
                    gene_symbol,
                    go_id,
                    edge_type='gene_annotation',
                    evidence_code=assoc['evidence_code'],
                    qualifier=assoc['qualifier'],
                    assigned_by=assoc['assigned_by'],
                    date=assoc['date']
                )
                association_count += 1
        
        logger.info(f"Added {association_count} gene-GO associations")
    
    def _calculate_stats(self):
        """Calculate graph statistics."""
        self.stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'go_terms': len([n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'go_term']),
            'genes': len([n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'gene']),
            'go_relationships': len([e for e in self.graph.edges(data=True) if e[2].get('edge_type') == 'go_hierarchy']),
            'gene_associations': len([e for e in self.graph.edges(data=True) if e[2].get('edge_type') == 'gene_annotation'])
        }
        
        logger.info(f"Graph statistics: {self.stats}")
    
    def get_stats(self) -> Dict:
        """Get graph statistics."""
        return self.stats.copy()
    
    def save_graph(self, filepath: str):
        """
        Save the NetworkX graph to disk.
        
        Args:
            filepath: Path to save the graph
        """
        logger.info(f"Saving graph to {filepath}")
        
        # Save as GraphML for interoperability
        if filepath.endswith('.graphml'):
            nx.write_graphml(self.graph, filepath)
        elif filepath.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(self.graph, f)
        else:
            # Default to pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self.graph, f)
        
        logger.info("Graph saved successfully")
    
    def load_graph(self, filepath: str):
        """
        Load a NetworkX graph from disk.
        
        Args:
            filepath: Path to load the graph from
        """
        logger.info(f"Loading graph from {filepath}")
        
        if filepath.endswith('.graphml'):
            self.graph = nx.read_graphml(filepath)
        elif filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                self.graph = pickle.load(f)
        else:
            # Try pickle first
            with open(filepath, 'rb') as f:
                self.graph = pickle.load(f)
        
        # Recalculate stats
        self._calculate_stats()
        logger.info("Graph loaded successfully")
    
    def query_gene_functions(self, gene_symbol: str) -> List[Dict]:
        """
        Query GO terms associated with a gene.
        
        Args:
            gene_symbol: Gene symbol to query
            
        Returns:
            List of GO terms with details
        """
        if gene_symbol not in self.graph:
            return []
        
        results = []
        for neighbor in self.graph.neighbors(gene_symbol):
            edge_data = self.graph[gene_symbol][neighbor]
            
            # Handle multiple edges between same nodes
            for edge_key, edge_attrs in edge_data.items():
                if edge_attrs.get('edge_type') == 'gene_annotation':
                    node_data = self.graph.nodes[neighbor]
                    results.append({
                        'go_id': neighbor,
                        'go_name': node_data.get('name', ''),
                        'evidence_code': edge_attrs.get('evidence_code', ''),
                        'qualifier': edge_attrs.get('qualifier', '')
                    })
        
        return results
    
    def query_go_term_genes(self, go_id: str) -> List[Dict]:
        """
        Query genes associated with a GO term.
        
        Args:
            go_id: GO term ID to query
            
        Returns:
            List of genes with details
        """
        if go_id not in self.graph:
            return []
        
        results = []
        # Look for incoming edges (genes pointing to GO terms)
        for predecessor in self.graph.predecessors(go_id):
            pred_data = self.graph.nodes[predecessor]
            if pred_data.get('node_type') == 'gene':
                edge_data = self.graph[predecessor][go_id]
                
                # Handle multiple edges
                for edge_key, edge_attrs in edge_data.items():
                    if edge_attrs.get('edge_type') == 'gene_annotation':
                        results.append({
                            'gene_symbol': predecessor,
                            'gene_name': pred_data.get('gene_name', ''),
                            'evidence_code': edge_attrs.get('evidence_code', ''),
                            'qualifier': edge_attrs.get('qualifier', '')
                        })
        
        return results
    
    def query_go_hierarchy(self, go_id: str, direction: str = 'children') -> List[Dict]:
        """
        Query GO term hierarchy.
        
        Args:
            go_id: GO term ID to query
            direction: 'children' for child terms, 'parents' for parent terms
            
        Returns:
            List of related GO terms
        """
        if go_id not in self.graph:
            return []
        
        results = []
        
        if direction == 'children':
            # Look for nodes that point to this GO term
            for predecessor in self.graph.predecessors(go_id):
                pred_data = self.graph.nodes[predecessor]
                if pred_data.get('node_type') == 'go_term':
                    edge_data = self.graph[predecessor][go_id]
                    
                    for edge_key, edge_attrs in edge_data.items():
                        if edge_attrs.get('edge_type') == 'go_hierarchy':
                            results.append({
                                'go_id': predecessor,
                                'go_name': pred_data.get('name', ''),
                                'relationship_type': edge_attrs.get('relationship_type', '')
                            })
        
        elif direction == 'parents':
            # Look for nodes this GO term points to
            for successor in self.graph.successors(go_id):
                succ_data = self.graph.nodes[successor]
                if succ_data.get('node_type') == 'go_term':
                    edge_data = self.graph[go_id][successor]
                    
                    for edge_key, edge_attrs in edge_data.items():
                        if edge_attrs.get('edge_type') == 'go_hierarchy':
                            results.append({
                                'go_id': successor,
                                'go_name': succ_data.get('name', ''),
                                'relationship_type': edge_attrs.get('relationship_type', '')
                            })
        
        return results


def main():
    """Example usage of the GO_BP knowledge graph."""
    
    # Build knowledge graph
    kg = GOBPKnowledgeGraph(use_neo4j=False)
    
    # Load data
    data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data/GO_BP"
    kg.load_data(data_dir)
    
    # Build graph
    kg.build_graph()
    
    # Print statistics
    stats = kg.get_stats()
    print("Knowledge Graph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    
    # Example queries
    print("\n" + "="*50)
    print("EXAMPLE QUERIES")
    print("="*50)
    
    # Query 1: What GO terms are associated with TP53?
    gene = "TP53"
    functions = kg.query_gene_functions(gene)
    print(f"\nGO terms for {gene} (showing first 5):")
    for func in functions[:5]:
        print(f"  {func['go_id']}: {func['go_name']} (Evidence: {func['evidence_code']})")
    
    # Query 2: What genes are associated with apoptosis?
    apoptosis_terms = [go_id for go_id, info in kg.go_terms.items() 
                      if 'apoptosis' in info['name'].lower()]
    if apoptosis_terms:
        go_term = apoptosis_terms[0]
        genes = kg.query_go_term_genes(go_term)
        print(f"\nGenes associated with '{kg.go_terms[go_term]['name']}' (showing first 5):")
        for gene_info in genes[:5]:
            print(f"  {gene_info['gene_symbol']}: {gene_info['gene_name']}")
    
    # Query 3: GO term hierarchy
    if apoptosis_terms:
        go_term = apoptosis_terms[0]
        parents = kg.query_go_hierarchy(go_term, 'parents')
        print(f"\nParent terms of '{kg.go_terms[go_term]['name']}' (showing first 3):")
        for parent in parents[:3]:
            print(f"  {parent['go_id']}: {parent['go_name']} ({parent['relationship_type']})")
    
    # Save the graph
    output_path = "/home/mreddy1/knowledge_graph/data/go_bp_kg.pkl"
    kg.save_graph(output_path)
    print(f"\nKnowledge graph saved to: {output_path}")


if __name__ == "__main__":
    main()