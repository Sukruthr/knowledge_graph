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
        Load and parse GO_BP data comprehensively.
        
        Args:
            data_dir: Path to GO_BP data directory
        """
        logger.info(f"Loading data from {data_dir}")
        self.parser = GOBPDataParser(data_dir)
        
        # Parse all data components (basic)
        self.go_terms = self.parser.parse_go_terms()
        self.go_relationships = self.parser.parse_go_relationships()
        self.gene_associations = self.parser.parse_gene_go_associations_from_gaf()
        
        # Parse enhanced data components
        self.go_alt_ids = self.parser.parse_go_alternative_ids()
        self.gene_id_mappings = self.parser.parse_gene_identifier_mappings()
        self.obo_terms = self.parser.parse_obo_ontology()
        
        # Parse collapsed GO data comprehensively
        self.collapsed_symbol = self.parser.parse_collapsed_go_file('symbol')
        self.collapsed_entrez = self.parser.parse_collapsed_go_file('entrez')
        self.collapsed_uniprot = self.parser.parse_collapsed_go_file('uniprot')
        self.all_collapsed_associations = self.parser.parse_all_gene_associations_from_collapsed_files()
        
        # Extract GO clusters (backward compatibility)
        self.go_clusters = self.collapsed_symbol['clusters']
        
        logger.info("Comprehensive data loading complete")
    
    def build_graph(self):
        """Build the comprehensive knowledge graph from parsed data."""
        logger.info("Building comprehensive knowledge graph...")
        
        # Add GO term nodes (with OBO enhancement)
        self._add_go_term_nodes()
        
        # Add comprehensive gene nodes (all identifier types)
        self._add_comprehensive_gene_nodes()
        
        # Add GO-GO hierarchical relationships
        self._add_go_relationships()
        
        # Add GO clustering relationships
        self._add_go_clusters()
        
        # Add comprehensive gene-GO associations
        self._add_comprehensive_gene_associations()
        
        # Add gene cross-reference edges
        self._add_gene_cross_references()
        
        # Add alternative GO ID mappings
        self._add_alternative_id_mappings()
        
        # Calculate enhanced statistics
        self._calculate_stats()
        
        # Validate graph integrity
        self.validate_graph_integrity()
        
        logger.info("Comprehensive knowledge graph construction complete")
    
    def _add_go_term_nodes(self):
        """Add enriched GO term nodes to the graph."""
        logger.info("Adding enriched GO term nodes...")
        
        enhanced_count = 0
        for go_id, go_info in self.go_terms.items():
            # Start with basic info
            node_attrs = {
                'node_type': 'go_term',
                'name': go_info['name'],
                'namespace': go_info['namespace'],
                'description': go_info.get('description', '')
            }
            
            # Enhance with OBO data if available
            if go_id in self.obo_terms:
                obo_info = self.obo_terms[go_id]
                node_attrs.update({
                    'definition': obo_info.get('definition', ''),
                    'synonyms': obo_info.get('synonyms', []),
                    'is_obsolete': obo_info.get('is_obsolete', False)
                })
                enhanced_count += 1
            
            # Check if this is an alternative ID
            is_alternative = go_id in self.go_alt_ids
            if is_alternative:
                node_attrs['is_alternative_id'] = True
                node_attrs['primary_id'] = self.go_alt_ids[go_id]
            
            self.graph.add_node(go_id, **node_attrs)
        
        logger.info(f"Added {len(self.go_terms)} GO term nodes ({enhanced_count} with OBO enrichment)")
    
    def _add_comprehensive_gene_nodes(self):
        """Add comprehensive gene nodes with all identifier types."""
        logger.info("Adding comprehensive gene nodes...")
        
        # Collect unique genes from all sources
        genes = {}
        
        # From GAF associations (primary source)
        for assoc in self.gene_associations:
            gene_symbol = assoc['gene_symbol']
            if gene_symbol not in genes:
                genes[gene_symbol] = {
                    'gene_symbol': gene_symbol,
                    'uniprot_id': assoc['uniprot_id'],
                    'gene_name': assoc['gene_name'],
                    'gene_type': assoc['gene_type'],
                    'taxon': assoc['taxon'],
                    'sources': ['gaf']
                }
            else:
                if 'gaf' not in genes[gene_symbol]['sources']:
                    genes[gene_symbol]['sources'].append('gaf')
        
        # Enhance with collapsed file data
        for id_type, associations in self.all_collapsed_associations.items():
            for assoc in associations:
                gene_id = assoc['gene_id']
                
                # For symbol type, merge directly
                if id_type == 'symbol':
                    if gene_id not in genes:
                        genes[gene_id] = {
                            'gene_symbol': gene_id,
                            'sources': ['collapsed_symbol']
                        }
                    else:
                        if 'collapsed_symbol' not in genes[gene_id]['sources']:
                            genes[gene_id]['sources'].append('collapsed_symbol')
                
                # For entrez and uniprot, use mappings to find corresponding symbols
                elif id_type == 'entrez':
                    # Try to find corresponding symbol
                    symbol = self.gene_id_mappings.get('entrez_to_symbol', {}).get(gene_id)
                    if symbol and symbol in genes:
                        genes[symbol]['entrez_id'] = gene_id
                        if 'collapsed_entrez' not in genes[symbol]['sources']:
                            genes[symbol]['sources'].append('collapsed_entrez')
                
                elif id_type == 'uniprot':
                    # Try to find corresponding symbol
                    symbol = self.gene_id_mappings.get('uniprot_to_symbol', {}).get(gene_id)
                    if symbol and symbol in genes:
                        genes[symbol]['mapped_uniprot_id'] = gene_id
                        if 'collapsed_uniprot' not in genes[symbol]['sources']:
                            genes[symbol]['sources'].append('collapsed_uniprot')
        
        # Add comprehensive gene nodes
        for gene_symbol, gene_info in genes.items():
            node_attrs = {
                'node_type': 'gene',
                'gene_symbol': gene_symbol,
                'sources': gene_info.get('sources', [])
            }
            
            # Add basic info if available
            for attr in ['uniprot_id', 'gene_name', 'gene_type', 'taxon']:
                if attr in gene_info:
                    node_attrs[attr] = gene_info[attr]
            
            # Add cross-reference IDs
            if gene_symbol in self.gene_id_mappings.get('symbol_to_entrez', {}):
                node_attrs['entrez_id'] = self.gene_id_mappings['symbol_to_entrez'][gene_symbol]
            
            if gene_symbol in self.gene_id_mappings.get('symbol_to_uniprot', {}):
                node_attrs['cross_ref_uniprot'] = self.gene_id_mappings['symbol_to_uniprot'][gene_symbol]
            
            # Add mapped IDs from collapsed files
            for attr in ['entrez_id', 'mapped_uniprot_id']:
                if attr in gene_info:
                    node_attrs[attr] = gene_info[attr]
            
            self.graph.add_node(gene_symbol, **node_attrs)
        
        logger.info(f"Added {len(genes)} comprehensive gene nodes")
    
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
    
    def _add_comprehensive_gene_associations(self):
        """Add comprehensive gene-GO associations from all sources."""
        logger.info("Adding comprehensive gene-GO associations...")
        
        gaf_count = 0
        collapsed_count = 0
        
        # Add GAF associations (high-quality with evidence codes)
        for assoc in self.gene_associations:
            gene_symbol = assoc['gene_symbol']
            go_id = assoc['go_id']
            
            # Only add if both nodes exist
            if gene_symbol in self.graph and go_id in self.graph:
                self.graph.add_edge(
                    gene_symbol,
                    go_id,
                    edge_type='gene_annotation',
                    source='gaf',
                    evidence_code=assoc['evidence_code'],
                    qualifier=assoc['qualifier'],
                    assigned_by=assoc['assigned_by'],
                    date=assoc['date'],
                    database=assoc['database']
                )
                gaf_count += 1
        
        # Add collapsed file associations (additional coverage)
        added_collapsed = set()  # Track to avoid duplicates
        
        for id_type, associations in self.all_collapsed_associations.items():
            for assoc in associations:
                go_id = assoc['go_id']
                gene_id = assoc['gene_id']
                
                # Map to gene symbol for graph consistency
                if id_type == 'symbol':
                    gene_symbol = gene_id
                elif id_type == 'entrez':
                    gene_symbol = self.gene_id_mappings.get('entrez_to_symbol', {}).get(gene_id)
                elif id_type == 'uniprot':
                    gene_symbol = self.gene_id_mappings.get('uniprot_to_symbol', {}).get(gene_id)
                else:
                    continue
                
                # Only add if both nodes exist and not already added
                edge_key = (gene_symbol, go_id, id_type)
                if (gene_symbol and gene_symbol in self.graph and go_id in self.graph and 
                    edge_key not in added_collapsed):
                    
                    self.graph.add_edge(
                        gene_symbol,
                        go_id,
                        edge_type='gene_annotation',
                        source=f'collapsed_{id_type}',
                        identifier_type=id_type,
                        original_gene_id=gene_id
                    )
                    collapsed_count += 1
                    added_collapsed.add(edge_key)
        
        logger.info(f"Added {gaf_count} GAF associations and {collapsed_count} collapsed file associations")
    
    def _add_alternative_id_mappings(self):
        """Add alternative GO ID mappings as edges."""
        logger.info("Adding alternative GO ID mappings...")
        
        mapping_count = 0
        for alt_id, primary_id in self.go_alt_ids.items():
            # Only add if both nodes exist
            if alt_id in self.graph and primary_id in self.graph:
                self.graph.add_edge(
                    alt_id,
                    primary_id,
                    edge_type='alternative_id_mapping',
                    relationship_type='alternative_to'
                )
                mapping_count += 1
        
        logger.info(f"Added {mapping_count} alternative GO ID mappings")
    
    def _add_go_clusters(self):
        """Add GO clustering relationships from collapsed_go files."""
        logger.info("Adding GO clustering relationships...")
        
        cluster_count = 0
        for parent_go, children in self.go_clusters.items():
            # Only add if parent GO exists
            if parent_go in self.graph:
                for child_info in children:
                    child_go = child_info['child_go']
                    cluster_type = child_info['cluster_type']
                    
                    # Only add if child GO exists
                    if child_go in self.graph:
                        self.graph.add_edge(
                            parent_go,
                            child_go,
                            edge_type='go_clustering',
                            cluster_type=cluster_type,
                            relationship_type='clusters'
                        )
                        cluster_count += 1
        
        logger.info(f"Added {cluster_count} GO clustering relationships")
    
    def _add_gene_cross_references(self):
        """Add gene cross-reference edges between different identifier types."""
        logger.info("Adding gene cross-reference edges...")
        
        cross_ref_count = 0
        
        # Add symbol-entrez cross-references
        for symbol, entrez in self.gene_id_mappings.get('symbol_to_entrez', {}).items():
            if symbol in self.graph:
                # Create virtual entrez node if it doesn't exist
                entrez_node_id = f"ENTREZ:{entrez}"
                if entrez_node_id not in self.graph:
                    self.graph.add_node(entrez_node_id, 
                                       node_type='gene_identifier',
                                       identifier_type='entrez',
                                       entrez_id=entrez,
                                       gene_symbol=symbol)
                
                self.graph.add_edge(
                    symbol,
                    entrez_node_id,
                    edge_type='gene_cross_reference',
                    reference_type='symbol_to_entrez'
                )
                cross_ref_count += 1
        
        # Add symbol-uniprot cross-references
        for symbol, uniprot in self.gene_id_mappings.get('symbol_to_uniprot', {}).items():
            if symbol in self.graph:
                # Create virtual uniprot node if it doesn't exist
                uniprot_node_id = f"UNIPROT:{uniprot}"
                if uniprot_node_id not in self.graph:
                    self.graph.add_node(uniprot_node_id,
                                       node_type='gene_identifier', 
                                       identifier_type='uniprot',
                                       uniprot_id=uniprot,
                                       gene_symbol=symbol)
                
                self.graph.add_edge(
                    symbol,
                    uniprot_node_id,
                    edge_type='gene_cross_reference',
                    reference_type='symbol_to_uniprot'
                )
                cross_ref_count += 1
        
        # Add entrez-uniprot cross-references
        for entrez, uniprot in self.gene_id_mappings.get('entrez_to_uniprot', {}).items():
            entrez_node_id = f"ENTREZ:{entrez}"
            uniprot_node_id = f"UNIPROT:{uniprot}"
            
            if entrez_node_id in self.graph and uniprot_node_id in self.graph:
                self.graph.add_edge(
                    entrez_node_id,
                    uniprot_node_id,
                    edge_type='gene_cross_reference',
                    reference_type='entrez_to_uniprot'
                )
                cross_ref_count += 1
        
        logger.info(f"Added {cross_ref_count} gene cross-reference edges")
    
    def _calculate_stats(self):
        """Calculate comprehensive graph statistics."""
        # Count nodes by type
        go_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'go_term']
        gene_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'gene']
        gene_id_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'gene_identifier']
        
        # Count enhanced GO terms
        enhanced_go_terms = len([n for n, d in self.graph.nodes(data=True) 
                                if d.get('node_type') == 'go_term' and d.get('definition')])
        
        # Count alternative GO IDs
        alternative_go_ids = len([n for n, d in self.graph.nodes(data=True) 
                                 if d.get('is_alternative_id')])
        
        # Count edges by type
        edges = list(self.graph.edges(data=True))
        go_relationships = len([e for e in edges if e[2].get('edge_type') == 'go_hierarchy'])
        go_clusters = len([e for e in edges if e[2].get('edge_type') == 'go_clustering'])
        gene_associations = len([e for e in edges if e[2].get('edge_type') == 'gene_annotation'])
        gene_cross_refs = len([e for e in edges if e[2].get('edge_type') == 'gene_cross_reference'])
        alt_id_mappings = len([e for e in edges if e[2].get('edge_type') == 'alternative_id_mapping'])
        
        # Count gene associations by source
        gaf_associations = len([e for e in edges if e[2].get('edge_type') == 'gene_annotation' and e[2].get('source') == 'gaf'])
        collapsed_associations = len([e for e in edges if e[2].get('edge_type') == 'gene_annotation' and e[2].get('source', '').startswith('collapsed')])
        
        self.stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'go_terms': len(go_nodes),
            'genes': len(gene_nodes),
            'gene_identifiers': len(gene_id_nodes),
            'enhanced_go_terms': enhanced_go_terms,
            'alternative_go_ids': alternative_go_ids,
            'go_relationships': go_relationships,
            'go_clusters': go_clusters,
            'gene_associations': gene_associations,
            'gaf_associations': gaf_associations,
            'collapsed_associations': collapsed_associations,
            'gene_cross_references': gene_cross_refs,
            'alternative_id_mappings': alt_id_mappings,
            'total_gene_id_mappings': sum(len(m) for m in self.gene_id_mappings.values())
        }
        
        logger.info(f"Comprehensive graph statistics: {self.stats}")
    
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
        
        # Initialize missing attributes for loaded graphs
        if not hasattr(self, 'gene_id_mappings'):
            self.gene_id_mappings = {}
        if not hasattr(self, 'go_terms'):
            self.go_terms = {}
        if not hasattr(self, 'go_alt_ids'):
            self.go_alt_ids = {}
        
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
    
    def search_go_terms_by_definition(self, search_term: str) -> List[Dict]:
        """
        Search GO terms by definition or synonym using enhanced OBO data.
        
        Args:
            search_term: Term to search for in definitions and synonyms
            
        Returns:
            List of matching GO terms with relevance score
        """
        search_term = search_term.lower()
        results = []
        
        for go_id, node_data in self.graph.nodes(data=True):
            if node_data.get('node_type') != 'go_term':
                continue
                
            score = 0
            match_types = []
            
            # Check name (highest weight)
            if search_term in node_data.get('name', '').lower():
                score += 10
                match_types.append('name')
            
            # Check definition (medium weight)
            if 'definition' in node_data and search_term in node_data['definition'].lower():
                score += 5
                match_types.append('definition')
            
            # Check synonyms (medium weight)
            synonyms = node_data.get('synonyms', [])
            for synonym in synonyms:
                if search_term in synonym.lower():
                    score += 3
                    match_types.append('synonym')
                    break
            
            if score > 0:
                results.append({
                    'go_id': go_id,
                    'name': node_data.get('name', ''),
                    'definition': node_data.get('definition', '')[:200] + '...' if len(node_data.get('definition', '')) > 200 else node_data.get('definition', ''),
                    'score': score,
                    'match_types': match_types
                })
        
        # Sort by relevance score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def resolve_alternative_go_id(self, go_id: str) -> str:
        """
        Resolve alternative/obsolete GO ID to primary ID.
        
        Args:
            go_id: GO ID to resolve
            
        Returns:
            Primary GO ID or original ID if no mapping exists
        """
        return self.go_alt_ids.get(go_id, go_id)
    
    def get_gene_cross_references(self, gene_symbol: str) -> Dict:
        """
        Get cross-reference information for a gene.
        
        Args:
            gene_symbol: Gene symbol to look up
            
        Returns:
            Dictionary with cross-reference IDs
        """
        cross_refs = {}
        
        if gene_symbol in self.gene_id_mappings.get('symbol_to_uniprot', {}):
            cross_refs['uniprot'] = self.gene_id_mappings['symbol_to_uniprot'][gene_symbol]
        
        # Add node data if gene exists in graph
        if gene_symbol in self.graph:
            node_data = self.graph.nodes[gene_symbol]
            cross_refs.update({
                'gene_name': node_data.get('gene_name', ''),
                'gene_type': node_data.get('gene_type', ''),
                'taxon': node_data.get('taxon', '')
            })
        
        return cross_refs
    
    def validate_graph_integrity(self) -> Dict[str, bool]:
        """Validate the integrity and consistency of the knowledge graph."""
        logger.info("Validating graph integrity...")
        
        validation = {
            'has_nodes': self.graph.number_of_nodes() > 0,
            'has_edges': self.graph.number_of_edges() > 0,
            'go_terms_valid': True,
            'gene_nodes_valid': True,
            'relationships_valid': True,
            'associations_valid': True,
            'cross_references_valid': True
        }
        
        # Validate GO term nodes
        go_nodes_with_issues = 0
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('node_type') == 'go_term':
                if not node_data.get('name') or not node_id.startswith('GO:'):
                    go_nodes_with_issues += 1
        
        validation['go_terms_valid'] = go_nodes_with_issues == 0
        
        # Validate gene nodes
        gene_nodes_with_issues = 0
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('node_type') == 'gene':
                if not node_data.get('gene_symbol'):
                    gene_nodes_with_issues += 1
        
        validation['gene_nodes_valid'] = gene_nodes_with_issues == 0
        
        # Validate relationships
        invalid_relationships = 0
        for source, target, edge_data in self.graph.edges(data=True):
            if edge_data.get('edge_type') == 'go_hierarchy':
                if (source not in self.graph or target not in self.graph or
                    not source.startswith('GO:') or not target.startswith('GO:')):
                    invalid_relationships += 1
        
        validation['relationships_valid'] = invalid_relationships == 0
        
        # Validate associations
        invalid_associations = 0
        for source, target, edge_data in self.graph.edges(data=True):
            if edge_data.get('edge_type') == 'gene_annotation':
                source_data = self.graph.nodes.get(source, {})
                target_data = self.graph.nodes.get(target, {})
                if (source_data.get('node_type') != 'gene' or
                    target_data.get('node_type') != 'go_term'):
                    invalid_associations += 1
        
        validation['associations_valid'] = invalid_associations == 0
        
        # Validate cross-references
        invalid_cross_refs = 0
        for source, target, edge_data in self.graph.edges(data=True):
            if edge_data.get('edge_type') == 'gene_cross_reference':
                if source not in self.graph or target not in self.graph:
                    invalid_cross_refs += 1
        
        validation['cross_references_valid'] = invalid_cross_refs == 0
        
        # Overall validation
        validation['overall_valid'] = all(validation.values())
        
        if validation['overall_valid']:
            logger.info("✅ Graph integrity validation passed")
        else:
            logger.warning(f"⚠️ Graph integrity issues found: {validation}")
        
        return validation


def main():
    """Example usage of the GO_BP knowledge graph."""
    
    # Build knowledge graph
    kg = GOBPKnowledgeGraph(use_neo4j=False)
    
    # Load data
    data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data/GO_BP"
    kg.load_data(data_dir)
    
    # Build graph
    kg.build_graph()
    
    # Print enhanced statistics
    stats = kg.get_stats()
    print("Enhanced Knowledge Graph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    
    # Example queries
    print("\n" + "="*50)
    print("ENHANCED QUERY EXAMPLES")
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
    
    # Enhanced Query 4: Search by definition
    print(f"\n4. Search GO terms containing 'DNA damage':")
    dna_damage_terms = kg.search_go_terms_by_definition('DNA damage')
    for term in dna_damage_terms[:3]:
        print(f"  {term['go_id']}: {term['name']} (Score: {term['score']}, Matches: {', '.join(term['match_types'])})")
    
    # Enhanced Query 5: Resolve alternative GO ID
    if kg.go_alt_ids:
        alt_id = list(kg.go_alt_ids.keys())[0]
        primary_id = kg.resolve_alternative_go_id(alt_id)
        print(f"\n5. Alternative ID resolution:")
        print(f"  {alt_id} -> {primary_id}")
    
    # Enhanced Query 6: Gene cross-references
    if "TP53" in kg.graph:
        cross_refs = kg.get_gene_cross_references("TP53")
        print(f"\n6. TP53 cross-references:")
        for key, value in cross_refs.items():
            print(f"  {key}: {value}")
    
    # Enhanced Query 8: GO clustering relationships
    if kg.go_clusters:
        sample_cluster = list(kg.go_clusters.keys())[0]
        cluster_children = kg.go_clusters[sample_cluster]
        print(f"\n8. GO clustering example - {sample_cluster}:")
        print(f"  Clusters {len(cluster_children)} child GO terms")
        for child in cluster_children[:3]:
            child_go = child['child_go']
            child_name = kg.go_terms.get(child_go, {}).get('name', 'Unknown')
            print(f"    -> {child_go}: {child_name}")
    
    # Enhanced Query 9: Validation results
    validation = kg.validate_graph_integrity()
    print(f"\n9. Graph validation:")
    for check, result in validation.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check}: {result}")
    
    # Enhanced Query 7: Show enriched GO term example
    apoptosis_terms = [go_id for go_id, info in kg.go_terms.items() 
                      if 'apoptosis' in info['name'].lower()]
    if apoptosis_terms:
        go_term = apoptosis_terms[0]
        node_data = kg.graph.nodes[go_term]
        print(f"\n7. Enhanced GO term example - {go_term}:")
        print(f"  Name: {node_data.get('name', 'N/A')}")
        if 'definition' in node_data:
            print(f"  Definition: {node_data['definition'][:150]}...")
        if 'synonyms' in node_data and node_data['synonyms']:
            print(f"  Synonyms: {', '.join(node_data['synonyms'][:3])}")
    
    # Save the comprehensive graph
    output_path = "/home/mreddy1/knowledge_graph/data/go_bp_comprehensive_kg.pkl"
    kg.save_graph(output_path)
    print(f"\nComprehensive knowledge graph saved to: {output_path}")


if __name__ == "__main__":
    main()