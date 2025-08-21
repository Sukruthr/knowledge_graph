"""
Knowledge Graph Builder for Gene Ontology Biological Process data.

This module builds a knowledge graph using either NetworkX (for local analysis)
or Neo4j (for persistent storage and complex queries).
"""

import networkx as nx
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import json
import pickle

try:
    from .data_parsers import GOBPDataParser, GODataParser, CombinedGOParser, OmicsDataParser, CombinedBiomedicalParser
except ImportError:
    from data_parsers import GOBPDataParser, GODataParser, CombinedGOParser, OmicsDataParser, CombinedBiomedicalParser

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


class GOKnowledgeGraph:
    """Generic Knowledge Graph for Gene Ontology data (supports GO_BP, GO_CC, GO_MF)."""
    
    def __init__(self, use_neo4j: bool = False, namespace: str = 'biological_process'):
        """
        Initialize the knowledge graph.
        
        Args:
            use_neo4j: Whether to use Neo4j database or NetworkX
            namespace: GO namespace to build graph for
        """
        self.use_neo4j = use_neo4j
        self.namespace = namespace
        self.graph = nx.MultiDiGraph()
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
        """Load and parse GO data for the specified namespace."""
        logger.info(f"Loading {self.namespace} data from {data_dir}")
        self.parser = GODataParser(data_dir, self.namespace)
        
        # Parse all data components
        self.go_terms = self.parser.parse_go_terms()
        self.go_relationships = self.parser.parse_go_relationships()
        self.gene_associations = self.parser.parse_gene_go_associations_from_gaf()
        self.go_alt_ids = self.parser.parse_go_alternative_ids()
        self.gene_id_mappings = self.parser.parse_gene_identifier_mappings()
        self.obo_terms = self.parser.parse_obo_ontology()
        
        # Parse collapsed GO data comprehensively
        self.collapsed_symbol = self.parser.parse_collapsed_go_file('symbol')
        self.collapsed_entrez = self.parser.parse_collapsed_go_file('entrez')
        self.collapsed_uniprot = self.parser.parse_collapsed_go_file('uniprot')
        self.all_collapsed_associations = self.parser.parse_all_gene_associations_from_collapsed_files()
        
        # Extract GO clusters
        self.go_clusters = self.collapsed_symbol['clusters']
        
        logger.info(f"Comprehensive {self.namespace} data loading complete")
    
    def build_graph(self):
        """Build the knowledge graph from parsed data."""
        logger.info(f"Building {self.namespace} knowledge graph...")
        
        # Use the same methods as GOBPKnowledgeGraph but with generic namespace
        self._add_go_term_nodes()
        self._add_comprehensive_gene_nodes()
        self._add_go_relationships()
        self._add_go_clusters()
        self._add_comprehensive_gene_associations()
        self._add_gene_cross_references()
        self._add_alternative_id_mappings()
        self._calculate_stats()
        self.validate_graph_integrity()
        
        logger.info(f"{self.namespace.title()} knowledge graph construction complete")
    
    # Copy all the private methods from GOBPKnowledgeGraph
    def _add_go_term_nodes(self):
        """Add enriched GO term nodes to the graph."""
        logger.info("Adding enriched GO term nodes...")
        
        enhanced_count = 0
        for go_id, go_info in self.go_terms.items():
            node_attrs = {
                'node_type': 'go_term',
                'name': go_info['name'],
                'namespace': go_info['namespace'],
                'description': go_info.get('description', '')
            }
            
            if go_id in self.obo_terms:
                obo_info = self.obo_terms[go_id]
                node_attrs.update({
                    'definition': obo_info.get('definition', ''),
                    'synonyms': obo_info.get('synonyms', []),
                    'is_obsolete': obo_info.get('is_obsolete', False)
                })
                enhanced_count += 1
            
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
                
                if id_type == 'symbol':
                    if gene_id not in genes:
                        genes[gene_id] = {
                            'gene_symbol': gene_id,
                            'sources': ['collapsed_symbol']
                        }
                    else:
                        if 'collapsed_symbol' not in genes[gene_id]['sources']:
                            genes[gene_id]['sources'].append('collapsed_symbol')
                
                elif id_type == 'entrez':
                    symbol = self.gene_id_mappings.get('entrez_to_symbol', {}).get(gene_id)
                    if symbol and symbol in genes:
                        genes[symbol]['entrez_id'] = gene_id
                        if 'collapsed_entrez' not in genes[symbol]['sources']:
                            genes[symbol]['sources'].append('collapsed_entrez')
                
                elif id_type == 'uniprot':
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
            
            for attr in ['uniprot_id', 'gene_name', 'gene_type', 'taxon']:
                if attr in gene_info:
                    node_attrs[attr] = gene_info[attr]
            
            if gene_symbol in self.gene_id_mappings.get('symbol_to_entrez', {}):
                node_attrs['entrez_id'] = self.gene_id_mappings['symbol_to_entrez'][gene_symbol]
            
            if gene_symbol in self.gene_id_mappings.get('symbol_to_uniprot', {}):
                node_attrs['cross_ref_uniprot'] = self.gene_id_mappings['symbol_to_uniprot'][gene_symbol]
            
            for attr in ['entrez_id', 'mapped_uniprot_id']:
                if attr in gene_info:
                    node_attrs[attr] = gene_info[attr]
            
            self.graph.add_node(gene_symbol, **node_attrs)
        
        logger.info(f"Added {len(genes)} comprehensive gene nodes")
    
    # Continue copying the other private methods...
    def _add_go_relationships(self):
        """Add GO-GO relationships to the graph."""
        logger.info("Adding GO-GO relationships...")
        
        relationship_count = 0
        for rel in self.go_relationships:
            parent_id = rel['parent_id']
            child_id = rel['child_id']
            rel_type = rel['relationship_type']
            
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
        
        # Add GAF associations
        for assoc in self.gene_associations:
            gene_symbol = assoc['gene_symbol']
            go_id = assoc['go_id']
            
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
        
        # Add collapsed file associations
        added_collapsed = set()
        
        for id_type, associations in self.all_collapsed_associations.items():
            for assoc in associations:
                go_id = assoc['go_id']
                gene_id = assoc['gene_id']
                
                if id_type == 'symbol':
                    gene_symbol = gene_id
                elif id_type == 'entrez':
                    gene_symbol = self.gene_id_mappings.get('entrez_to_symbol', {}).get(gene_id)
                elif id_type == 'uniprot':
                    gene_symbol = self.gene_id_mappings.get('uniprot_to_symbol', {}).get(gene_id)
                else:
                    continue
                
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
    
    def _add_go_clusters(self):
        """Add GO clustering relationships from collapsed_go files."""
        logger.info("Adding GO clustering relationships...")
        
        cluster_count = 0
        for parent_go, children in self.go_clusters.items():
            if parent_go in self.graph:
                for child_info in children:
                    child_go = child_info['child_go']
                    cluster_type = child_info['cluster_type']
                    
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
        
        for symbol, entrez in self.gene_id_mappings.get('symbol_to_entrez', {}).items():
            if symbol in self.graph:
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
        
        for symbol, uniprot in self.gene_id_mappings.get('symbol_to_uniprot', {}).items():
            if symbol in self.graph:
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
    
    def _add_alternative_id_mappings(self):
        """Add alternative GO ID mappings as edges."""
        logger.info("Adding alternative GO ID mappings...")
        
        mapping_count = 0
        for alt_id, primary_id in self.go_alt_ids.items():
            if alt_id in self.graph and primary_id in self.graph:
                self.graph.add_edge(
                    alt_id,
                    primary_id,
                    edge_type='alternative_id_mapping',
                    relationship_type='alternative_to'
                )
                mapping_count += 1
        
        logger.info(f"Added {mapping_count} alternative GO ID mappings")
    
    def _calculate_stats(self):
        """Calculate comprehensive graph statistics."""
        go_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'go_term']
        gene_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'gene']
        gene_id_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'gene_identifier']
        
        enhanced_go_terms = len([n for n, d in self.graph.nodes(data=True) 
                                if d.get('node_type') == 'go_term' and d.get('definition')])
        
        alternative_go_ids = len([n for n, d in self.graph.nodes(data=True) 
                                 if d.get('is_alternative_id')])
        
        edges = list(self.graph.edges(data=True))
        go_relationships = len([e for e in edges if e[2].get('edge_type') == 'go_hierarchy'])
        go_clusters = len([e for e in edges if e[2].get('edge_type') == 'go_clustering'])
        gene_associations = len([e for e in edges if e[2].get('edge_type') == 'gene_annotation'])
        gene_cross_refs = len([e for e in edges if e[2].get('edge_type') == 'gene_cross_reference'])
        alt_id_mappings = len([e for e in edges if e[2].get('edge_type') == 'alternative_id_mapping'])
        
        gaf_associations = len([e for e in edges if e[2].get('edge_type') == 'gene_annotation' and e[2].get('source') == 'gaf'])
        collapsed_associations = len([e for e in edges if e[2].get('edge_type') == 'gene_annotation' and e[2].get('source', '').startswith('collapsed')])
        
        self.stats = {
            'namespace': self.namespace,
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
        
        logger.info(f"{self.namespace.title()} graph statistics: {self.stats}")
    
    def get_stats(self) -> Dict:
        """Get graph statistics."""
        return self.stats.copy()
    
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
        
        validation['overall_valid'] = all(validation.values())
        
        if validation['overall_valid']:
            logger.info("✅ Graph integrity validation passed")
        else:
            logger.warning(f"⚠️ Graph integrity issues found: {validation}")
        
        return validation
    
    # Copy query methods from GOBPKnowledgeGraph
    def query_gene_functions(self, gene_symbol: str) -> List[Dict]:
        """Query GO terms associated with a gene."""
        if gene_symbol not in self.graph:
            return []
        
        results = []
        for neighbor in self.graph.neighbors(gene_symbol):
            edge_data = self.graph[gene_symbol][neighbor]
            
            for edge_key, edge_attrs in edge_data.items():
                if edge_attrs.get('edge_type') == 'gene_annotation':
                    node_data = self.graph.nodes[neighbor]
                    results.append({
                        'go_id': neighbor,
                        'go_name': node_data.get('name', ''),
                        'evidence_code': edge_attrs.get('evidence_code', ''),
                        'qualifier': edge_attrs.get('qualifier', ''),
                        'namespace': node_data.get('namespace', self.namespace)
                    })
        
        return results
    
    def query_go_term_genes(self, go_id: str) -> List[Dict]:
        """Query genes associated with a GO term."""
        if go_id not in self.graph:
            return []
        
        results = []
        for predecessor in self.graph.predecessors(go_id):
            pred_data = self.graph.nodes[predecessor]
            if pred_data.get('node_type') == 'gene':
                edge_data = self.graph[predecessor][go_id]
                
                for edge_key, edge_attrs in edge_data.items():
                    if edge_attrs.get('edge_type') == 'gene_annotation':
                        results.append({
                            'gene_symbol': predecessor,
                            'gene_name': pred_data.get('gene_name', ''),
                            'evidence_code': edge_attrs.get('evidence_code', ''),
                            'qualifier': edge_attrs.get('qualifier', '')
                        })
        
        return results
    
    def save_graph(self, filepath: str):
        """Save the NetworkX graph to disk."""
        logger.info(f"Saving graph to {filepath}")
        
        if filepath.endswith('.graphml'):
            nx.write_graphml(self.graph, filepath)
        elif filepath.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(self.graph, f)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(self.graph, f)
        
        logger.info("Graph saved successfully")
    
    def load_graph(self, filepath: str):
        """Load a NetworkX graph from disk."""
        logger.info(f"Loading graph from {filepath}")
        
        if filepath.endswith('.graphml'):
            self.graph = nx.read_graphml(filepath)
        elif filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                self.graph = pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                self.graph = pickle.load(f)
        
        if not hasattr(self, 'gene_id_mappings'):
            self.gene_id_mappings = {}
        if not hasattr(self, 'go_terms'):
            self.go_terms = {}
        if not hasattr(self, 'go_alt_ids'):
            self.go_alt_ids = {}
        
        self._calculate_stats()
        logger.info("Graph loaded successfully")


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
        logger.info(f"Saving combined graph to {filepath}")
        
        if filepath.endswith('.graphml'):
            nx.write_graphml(self.graph, filepath)
        elif filepath.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(self.graph, f)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(self.graph, f)
        
        logger.info("Combined graph saved successfully")


class ComprehensiveBiomedicalKnowledgeGraph:
    """Comprehensive Knowledge Graph integrating GO ontology with Omics data."""
    
    def __init__(self, use_neo4j: bool = False):
        """
        Initialize the comprehensive biomedical knowledge graph.
        
        Args:
            use_neo4j: Whether to use Neo4j database or NetworkX
        """
        self.use_neo4j = use_neo4j
        self.graph = nx.MultiDiGraph()
        self.parser = None
        self.parsed_data = {}
        self.stats = {}
        
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
        Load and parse comprehensive biomedical data (GO + Omics).
        
        Args:
            base_data_dir: Base directory containing GO_BP, GO_CC, GO_MF, and Omics_data subdirectories
        """
        logger.info(f"Loading comprehensive biomedical data from {base_data_dir}")
        
        # Initialize comprehensive parser
        self.parser = CombinedBiomedicalParser(base_data_dir)
        
        # Parse all biomedical data
        self.parsed_data = self.parser.parse_all_biomedical_data()
        
        logger.info("Comprehensive biomedical data loading complete")
    
    def build_comprehensive_graph(self):
        """Build the comprehensive biomedical knowledge graph."""
        logger.info("Building comprehensive biomedical knowledge graph...")
        
        # Build base GO knowledge graph
        self._build_go_component()
        
        # Add Omics data components
        self._add_omics_nodes()
        self._add_omics_associations()
        self._add_cluster_relationships()
        
        # Add enhanced semantic data from Omics_data2
        self._add_semantic_enhancements()
        
        # Add model comparison data integration
        self._add_model_comparison_data()
        
        # Add CC_MF_Branch data integration
        self._add_cc_mf_branch_data()
        
        # Calculate comprehensive statistics
        self._calculate_comprehensive_stats()
        
        # Validate comprehensive graph
        self._validate_comprehensive_graph()
        
        logger.info("Comprehensive biomedical knowledge graph construction complete")
    
    def _build_go_component(self):
        """Build the GO component of the knowledge graph."""
        logger.info("Building GO component...")
        
        if 'go_data' not in self.parsed_data:
            logger.warning("No GO data available for graph construction")
            return
        
        go_data = self.parsed_data['go_data']
        
        # Add GO term nodes from all namespaces
        for namespace, namespace_data in go_data.items():
            logger.info(f"Adding {namespace} GO terms...")
            
            # Add GO term nodes
            for go_id, go_info in namespace_data.get('go_terms', {}).items():
                node_attrs = {
                    'node_type': 'go_term',
                    'name': go_info['name'],
                    'namespace': go_info['namespace'],
                    'description': go_info.get('description', '')
                }
                
                # Enhance with OBO data if available
                obo_terms = namespace_data.get('obo_terms', {})
                if go_id in obo_terms:
                    obo_info = obo_terms[go_id]
                    node_attrs.update({
                        'definition': obo_info.get('definition', ''),
                        'synonyms': obo_info.get('synonyms', []),
                        'is_obsolete': obo_info.get('is_obsolete', False)
                    })
                
                self.graph.add_node(go_id, **node_attrs)
            
            # Add GO relationships
            for rel in namespace_data.get('go_relationships', []):
                parent_id = rel['parent_id']
                child_id = rel['child_id']
                
                if parent_id in self.graph and child_id in self.graph:
                    self.graph.add_edge(
                        child_id,
                        parent_id,
                        edge_type='go_hierarchy',
                        relationship_type=rel['relationship_type'],
                        namespace=rel['namespace']
                    )
            
            # Add gene nodes and associations
            genes_added = set()
            for assoc in namespace_data.get('gene_associations', []):
                gene_symbol = assoc['gene_symbol']
                go_id = assoc['go_id']
                
                # Add gene node if not already added
                if gene_symbol not in genes_added:
                    gene_attrs = {
                        'node_type': 'gene',
                        'gene_symbol': gene_symbol,
                        'uniprot_id': assoc.get('uniprot_id', ''),
                        'gene_name': assoc.get('gene_name', ''),
                        'gene_type': assoc.get('gene_type', ''),
                        'taxon': assoc.get('taxon', ''),
                        'sources': ['gaf']
                    }
                    self.graph.add_node(gene_symbol, **gene_attrs)
                    genes_added.add(gene_symbol)
                
                # Add gene-GO association
                if gene_symbol in self.graph and go_id in self.graph:
                    self.graph.add_edge(
                        gene_symbol,
                        go_id,
                        edge_type='gene_annotation',
                        source='gaf',
                        evidence_code=assoc.get('evidence_code', ''),
                        qualifier=assoc.get('qualifier', ''),
                        namespace=namespace
                    )
    
    def _add_omics_nodes(self):
        """Add Omics-specific nodes (diseases, drugs, viral conditions, clusters)."""
        logger.info("Adding Omics nodes...")
        
        if 'omics_data' not in self.parsed_data:
            logger.warning("No Omics data available")
            return
        
        omics_data = self.parsed_data['omics_data']
        unique_entities = omics_data.get('unique_entities', {})
        
        # Add disease nodes
        for disease in unique_entities.get('diseases', []):
            self.graph.add_node(
                f"DISEASE:{disease}",
                node_type='disease',
                disease_name=disease,
                entity_type='disease'
            )
        
        # Add drug nodes
        for drug in unique_entities.get('drugs', []):
            self.graph.add_node(
                f"DRUG:{drug}",
                node_type='drug',
                drug_name=drug,
                entity_type='small_molecule'
            )
        
        # Add viral condition nodes
        for viral_condition in unique_entities.get('viral_conditions', []):
            self.graph.add_node(
                f"VIRAL:{viral_condition}",
                node_type='viral_condition',
                viral_condition=viral_condition,
                entity_type='viral_perturbation'
            )
        
        # Add cluster nodes
        for cluster in unique_entities.get('clusters', []):
            self.graph.add_node(
                f"CLUSTER:{cluster}",
                node_type='network_cluster',
                cluster_id=cluster,
                entity_type='network_module'
            )
        
        # Add study context nodes
        for study in unique_entities.get('studies', []):
            if study != '-666':  # Filter out placeholder IDs
                self.graph.add_node(
                    f"STUDY:GSE{study}",
                    node_type='study',
                    gse_id=study,
                    entity_type='expression_study'
                )
        
        logger.info(f"Added Omics nodes: "
                   f"Diseases={len(unique_entities.get('diseases', []))}, "
                   f"Drugs={len(unique_entities.get('drugs', []))}, "
                   f"Viral={len(unique_entities.get('viral_conditions', []))}, "
                   f"Clusters={len(unique_entities.get('clusters', []))}")
    
    def _add_omics_associations(self):
        """Add gene-omics associations (disease, drug, viral)."""
        logger.info("Adding Omics associations...")
        
        if 'omics_data' not in self.parsed_data:
            return
        
        omics_data = self.parsed_data['omics_data']
        association_counts = {'disease': 0, 'drug': 0, 'viral': 0}
        
        # Add gene-disease associations
        for assoc in omics_data.get('disease_associations', []):
            gene_symbol = assoc['gene_symbol']
            disease_node = f"DISEASE:{assoc['disease_name']}"
            
            if gene_symbol in self.graph and disease_node in self.graph:
                self.graph.add_edge(
                    gene_symbol,
                    disease_node,
                    edge_type='gene_disease_association',
                    disease_condition=assoc['disease_condition'],
                    gse_id=assoc['gse_id'],
                    weight=assoc['weight'],
                    source='omics_disease'
                )
                association_counts['disease'] += 1
        
        # Add gene-drug associations
        for assoc in omics_data.get('drug_associations', []):
            gene_symbol = assoc['gene_symbol']
            drug_node = f"DRUG:{assoc['drug_name']}"
            
            if gene_symbol in self.graph and drug_node in self.graph:
                self.graph.add_edge(
                    gene_symbol,
                    drug_node,
                    edge_type='gene_drug_perturbation',
                    drug_condition=assoc['drug_condition'],
                    perturbation_id=assoc['perturbation_id'],
                    weight=assoc['weight'],
                    source='omics_drug'
                )
                association_counts['drug'] += 1
        
        # Add gene-viral associations
        for assoc in omics_data.get('viral_associations', []):
            gene_symbol = assoc['gene_symbol']
            viral_node = f"VIRAL:{assoc['viral_perturbation']}"
            
            if gene_symbol in self.graph and viral_node in self.graph:
                self.graph.add_edge(
                    gene_symbol,
                    viral_node,
                    edge_type='gene_viral_response',
                    viral_condition=assoc['viral_condition'],
                    gse_id=assoc['gse_id'],
                    weight=assoc['weight'],
                    source='omics_viral'
                )
                association_counts['viral'] += 1
        
        # Add viral expression matrix data
        viral_expression_count = 0
        viral_expression_matrix = omics_data.get('viral_expression_matrix', {})
        for gene_symbol, expression_data in viral_expression_matrix.items():
            if gene_symbol in self.graph:
                for expr in expression_data['expressions']:
                    viral_condition = expr['viral_perturbation']
                    viral_node = f"VIRAL:{viral_condition}"
                    
                    if viral_node in self.graph:
                        self.graph.add_edge(
                            gene_symbol,
                            viral_node,
                            edge_type='gene_viral_expression',
                            condition=expr['condition'],
                            expression_value=expr['expression_value'],
                            expression_direction=expr['expression_direction'],
                            expression_magnitude=expr['expression_magnitude'],
                            weight=abs(expr['expression_value']),
                            source='viral_expression_matrix'
                        )
                        viral_expression_count += 1
        
        association_counts['viral'] += viral_expression_count
        
        logger.info(f"Added Omics associations: "
                   f"Disease={association_counts['disease']}, "
                   f"Drug={association_counts['drug']}, "
                   f"Viral={association_counts['viral']}")
    
    def _add_cluster_relationships(self):
        """Add network cluster hierarchy relationships."""
        logger.info("Adding cluster relationships...")
        
        if 'omics_data' not in self.parsed_data:
            return
        
        omics_data = self.parsed_data['omics_data']
        cluster_count = 0
        
        for rel in omics_data.get('cluster_relationships', []):
            parent_cluster = f"CLUSTER:{rel['parent_cluster']}"
            child_cluster = f"CLUSTER:{rel['child_cluster']}"
            
            if parent_cluster in self.graph and child_cluster in self.graph:
                self.graph.add_edge(
                    child_cluster,
                    parent_cluster,
                    edge_type='cluster_hierarchy',
                    relationship_type=rel['relationship_type'],
                    source='nest_network'
                )
                cluster_count += 1
        
        logger.info(f"Added {cluster_count} cluster relationships")
    
    def _add_semantic_enhancements(self):
        """Add enhanced semantic data from Omics_data2."""
        logger.info("Adding semantic enhancements...")
        
        if 'omics_data' not in self.parsed_data:
            logger.warning("No Omics data available for semantic enhancements")
            return
            
        omics_data = self.parsed_data['omics_data']
        enhanced_data = omics_data.get('enhanced_data', {})
        
        if not enhanced_data:
            logger.info("No enhanced semantic data available, skipping")
            return
        
        # Add gene set annotations
        self._add_gene_set_annotations(enhanced_data)
        
        # Add literature references
        self._add_literature_references(enhanced_data)
        
        # Add GO term validations
        self._add_go_term_validations(enhanced_data)
        
        # Add experimental metadata
        self._add_experimental_metadata(enhanced_data)
        
        logger.info("Semantic enhancements integration complete")
    
    def _add_gene_set_annotations(self, enhanced_data):
        """Add LLM-enhanced gene set annotations to the graph."""
        annotations = enhanced_data.get('gene_set_annotations', {})
        if not annotations:
            return
        
        logger.info(f"Adding {len(annotations)} gene set annotations...")
        
        annotation_count = 0
        for gene_set_id, annotation in annotations.items():
            # Create or update gene set node
            gene_set_node = f"GENESET:{gene_set_id}"
            
            # Add enhanced attributes to the gene set node
            node_attrs = {
                'node_type': 'gene_set',
                'gene_set_id': gene_set_id,
                'source': annotation.get('source', ''),
                'gene_set_name': annotation.get('gene_set_name', ''),
                'llm_name': annotation.get('llm_name', ''),
                'llm_analysis': annotation.get('llm_analysis', ''),
                'llm_score': annotation.get('score', 0.0),
                'n_genes': annotation.get('n_genes', 0),
                'supporting_count': annotation.get('supporting_count', 0),
                'llm_coverage': annotation.get('llm_coverage', 0.0)
            }
            
            if gene_set_node in self.graph:
                # Update existing node
                self.graph.nodes[gene_set_node].update(node_attrs)
            else:
                # Add new node
                self.graph.add_node(gene_set_node, **node_attrs)
            
            # Add edges to genes in the set
            for gene_symbol in annotation.get('gene_list', []):
                if gene_symbol in self.graph:
                    self.graph.add_edge(
                        gene_symbol,
                        gene_set_node,
                        edge_type='gene_in_set',
                        support_level='annotated',
                        source='llm_annotation'
                    )
            
            # Add edges to supporting genes with higher confidence
            for gene_symbol in annotation.get('supporting_genes', []):
                if gene_symbol in self.graph:
                    self.graph.add_edge(
                        gene_symbol,
                        gene_set_node,
                        edge_type='gene_supports_set',
                        support_level='high_confidence',
                        source='llm_validation'
                    )
            
            annotation_count += 1
        
        logger.info(f"Added {annotation_count} gene set annotations")
    
    def _add_literature_references(self, enhanced_data):
        """Add literature references to gene sets."""
        references = enhanced_data.get('literature_references', {})
        if not references:
            return
        
        logger.info(f"Adding literature references for {len(references)} gene sets...")
        
        ref_count = 0
        for gene_set_id, ref_list in references.items():
            gene_set_node = f"GENESET:{gene_set_id}"
            
            if gene_set_node in self.graph:
                # Add literature metadata to gene set node
                literature_data = {
                    'has_literature': True,
                    'num_references': len([ref for ref_entry in ref_list 
                                         for ref in ref_entry.get('references', [])]),
                    'keywords': [ref_entry.get('keyword', '') for ref_entry in ref_list],
                    'paragraphs': [ref_entry.get('paragraph', '') for ref_entry in ref_list]
                }
                
                self.graph.nodes[gene_set_node].update(literature_data)
                ref_count += 1
        
        logger.info(f"Added literature references for {ref_count} gene sets")
    
    def _add_go_term_validations(self, enhanced_data):
        """Add GO term validation data."""
        validations = enhanced_data.get('go_term_validations', {})
        if not validations:
            return
        
        logger.info(f"Adding GO term validations for {len(validations)} gene sets...")
        
        validation_count = 0
        for gene_set_id, validation in validations.items():
            gene_set_node = f"GENESET:{gene_set_id}"
            
            if gene_set_node in self.graph:
                # Add GO validation metadata
                go_validation = {
                    'validated_go_term': validation.get('go_term', ''),
                    'validated_go_id': validation.get('go_id', ''),
                    'go_p_value': validation.get('adjusted_p_value', 1.0),
                    'go_intersection_size': validation.get('intersection_size', 0),
                    'go_term_size': validation.get('term_size', 0),
                    'gprofiler_ji': validation.get('gprofiler_ji', 0.0),
                    'llm_go_match': validation.get('llm_best_matching_go', ''),
                    'llm_ji': validation.get('llm_ji', 0.0),
                    'llm_validation_success': validation.get('llm_success_tf', False),
                    'gprofiler_validation_success': validation.get('gprofiler_success_tf', False)
                }
                
                self.graph.nodes[gene_set_node].update(go_validation)
                
                # If there's a validated GO term, create connection
                go_id = validation.get('go_id', '')
                if go_id and go_id in self.graph:
                    self.graph.add_edge(
                        gene_set_node,
                        go_id,
                        edge_type='validated_by_go_term',
                        p_value=validation.get('adjusted_p_value', 1.0),
                        intersection_size=validation.get('intersection_size', 0),
                        validation_source='gprofiler'
                    )
                
                validation_count += 1
        
        logger.info(f"Added GO validations for {validation_count} gene sets")
    
    def _add_experimental_metadata(self, enhanced_data):
        """Add experimental metadata to gene sets."""
        metadata = enhanced_data.get('experimental_metadata', {})
        if not metadata:
            return
        
        logger.info(f"Adding experimental metadata for {len(metadata)} gene sets...")
        
        metadata_count = 0
        for gene_set_id, meta in metadata.items():
            gene_set_node = f"GENESET:{gene_set_id}"
            
            if gene_set_node in self.graph:
                # Add experimental metadata
                experimental_data = {
                    'experimental_overlap': meta.get('overlap', 0),
                    'experimental_p_value': meta.get('p_value', 1.0),
                    'experimental_genes': meta.get('genes', []),
                    'go_term_similarity': meta.get('llm_name_go_term_sim', 0.0),
                    'referenced_analysis': meta.get('referenced_analysis', '')
                }
                
                self.graph.nodes[gene_set_node].update(experimental_data)
                metadata_count += 1
        
        logger.info(f"Added experimental metadata for {metadata_count} gene sets")
    
    def _add_model_comparison_data(self):
        """Add model comparison data and LLM evaluation results to the knowledge graph."""
        logger.info("Adding model comparison data...")
        
        if 'model_compare_data' not in self.parsed_data:
            logger.info("No model comparison data available")
            return
        
        model_data = self.parsed_data['model_compare_data']
        
        # Add model nodes
        self._add_model_nodes(model_data)
        
        # Add model predictions and confidence scores
        self._add_model_predictions(model_data)
        
        # Add similarity rankings
        self._add_similarity_rankings(model_data)
        
        # Add contamination analysis
        self._add_contamination_analysis(model_data)
        
        logger.info("Model comparison data integration complete")
    
    def _add_model_nodes(self, model_data):
        """Add LLM model nodes to the graph."""
        available_models = model_data.get('available_models', [])
        
        for model_name in available_models:
            model_node_id = f"LLM_MODEL:{model_name}"
            
            # Get model statistics from evaluation metrics
            model_metrics = model_data.get('evaluation_metrics', {}).get(model_name, {})
            confidence_stats = model_metrics.get('confidence_stats', {})
            similarity_stats = model_metrics.get('similarity_stats', {})
            
            model_attrs = {
                'node_type': 'llm_model',
                'model_name': model_name,
                'entity_type': 'language_model',
                'mean_confidence': confidence_stats.get('mean_confidence', 0.0),
                'median_confidence': confidence_stats.get('median_confidence', 0.0),
                'high_confidence_count': confidence_stats.get('high_confidence_count', 0),
                'low_confidence_count': confidence_stats.get('low_confidence_count', 0),
                'mean_similarity': similarity_stats.get('mean_similarity', 0.0),
                'mean_percentile': similarity_stats.get('mean_percentile', 0.0),
                'mean_rank': similarity_stats.get('mean_rank', 0),
                'top_10_percent_count': similarity_stats.get('top_10_percent_count', 0),
                'bottom_50_percent_count': similarity_stats.get('bottom_50_percent_count', 0)
            }
            
            self.graph.add_node(model_node_id, **model_attrs)
    
    def _add_model_predictions(self, model_data):
        """Add model prediction nodes and relationships."""
        model_predictions = model_data.get('model_predictions', {})
        prediction_count = 0
        
        for model_name, model_info in model_predictions.items():
            model_node_id = f"LLM_MODEL:{model_name}"
            go_predictions = model_info.get('go_predictions', {})
            
            for go_id, prediction_data in go_predictions.items():
                if go_id in self.graph:  # Only process if GO term exists in graph
                    # Add prediction node for each scenario
                    scenarios = prediction_data.get('scenarios', {})
                    
                    for scenario, scenario_data in scenarios.items():
                        prediction_node_id = f"PREDICTION:{model_name}:{go_id}:{scenario}"
                        
                        prediction_attrs = {
                            'node_type': 'model_prediction',
                            'model_name': model_name,
                            'go_id': go_id,
                            'scenario': scenario,
                            'predicted_name': scenario_data.get('predicted_name', ''),
                            'analysis': scenario_data.get('analysis', ''),
                            'confidence_score': scenario_data.get('confidence_score', 0.0),
                            'confidence_bin': scenario_data.get('confidence_bin', ''),
                            'genes_used': scenario_data.get('genes_used', []),
                            'true_description': prediction_data.get('true_description', ''),
                            'gene_count': prediction_data.get('gene_count', 0)
                        }
                        
                        self.graph.add_node(prediction_node_id, **prediction_attrs)
                        
                        # Add edges: model -> prediction
                        self.graph.add_edge(
                            model_node_id,
                            prediction_node_id,
                            edge_type='model_predicts',
                            confidence_score=scenario_data.get('confidence_score', 0.0),
                            scenario=scenario,
                            prediction_type='go_term_prediction'
                        )
                        
                        # Add edges: prediction -> GO term
                        self.graph.add_edge(
                            prediction_node_id,
                            go_id,
                            edge_type='predicts_go_term',
                            confidence_score=scenario_data.get('confidence_score', 0.0),
                            scenario=scenario,
                            prediction_accuracy='evaluated'
                        )
                        
                        # Add edges: prediction -> genes (if genes exist in graph)
                        for gene_symbol in scenario_data.get('genes_used', []):
                            if gene_symbol in self.graph:
                                self.graph.add_edge(
                                    prediction_node_id,
                                    gene_symbol,
                                    edge_type='prediction_uses_gene',
                                    scenario=scenario,
                                    usage_context='llm_analysis'
                                )
                        
                        prediction_count += 1
        
        logger.info(f"Added {prediction_count} model predictions")
    
    def _add_similarity_rankings(self, model_data):
        """Add similarity ranking data."""
        similarity_rankings = model_data.get('similarity_rankings', {})
        ranking_count = 0
        
        for model_name, ranking_info in similarity_rankings.items():
            model_node_id = f"LLM_MODEL:{model_name}"
            similarity_metrics = ranking_info.get('similarity_metrics', {})
            
            for go_id, similarity_data in similarity_metrics.items():
                if go_id in self.graph:  # Only process if GO term exists in graph
                    ranking_node_id = f"SIMILARITY:{model_name}:{go_id}"
                    
                    ranking_attrs = {
                        'node_type': 'similarity_ranking',
                        'model_name': model_name,
                        'go_id': go_id,
                        'llm_go_similarity': similarity_data.get('llm_go_similarity', 0.0),
                        'similarity_rank': similarity_data.get('similarity_rank', 0),
                        'true_percentile': similarity_data.get('true_percentile', 0.0),
                        'random_go_name': similarity_data.get('random_comparison', {}).get('random_go_name', ''),
                        'random_similarity': similarity_data.get('random_comparison', {}).get('random_similarity', 0.0),
                        'random_rank': similarity_data.get('random_comparison', {}).get('random_rank', 0),
                        'random_percentile': similarity_data.get('random_comparison', {}).get('random_percentile', 0.0),
                        'top_3_hits': similarity_data.get('top_matches', {}).get('top_3_hits', []),
                        'top_3_similarities': similarity_data.get('top_matches', {}).get('top_3_similarities', [])
                    }
                    
                    self.graph.add_node(ranking_node_id, **ranking_attrs)
                    
                    # Add edges: model -> ranking
                    self.graph.add_edge(
                        model_node_id,
                        ranking_node_id,
                        edge_type='model_similarity_ranking',
                        similarity_score=similarity_data.get('llm_go_similarity', 0.0),
                        ranking_percentile=similarity_data.get('true_percentile', 0.0)
                    )
                    
                    # Add edges: ranking -> GO term
                    self.graph.add_edge(
                        ranking_node_id,
                        go_id,
                        edge_type='similarity_evaluated_for',
                        similarity_score=similarity_data.get('llm_go_similarity', 0.0),
                        ranking_position=similarity_data.get('similarity_rank', 0),
                        percentile_rank=similarity_data.get('true_percentile', 0.0)
                    )
                    
                    ranking_count += 1
        
        logger.info(f"Added {ranking_count} similarity rankings")
    
    def _add_contamination_analysis(self, model_data):
        """Add contamination analysis results."""
        contamination_results = model_data.get('contamination_results', {})
        
        for model_name, contamination_info in contamination_results.items():
            model_node_id = f"LLM_MODEL:{model_name}"
            
            if model_node_id in self.graph:
                # Add contamination analysis metadata to model node
                contamination_attrs = {
                    'contamination_robustness_score': contamination_info.get('robustness_score', 0.0),
                    'severe_drops': contamination_info.get('performance_degradation', {}).get('severe_drop', 0),
                    'moderate_drops': contamination_info.get('performance_degradation', {}).get('moderate_drop', 0),
                    'stable_performance': contamination_info.get('performance_degradation', {}).get('stable', 0),
                    'performance_improvements': contamination_info.get('performance_degradation', {}).get('improvement', 0)
                }
                
                self.graph.nodes[model_node_id].update(contamination_attrs)
        
        logger.info("Added contamination analysis results")
    
    def _add_cc_mf_branch_data(self):
        """Add CC_MF_Branch data (CC and MF GO terms with LLM interpretations) to the knowledge graph."""
        logger.info("Adding CC_MF_Branch data...")
        
        if 'cc_mf_branch_data' not in self.parsed_data:
            logger.info("No CC_MF_Branch data available")
            return
        
        cc_mf_data = self.parsed_data['cc_mf_branch_data']
        
        # Add CC and MF GO term nodes
        self._add_cc_mf_go_term_nodes(cc_mf_data)
        
        # Add gene-GO term associations for CC and MF
        self._add_cc_mf_gene_associations(cc_mf_data)
        
        # Add LLM interpretations and confidence scores
        self._add_cc_mf_llm_interpretations(cc_mf_data)
        
        # Add similarity rankings for CC and MF terms
        self._add_cc_mf_similarity_rankings(cc_mf_data)
        
        logger.info("CC_MF_Branch data integration complete")
    
    def _add_cc_mf_go_term_nodes(self, cc_mf_data):
        """Add CC and MF GO term nodes to the graph."""
        cc_terms = cc_mf_data.get('cc_go_terms', {})
        mf_terms = cc_mf_data.get('mf_go_terms', {})
        
        # Add CC GO terms
        for go_id, term_data in cc_terms.items():
            node_attrs = {
                'node_type': 'go_term',
                'name': term_data.get('term_description', ''),
                'namespace': 'CC',
                'description': term_data.get('term_description', ''),
                'gene_count': term_data.get('gene_count', 0),
                'source': 'CC_MF_branch'
            }
            self.graph.add_node(go_id, **node_attrs)
        
        # Add MF GO terms  
        for go_id, term_data in mf_terms.items():
            node_attrs = {
                'node_type': 'go_term',
                'name': term_data.get('term_description', ''),
                'namespace': 'MF',
                'description': term_data.get('term_description', ''),
                'gene_count': term_data.get('gene_count', 0),
                'source': 'CC_MF_branch'
            }
            self.graph.add_node(go_id, **node_attrs)
        
        logger.info(f"Added {len(cc_terms)} CC and {len(mf_terms)} MF GO terms")
    
    def _add_cc_mf_gene_associations(self, cc_mf_data):
        """Add gene-GO term associations for CC and MF terms."""
        cc_terms = cc_mf_data.get('cc_go_terms', {})
        mf_terms = cc_mf_data.get('mf_go_terms', {})
        
        gene_association_count = 0
        
        # Process CC terms
        for go_id, term_data in cc_terms.items():
            genes = term_data.get('genes', [])
            for gene_symbol in genes:
                # Ensure gene node exists
                gene_node_id = f"GENE:{gene_symbol}"
                if gene_node_id not in self.graph:
                    self.graph.add_node(gene_node_id, 
                                      node_type='gene',
                                      gene_symbol=gene_symbol,
                                      entity_type='gene')
                
                # Add association edge
                self.graph.add_edge(
                    gene_node_id,
                    go_id,
                    edge_type='gene_go_association',
                    namespace='CC',
                    association_type='annotated_to',
                    source='CC_MF_branch'
                )
                gene_association_count += 1
        
        # Process MF terms
        for go_id, term_data in mf_terms.items():
            genes = term_data.get('genes', [])
            for gene_symbol in genes:
                # Ensure gene node exists
                gene_node_id = f"GENE:{gene_symbol}"
                if gene_node_id not in self.graph:
                    self.graph.add_node(gene_node_id,
                                      node_type='gene', 
                                      gene_symbol=gene_symbol,
                                      entity_type='gene')
                
                # Add association edge
                self.graph.add_edge(
                    gene_node_id,
                    go_id,
                    edge_type='gene_go_association',
                    namespace='MF',
                    association_type='annotated_to',
                    source='CC_MF_branch'
                )
                gene_association_count += 1
        
        logger.info(f"Added {gene_association_count} CC/MF gene-GO associations")
    
    def _add_cc_mf_llm_interpretations(self, cc_mf_data):
        """Add LLM interpretations and confidence scores for CC and MF terms."""
        cc_interpretations = cc_mf_data.get('cc_llm_interpretations', {})
        mf_interpretations = cc_mf_data.get('mf_llm_interpretations', {})
        
        interpretation_count = 0
        
        # Process CC interpretations
        for go_id, interp_data in cc_interpretations.items():
            interpretation_node_id = f"CC_INTERPRETATION:{go_id}"
            
            interp_attrs = {
                'node_type': 'llm_interpretation',
                'go_term_id': go_id,
                'namespace': 'CC',
                'llm_name': interp_data.get('llm_name', ''),
                'llm_analysis': interp_data.get('llm_analysis', ''),
                'llm_score': interp_data.get('llm_score', 0.0),
                'term_description': interp_data.get('term_description', ''),
                'gene_count': interp_data.get('gene_count', 0),
                'source': 'CC_MF_branch'
            }
            
            self.graph.add_node(interpretation_node_id, **interp_attrs)
            
            # Link interpretation to GO term
            if go_id in self.graph:
                self.graph.add_edge(
                    interpretation_node_id,
                    go_id,
                    edge_type='interprets_go_term',
                    confidence_score=interp_data.get('llm_score', 0.0),
                    namespace='CC'
                )
            
            interpretation_count += 1
        
        # Process MF interpretations
        for go_id, interp_data in mf_interpretations.items():
            interpretation_node_id = f"MF_INTERPRETATION:{go_id}"
            
            interp_attrs = {
                'node_type': 'llm_interpretation',
                'go_term_id': go_id,
                'namespace': 'MF',
                'llm_name': interp_data.get('llm_name', ''),
                'llm_analysis': interp_data.get('llm_analysis', ''),
                'llm_score': interp_data.get('llm_score', 0.0),
                'term_description': interp_data.get('term_description', ''),
                'gene_count': interp_data.get('gene_count', 0),
                'source': 'CC_MF_branch'
            }
            
            self.graph.add_node(interpretation_node_id, **interp_attrs)
            
            # Link interpretation to GO term
            if go_id in self.graph:
                self.graph.add_edge(
                    interpretation_node_id,
                    go_id,
                    edge_type='interprets_go_term',
                    confidence_score=interp_data.get('llm_score', 0.0),
                    namespace='MF'
                )
            
            interpretation_count += 1
        
        logger.info(f"Added {interpretation_count} CC/MF LLM interpretations")
    
    def _add_cc_mf_similarity_rankings(self, cc_mf_data):
        """Add similarity rankings for CC and MF terms."""
        cc_rankings = cc_mf_data.get('cc_similarity_rankings', {})
        mf_rankings = cc_mf_data.get('mf_similarity_rankings', {})
        
        ranking_count = 0
        
        # Process CC similarity rankings
        for go_id, ranking_data in cc_rankings.items():
            ranking_node_id = f"CC_SIMILARITY_RANKING:{go_id}"
            
            ranking_attrs = {
                'node_type': 'similarity_ranking',
                'go_term_id': go_id,
                'namespace': 'CC',
                'llm_name_go_term_sim': ranking_data.get('llm_name_go_term_sim', 0.0),
                'sim_rank': ranking_data.get('sim_rank', 0),
                'true_go_term_sim_percentile': ranking_data.get('true_go_term_sim_percentile', 0.0),
                'random_go_name': ranking_data.get('random_go_name', ''),
                'random_go_llm_sim': ranking_data.get('random_go_llm_sim', 0.0),
                'random_sim_rank': ranking_data.get('random_sim_rank', 0),
                'random_sim_percentile': ranking_data.get('random_sim_percentile', 0.0),
                'top_3_hits': ranking_data.get('top_3_hits', []),
                'top_3_similarities': ranking_data.get('top_3_similarities', []),
                'source': 'CC_MF_branch'
            }
            
            self.graph.add_node(ranking_node_id, **ranking_attrs)
            
            # Link ranking to GO term
            if go_id in self.graph:
                self.graph.add_edge(
                    ranking_node_id,
                    go_id,
                    edge_type='similarity_ranking_for',
                    similarity_score=ranking_data.get('llm_name_go_term_sim', 0.0),
                    percentile_rank=ranking_data.get('true_go_term_sim_percentile', 0.0),
                    namespace='CC'
                )
            
            ranking_count += 1
        
        # Process MF similarity rankings
        for go_id, ranking_data in mf_rankings.items():
            ranking_node_id = f"MF_SIMILARITY_RANKING:{go_id}"
            
            ranking_attrs = {
                'node_type': 'similarity_ranking',
                'go_term_id': go_id,
                'namespace': 'MF',
                'llm_name_go_term_sim': ranking_data.get('llm_name_go_term_sim', 0.0),
                'sim_rank': ranking_data.get('sim_rank', 0),
                'true_go_term_sim_percentile': ranking_data.get('true_go_term_sim_percentile', 0.0),
                'random_go_name': ranking_data.get('random_go_name', ''),
                'random_go_llm_sim': ranking_data.get('random_go_llm_sim', 0.0),
                'random_sim_rank': ranking_data.get('random_sim_rank', 0),
                'random_sim_percentile': ranking_data.get('random_sim_percentile', 0.0),
                'top_3_hits': ranking_data.get('top_3_hits', []),
                'top_3_similarities': ranking_data.get('top_3_similarities', []),
                'source': 'CC_MF_branch'
            }
            
            self.graph.add_node(ranking_node_id, **ranking_attrs)
            
            # Link ranking to GO term
            if go_id in self.graph:
                self.graph.add_edge(
                    ranking_node_id,
                    go_id,
                    edge_type='similarity_ranking_for',
                    similarity_score=ranking_data.get('llm_name_go_term_sim', 0.0),
                    percentile_rank=ranking_data.get('true_go_term_sim_percentile', 0.0),
                    namespace='MF'
                )
            
            ranking_count += 1
        
        logger.info(f"Added {ranking_count} CC/MF similarity rankings")
    
    def _calculate_comprehensive_stats(self):
        """Calculate comprehensive statistics for the integrated graph."""
        logger.info("Calculating comprehensive statistics...")
        
        # Count nodes by type
        node_counts = {}
        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data.get('node_type', 'unknown')
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
        
        # Count edges by type
        edge_counts = {}
        for source, target, edge_data in self.graph.edges(data=True):
            edge_type = edge_data.get('edge_type', 'unknown')
            edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1
        
        # Calculate integration metrics
        go_genes = set()
        omics_genes = set()
        
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('node_type') == 'gene':
                # Check if connected to GO terms
                for neighbor in self.graph.neighbors(node_id):
                    neighbor_data = self.graph.nodes.get(neighbor, {})
                    if neighbor_data.get('node_type') == 'go_term':
                        go_genes.add(node_id)
                    elif neighbor_data.get('node_type') in ['disease', 'drug', 'viral_condition']:
                        omics_genes.add(node_id)
        
        integrated_genes = go_genes & omics_genes
        
        self.stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_counts': node_counts,
            'edge_counts': edge_counts,
            'integration_metrics': {
                'go_connected_genes': len(go_genes),
                'omics_connected_genes': len(omics_genes),
                'integrated_genes': len(integrated_genes),
                'integration_ratio': len(integrated_genes) / len(go_genes) if go_genes else 0
            }
        }
        
        logger.info(f"Comprehensive graph statistics calculated: {self.stats}")
    
    def _validate_comprehensive_graph(self):
        """Validate the comprehensive biomedical knowledge graph."""
        logger.info("Validating comprehensive graph...")
        
        validation = {
            'has_nodes': self.graph.number_of_nodes() > 0,
            'has_edges': self.graph.number_of_edges() > 0,
            'has_go_component': False,
            'has_omics_component': False,
            'has_model_comparison_component': False,
            'integration_successful': False
        }
        
        # Check for GO component
        go_terms = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'go_term']
        validation['has_go_component'] = len(go_terms) > 0
        
        # Check for Omics component
        omics_nodes = [n for n, d in self.graph.nodes(data=True) 
                      if d.get('node_type') in ['disease', 'drug', 'viral_condition', 'network_cluster']]
        validation['has_omics_component'] = len(omics_nodes) > 0
        
        # Check for Model Comparison component
        model_nodes = [n for n, d in self.graph.nodes(data=True) 
                      if d.get('node_type') in ['llm_model', 'model_prediction', 'similarity_ranking']]
        validation['has_model_comparison_component'] = len(model_nodes) > 0
        
        # Check integration
        integration_metrics = self.stats.get('integration_metrics', {})
        validation['integration_successful'] = integration_metrics.get('integrated_genes', 0) > 0
        
        validation['overall_valid'] = all(validation.values())
        
        if validation['overall_valid']:
            logger.info("✅ Comprehensive graph validation passed")
        else:
            logger.warning(f"⚠️ Comprehensive graph validation issues: {validation}")
        
        return validation
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive graph statistics."""
        return self.stats.copy()
    
    def query_gene_comprehensive(self, gene_symbol: str) -> Dict:
        """
        Query all associations for a gene across GO and Omics data.
        
        Args:
            gene_symbol: Gene symbol to query
            
        Returns:
            Dictionary with comprehensive gene information
        """
        if gene_symbol not in self.graph:
            return {}
        
        results = {
            'gene_symbol': gene_symbol,
            'go_annotations': [],
            'disease_associations': [],
            'drug_perturbations': [],
            'viral_responses': [],
            'cluster_memberships': [],
            'gene_set_memberships': [],
            'semantic_annotations': [],
            'model_predictions': []
        }
        
        # Get all neighbors and their relationships
        for neighbor in self.graph.neighbors(gene_symbol):
            edge_data = self.graph[gene_symbol][neighbor]
            neighbor_data = self.graph.nodes[neighbor]
            
            # Handle multiple edges between same nodes
            for edge_key, edge_attrs in edge_data.items():
                edge_type = edge_attrs.get('edge_type')
                
                if edge_type == 'gene_annotation':
                    results['go_annotations'].append({
                        'go_id': neighbor,
                        'go_name': neighbor_data.get('name', ''),
                        'namespace': neighbor_data.get('namespace', ''),
                        'evidence_code': edge_attrs.get('evidence_code', '')
                    })
                
                elif edge_type == 'gene_disease_association':
                    results['disease_associations'].append({
                        'disease': neighbor_data.get('disease_name', ''),
                        'condition': edge_attrs.get('disease_condition', ''),
                        'gse_id': edge_attrs.get('gse_id', '')
                    })
                
                elif edge_type == 'gene_drug_perturbation':
                    results['drug_perturbations'].append({
                        'drug': neighbor_data.get('drug_name', ''),
                        'condition': edge_attrs.get('drug_condition', ''),
                        'perturbation_id': edge_attrs.get('perturbation_id', '')
                    })
                
                elif edge_type == 'gene_viral_response':
                    results['viral_responses'].append({
                        'viral_condition': neighbor_data.get('viral_condition', ''),
                        'condition': edge_attrs.get('viral_condition', ''),
                        'gse_id': edge_attrs.get('gse_id', ''),
                        'type': 'response'
                    })
                
                elif edge_type == 'gene_viral_expression':
                    results['viral_responses'].append({
                        'viral_condition': neighbor_data.get('viral_condition', ''),
                        'condition': edge_attrs.get('condition', ''),
                        'expression_value': edge_attrs.get('expression_value', 0),
                        'expression_direction': edge_attrs.get('expression_direction', ''),
                        'expression_magnitude': edge_attrs.get('expression_magnitude', 0),
                        'type': 'expression'
                    })
                
                elif edge_type == 'gene_in_set':
                    results['gene_set_memberships'].append({
                        'gene_set_id': neighbor_data.get('gene_set_id', ''),
                        'gene_set_name': neighbor_data.get('gene_set_name', ''),
                        'llm_name': neighbor_data.get('llm_name', ''),
                        'llm_score': neighbor_data.get('llm_score', 0.0),
                        'support_level': edge_attrs.get('support_level', ''),
                        'membership_type': 'annotated'
                    })
                
                elif edge_type == 'gene_supports_set':
                    results['gene_set_memberships'].append({
                        'gene_set_id': neighbor_data.get('gene_set_id', ''),
                        'gene_set_name': neighbor_data.get('gene_set_name', ''),
                        'llm_name': neighbor_data.get('llm_name', ''),
                        'llm_score': neighbor_data.get('llm_score', 0.0),
                        'support_level': edge_attrs.get('support_level', ''),
                        'membership_type': 'high_confidence_support'
                    })
                
                elif edge_type == 'prediction_uses_gene':
                    results['model_predictions'].append({
                        'prediction_id': neighbor,
                        'model_name': neighbor_data.get('model_name', ''),
                        'go_id': neighbor_data.get('go_id', ''),
                        'scenario': neighbor_data.get('scenario', ''),
                        'predicted_name': neighbor_data.get('predicted_name', ''),
                        'confidence_score': neighbor_data.get('confidence_score', 0.0),
                        'confidence_bin': neighbor_data.get('confidence_bin', ''),
                        'usage_context': edge_attrs.get('usage_context', ''),
                        'true_description': neighbor_data.get('true_description', '')
                    })
        
        return results
    
    def query_model_predictions(self, go_id: str = None, model_name: str = None) -> List[Dict]:
        """
        Query model predictions with optional filtering.
        
        Args:
            go_id: Optional GO term ID to filter predictions
            model_name: Optional model name to filter predictions
            
        Returns:
            List of model predictions with metadata
        """
        results = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('node_type') == 'model_prediction':
                # Apply filters if provided
                if go_id and node_data.get('go_id') != go_id:
                    continue
                if model_name and node_data.get('model_name') != model_name:
                    continue
                
                prediction_info = {
                    'prediction_id': node_id,
                    'model_name': node_data.get('model_name', ''),
                    'go_id': node_data.get('go_id', ''),
                    'scenario': node_data.get('scenario', ''),
                    'predicted_name': node_data.get('predicted_name', ''),
                    'analysis': node_data.get('analysis', ''),
                    'confidence_score': node_data.get('confidence_score', 0.0),
                    'confidence_bin': node_data.get('confidence_bin', ''),
                    'genes_used': node_data.get('genes_used', []),
                    'true_description': node_data.get('true_description', ''),
                    'gene_count': node_data.get('gene_count', 0)
                }
                
                results.append(prediction_info)
        
        # Sort by confidence score (highest first)
        results.sort(key=lambda x: x['confidence_score'], reverse=True)
        return results
    
    def query_model_comparison_summary(self) -> Dict:
        """
        Get a summary of model comparison data in the knowledge graph.
        
        Returns:
            Dictionary with model comparison statistics
        """
        summary = {
            'total_models': 0,
            'total_predictions': 0,
            'total_similarity_rankings': 0,
            'model_performance': {},
            'scenario_coverage': {},
            'go_term_coverage': 0
        }
        
        # Count models
        models = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'llm_model']
        summary['total_models'] = len(models)
        
        # Count predictions and analyze by model
        predictions = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'model_prediction']
        summary['total_predictions'] = len(predictions)
        
        # Count similarity rankings
        rankings = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'similarity_ranking']
        summary['total_similarity_rankings'] = len(rankings)
        
        # Analyze model performance
        for model_node in models:
            model_data = self.graph.nodes[model_node]
            model_name = model_data.get('model_name', '')
            
            summary['model_performance'][model_name] = {
                'mean_confidence': model_data.get('mean_confidence', 0.0),
                'mean_similarity': model_data.get('mean_similarity', 0.0),
                'mean_percentile': model_data.get('mean_percentile', 0.0),
                'contamination_robustness': model_data.get('contamination_robustness_score', 0.0),
                'stable_performance': model_data.get('stable_performance', 0),
                'severe_drops': model_data.get('severe_drops', 0)
            }
        
        # Count scenarios
        scenario_counts = {}
        go_terms_with_predictions = set()
        
        for prediction_node in predictions:
            prediction_data = self.graph.nodes[prediction_node]
            scenario = prediction_data.get('scenario', '')
            go_id = prediction_data.get('go_id', '')
            
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
            if go_id:
                go_terms_with_predictions.add(go_id)
        
        summary['scenario_coverage'] = scenario_counts
        summary['go_term_coverage'] = len(go_terms_with_predictions)
        
        return summary
    
    def query_cc_mf_terms(self, namespace: str = None) -> List[Dict]:
        """Query CC and MF GO terms.
        
        Args:
            namespace: 'CC', 'MF', or None for both
            
        Returns:
            List of GO term information
        """
        terms = []
        
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get('node_type') == 'go_term' and attrs.get('source') == 'CC_MF_branch':
                if namespace is None or attrs.get('namespace') == namespace:
                    term_info = {
                        'go_id': node_id,
                        'name': attrs.get('name', ''),
                        'namespace': attrs.get('namespace'),
                        'description': attrs.get('description', ''),
                        'gene_count': attrs.get('gene_count', 0)
                    }
                    terms.append(term_info)
        
        return sorted(terms, key=lambda x: x['gene_count'], reverse=True)
    
    def query_cc_mf_llm_interpretations(self, go_id: str = None, namespace: str = None) -> List[Dict]:
        """Query LLM interpretations for CC and MF terms.
        
        Args:
            go_id: Specific GO term to query
            namespace: 'CC', 'MF', or None for both
            
        Returns:
            List of LLM interpretation information
        """
        interpretations = []
        
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get('node_type') == 'llm_interpretation' and attrs.get('source') == 'CC_MF_branch':
                if go_id is None or attrs.get('go_term_id') == go_id:
                    if namespace is None or attrs.get('namespace') == namespace:
                        interp_info = {
                            'interpretation_id': node_id,
                            'go_term_id': attrs.get('go_term_id'),
                            'namespace': attrs.get('namespace'),
                            'llm_name': attrs.get('llm_name', ''),
                            'llm_analysis': attrs.get('llm_analysis', ''),
                            'llm_score': attrs.get('llm_score', 0.0),
                            'term_description': attrs.get('term_description', ''),
                            'gene_count': attrs.get('gene_count', 0)
                        }
                        interpretations.append(interp_info)
        
        return sorted(interpretations, key=lambda x: x['llm_score'], reverse=True)
    
    def query_cc_mf_similarity_rankings(self, go_id: str = None, namespace: str = None) -> List[Dict]:
        """Query similarity rankings for CC and MF terms.
        
        Args:
            go_id: Specific GO term to query
            namespace: 'CC', 'MF', or None for both
            
        Returns:
            List of similarity ranking information
        """
        rankings = []
        
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get('node_type') == 'similarity_ranking' and attrs.get('source') == 'CC_MF_branch':
                if go_id is None or attrs.get('go_term_id') == go_id:
                    if namespace is None or attrs.get('namespace') == namespace:
                        ranking_info = {
                            'ranking_id': node_id,
                            'go_term_id': attrs.get('go_term_id'),
                            'namespace': attrs.get('namespace'),
                            'llm_name_go_term_sim': attrs.get('llm_name_go_term_sim', 0.0),
                            'sim_rank': attrs.get('sim_rank', 0),
                            'true_go_term_sim_percentile': attrs.get('true_go_term_sim_percentile', 0.0),
                            'random_go_name': attrs.get('random_go_name', ''),
                            'top_3_hits': attrs.get('top_3_hits', []),
                            'top_3_similarities': attrs.get('top_3_similarities', [])
                        }
                        rankings.append(ranking_info)
        
        return sorted(rankings, key=lambda x: x['true_go_term_sim_percentile'], reverse=True)
    
    def query_gene_cc_mf_profile(self, gene_symbol: str) -> Dict[str, Any]:
        """Query comprehensive CC and MF profile for a gene.
        
        Args:
            gene_symbol: Gene symbol to query
            
        Returns:
            Dictionary containing CC and MF associations, interpretations, and rankings
        """
        gene_node_id = f"GENE:{gene_symbol}"
        
        if gene_node_id not in self.graph:
            return {'error': f'Gene {gene_symbol} not found in knowledge graph'}
        
        profile = {
            'gene_symbol': gene_symbol,
            'cc_associations': [],
            'mf_associations': [],
            'cc_interpretations': [],
            'mf_interpretations': [],
            'cc_similarity_rankings': [],
            'mf_similarity_rankings': []
        }
        
        # Find GO term associations
        for neighbor in self.graph.neighbors(gene_node_id):
            edge_data = self.graph.get_edge_data(gene_node_id, neighbor, 0)
            
            if edge_data and edge_data.get('edge_type') == 'gene_go_association':
                namespace = edge_data.get('namespace')
                go_node_attrs = self.graph.nodes[neighbor]
                
                association_info = {
                    'go_id': neighbor,
                    'name': go_node_attrs.get('name', ''),
                    'description': go_node_attrs.get('description', ''),
                    'gene_count': go_node_attrs.get('gene_count', 0)
                }
                
                if namespace == 'CC':
                    profile['cc_associations'].append(association_info)
                elif namespace == 'MF':
                    profile['mf_associations'].append(association_info)
                
                # Find related interpretations and rankings
                go_id = neighbor
                interpretations = self.query_cc_mf_llm_interpretations(go_id=go_id, namespace=namespace)
                rankings = self.query_cc_mf_similarity_rankings(go_id=go_id, namespace=namespace)
                
                if namespace == 'CC':
                    profile['cc_interpretations'].extend(interpretations)
                    profile['cc_similarity_rankings'].extend(rankings)
                elif namespace == 'MF':
                    profile['mf_interpretations'].extend(interpretations)
                    profile['mf_similarity_rankings'].extend(rankings)
        
        # Sort results
        profile['cc_associations'] = sorted(profile['cc_associations'], key=lambda x: x['gene_count'], reverse=True)
        profile['mf_associations'] = sorted(profile['mf_associations'], key=lambda x: x['gene_count'], reverse=True)
        profile['cc_interpretations'] = sorted(profile['cc_interpretations'], key=lambda x: x['llm_score'], reverse=True)
        profile['mf_interpretations'] = sorted(profile['mf_interpretations'], key=lambda x: x['llm_score'], reverse=True)
        
        return profile
    
    def get_cc_mf_branch_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for CC_MF_Branch data integration."""
        stats = {
            'cc_go_terms': 0,
            'mf_go_terms': 0,
            'cc_interpretations': 0,
            'mf_interpretations': 0,
            'cc_similarity_rankings': 0,
            'mf_similarity_rankings': 0,
            'cc_mf_gene_associations': 0,
            'unique_cc_mf_genes': 0
        }
        
        unique_genes = set()
        
        for node_id, attrs in self.graph.nodes(data=True):
            source = attrs.get('source')
            node_type = attrs.get('node_type')
            namespace = attrs.get('namespace')
            
            if source == 'CC_MF_branch':
                if node_type == 'go_term':
                    if namespace == 'CC':
                        stats['cc_go_terms'] += 1
                    elif namespace == 'MF':
                        stats['mf_go_terms'] += 1
                elif node_type == 'llm_interpretation':
                    if namespace == 'CC':
                        stats['cc_interpretations'] += 1
                    elif namespace == 'MF':
                        stats['mf_interpretations'] += 1
                elif node_type == 'similarity_ranking':
                    if namespace == 'CC':
                        stats['cc_similarity_rankings'] += 1
                    elif namespace == 'MF':
                        stats['mf_similarity_rankings'] += 1
        
        # Count gene associations and unique genes
        for u, v, edge_attrs in self.graph.edges(data=True):
            if (edge_attrs.get('source') == 'CC_MF_branch' and 
                edge_attrs.get('edge_type') == 'gene_go_association'):
                stats['cc_mf_gene_associations'] += 1
                
                # Extract gene symbol from node ID
                if u.startswith('GENE:'):
                    unique_genes.add(u.split(':', 1)[1])
        
        stats['unique_cc_mf_genes'] = len(unique_genes)
        stats['total_cc_mf_terms'] = stats['cc_go_terms'] + stats['mf_go_terms']
        stats['total_cc_mf_interpretations'] = stats['cc_interpretations'] + stats['mf_interpretations']
        stats['total_cc_mf_rankings'] = stats['cc_similarity_rankings'] + stats['mf_similarity_rankings']
        
        return stats
    
    def save_comprehensive_graph(self, filepath: str):
        """Save the comprehensive biomedical knowledge graph."""
        logger.info(f"Saving comprehensive graph to {filepath}")
        
        if filepath.endswith('.graphml'):
            nx.write_graphml(self.graph, filepath)
        elif filepath.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(self.graph, f)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(self.graph, f)
        
        logger.info("Comprehensive graph saved successfully")


def main():
    """Example usage of the combined GO knowledge graph (BP + CC + MF)."""
    
    print("=" * 80)
    print("COMBINED GO KNOWLEDGE GRAPH DEMONSTRATION")
    print("=" * 80)
    
    # Build combined knowledge graph
    combined_kg = CombinedGOKnowledgeGraph(use_neo4j=False)
    
    # Load data from multiple namespaces
    base_data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data"
    combined_kg.load_data(base_data_dir)
    
    # Build the combined graph
    combined_kg.build_combined_graph()
    
    # Print combined statistics
    combined_stats = combined_kg.get_combined_stats()
    print("\n" + "="*60)
    print("COMBINED GRAPH STATISTICS")
    print("="*60)
    
    print(f"Total Nodes: {combined_stats['total_nodes']:,}")
    print(f"Total Edges: {combined_stats['total_edges']:,}")
    print(f"GO Terms: {combined_stats['go_terms']:,}")
    print(f"Genes: {combined_stats['genes']:,}")
    print(f"Gene Identifiers: {combined_stats['gene_identifiers']:,}")
    
    print("\nBy Namespace:")
    for namespace, count in combined_stats['namespace_counts'].items():
        print(f"  {namespace}: {count:,} terms")
    
    print("\nIndividual Graph Statistics:")
    for namespace, stats in combined_stats['individual_stats'].items():
        print(f"  {namespace}:")
        print(f"    Nodes: {stats['total_nodes']:,}, Edges: {stats['total_edges']:,}")
        print(f"    GO Terms: {stats['go_terms']:,}, Gene Associations: {stats['gene_associations']:,}")
    
    # Demonstrate cross-namespace queries
    print("\n" + "="*60)
    print("CROSS-NAMESPACE QUERY EXAMPLES")  
    print("="*60)
    
    # Query gene functions across all namespaces
    gene = "TP53"
    all_functions = combined_kg.query_gene_functions_all_namespaces(gene)
    
    print(f"\n{gene} functions across all namespaces:")
    for namespace, functions in all_functions.items():
        print(f"  {namespace.upper().replace('_', ' ')} ({len(functions)} terms):")
        for func in functions[:3]:  # Show first 3
            print(f"    {func['go_id']}: {func['go_name']}")
    
    # Save combined graph
    combined_output_path = "/home/mreddy1/knowledge_graph/data/combined_go_kg.pkl"
    combined_kg.save_combined_graph(combined_output_path)
    print(f"\nCombined knowledge graph saved to: {combined_output_path}")
    
    # Also demonstrate individual namespace graphs
    print("\n" + "="*60)
    print("INDIVIDUAL NAMESPACE DEMONSTRATIONS")
    print("="*60)
    
    for namespace in ['biological_process', 'cellular_component']:
        if namespace in combined_kg.individual_graphs:
            kg = combined_kg.individual_graphs[namespace]
            stats = kg.get_stats()
            
            print(f"\n{namespace.upper().replace('_', ' ')}:")
            print(f"  Nodes: {stats['total_nodes']:,}, Edges: {stats['total_edges']:,}")
            print(f"  GO Terms: {stats['go_terms']:,}")
            print(f"  Gene Associations: {stats['gene_associations']:,}")
            
            # Example queries for this namespace
            gene_functions = kg.query_gene_functions('TP53')
            if gene_functions:
                print(f"  TP53 annotations (first 2):")
                for func in gene_functions[:2]:
                    print(f"    {func['go_id']}: {func['go_name']}")
    
    print("\n" + "="*80)
    print("COMBINED KNOWLEDGE GRAPH DEMONSTRATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()