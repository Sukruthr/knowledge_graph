"""
Generic Knowledge Graph for Gene Ontology data (supports GO_BP, GO_CC, GO_MF).

Extracted from kg_builder.py for better modularity and maintainability.
"""

import networkx as nx
from typing import Dict, List, Optional, Any
import logging

try:
    from ..parsers import GODataParser
except ImportError:
    from parsers import GODataParser

from .shared_utils import (
    save_graph_to_file, 
    load_graph_from_file, 
    initialize_graph_attributes
)

logger = logging.getLogger(__name__)


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
        save_graph_to_file(self.graph, filepath)
    
    def load_graph(self, filepath: str):
        """Load a NetworkX graph from disk."""
        self.graph = load_graph_from_file(filepath)
        initialize_graph_attributes(self)
        self._calculate_stats()