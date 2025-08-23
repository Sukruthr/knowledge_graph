"""
Knowledge Graph Service for LLM Integration

This service provides a high-level interface to the comprehensive biomedical knowledge graph,
optimized for LLM consumption and query processing.
"""

import pickle
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import time
import networkx as nx

logger = logging.getLogger(__name__)

class KnowledgeGraphService:
    """
    Service for loading and querying the comprehensive biomedical knowledge graph.
    
    Provides a unified interface for LLM systems to access structured biological data
    including genes, GO terms, diseases, drugs, viral conditions, and their relationships.
    """
    
    def __init__(self, graph_path: str, enable_caching: bool = True, cache_size: int = 1000):
        """
        Initialize the Knowledge Graph Service.
        
        Args:
            graph_path: Path to the pickled NetworkX graph file
            enable_caching: Whether to enable query result caching
            cache_size: Maximum number of cached query results
        """
        self.graph_path = Path(graph_path)
        self.graph = None
        self.kg_instance = None
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self._query_cache = {}
        self._load_times = {}
        
        # Load the knowledge graph
        self._load_knowledge_graph()
    
    def _load_knowledge_graph(self):
        """Load the knowledge graph from pickle file."""
        logger.info(f"Loading knowledge graph from {self.graph_path}")
        start_time = time.time()
        
        try:
            with open(self.graph_path, 'rb') as f:
                # Load the saved graph data
                saved_data = pickle.load(f)
                
            # Check for NetworkX graphs first (since they also have 'graph' attribute for metadata)
            if isinstance(saved_data, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
                # Raw NetworkX graph
                self.graph = saved_data
                self.kg_instance = None  # No KG instance available
                logger.info("Loaded raw NetworkX graph - some advanced features may not be available")
            elif isinstance(saved_data, dict) and 'kg_instance' in saved_data:
                self.kg_instance = saved_data['kg_instance']
                self.graph = self.kg_instance.graph
                logger.info("Loaded KG instance from saved data")
            elif hasattr(saved_data, 'graph') and not isinstance(saved_data, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
                # Direct KG instance (but not a NetworkX graph)
                self.kg_instance = saved_data
                self.graph = saved_data.graph
                logger.info("Loaded KG instance directly")
            else:
                raise ValueError(f"Unknown saved data format: {type(saved_data)}")
                
            load_time = time.time() - start_time
            self._load_times['graph_load'] = load_time
            
            logger.info(f"Knowledge graph loaded in {load_time:.2f}s")
            
            # Debug: Check what self.graph actually is
            logger.info(f"Graph object type: {type(self.graph)}")
            
            if hasattr(self.graph, 'number_of_nodes'):
                logger.info(f"Graph stats: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            else:
                logger.warning(f"Graph object does not have number_of_nodes method. Available methods: {[m for m in dir(self.graph) if not m.startswith('_')][:10]}")
            
        except Exception as e:
            logger.error(f"Failed to load knowledge graph: {e}")
            raise
    
    def _cache_key(self, method: str, **kwargs) -> str:
        """Generate cache key for method and parameters."""
        key_parts = [method] + [f"{k}={v}" for k, v in sorted(kwargs.items()) if v is not None]
        return "|".join(key_parts)
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get result from cache if available."""
        if not self.enable_caching:
            return None
        return self._query_cache.get(cache_key)
    
    def _add_to_cache(self, cache_key: str, result: Any):
        """Add result to cache with size management."""
        if not self.enable_caching:
            return
            
        # Simple cache size management - remove oldest entries
        if len(self._query_cache) >= self.cache_size:
            # Remove first (oldest) entry
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        
        self._query_cache[cache_key] = result
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        if not self.graph:
            return {"error": "Graph not loaded"}
        
        cache_key = self._cache_key("get_graph_stats")
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Safely get graph statistics
        if hasattr(self.graph, 'number_of_nodes'):
            stats = {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "load_time_seconds": self._load_times.get('graph_load', 0),
                "is_directed": getattr(self.graph, 'is_directed', lambda: False)(),
                "is_multigraph": getattr(self.graph, 'is_multigraph', lambda: False)()
            }
        else:
            stats = {
                "total_nodes": 0,
                "total_edges": 0,
                "load_time_seconds": self._load_times.get('graph_load', 0),
                "is_directed": False,
                "is_multigraph": False,
                "error": f"Graph object type {type(self.graph)} does not support standard graph operations"
            }
        
        # Add node type distribution if available
        node_types = {}
        if hasattr(self.graph, 'nodes') and callable(self.graph.nodes):
            try:
                for node, data in self.graph.nodes(data=True):
                    node_type = data.get('type', 'unknown') if data else 'unknown'
                    node_types[node_type] = node_types.get(node_type, 0) + 1
                stats["node_type_distribution"] = node_types
            except Exception as e:
                logger.warning(f"Failed to get node type distribution: {e}")
                stats["node_type_distribution"] = {"error": str(e)}
        
        # Add edge type distribution if available  
        edge_types = {}
        if hasattr(self.graph, 'edges') and callable(self.graph.edges):
            try:
                for u, v, data in self.graph.edges(data=True):
                    edge_type = data.get('type', 'unknown') if data else 'unknown'
                    edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
                stats["edge_type_distribution"] = edge_types
            except Exception as e:
                logger.warning(f"Failed to get edge type distribution: {e}")
                stats["edge_type_distribution"] = {"error": str(e)}
        
        self._add_to_cache(cache_key, stats)
        return stats
    
    def query_gene_information(self, gene: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a gene.
        
        Args:
            gene: Gene symbol or identifier
            
        Returns:
            Dictionary containing gene information and all associations
        """
        cache_key = self._cache_key("query_gene_information", gene=gene)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        logger.info(f"Querying gene information for: {gene}")
        
        # Use the KG instance's comprehensive query method if available
        if self.kg_instance and hasattr(self.kg_instance, 'query_gene_comprehensive'):
            try:
                result = self.kg_instance.query_gene_comprehensive(gene)
                self._add_to_cache(cache_key, result)
                return result
            except Exception as e:
                logger.warning(f"KG comprehensive query failed: {e}")
        
        # Fallback to direct graph queries
        result = self._direct_gene_query(gene)
        self._add_to_cache(cache_key, result)
        return result
    
    def _direct_gene_query(self, gene: str) -> Dict[str, Any]:
        """Direct NetworkX graph query for gene information."""
        result = {
            "gene": gene,
            "found": False,
            "go_annotations": [],
            "associations": [],
            "neighbors": [],
            "node_data": {}
        }
        
        # Find gene node - handle structured identifiers
        gene_nodes = []
        
        # First try structured identifier
        structured_id = f"GENE:{gene.upper()}"
        if structured_id in self.graph.nodes:
            gene_nodes.append(structured_id)
        
        # Try exact matches
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            node_type = node_data.get('node_type', node_data.get('type', ''))
            
            if node_type == 'gene':
                # Check node ID
                if str(node).upper() == gene.upper():
                    gene_nodes.append(node)
                # Check gene symbol/name fields  
                elif (gene.upper() == node_data.get('gene_symbol', '').upper() or
                      gene.upper() == node_data.get('name', '').upper()):
                    gene_nodes.append(node)
        
        # If no exact match, try partial matching
        if not gene_nodes:
            for node in self.graph.nodes():
                node_data = self.graph.nodes[node]
                node_type = node_data.get('node_type', node_data.get('type', ''))
                
                if node_type == 'gene' and gene.upper() in str(node).upper():
                    gene_nodes.append(node)
        
        if gene_nodes:
            gene_node = gene_nodes[0]
            result["found"] = True
            result["node_data"] = dict(self.graph.nodes[gene_node])
            
            # Get all neighbors and their relationships (limit for performance)
            if self.graph.is_directed():
                predecessors = list(self.graph.predecessors(gene_node))[:100]
                successors = list(self.graph.successors(gene_node))[:100]
                neighbors = predecessors + successors
            else:
                neighbors = list(self.graph.neighbors(gene_node))[:200]
            for neighbor in neighbors:
                neighbor_data = dict(self.graph.nodes[neighbor])
                
                # Get edge data - handle directed multigraphs carefully
                edge_data = {}
                try:
                    if gene_node in self.graph[neighbor]:  # edge from neighbor to gene
                        edge_data = list(self.graph[neighbor][gene_node].values())[0] if self.graph.is_multigraph() else dict(self.graph[neighbor][gene_node])
                    elif neighbor in self.graph[gene_node]:  # edge from gene to neighbor
                        edge_data = list(self.graph[gene_node][neighbor].values())[0] if self.graph.is_multigraph() else dict(self.graph[gene_node][neighbor])
                except (KeyError, IndexError):
                    edge_data = {}
                
                result["neighbors"].append({
                    "node": neighbor,
                    "node_data": neighbor_data,
                    "relationship": edge_data
                })
                
                # Categorize by type
                node_type = neighbor_data.get('node_type', neighbor_data.get('type', 'unknown'))
                if node_type == 'go_term':
                    result["go_annotations"].append({
                        "go_id": neighbor,
                        "name": neighbor_data.get('name', ''),
                        "namespace": neighbor_data.get('namespace', ''),
                        "evidence": edge_data.get('evidence_code', '')
                    })
                elif node_type in ['disease', 'drug', 'viral_condition']:
                    result["associations"].append({
                        "entity": neighbor,
                        "type": node_type,
                        "name": neighbor_data.get('name', ''),
                        "relationship_type": edge_data.get('edge_type', edge_data.get('type', '')),
                        "details": edge_data
                    })
        
        return result
    
    def query_pathway_information(self, pathway: str) -> Dict[str, Any]:
        """
        Get information about biological pathways.
        
        Args:
            pathway: Pathway name or GO term
            
        Returns:
            Dictionary containing pathway information and associated genes
        """
        cache_key = self._cache_key("query_pathway_information", pathway=pathway)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        logger.info(f"Querying pathway information for: {pathway}")
        
        result = {
            "pathway": pathway,
            "found": False,
            "pathway_nodes": [],
            "associated_genes": [],
            "go_terms": []
        }
        
        # Search for pathway-related nodes
        pathway_nodes = []
        for node, data in self.graph.nodes(data=True):
            node_name = data.get('name', str(node)).lower()
            if pathway.lower() in node_name or any(term in node_name for term in pathway.lower().split()):
                pathway_nodes.append((node, data))
        
        if pathway_nodes:
            result["found"] = True
            result["pathway_nodes"] = [{"node": n, "data": d} for n, d in pathway_nodes]
            
            # Get associated genes for each pathway node
            for node, data in pathway_nodes:
                for neighbor in self.graph.neighbors(node):
                    neighbor_data = dict(self.graph.nodes[neighbor])
                    if neighbor_data.get('type') == 'gene':
                        edge_data = dict(self.graph[node][neighbor])
                        result["associated_genes"].append({
                            "gene": neighbor,
                            "gene_data": neighbor_data,
                            "pathway_node": node,
                            "relationship": edge_data
                        })
        
        self._add_to_cache(cache_key, result)
        return result
    
    def query_disease_associations(self, disease: str) -> Dict[str, Any]:
        """
        Get disease-related information and gene associations.
        
        Args:
            disease: Disease name or identifier
            
        Returns:
            Dictionary containing disease information and associations
        """
        cache_key = self._cache_key("query_disease_associations", disease=disease)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        logger.info(f"Querying disease associations for: {disease}")
        
        result = {
            "disease": disease,
            "found": False,
            "disease_nodes": [],
            "associated_genes": [],
            "pathways": []
        }
        
        # Find disease nodes
        disease_nodes = []
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'disease':
                node_name = data.get('name', str(node)).lower()
                if disease.lower() in node_name:
                    disease_nodes.append((node, data))
        
        if disease_nodes:
            result["found"] = True
            result["disease_nodes"] = [{"node": n, "data": d} for n, d in disease_nodes]
            
            # Get associated genes and pathways
            for node, data in disease_nodes:
                for neighbor in self.graph.neighbors(node):
                    neighbor_data = dict(self.graph.nodes[neighbor])
                    edge_data = dict(self.graph[node][neighbor])
                    
                    if neighbor_data.get('type') == 'gene':
                        result["associated_genes"].append({
                            "gene": neighbor,
                            "gene_data": neighbor_data,
                            "relationship": edge_data
                        })
                    elif neighbor_data.get('type') == 'go_term':
                        result["pathways"].append({
                            "go_term": neighbor,
                            "go_data": neighbor_data,
                            "relationship": edge_data
                        })
        
        self._add_to_cache(cache_key, result)
        return result
    
    def query_drug_interactions(self, drug: str) -> Dict[str, Any]:
        """
        Get drug interaction and target information.
        
        Args:
            drug: Drug name or identifier
            
        Returns:
            Dictionary containing drug information and interactions
        """
        cache_key = self._cache_key("query_drug_interactions", drug=drug)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        logger.info(f"Querying drug interactions for: {drug}")
        
        result = {
            "drug": drug,
            "found": False,
            "drug_nodes": [],
            "target_genes": [],
            "affected_pathways": []
        }
        
        # Find drug nodes - handle structured identifiers
        drug_nodes = []
        
        # First try exact structured identifier match
        structured_id = f"DRUG:{drug.lower()}"
        if structured_id in self.graph.nodes:
            node_data = dict(self.graph.nodes[structured_id])
            drug_nodes.append((structured_id, node_data))
        
        # Then search through all drug nodes
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('node_type', data.get('type', ''))
            if node_type == 'drug':
                # Check node ID contains drug name
                if drug.lower() in str(node).lower():
                    drug_nodes.append((node, data))
                # Check drug_name field
                elif drug.lower() in data.get('drug_name', '').lower():
                    drug_nodes.append((node, data))
                # Check name field
                elif drug.lower() in data.get('name', '').lower():
                    drug_nodes.append((node, data))
        
        if drug_nodes:
            result["found"] = True
            result["drug_nodes"] = [{"node": n, "data": d} for n, d in drug_nodes]
            
            # Get target genes and affected pathways
            for node, data in drug_nodes:
                # Handle directed graphs - get both predecessors and successors
                if self.graph.is_directed():
                    predecessors = list(self.graph.predecessors(node))[:50]
                    successors = list(self.graph.successors(node))[:50]  
                    neighbors = predecessors + successors
                else:
                    neighbors = list(self.graph.neighbors(node))[:100]
                    
                for neighbor in neighbors:
                    neighbor_data = dict(self.graph.nodes[neighbor])
                    
                    # Get edge data - handle directed multigraphs carefully
                    edge_data = {}
                    try:
                        if node in self.graph[neighbor]:  # edge from neighbor to drug
                            edge_data = list(self.graph[neighbor][node].values())[0] if self.graph.is_multigraph() else dict(self.graph[neighbor][node])
                        elif neighbor in self.graph[node]:  # edge from drug to neighbor  
                            edge_data = list(self.graph[node][neighbor].values())[0] if self.graph.is_multigraph() else dict(self.graph[node][neighbor])
                    except (KeyError, IndexError):
                        edge_data = {}
                    
                    neighbor_type = neighbor_data.get('node_type', neighbor_data.get('type', ''))
                    if neighbor_type == 'gene':
                        result["target_genes"].append({
                            "gene": neighbor,
                            "gene_data": neighbor_data,
                            "interaction_type": edge_data.get('edge_type', edge_data.get('type', '')),
                            "details": edge_data
                        })
                    elif neighbor_type == 'go_term':
                        result["affected_pathways"].append({
                            "pathway": neighbor,
                            "pathway_data": neighbor_data,
                            "relationship": edge_data
                        })
        
        self._add_to_cache(cache_key, result)
        return result
    
    def smart_entity_search(self, text: str) -> Dict[str, Any]:
        """
        Intelligent search that tries to find relevant entities in the text
        even when entity extraction fails.
        """
        result = {
            "text": text,
            "drugs_found": [],
            "genes_found": [],
            "diseases_found": [],
            "go_terms_found": []
        }
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Look for known drug names in the KG
        for node, data in list(self.graph.nodes(data=True))[:5000]:  # Limit for performance
            if data.get('node_type') == 'drug':
                drug_name = data.get('drug_name', str(node)).lower()
                if drug_name in text_lower or any(word in drug_name for word in words):
                    result["drugs_found"].append({
                        "node": node,
                        "name": drug_name,
                        "data": data,
                        "connections": self.graph.degree(node)
                    })
        
        # Look for known gene names in the KG  
        for node, data in list(self.graph.nodes(data=True))[:5000]:  # Limit for performance
            if data.get('node_type') == 'gene':
                gene_name = data.get('gene_symbol', str(node)).lower()
                if gene_name in text_lower or any(word in gene_name for word in words):
                    result["genes_found"].append({
                        "node": node,
                        "name": gene_name,
                        "data": data,
                        "connections": self.graph.degree(node)
                    })
        
        # Sort by connection count (most connected = most relevant)
        result["drugs_found"].sort(key=lambda x: x["connections"], reverse=True)
        result["genes_found"].sort(key=lambda x: x["connections"], reverse=True)
        
        return result
    
    def search_by_keywords(self, keywords: List[str], limit: int = 50) -> Dict[str, Any]:
        """
        Search the knowledge graph using keywords.
        
        Args:
            keywords: List of keywords to search for
            limit: Maximum number of results to return
            
        Returns:
            Dictionary containing search results categorized by entity type
        """
        cache_key = self._cache_key("search_by_keywords", keywords=tuple(keywords), limit=limit)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        logger.info(f"Searching by keywords: {keywords}")
        
        result = {
            "keywords": keywords,
            "total_matches": 0,
            "genes": [],
            "go_terms": [],
            "diseases": [],
            "drugs": [],
            "other": []
        }
        
        matches = []
        keywords_lower = [k.lower() for k in keywords]
        
        for node, data in self.graph.nodes(data=True):
            # Search in node ID and name
            searchable_text = [str(node).lower()]
            if 'name' in data:
                searchable_text.append(data['name'].lower())
            if 'definition' in data:
                searchable_text.append(data['definition'].lower())
            
            # Check if any keyword matches
            text_to_search = ' '.join(searchable_text)
            match_score = 0
            matched_keywords = []
            
            for keyword in keywords_lower:
                if keyword in text_to_search:
                    match_score += 1
                    matched_keywords.append(keyword)
            
            if match_score > 0:
                matches.append({
                    "node": node,
                    "data": data,
                    "match_score": match_score,
                    "matched_keywords": matched_keywords
                })
        
        # Sort by match score and limit results
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        matches = matches[:limit]
        
        result["total_matches"] = len(matches)
        
        # Categorize by node type
        for match in matches:
            node_type = match["data"].get('type', 'unknown')
            match_info = {
                "node": match["node"],
                "name": match["data"].get('name', ''),
                "match_score": match["match_score"],
                "matched_keywords": match["matched_keywords"],
                "data": match["data"]
            }
            
            if node_type == 'gene':
                result["genes"].append(match_info)
            elif node_type == 'go_term':
                result["go_terms"].append(match_info)
            elif node_type == 'disease':
                result["diseases"].append(match_info)
            elif node_type == 'drug':
                result["drugs"].append(match_info)
            else:
                result["other"].append(match_info)
        
        self._add_to_cache(cache_key, result)
        return result
    
    def query_viral_expression(self, limit: int = 50) -> Dict[str, Any]:
        """
        Get genes with highest viral expression levels.
        
        Args:
            limit: Maximum number of genes to return
            
        Returns:
            Dictionary containing ranked genes by viral expression
        """
        cache_key = self._cache_key("query_viral_expression", limit=limit)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        logger.info(f"Querying viral expression for top {limit} genes")
        
        result = {
            "found": False,
            "total_genes": 0,
            "top_genes": [],
            "viral_conditions": []
        }
        
        try:
            # Find genes with viral expression edges
            gene_expression_scores = {}
            viral_conditions_set = set()
            
            for u, v, edge_data in self.graph.edges(data=True):
                edge_type = edge_data.get('edge_type', '')
                
                # Look for viral expression edges
                if 'viral' in edge_type.lower() or 'virus' in edge_type.lower():
                    # Determine which node is gene and which is viral condition
                    gene_node = None
                    viral_condition = None
                    
                    u_data = self.graph.nodes.get(u, {})
                    v_data = self.graph.nodes.get(v, {})
                    
                    if u_data.get('node_type') == 'gene':
                        gene_node = u
                        viral_condition = v
                    elif v_data.get('node_type') == 'gene':
                        gene_node = v  
                        viral_condition = u
                    elif 'viral' in str(v).lower() or 'virus' in str(v).lower():
                        gene_node = u
                        viral_condition = v
                    elif 'viral' in str(u).lower() or 'virus' in str(u).lower():
                        gene_node = v
                        viral_condition = u
                        
                    if gene_node and viral_condition:
                        gene_symbol = self.graph.nodes.get(gene_node, {}).get('gene_symbol', gene_node)
                        expression_value = float(edge_data.get('weight', edge_data.get('expression_value', 0)))
                        
                        if gene_symbol not in gene_expression_scores:
                            gene_expression_scores[gene_symbol] = {
                                'max_expression': 0,
                                'total_expression': 0,
                                'condition_count': 0,
                                'conditions': []
                            }
                        
                        gene_data = gene_expression_scores[gene_symbol]
                        gene_data['max_expression'] = max(gene_data['max_expression'], abs(expression_value))
                        gene_data['total_expression'] += expression_value
                        gene_data['condition_count'] += 1
                        gene_data['conditions'].append(viral_condition)
                        viral_conditions_set.add(viral_condition)
            
            if gene_expression_scores:
                result["found"] = True
                result["total_genes"] = len(gene_expression_scores)
                result["viral_conditions"] = list(viral_conditions_set)[:20]  # Sample conditions
                
                # Calculate average expression and rank genes
                for gene, data in gene_expression_scores.items():
                    data['avg_expression'] = data['total_expression'] / data['condition_count'] if data['condition_count'] > 0 else 0
                
                # Sort by max expression, then by average expression
                sorted_genes = sorted(
                    gene_expression_scores.items(), 
                    key=lambda x: (x[1]['max_expression'], x[1]['avg_expression']), 
                    reverse=True
                )
                
                # Format top genes
                for gene, data in sorted_genes[:limit]:
                    gene_info = {
                        'gene_symbol': gene,
                        'max_expression': data['max_expression'],
                        'avg_expression': data['avg_expression'],
                        'condition_count': data['condition_count'],
                        'sample_conditions': data['conditions'][:5]  # First 5 conditions
                    }
                    result["top_genes"].append(gene_info)
                
                logger.info(f"Found {len(sorted_genes)} genes with viral expression data")
            
        except Exception as e:
            logger.error(f"Error querying viral expression: {e}")
            
        self._add_to_cache(cache_key, result)
        return result
    
    def get_related_entities(self, entity: str, relation_types: List[str] = None, max_depth: int = 2) -> Dict[str, Any]:
        """
        Get entities related to a given entity through specified relationship types.
        
        Args:
            entity: Starting entity (node ID)
            relation_types: Types of relationships to follow (None for all)
            max_depth: Maximum depth to traverse
            
        Returns:
            Dictionary containing related entities by depth level
        """
        cache_key = self._cache_key("get_related_entities", 
                                  entity=entity, 
                                  relation_types=tuple(relation_types) if relation_types else None,
                                  max_depth=max_depth)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        logger.info(f"Finding related entities for: {entity}")
        
        result = {
            "root_entity": entity,
            "found": False,
            "levels": {},
            "total_related": 0
        }
        
        # Check if entity exists
        if entity not in self.graph:
            self._add_to_cache(cache_key, result)
            return result
        
        result["found"] = True
        visited = {entity}
        current_level = {entity}
        
        for depth in range(max_depth):
            next_level = set()
            level_entities = []
            
            for node in current_level:
                for neighbor in self.graph.neighbors(node):
                    if neighbor in visited:
                        continue
                    
                    # Check relationship type if specified
                    if relation_types:
                        edge_data = dict(self.graph[node][neighbor])
                        edge_type = edge_data.get('type', '')
                        if edge_type not in relation_types:
                            continue
                    
                    next_level.add(neighbor)
                    visited.add(neighbor)
                    
                    neighbor_data = dict(self.graph.nodes[neighbor])
                    edge_data = dict(self.graph[node][neighbor])
                    
                    level_entities.append({
                        "entity": neighbor,
                        "data": neighbor_data,
                        "relationship": edge_data,
                        "connected_to": node
                    })
            
            if level_entities:
                result["levels"][f"depth_{depth + 1}"] = level_entities
                result["total_related"] += len(level_entities)
            
            current_level = next_level
            if not current_level:
                break
        
        self._add_to_cache(cache_key, result)
        return result
    
    def get_shortest_path(self, source: str, target: str, cutoff: int = 5) -> Dict[str, Any]:
        """
        Find shortest path between two entities in the knowledge graph.
        
        Args:
            source: Source entity
            target: Target entity
            cutoff: Maximum path length to search
            
        Returns:
            Dictionary containing path information
        """
        cache_key = self._cache_key("get_shortest_path", source=source, target=target, cutoff=cutoff)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        logger.info(f"Finding shortest path from {source} to {target}")
        
        result = {
            "source": source,
            "target": target,
            "path_found": False,
            "path_length": 0,
            "path": [],
            "path_details": []
        }
        
        try:
            if source in self.graph and target in self.graph:
                path = nx.shortest_path(self.graph, source, target, cutoff=cutoff)
                result["path_found"] = True
                result["path_length"] = len(path) - 1
                result["path"] = path
                
                # Get detailed information for each edge in the path
                for i in range(len(path) - 1):
                    current_node = path[i]
                    next_node = path[i + 1]
                    
                    current_data = dict(self.graph.nodes[current_node])
                    next_data = dict(self.graph.nodes[next_node])
                    edge_data = dict(self.graph[current_node][next_node])
                    
                    result["path_details"].append({
                        "from_node": current_node,
                        "from_data": current_data,
                        "to_node": next_node,
                        "to_data": next_data,
                        "edge_data": edge_data
                    })
                    
        except nx.NetworkXNoPath:
            logger.info(f"No path found between {source} and {target}")
        except Exception as e:
            logger.warning(f"Error finding path: {e}")
        
        self._add_to_cache(cache_key, result)
        return result
    
    def clear_cache(self):
        """Clear the query cache."""
        self._query_cache.clear()
        logger.info("Query cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._query_cache),
            "max_cache_size": self.cache_size,
            "cache_enabled": self.enable_caching
        }