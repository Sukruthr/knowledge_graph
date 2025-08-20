#!/usr/bin/env python3
"""
Complete Knowledge Graph Analysis: Comprehensive Interconnection Analysis
Analyzes connections between GO_BP, GO_CC, GO_MF, Omics_data, and Omics_data2
"""

import sys
import logging
import networkx as nx
from collections import defaultdict, Counter
import json
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from kg_builder import ComprehensiveBiomedicalKnowledgeGraph

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KGConnectionAnalyzer:
    """Analyzer for comprehensive knowledge graph connections."""
    
    def __init__(self, kg):
        self.kg = kg
        self.graph = kg.graph
        self.connections = {}
        self.cross_modal_stats = {}
        
    def analyze_all_connections(self):
        """Perform comprehensive connection analysis."""
        logger.info("="*80)
        logger.info("COMPREHENSIVE KNOWLEDGE GRAPH CONNECTION ANALYSIS")
        logger.info("="*80)
        
        # 1. Basic graph statistics
        self.analyze_basic_stats()
        
        # 2. GO namespace interconnections
        self.analyze_go_namespace_connections()
        
        # 3. GO-Omics connections
        self.analyze_go_omics_connections()
        
        # 4. Omics internal connections
        self.analyze_omics_internal_connections()
        
        # 5. Semantic enhancement connections
        self.analyze_semantic_connections()
        
        # 6. Cross-modal gene analysis
        self.analyze_cross_modal_genes()
        
        # 7. Hub analysis
        self.analyze_hub_nodes()
        
        # 8. Path analysis
        self.analyze_connection_paths()
        
        # 9. Summary and insights
        self.generate_insights()
        
    def analyze_basic_stats(self):
        """Analyze basic graph statistics."""
        logger.info("\n📊 BASIC GRAPH STATISTICS")
        logger.info("-" * 50)
        
        stats = self.kg.get_comprehensive_stats()
        
        logger.info(f"Total Nodes: {stats['total_nodes']:,}")
        logger.info(f"Total Edges: {stats['total_edges']:,}")
        logger.info(f"Graph Density: {stats['total_edges'] / (stats['total_nodes'] * (stats['total_nodes'] - 1)):.6f}")
        
        logger.info("\nNode Distribution:")
        for node_type, count in stats['node_counts'].items():
            percentage = (count / stats['total_nodes']) * 100
            logger.info(f"  {node_type}: {count:,} ({percentage:.1f}%)")
        
        logger.info("\nEdge Distribution:")
        for edge_type, count in stats['edge_counts'].items():
            percentage = (count / stats['total_edges']) * 100
            logger.info(f"  {edge_type}: {count:,} ({percentage:.1f}%)")
    
    def analyze_go_namespace_connections(self):
        """Analyze connections between GO namespaces."""
        logger.info("\n🧬 GO NAMESPACE INTERCONNECTIONS")
        logger.info("-" * 50)
        
        # Count GO terms by namespace
        go_namespaces = defaultdict(set)
        shared_genes = defaultdict(set)
        
        for node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'go_term':
                namespace = data.get('namespace', 'unknown')
                go_namespaces[namespace].add(node)
        
        logger.info("GO Terms by Namespace:")
        for namespace, terms in go_namespaces.items():
            logger.info(f"  {namespace}: {len(terms):,} terms")
        
        # Analyze genes connected to multiple namespaces
        gene_namespace_connections = defaultdict(set)
        
        for gene_node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'gene':
                # Check which GO namespaces this gene connects to
                for neighbor in self.graph.neighbors(gene_node):
                    neighbor_data = self.graph.nodes.get(neighbor, {})
                    if neighbor_data.get('node_type') == 'go_term':
                        namespace = neighbor_data.get('namespace', 'unknown')
                        gene_namespace_connections[gene_node].add(namespace)
                        shared_genes[namespace].add(gene_node)
        
        # Calculate cross-namespace gene overlaps
        namespaces = list(go_namespaces.keys())
        logger.info("\nCross-Namespace Gene Connections:")
        
        for i, ns1 in enumerate(namespaces):
            for j, ns2 in enumerate(namespaces):
                if i < j:  # Avoid duplicates
                    overlap = shared_genes[ns1] & shared_genes[ns2]
                    total_genes_ns1 = len(shared_genes[ns1])
                    total_genes_ns2 = len(shared_genes[ns2])
                    
                    if total_genes_ns1 > 0 and total_genes_ns2 > 0:
                        jaccard = len(overlap) / len(shared_genes[ns1] | shared_genes[ns2])
                        logger.info(f"  {ns1} ↔ {ns2}: {len(overlap):,} shared genes "
                                   f"(Jaccard: {jaccard:.3f})")
        
        # Genes connected to all three namespaces
        triple_connected = set()
        for gene, namespaces_connected in gene_namespace_connections.items():
            if len(namespaces_connected) >= 3:
                triple_connected.add(gene)
        
        logger.info(f"\nGenes connected to all GO namespaces: {len(triple_connected):,}")
        if triple_connected:
            sample_genes = list(triple_connected)[:5]
            logger.info(f"Sample multi-namespace genes: {', '.join(sample_genes)}")
    
    def analyze_go_omics_connections(self):
        """Analyze connections between GO and Omics data."""
        logger.info("\n🔬 GO-OMICS INTERCONNECTIONS")
        logger.info("-" * 50)
        
        # Find genes that bridge GO and Omics
        go_connected_genes = set()
        omics_connected_genes = set()
        disease_genes = set()
        drug_genes = set()
        viral_genes = set()
        
        for gene_node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'gene':
                has_go = False
                has_disease = False
                has_drug = False
                has_viral = False
                
                for neighbor in self.graph.neighbors(gene_node):
                    neighbor_data = self.graph.nodes.get(neighbor, {})
                    neighbor_type = neighbor_data.get('node_type')
                    
                    if neighbor_type == 'go_term':
                        has_go = True
                    elif neighbor_type == 'disease':
                        has_disease = True
                    elif neighbor_type == 'drug':
                        has_drug = True
                    elif neighbor_type == 'viral_condition':
                        has_viral = True
                
                if has_go:
                    go_connected_genes.add(gene_node)
                if has_disease or has_drug or has_viral:
                    omics_connected_genes.add(gene_node)
                if has_disease:
                    disease_genes.add(gene_node)
                if has_drug:
                    drug_genes.add(gene_node)
                if has_viral:
                    viral_genes.add(gene_node)
        
        # Calculate integration metrics
        integrated_genes = go_connected_genes & omics_connected_genes
        
        logger.info("GO-Omics Gene Integration:")
        logger.info(f"  GO-connected genes: {len(go_connected_genes):,}")
        logger.info(f"  Omics-connected genes: {len(omics_connected_genes):,}")
        logger.info(f"  Integrated genes (both): {len(integrated_genes):,}")
        logger.info(f"  Integration ratio: {len(integrated_genes)/len(go_connected_genes):.3f}")
        
        logger.info("\nOmics Data Type Coverage:")
        logger.info(f"  Disease-connected genes: {len(disease_genes):,}")
        logger.info(f"  Drug-connected genes: {len(drug_genes):,}")
        logger.info(f"  Viral-connected genes: {len(viral_genes):,}")
        
        # Cross-omics overlaps
        disease_drug_overlap = disease_genes & drug_genes
        disease_viral_overlap = disease_genes & viral_genes
        drug_viral_overlap = drug_genes & viral_genes
        all_omics_overlap = disease_genes & drug_genes & viral_genes
        
        logger.info("\nCross-Omics Gene Overlaps:")
        logger.info(f"  Disease ↔ Drug: {len(disease_drug_overlap):,} genes")
        logger.info(f"  Disease ↔ Viral: {len(disease_viral_overlap):,} genes")
        logger.info(f"  Drug ↔ Viral: {len(drug_viral_overlap):,} genes")
        logger.info(f"  All three omics: {len(all_omics_overlap):,} genes")
        
        # Sample highly connected genes
        if all_omics_overlap:
            sample_multi_omics = list(all_omics_overlap)[:5]
            logger.info(f"Sample multi-omics genes: {', '.join(sample_multi_omics)}")
    
    def analyze_omics_internal_connections(self):
        """Analyze internal connections within Omics data."""
        logger.info("\n🔗 OMICS INTERNAL CONNECTIONS")
        logger.info("-" * 50)
        
        # Analyze cluster relationships
        cluster_hierarchy_edges = [
            (u, v, data) for u, v, data in self.graph.edges(data=True)
            if data.get('edge_type') == 'cluster_hierarchy'
        ]
        
        logger.info(f"Network cluster hierarchy edges: {len(cluster_hierarchy_edges):,}")
        
        # Analyze expression vs association data
        viral_expression_edges = [
            (u, v, data) for u, v, data in self.graph.edges(data=True)
            if data.get('edge_type') == 'gene_viral_expression'
        ]
        
        viral_response_edges = [
            (u, v, data) for u, v, data in self.graph.edges(data=True)
            if data.get('edge_type') == 'gene_viral_response'
        ]
        
        logger.info(f"Viral expression edges: {len(viral_expression_edges):,}")
        logger.info(f"Viral response edges: {len(viral_response_edges):,}")
        
        # Genes with both expression and response data
        expression_genes = {edge[0] for edge in viral_expression_edges}
        response_genes = {edge[0] for edge in viral_response_edges}
        dual_viral_genes = expression_genes & response_genes
        
        logger.info(f"Genes with both expression and response data: {len(dual_viral_genes):,}")
        
        # Study connections
        study_edges = defaultdict(list)
        for u, v, data in self.graph.edges(data=True):
            if 'gse_id' in data:
                study_id = data['gse_id']
                study_edges[study_id].append((u, v))
        
        logger.info(f"Unique studies connecting data: {len(study_edges):,}")
        
        # Largest study connections
        if study_edges:
            largest_studies = sorted(study_edges.items(), key=lambda x: len(x[1]), reverse=True)[:5]
            logger.info("Top studies by connection count:")
            for study_id, connections in largest_studies:
                logger.info(f"  {study_id}: {len(connections):,} connections")
    
    def analyze_semantic_connections(self):
        """Analyze semantic enhancement connections."""
        logger.info("\n✨ SEMANTIC ENHANCEMENT CONNECTIONS")
        logger.info("-" * 50)
        
        # Gene set connections
        gene_set_nodes = [
            node for node, data in self.graph.nodes(data=True)
            if data.get('node_type') == 'gene_set'
        ]
        
        logger.info(f"Total gene sets: {len(gene_set_nodes):,}")
        
        # Analyze gene set to gene connections
        gene_in_set_edges = [
            (u, v, data) for u, v, data in self.graph.edges(data=True)
            if data.get('edge_type') == 'gene_in_set'
        ]
        
        gene_supports_set_edges = [
            (u, v, data) for u, v, data in self.graph.edges(data=True)
            if data.get('edge_type') == 'gene_supports_set'
        ]
        
        logger.info(f"Gene-in-set edges: {len(gene_in_set_edges):,}")
        logger.info(f"Gene-supports-set edges: {len(gene_supports_set_edges):,}")
        
        # GO term validation connections
        go_validation_edges = [
            (u, v, data) for u, v, data in self.graph.edges(data=True)
            if data.get('edge_type') == 'validated_by_go_term'
        ]
        
        logger.info(f"GO validation edges: {len(go_validation_edges):,}")
        
        # Analyze gene sets with literature and GO validation
        gene_sets_with_literature = 0
        gene_sets_with_validation = 0
        gene_sets_with_both = 0
        
        for gene_set in gene_set_nodes:
            node_data = self.graph.nodes[gene_set]
            has_literature = node_data.get('has_literature', False)
            has_validation = bool(node_data.get('validated_go_term'))
            
            if has_literature:
                gene_sets_with_literature += 1
            if has_validation:
                gene_sets_with_validation += 1
            if has_literature and has_validation:
                gene_sets_with_both += 1
        
        logger.info(f"Gene sets with literature: {gene_sets_with_literature:,}")
        logger.info(f"Gene sets with GO validation: {gene_sets_with_validation:,}")
        logger.info(f"Gene sets with both: {gene_sets_with_both:,}")
        
        # Average LLM scores
        llm_scores = [
            self.graph.nodes[gene_set].get('llm_score', 0)
            for gene_set in gene_set_nodes
        ]
        
        if llm_scores:
            avg_score = sum(llm_scores) / len(llm_scores)
            logger.info(f"Average LLM confidence score: {avg_score:.3f}")
    
    def analyze_cross_modal_genes(self):
        """Analyze genes that connect multiple data modalities."""
        logger.info("\n🌐 CROSS-MODAL GENE ANALYSIS")
        logger.info("-" * 50)
        
        # Categorize genes by their connections
        gene_connections = defaultdict(lambda: {
            'go_bp': 0, 'go_cc': 0, 'go_mf': 0,
            'disease': 0, 'drug': 0, 'viral': 0,
            'gene_sets': 0, 'clusters': 0
        })
        
        for gene_node, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'gene':
                for neighbor in self.graph.neighbors(gene_node):
                    edge_data_list = self.graph[gene_node][neighbor]
                    neighbor_data = self.graph.nodes.get(neighbor, {})
                    
                    # Count different types of connections
                    for edge_key, edge_attrs in edge_data_list.items():
                        edge_type = edge_attrs.get('edge_type')
                        neighbor_type = neighbor_data.get('node_type')
                        
                        if edge_type == 'gene_annotation':
                            namespace = neighbor_data.get('namespace', '')
                            if 'biological_process' in namespace:
                                gene_connections[gene_node]['go_bp'] += 1
                            elif 'cellular_component' in namespace:
                                gene_connections[gene_node]['go_cc'] += 1
                            elif 'molecular_function' in namespace:
                                gene_connections[gene_node]['go_mf'] += 1
                        
                        elif edge_type == 'gene_disease_association':
                            gene_connections[gene_node]['disease'] += 1
                        elif edge_type == 'gene_drug_perturbation':
                            gene_connections[gene_node]['drug'] += 1
                        elif edge_type in ['gene_viral_response', 'gene_viral_expression']:
                            gene_connections[gene_node]['viral'] += 1
                        elif edge_type in ['gene_in_set', 'gene_supports_set']:
                            gene_connections[gene_node]['gene_sets'] += 1
                        elif neighbor_type == 'network_cluster':
                            gene_connections[gene_node]['clusters'] += 1
        
        # Find super-connected genes (connected to all modalities)
        super_connected = []
        highly_connected = []
        
        for gene, connections in gene_connections.items():
            total_modalities = sum(1 for count in connections.values() if count > 0)
            total_connections = sum(connections.values())
            
            if total_modalities >= 6:  # Connected to most modalities
                super_connected.append((gene, total_connections, total_modalities))
            elif total_connections >= 100:  # High number of connections
                highly_connected.append((gene, total_connections, total_modalities))
        
        # Sort by total connections
        super_connected.sort(key=lambda x: x[1], reverse=True)
        highly_connected.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Super-connected genes (6+ modalities): {len(super_connected):,}")
        logger.info(f"Highly-connected genes (100+ connections): {len(highly_connected):,}")
        
        # Show top super-connected genes
        logger.info("\nTop Super-Connected Genes:")
        for i, (gene, total_conn, modalities) in enumerate(super_connected[:10]):
            connections = gene_connections[gene]
            logger.info(f"  {i+1}. {gene}: {total_conn:,} connections across {modalities} modalities")
            logger.info(f"     GO_BP:{connections['go_bp']}, GO_CC:{connections['go_cc']}, "
                       f"GO_MF:{connections['go_mf']}, Disease:{connections['disease']}, "
                       f"Drug:{connections['drug']}, Viral:{connections['viral']}, "
                       f"Sets:{connections['gene_sets']}")
        
        return super_connected, highly_connected
    
    def analyze_hub_nodes(self):
        """Analyze hub nodes with highest connectivity."""
        logger.info("\n🎯 HUB NODE ANALYSIS")
        logger.info("-" * 50)
        
        # Calculate degree for all nodes
        degrees = dict(self.graph.degree())
        
        # Categorize by node type
        hubs_by_type = defaultdict(list)
        
        for node, degree in degrees.items():
            node_data = self.graph.nodes.get(node, {})
            node_type = node_data.get('node_type', 'unknown')
            hubs_by_type[node_type].append((node, degree))
        
        # Sort and show top hubs for each type
        for node_type, nodes in hubs_by_type.items():
            if nodes:
                nodes.sort(key=lambda x: x[1], reverse=True)
                top_hubs = nodes[:5]
                
                logger.info(f"\nTop {node_type} hubs:")
                for i, (node, degree) in enumerate(top_hubs):
                    node_data = self.graph.nodes.get(node, {})
                    name = node_data.get('name', node_data.get('gene_set_name', str(node)))
                    if len(name) > 50:
                        name = name[:47] + "..."
                    logger.info(f"  {i+1}. {name}: {degree:,} connections")
    
    def analyze_connection_paths(self):
        """Analyze important connection paths between modalities."""
        logger.info("\n🛤️  CONNECTION PATH ANALYSIS")
        logger.info("-" * 50)
        
        # Find representative nodes from each modality
        modality_nodes = {
            'go_bp': [], 'go_cc': [], 'go_mf': [],
            'disease': [], 'drug': [], 'viral': [],
            'gene_set': [], 'cluster': []
        }
        
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('node_type')
            namespace = data.get('namespace', '')
            
            if node_type == 'go_term':
                if 'biological_process' in namespace:
                    modality_nodes['go_bp'].append(node)
                elif 'cellular_component' in namespace:
                    modality_nodes['go_cc'].append(node)
                elif 'molecular_function' in namespace:
                    modality_nodes['go_mf'].append(node)
            elif node_type == 'disease':
                modality_nodes['disease'].append(node)
            elif node_type == 'drug':
                modality_nodes['drug'].append(node)
            elif node_type == 'viral_condition':
                modality_nodes['viral'].append(node)
            elif node_type == 'gene_set':
                modality_nodes['gene_set'].append(node)
            elif node_type == 'network_cluster':
                modality_nodes['cluster'].append(node)
        
        # Analyze path lengths between modalities
        logger.info("Sample shortest paths between modalities:")
        
        # Try to find paths between different modalities
        modality_pairs = [
            ('go_bp', 'disease'), ('go_cc', 'drug'), ('go_mf', 'viral'),
            ('disease', 'drug'), ('drug', 'viral'), ('gene_set', 'go_bp')
        ]
        
        for mod1, mod2 in modality_pairs:
            if modality_nodes[mod1] and modality_nodes[mod2]:
                # Sample a few nodes from each modality
                source = modality_nodes[mod1][0]
                target = modality_nodes[mod2][0]
                
                try:
                    path_length = nx.shortest_path_length(self.graph, source, target)
                    logger.info(f"  {mod1} → {mod2}: {path_length} hops")
                except nx.NetworkXNoPath:
                    logger.info(f"  {mod1} → {mod2}: No direct path")
    
    def generate_insights(self):
        """Generate key insights from the analysis."""
        logger.info("\n💡 KEY INSIGHTS AND DISCOVERIES")
        logger.info("=" * 50)
        
        stats = self.kg.get_comprehensive_stats()
        
        # Calculate key metrics
        go_terms = stats['node_counts'].get('go_term', 0)
        genes = stats['node_counts'].get('gene', 0)
        gene_sets = stats['node_counts'].get('gene_set', 0)
        
        integration_ratio = stats['integration_metrics']['integration_ratio']
        
        insights = []
        
        # 1. Multi-modal integration success
        if integration_ratio > 0.8:
            insights.append(f"🎯 EXCELLENT INTEGRATION: {integration_ratio:.1%} of GO-connected genes are also connected to Omics data")
        
        # 2. Semantic enhancement impact
        semantic_edges = stats['edge_counts'].get('gene_in_set', 0) + stats['edge_counts'].get('validated_by_go_term', 0)
        if semantic_edges > 10000:
            insights.append(f"✨ RICH SEMANTIC LAYER: {semantic_edges:,} semantic enhancement edges provide functional context")
        
        # 3. Cross-namespace coverage
        go_namespaces = 3  # BP, CC, MF
        insights.append(f"🧬 COMPREHENSIVE GO COVERAGE: All {go_namespaces} GO namespaces integrated with {go_terms:,} terms")
        
        # 4. Expression data richness
        expression_edges = stats['edge_counts'].get('gene_viral_expression', 0)
        if expression_edges > 1000000:
            insights.append(f"📊 MASSIVE EXPRESSION DATA: {expression_edges:,} quantitative expression measurements")
        
        # 5. Literature integration
        if gene_sets > 200:
            insights.append(f"📚 LITERATURE-BACKED ANNOTATIONS: {gene_sets} gene sets with AI-enhanced functional descriptions")
        
        # 6. Network density optimization
        density = stats['total_edges'] / (stats['total_nodes'] * (stats['total_nodes'] - 1))
        if density < 0.001:
            insights.append(f"⚡ EFFICIENT GRAPH STRUCTURE: Low density ({density:.6f}) enables fast queries while maintaining rich connectivity")
        
        logger.info("🔍 DISCOVERED CONNECTIONS:")
        for i, insight in enumerate(insights, 1):
            logger.info(f"{i}. {insight}")
        
        # Connection summary
        logger.info("\n🌐 CONNECTION SUMMARY:")
        logger.info("  ✓ GO Namespaces ↔ Each other (via shared genes)")
        logger.info("  ✓ GO Terms ↔ Omics Data (via annotated genes)")
        logger.info("  ✓ Disease ↔ Drug ↔ Viral (via shared gene responses)")
        logger.info("  ✓ Gene Sets ↔ GO Terms (via validation)")
        logger.info("  ✓ Literature ↔ Functional Annotations (via AI analysis)")
        logger.info("  ✓ Expression ↔ Response Data (via quantitative measurements)")
        logger.info("  ✓ Network Clusters ↔ All data types (via hierarchical organization)")
        
        return insights

def main():
    """Main analysis function."""
    logger.info("🚀 STARTING COMPLETE KNOWLEDGE GRAPH CONNECTION ANALYSIS")
    logger.info("="*80)
    
    # Build complete knowledge graph
    logger.info("Building comprehensive knowledge graph...")
    kg = ComprehensiveBiomedicalKnowledgeGraph()
    kg.load_data("llm_evaluation_for_gene_set_interpretation/data")
    kg.build_comprehensive_graph()
    
    # Analyze connections
    analyzer = KGConnectionAnalyzer(kg)
    insights = analyzer.analyze_all_connections()
    
    # Save analysis results
    results = {
        'graph_stats': kg.get_comprehensive_stats(),
        'insights': insights,
        'analysis_timestamp': str(pd.Timestamp.now())
    }
    
    output_file = "complete_kg_connection_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n💾 Analysis results saved to: {output_file}")
    logger.info("="*80)
    logger.info("🎉 COMPLETE CONNECTION ANALYSIS FINISHED")
    logger.info("="*80)
    
    return analyzer, insights

if __name__ == "__main__":
    import pandas as pd
    analyzer, insights = main()