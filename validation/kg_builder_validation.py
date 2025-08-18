#!/usr/bin/env python3
"""
Validation script to identify missing functionality in kg_builder.py 
based on the comprehensive parser capabilities.
"""

import sys
sys.path.append('/home/mreddy1/knowledge_graph/src')

from data_parsers import GOBPDataParser
from kg_builder import GOBPKnowledgeGraph

def analyze_kg_builder_gaps():
    """Analyze what the KG builder is missing based on parser capabilities."""
    
    data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data/GO_BP"
    parser = GOBPDataParser(data_dir)
    
    print("=" * 80)
    print("KG BUILDER ANALYSIS - IDENTIFYING MISSING FUNCTIONALITY")
    print("=" * 80)
    
    # Parse all data to see what's available
    go_terms = parser.parse_go_terms()
    relationships = parser.parse_go_relationships()
    associations = parser.parse_gene_go_associations_from_gaf()
    alt_ids = parser.parse_go_alternative_ids()
    obo_terms = parser.parse_obo_ontology()
    gene_mappings = parser.parse_gene_identifier_mappings()
    
    # Check collapsed_go data
    collapsed_symbol = parser.parse_collapsed_go_file('symbol')
    collapsed_entrez = parser.parse_collapsed_go_file('entrez')
    collapsed_uniprot = parser.parse_collapsed_go_file('uniprot')
    
    all_collapsed_assoc = parser.parse_all_gene_associations_from_collapsed_files()
    
    print(f"Parser provides:")
    print(f"  GO terms: {len(go_terms):,}")
    print(f"  GO relationships: {len(relationships):,}")
    print(f"  GAF gene associations: {len(associations):,}")
    print(f"  Alternative GO IDs: {len(alt_ids):,}")
    print(f"  OBO enhanced terms: {len(obo_terms):,}")
    print(f"  Gene ID mappings: {sum(len(m) for m in gene_mappings.values()):,}")
    print(f"  Collapsed GO clusters (symbol): {len(collapsed_symbol['clusters']):,}")
    print(f"  Collapsed gene associations:")
    for id_type, assocs in all_collapsed_assoc.items():
        print(f"    {id_type}: {len(assocs):,}")
    
    # Now test KG builder
    print(f"\n" + "=" * 80)
    print("TESTING CURRENT KG BUILDER")
    print("=" * 80)
    
    kg = GOBPKnowledgeGraph(use_neo4j=False)
    kg.load_data(data_dir)
    kg.build_graph()
    
    stats = kg.get_stats()
    print(f"KG Builder creates:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    
    # Identify gaps
    print(f"\n" + "=" * 80)
    print("IDENTIFYING GAPS AND MISSING FUNCTIONALITY")
    print("=" * 80)
    
    gaps = []
    recommendations = []
    
    # Gap 1: Collapsed GO clustering relationships
    clusters_in_kg = len([e for e in kg.graph.edges(data=True) 
                         if e[2].get('edge_type') == 'go_clustering'])
    
    if clusters_in_kg == 0:
        gaps.append("Missing GO clustering relationships from collapsed_go files")
        recommendations.append("Add _add_go_clusters() method to include GO-GO clustering relationships")
    
    # Gap 2: Multiple gene identifier support
    entrez_genes = set()
    uniprot_genes = set()
    for assoc in all_collapsed_assoc['entrez']:
        entrez_genes.add(assoc['gene_id'])
    for assoc in all_collapsed_assoc['uniprot']:
        uniprot_genes.add(assoc['gene_id'])
    
    gene_nodes = [n for n, d in kg.graph.nodes(data=True) if d.get('node_type') == 'gene']
    
    if len(entrez_genes) > 0 and not any('entrez_id' in kg.graph.nodes[g] for g in gene_nodes[:10]):
        gaps.append("Missing Entrez ID support in gene nodes")
        recommendations.append("Enhance gene nodes with Entrez and UniProt ID attributes")
    
    # Gap 3: Comprehensive gene associations
    total_collapsed_assoc = sum(len(assocs) for assocs in all_collapsed_assoc.values())
    gaf_assoc_count = len(associations)
    
    if total_collapsed_assoc > gaf_assoc_count:
        gaps.append(f"Only using GAF associations ({gaf_assoc_count:,}) but collapsed files have {total_collapsed_assoc:,}")
        recommendations.append("Add method to integrate collapsed file gene associations")
    
    # Gap 4: Gene cross-reference edges
    cross_ref_edges = len([e for e in kg.graph.edges(data=True) 
                          if e[2].get('edge_type') == 'gene_cross_reference'])
    
    if cross_ref_edges == 0:
        gaps.append("Missing gene cross-reference edges between different identifier types")
        recommendations.append("Add _add_gene_cross_references() method")
    
    # Gap 5: Enhanced validation and data quality methods
    if not hasattr(kg, 'validate_graph_integrity'):
        gaps.append("Missing graph validation methods")
        recommendations.append("Add comprehensive graph validation methods")
    
    # Gap 6: Alternative ID nodes
    alt_id_nodes = len([n for n, d in kg.graph.nodes(data=True) if d.get('is_alternative_id')])
    if alt_id_nodes == 0 and len(alt_ids) > 0:
        gaps.append("Alternative GO IDs not added as separate nodes")
        recommendations.append("Consider adding alternative GO IDs as separate nodes with mappings")
    
    # Report findings
    if gaps:
        print(f"❌ FOUND {len(gaps)} GAPS:")
        for i, gap in enumerate(gaps, 1):
            print(f"{i}. {gap}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("✅ NO GAPS FOUND - KG builder fully utilizes parser capabilities")
    
    # Detailed analysis of what should be added
    print(f"\n" + "=" * 80)
    print("DETAILED ENHANCEMENT RECOMMENDATIONS")
    print("=" * 80)
    
    print("1. GO CLUSTERING RELATIONSHIPS:")
    print(f"   Should add {len(collapsed_symbol['clusters']):,} cluster parent-child relationships")
    print(f"   Format: cluster_parent --[clusters]--> child_go_term")
    
    print("2. MULTIPLE GENE IDENTIFIER SUPPORT:")
    print(f"   Symbol genes: {len(set(a['gene_id'] for a in all_collapsed_assoc['symbol'])):,}")
    print(f"   Entrez genes: {len(entrez_genes):,}")
    print(f"   UniProt genes: {len(uniprot_genes):,}")
    print(f"   Should create unified gene nodes with all identifier types")
    
    print("3. COMPREHENSIVE GENE ASSOCIATIONS:")
    print(f"   GAF: {len(associations):,} associations")
    print(f"   Collapsed symbol: {len(all_collapsed_assoc['symbol']):,} associations")
    print(f"   Collapsed entrez: {len(all_collapsed_assoc['entrez']):,} associations")
    print(f"   Collapsed uniprot: {len(all_collapsed_assoc['uniprot']):,} associations")
    print(f"   Should integrate all association sources")
    
    print("4. GENE CROSS-REFERENCES:")
    print(f"   Should add {sum(len(m) for m in gene_mappings.values()):,} cross-reference edges")
    print(f"   Types: {list(gene_mappings.keys())}")
    
    return len(gaps) == 0

if __name__ == "__main__":
    success = analyze_kg_builder_gaps()
    exit(0 if success else 1)