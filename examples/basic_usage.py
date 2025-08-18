#!/usr/bin/env python3
"""
Basic usage example for the GO_BP Knowledge Graph.

This example demonstrates how to:
1. Load and parse GO_BP data
2. Build the comprehensive knowledge graph
3. Query gene functions and GO relationships
4. Search for specific biological processes
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from kg_builder import GOBPKnowledgeGraph

def main():
    """Demonstrate basic usage of the GO_BP knowledge graph."""
    
    print("=" * 60)
    print("GO_BP KNOWLEDGE GRAPH - BASIC USAGE EXAMPLE")
    print("=" * 60)
    
    # Initialize the knowledge graph
    kg = GOBPKnowledgeGraph(use_neo4j=False)
    
    # Load data (update path as needed)
    data_dir = "../llm_evaluation_for_gene_set_interpretation/data/GO_BP"
    print(f"Loading data from: {data_dir}")
    
    try:
        kg.load_data(data_dir)
        kg.build_graph()
        
        # Get statistics
        stats = kg.get_stats()
        print(f"\nKnowledge Graph Statistics:")
        print(f"  Total nodes: {stats['total_nodes']:,}")
        print(f"  Total edges: {stats['total_edges']:,}")
        print(f"  GO terms: {stats['go_terms']:,}")
        print(f"  Genes: {stats['genes']:,}")
        print(f"  Gene associations: {stats['gene_associations']:,}")
        
        # Example 1: Query gene functions
        print(f"\n" + "="*60)
        print("EXAMPLE 1: Gene Function Query")
        print("="*60)
        
        gene = "TP53"
        if gene in kg.graph:
            functions = kg.query_gene_functions(gene)
            print(f"\n{gene} is associated with {len(functions)} GO terms:")
            for func in functions[:5]:  # Show first 5
                print(f"  {func['go_id']}: {func['go_name']}")
                print(f"    Evidence: {func['evidence_code']}")
        else:
            print(f"Gene {gene} not found in graph")
        
        # Example 2: Search GO terms by definition
        print(f"\n" + "="*60)
        print("EXAMPLE 2: GO Term Search")
        print("="*60)
        
        search_term = "DNA repair"
        results = kg.search_go_terms_by_definition(search_term)
        print(f"\nFound {len(results)} GO terms matching '{search_term}':")
        for result in results[:3]:  # Show top 3
            print(f"  {result['go_id']}: {result['name']}")
            print(f"    Relevance score: {result['score']}")
            print(f"    Definition: {result['definition'][:100]}...")
        
        # Example 3: GO term hierarchy
        print(f"\n" + "="*60)
        print("EXAMPLE 3: GO Term Hierarchy")
        print("="*60)
        
        if results:
            go_term = results[0]['go_id']
            parents = kg.query_go_hierarchy(go_term, 'parents')
            children = kg.query_go_hierarchy(go_term, 'children')
            
            print(f"\nHierarchy for {go_term}:")
            print(f"  Parent terms: {len(parents)}")
            for parent in parents[:3]:
                print(f"    {parent['go_id']}: {parent['go_name']} ({parent['relationship_type']})")
            
            print(f"  Child terms: {len(children)}")
            for child in children[:3]:
                print(f"    {child['go_id']}: {child['go_name']} ({child['relationship_type']})")
        
        # Example 4: Gene cross-references
        print(f"\n" + "="*60)
        print("EXAMPLE 4: Gene Cross-References")
        print("="*60)
        
        if gene in kg.graph:
            cross_refs = kg.get_gene_cross_references(gene)
            print(f"\nCross-references for {gene}:")
            for key, value in cross_refs.items():
                print(f"  {key}: {value}")
        
        # Example 5: Graph validation
        print(f"\n" + "="*60)
        print("EXAMPLE 5: Graph Validation")
        print("="*60)
        
        validation = kg.validate_graph_integrity()
        print(f"\nGraph integrity check:")
        for check, result in validation.items():
            status = "✓" if result else "✗"
            print(f"  {status} {check}: {result}")
        
        print(f"\n" + "="*60)
        print("BASIC USAGE EXAMPLE COMPLETE")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"Error: Data directory not found. Please update the data_dir path.")
        print(f"Looking for: {data_dir}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()