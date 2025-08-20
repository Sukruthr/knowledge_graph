#!/usr/bin/env python3
"""
Demonstrate the power of interconnections in the comprehensive knowledge graph.
Shows how connections enable discovery across all data modalities.
"""

import sys
sys.path.append('src')

from kg_builder import ComprehensiveBiomedicalKnowledgeGraph

def demonstrate_cross_modal_discovery():
    """Demonstrate cross-modal gene discovery capabilities."""
    print("🧬 BUILDING COMPREHENSIVE KNOWLEDGE GRAPH...")
    kg = ComprehensiveBiomedicalKnowledgeGraph()
    kg.load_data("llm_evaluation_for_gene_set_interpretation/data")
    kg.build_comprehensive_graph()
    
    print("\n🔍 DEMONSTRATING CROSS-MODAL CONNECTIONS")
    print("="*60)
    
    # Example 1: TP53 - Show all modality connections
    print("\n1. 🎯 TP53 COMPREHENSIVE PROFILE")
    print("-" * 40)
    
    tp53_profile = kg.query_gene_comprehensive('TP53')
    
    print(f"GO Annotations: {len(tp53_profile['go_annotations'])} across all namespaces")
    print(f"Disease Associations: {len(tp53_profile['disease_associations'])}")
    print(f"Drug Perturbations: {len(tp53_profile['drug_perturbations'])}")
    print(f"Viral Responses: {len(tp53_profile['viral_responses'])}")
    print(f"Gene Set Memberships: {len(tp53_profile['gene_set_memberships'])}")
    
    # Show sample from each modality
    if tp53_profile['disease_associations']:
        sample_disease = tp53_profile['disease_associations'][0]
        print(f"Sample Disease: {sample_disease['disease']}")
    
    if tp53_profile['drug_perturbations']:
        sample_drug = tp53_profile['drug_perturbations'][0]
        print(f"Sample Drug: {sample_drug['drug']}")
    
    if tp53_profile['gene_set_memberships']:
        sample_set = tp53_profile['gene_set_memberships'][0]
        print(f"Sample Function: {sample_set['llm_name']} (score: {sample_set['llm_score']:.3f})")
    
    # Example 2: Cross-modal therapeutic discovery
    print("\n2. 💊 CROSS-MODAL THERAPEUTIC DISCOVERY")
    print("-" * 40)
    
    # Find genes affected by both cancer drugs and cancer diseases
    cancer_genes = set()
    
    # Sample a few major genes to demonstrate
    test_genes = ['BRCA1', 'EGFR', 'MYC', 'TP53', 'PTEN']
    
    for gene in test_genes:
        profile = kg.query_gene_comprehensive(gene)
        
        has_cancer_disease = any('cancer' in disease['disease'].lower() or 'tumor' in disease['disease'].lower() 
                               for disease in profile.get('disease_associations', []))
        
        has_cancer_drug = any('cisplatin' in drug['drug'].lower() or 'doxorubicin' in drug['drug'].lower()
                            for drug in profile.get('drug_perturbations', []))
        
        if has_cancer_disease and has_cancer_drug:
            cancer_genes.add(gene)
            print(f"✓ {gene}: Connected to cancer diseases AND cancer drugs")
    
    print(f"\nFound {len(cancer_genes)} genes bridging cancer diseases and treatments")
    
    # Example 3: Viral-Drug interaction discovery
    print("\n3. 🦠 VIRAL-DRUG INTERACTION ANALYSIS")
    print("-" * 40)
    
    # Find genes that respond to both antiviral drugs and viral infections
    antiviral_responsive_genes = set()
    
    for gene in test_genes:
        profile = kg.query_gene_comprehensive(gene)
        
        has_viral_response = len(profile.get('viral_responses', [])) > 0
        has_drug_response = len(profile.get('drug_perturbations', [])) > 0
        
        if has_viral_response and has_drug_response:
            antiviral_responsive_genes.add(gene)
            viral_count = len(profile['viral_responses'])
            drug_count = len(profile['drug_perturbations'])
            print(f"✓ {gene}: {viral_count} viral responses, {drug_count} drug responses")
    
    # Example 4: GO-Semantic validation
    print("\n4. ✨ GO-SEMANTIC VALIDATION")
    print("-" * 40)
    
    for gene in test_genes:
        profile = kg.query_gene_comprehensive(gene)
        
        go_functions = [ann['go_name'] for ann in profile.get('go_annotations', [])]
        semantic_functions = [mem['llm_name'] for mem in profile.get('gene_set_memberships', [])]
        
        if go_functions and semantic_functions:
            print(f"\n{gene}:")
            print(f"  GO functions (sample): {go_functions[0] if go_functions else 'None'}")
            print(f"  Semantic functions: {', '.join(semantic_functions) if semantic_functions else 'None'}")
    
    # Example 5: Network connectivity statistics
    print("\n5. 🌐 CONNECTIVITY STATISTICS")
    print("-" * 40)
    
    stats = kg.get_comprehensive_stats()
    
    print(f"Total interconnected nodes: {stats['total_nodes']:,}")
    print(f"Total connection edges: {stats['total_edges']:,}")
    print(f"Cross-modal integration: {stats['integration_metrics']['integration_ratio']:.1%}")
    
    print(f"\nData modality breakdown:")
    for node_type, count in stats['node_counts'].items():
        print(f"  {node_type}: {count:,}")
    
    print(f"\nConnection type breakdown:")
    for edge_type, count in stats['edge_counts'].items():
        percentage = (count / stats['total_edges']) * 100
        print(f"  {edge_type}: {count:,} ({percentage:.1f}%)")
    
    print("\n🎉 DEMONSTRATION COMPLETE!")
    print("="*60)
    print("This knowledge graph enables:")
    print("✓ Multi-modal gene analysis")
    print("✓ Cross-domain therapeutic discovery") 
    print("✓ Viral-drug interaction mapping")
    print("✓ AI-enhanced functional validation")
    print("✓ Evidence-based biomedical research")

if __name__ == "__main__":
    demonstrate_cross_modal_discovery()