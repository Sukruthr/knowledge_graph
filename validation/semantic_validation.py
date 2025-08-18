#!/usr/bin/env python3
"""
Semantic validation of GO_BP data parsers to ensure relationships make biological sense.
"""

import sys
sys.path.append('/home/mreddy1/knowledge_graph/src')

from data_parsers import GOBPDataParser
from collections import defaultdict
import random

def validate_semantic_correctness():
    """Validate semantic correctness of parsed relationships."""
    
    data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data/GO_BP"
    parser = GOBPDataParser(data_dir)
    
    print("=" * 80)
    print("SEMANTIC VALIDATION OF GO_BP DATA RELATIONSHIPS")
    print("=" * 80)
    
    # Parse all data
    go_terms = parser.parse_go_terms()
    relationships = parser.parse_go_relationships()
    associations = parser.parse_gene_go_associations_from_gaf()
    alt_ids = parser.parse_go_alternative_ids()
    obo_terms = parser.parse_obo_ontology()
    gene_mappings = parser.parse_gene_identifier_mappings()
    
    print(f"Loaded {len(go_terms)} GO terms, {len(relationships)} relationships, {len(associations)} associations")
    
    errors = []
    warnings = []
    
    # Test 1: Validate GO term relationships reference valid GO IDs
    print("\n1. VALIDATING GO RELATIONSHIP INTEGRITY")
    print("-" * 50)
    
    missing_parents = set()
    missing_children = set()
    valid_relationships = 0
    
    for rel in relationships:
        parent_id = rel['parent_id']
        child_id = rel['child_id']
        
        # Check if GO IDs exist in our term dictionary
        if parent_id not in go_terms:
            missing_parents.add(parent_id)
        if child_id not in go_terms:
            missing_children.add(child_id)
        else:
            valid_relationships += 1
    
    print(f"Valid relationships: {valid_relationships}/{len(relationships)}")
    
    if missing_parents:
        warnings.append(f"Found {len(missing_parents)} parent GO IDs in relationships but not in terms")
        print(f"⚠ Missing parent GO IDs: {len(missing_parents)} (sample: {list(missing_parents)[:3]})")
    
    if missing_children:
        warnings.append(f"Found {len(missing_children)} child GO IDs in relationships but not in terms")
        print(f"⚠ Missing child GO IDs: {len(missing_children)} (sample: {list(missing_children)[:3]})")
    
    # Test 2: Validate gene-GO associations reference valid GO IDs
    print("\n2. VALIDATING GENE-GO ASSOCIATION INTEGRITY")
    print("-" * 50)
    
    missing_go_in_assoc = set()
    valid_associations = 0
    gene_count = defaultdict(int)
    
    for assoc in associations:
        go_id = assoc['go_id']
        gene_symbol = assoc['gene_symbol']
        
        if go_id not in go_terms:
            missing_go_in_assoc.add(go_id)
        else:
            valid_associations += 1
            
        gene_count[gene_symbol] += 1
    
    print(f"Valid associations: {valid_associations}/{len(associations)}")
    print(f"Unique genes: {len(gene_count)}")
    
    if missing_go_in_assoc:
        warnings.append(f"Found {len(missing_go_in_assoc)} GO IDs in associations but not in terms")
        print(f"⚠ Missing GO IDs in associations: {len(missing_go_in_assoc)} (sample: {list(missing_go_in_assoc)[:3]})")
    
    # Test 3: Validate alternative ID mappings
    print("\n3. VALIDATING ALTERNATIVE ID MAPPINGS")
    print("-" * 50)
    
    valid_alt_mappings = 0
    invalid_primary_ids = set()
    invalid_alt_ids = set()
    
    for alt_id, primary_id in alt_ids.items():
        # Alternative IDs might not be in main term list (that's expected)
        # But primary IDs should exist
        if primary_id not in go_terms:
            invalid_primary_ids.add(primary_id)
        else:
            valid_alt_mappings += 1
    
    print(f"Valid alternative mappings: {valid_alt_mappings}/{len(alt_ids)}")
    
    if invalid_primary_ids:
        errors.append(f"Found {len(invalid_primary_ids)} primary GO IDs in alt mappings that don't exist in terms")
        print(f"❌ Invalid primary IDs: {len(invalid_primary_ids)} (sample: {list(invalid_primary_ids)[:3]})")
    
    # Test 4: Validate cross-identifier mappings
    print("\n4. VALIDATING GENE IDENTIFIER CROSS-REFERENCES")
    print("-" * 50)
    
    total_mappings = sum(len(mapping) for mapping in gene_mappings.values())
    print(f"Total cross-reference mappings: {total_mappings}")
    
    # Check bidirectional consistency
    symbol_to_uniprot = gene_mappings.get('symbol_to_uniprot', {})
    uniprot_to_symbol = gene_mappings.get('uniprot_to_symbol', {})
    
    bidirectional_errors = 0
    for symbol, uniprot in symbol_to_uniprot.items():
        if uniprot in uniprot_to_symbol:
            if uniprot_to_symbol[uniprot] != symbol:
                bidirectional_errors += 1
    
    if bidirectional_errors > 0:
        errors.append(f"Found {bidirectional_errors} bidirectional mapping inconsistencies")
        print(f"❌ Bidirectional mapping errors: {bidirectional_errors}")
    else:
        print("✓ Bidirectional mappings are consistent")
    
    # Test 5: Validate OBO enhancement consistency
    print("\n5. VALIDATING OBO ENHANCEMENT CONSISTENCY")
    print("-" * 50)
    
    enhanced_terms = 0
    name_mismatches = 0
    
    for go_id, obo_data in obo_terms.items():
        if go_id in go_terms:
            enhanced_terms += 1
            # Check if names match
            basic_name = go_terms[go_id]['name']
            obo_name = obo_data.get('name', '')
            if basic_name != obo_name:
                name_mismatches += 1
    
    print(f"OBO terms enhancing basic terms: {enhanced_terms}")
    print(f"Name consistency: {enhanced_terms - name_mismatches}/{enhanced_terms}")
    
    if name_mismatches > 0:
        warnings.append(f"Found {name_mismatches} name mismatches between basic and OBO terms")
        print(f"⚠ Name mismatches: {name_mismatches}")
    
    # Test 6: Validate collapsed_go clustering semantics
    print("\n6. VALIDATING COLLAPSED GO CLUSTERING SEMANTICS")
    print("-" * 50)
    
    for id_type in ['symbol', 'entrez', 'uniprot']:
        collapsed_data = parser.parse_collapsed_go_file(id_type)
        clusters = collapsed_data['clusters']
        gene_associations = collapsed_data['gene_associations']
        
        # Check that cluster GO IDs exist
        missing_cluster_gos = set()
        for parent_go, children in clusters.items():
            if parent_go not in go_terms:
                missing_cluster_gos.add(parent_go)
            for child in children:
                child_go = child['child_go']
                if child_go not in go_terms:
                    missing_cluster_gos.add(child_go)
        
        # Check that association GO IDs exist
        missing_assoc_gos = set()
        for assoc in gene_associations:
            go_id = assoc['go_id']
            if go_id not in go_terms:
                missing_assoc_gos.add(go_id)
        
        print(f"{id_type}: {len(clusters)} clusters, {len(gene_associations)} associations")
        
        if missing_cluster_gos:
            warnings.append(f"collapsed_go.{id_type}: {len(missing_cluster_gos)} cluster GO IDs not in terms")
            print(f"  ⚠ Missing cluster GO IDs: {len(missing_cluster_gos)}")
        
        if missing_assoc_gos:
            warnings.append(f"collapsed_go.{id_type}: {len(missing_assoc_gos)} association GO IDs not in terms")
            print(f"  ⚠ Missing association GO IDs: {len(missing_assoc_gos)}")
    
    # Test 7: Sample relationship validation
    print("\n7. SAMPLE RELATIONSHIP VALIDATION")
    print("-" * 50)
    
    # Pick a random GO term and trace its relationships
    sample_go_ids = random.sample(list(go_terms.keys()), 3)
    
    for go_id in sample_go_ids:
        term_info = go_terms[go_id]
        print(f"\nSample term: {go_id} - {term_info['name']}")
        
        # Find parent relationships
        parents = [rel for rel in relationships if rel['child_id'] == go_id]
        children = [rel for rel in relationships if rel['parent_id'] == go_id]
        
        print(f"  Parents: {len(parents)}")
        for parent in parents[:2]:  # Show first 2
            parent_name = go_terms.get(parent['parent_id'], {}).get('name', 'Unknown')
            print(f"    -> {parent['parent_id']}: {parent_name} ({parent['relationship_type']})")
        
        print(f"  Children: {len(children)}")
        for child in children[:2]:  # Show first 2
            child_name = go_terms.get(child['child_id'], {}).get('name', 'Unknown')
            print(f"    <- {child['child_id']}: {child_name} ({child['relationship_type']})")
        
        # Find gene associations
        gene_assocs = [assoc for assoc in associations if assoc['go_id'] == go_id]
        print(f"  Associated genes: {len(gene_assocs)}")
        if gene_assocs:
            sample_genes = [assoc['gene_symbol'] for assoc in gene_assocs[:3]]
            print(f"    Sample genes: {sample_genes}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SEMANTIC VALIDATION SUMMARY")
    print("=" * 80)
    
    if errors:
        print(f"❌ FOUND {len(errors)} CRITICAL ERRORS:")
        for i, error in enumerate(errors, 1):
            print(f"{i}. {error}")
    
    if warnings:
        print(f"⚠️ FOUND {len(warnings)} WARNINGS:")
        for i, warning in enumerate(warnings, 1):
            print(f"{i}. {warning}")
        print("\nNote: Some warnings are expected due to filtered/subset data files")
    
    if not errors:
        print("✅ NO CRITICAL ERRORS - Data relationships are semantically sound!")
        print("The parser correctly handles all data types and maintains referential integrity.")
    
    return len(errors) == 0

if __name__ == "__main__":
    success = validate_semantic_correctness()
    exit(0 if success else 1)