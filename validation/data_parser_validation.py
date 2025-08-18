#!/usr/bin/env python3
"""
Comprehensive validation script for GO_BP data parsers.
This script cross-checks each file against its parser function to ensure correctness.
"""

import sys
sys.path.append('/home/mreddy1/knowledge_graph/src')

from data_parsers import GOBPDataParser
import gzip
from pathlib import Path
import pandas as pd

def validate_file_parser_mapping():
    """Validate each file against its corresponding parser function."""
    
    data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data/GO_BP"
    parser = GOBPDataParser(data_dir)
    
    print("=" * 80)
    print("COMPREHENSIVE DATA PARSER VALIDATION")
    print("=" * 80)
    
    errors = []
    
    # Test 1: goID_2_name.tab validation
    print("\n1. VALIDATING goID_2_name.tab")
    print("-" * 40)
    
    try:
        # Read raw file
        names_file = Path(data_dir) / "goID_2_name.tab"
        with open(names_file, 'r') as f:
            first_line = f.readline().strip()
            raw_lines = []
            for i, line in enumerate(f):
                if i < 5:  # Check first 5 data lines
                    raw_lines.append(line.strip().split('\t'))
        
        # Use parser
        parsed_terms = parser.parse_go_terms()
        
        print(f"Raw file first line: {first_line}")
        print(f"Raw file sample: {raw_lines[0] if raw_lines else 'No data'}")
        print(f"Parser extracted {len(parsed_terms)} GO terms")
        
        # Validate specific entries
        if raw_lines:
            raw_go_id = raw_lines[0][0]
            raw_name = raw_lines[0][1]
            if raw_go_id in parsed_terms:
                parsed_name = parsed_terms[raw_go_id]['name']
                if raw_name == parsed_name:
                    print(f"✓ Validation passed: {raw_go_id} -> {raw_name}")
                else:
                    errors.append(f"Name mismatch for {raw_go_id}: raw='{raw_name}' vs parsed='{parsed_name}'")
            else:
                errors.append(f"GO ID {raw_go_id} missing from parsed data")
                
    except Exception as e:
        errors.append(f"Error validating goID_2_name.tab: {e}")
    
    # Test 2: go.tab validation (relationships)
    print("\n2. VALIDATING go.tab")
    print("-" * 40)
    
    try:
        relationships_file = Path(data_dir) / "go.tab"
        relationships_df = pd.read_csv(relationships_file, sep='\t', header=None,
                                     names=['parent_id', 'child_id', 'relationship_type', 'namespace'])
        
        parsed_relationships = parser.parse_go_relationships()
        
        print(f"Raw file contains {len(relationships_df)} relationships")
        print(f"Parser extracted {len(parsed_relationships)} relationships")
        
        # Check first relationship
        first_raw = relationships_df.iloc[0]
        first_parsed = parsed_relationships[0] if parsed_relationships else None
        
        if first_parsed:
            if (first_raw['parent_id'] == first_parsed['parent_id'] and
                first_raw['child_id'] == first_parsed['child_id'] and
                first_raw['relationship_type'] == first_parsed['relationship_type']):
                print(f"✓ Validation passed: {first_raw['child_id']} -> {first_raw['parent_id']} ({first_raw['relationship_type']})")
            else:
                errors.append("First relationship mismatch between raw and parsed data")
        
        # Validate relationship types
        raw_rel_types = set(relationships_df['relationship_type'].unique())
        parsed_rel_types = set(rel['relationship_type'] for rel in parsed_relationships)
        if raw_rel_types == parsed_rel_types:
            print(f"✓ Relationship types match: {raw_rel_types}")
        else:
            errors.append(f"Relationship types mismatch: raw={raw_rel_types} vs parsed={parsed_rel_types}")
            
    except Exception as e:
        errors.append(f"Error validating go.tab: {e}")
    
    # Test 3: GAF file validation
    print("\n3. VALIDATING goa_human.gaf.gz")
    print("-" * 40)
    
    try:
        gaf_file = Path(data_dir) / "goa_human.gaf.gz"
        
        # Count raw GAF entries
        raw_count = 0
        raw_bp_count = 0
        sample_raw = None
        
        with gzip.open(gaf_file, 'rt') as f:
            for line in f:
                if line.startswith('!'):
                    continue
                raw_count += 1
                parts = line.strip().split('\t')
                if len(parts) >= 15:
                    if parts[8] == 'P':  # biological process
                        raw_bp_count += 1
                        if sample_raw is None:
                            sample_raw = {
                                'uniprot_id': parts[1],
                                'gene_symbol': parts[2], 
                                'go_id': parts[4],
                                'aspect': parts[8]
                            }
        
        # Use parser
        parsed_associations = parser.parse_gene_go_associations_from_gaf()
        
        print(f"Raw GAF entries: {raw_count}")
        print(f"Raw biological process entries: {raw_bp_count}")
        print(f"Parser extracted: {len(parsed_associations)} BP associations")
        
        if sample_raw and parsed_associations:
            # Find matching association in parsed data
            found_match = False
            for assoc in parsed_associations[:100]:  # Check first 100
                if (assoc['uniprot_id'] == sample_raw['uniprot_id'] and
                    assoc['gene_symbol'] == sample_raw['gene_symbol'] and
                    assoc['go_id'] == sample_raw['go_id']):
                    found_match = True
                    print(f"✓ Sample match found: {sample_raw['gene_symbol']} -> {sample_raw['go_id']}")
                    break
            
            if not found_match:
                print(f"⚠ Sample not found in first 100 parsed entries")
                
    except Exception as e:
        errors.append(f"Error validating GAF file: {e}")
    
    # Test 4: collapsed_go files validation
    print("\n4. VALIDATING collapsed_go files")
    print("-" * 40)
    
    for id_type in ['symbol', 'entrez', 'uniprot']:
        try:
            collapsed_file = Path(data_dir) / f"collapsed_go.{id_type}"
            
            # Count raw entries
            go_go_count = 0
            gene_go_count = 0
            transition_line = None
            
            with open(collapsed_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        if parts[2] == 'default':
                            go_go_count += 1
                        elif parts[2] == 'gene':
                            if transition_line is None:
                                transition_line = line_num
                            gene_go_count += 1
            
            # Use parser
            parsed_data = parser.parse_collapsed_go_file(id_type)
            parsed_clusters = len(parsed_data['clusters'])
            parsed_gene_assoc = len(parsed_data['gene_associations'])
            
            print(f"{id_type}:")
            print(f"  Raw GO-GO clusters: {go_go_count}")
            print(f"  Raw gene associations: {gene_go_count}")
            print(f"  Transition at line: {transition_line}")
            print(f"  Parsed clusters: {parsed_clusters}")
            print(f"  Parsed associations: {parsed_gene_assoc}")
            
            # Validate cluster extraction - this needs fixing!
            if go_go_count > 0 and parsed_clusters == 0:
                errors.append(f"collapsed_go.{id_type}: No clusters parsed despite {go_go_count} raw GO-GO relationships")
            
            if gene_go_count != parsed_gene_assoc:
                errors.append(f"collapsed_go.{id_type}: Gene association count mismatch - raw={gene_go_count}, parsed={parsed_gene_assoc}")
                
        except Exception as e:
            errors.append(f"Error validating collapsed_go.{id_type}: {e}")
    
    # Test 5: Alternative ID validation
    print("\n5. VALIDATING goID_2_alt_id.tab")
    print("-" * 40)
    
    try:
        alt_ids_file = Path(data_dir) / "goID_2_alt_id.tab"
        
        # Count raw entries
        raw_count = 0
        sample_mapping = None
        
        with open(alt_ids_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and parts[0].startswith('GO:') and parts[1].startswith('GO:'):
                    raw_count += 1
                    if sample_mapping is None:
                        sample_mapping = (parts[0], parts[1])  # (primary, alternative)
        
        # Use parser
        parsed_alt_ids = parser.parse_go_alternative_ids()
        
        print(f"Raw alternative ID mappings: {raw_count}")
        print(f"Parsed alternative ID mappings: {len(parsed_alt_ids)}")
        
        if sample_mapping and sample_mapping[1] in parsed_alt_ids:
            if parsed_alt_ids[sample_mapping[1]] == sample_mapping[0]:
                print(f"✓ Sample mapping correct: {sample_mapping[1]} -> {sample_mapping[0]}")
            else:
                errors.append(f"Alternative ID mapping incorrect: expected {sample_mapping[1]} -> {sample_mapping[0]}, got {parsed_alt_ids[sample_mapping[1]]}")
                
    except Exception as e:
        errors.append(f"Error validating goID_2_alt_id.tab: {e}")
    
    # Test 6: OBO file validation
    print("\n6. VALIDATING go-basic-filtered.obo")
    print("-" * 40)
    
    try:
        obo_file = Path(data_dir) / "go-basic-filtered.obo"
        
        # Count raw terms
        raw_terms = 0
        sample_term = None
        
        with open(obo_file, 'r') as f:
            current_term = {}
            for line in f:
                line = line.strip()
                if line == "[Term]":
                    if current_term and 'id' in current_term:
                        raw_terms += 1
                        if sample_term is None:
                            sample_term = current_term.copy()
                    current_term = {}
                elif line.startswith("id: GO:"):
                    current_term['id'] = line.split(": ")[1]
                elif line.startswith("name: "):
                    current_term['name'] = line.split(": ", 1)[1]
                elif line.startswith("def: "):
                    definition = line.split(": \"", 1)[1]
                    if "\" [" in definition:
                        definition = definition.split("\" [")[0]
                    current_term['definition'] = definition
        
        # Use parser
        parsed_obo_terms = parser.parse_obo_ontology()
        
        print(f"Raw OBO terms: {raw_terms}")
        print(f"Parsed OBO terms: {len(parsed_obo_terms)}")
        
        if sample_term and sample_term['id'] in parsed_obo_terms:
            parsed_sample = parsed_obo_terms[sample_term['id']]
            if (sample_term['name'] == parsed_sample.get('name') and
                sample_term.get('definition') == parsed_sample.get('definition')):
                print(f"✓ Sample OBO term correct: {sample_term['id']}")
            else:
                errors.append(f"OBO term parsing mismatch for {sample_term['id']}")
                
    except Exception as e:
        errors.append(f"Error validating OBO file: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    if errors:
        print(f"❌ FOUND {len(errors)} ISSUES:")
        for i, error in enumerate(errors, 1):
            print(f"{i}. {error}")
    else:
        print("✅ ALL VALIDATIONS PASSED - Data parsers are working correctly!")
    
    return len(errors) == 0

if __name__ == "__main__":
    success = validate_file_parser_mapping()
    exit(0 if success else 1)