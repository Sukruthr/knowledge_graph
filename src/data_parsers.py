"""
Data parsers for Gene Ontology Biological Process (GO_BP) data.

This module provides comprehensive utilities to parse all GO_BP data files and extract
structured information for knowledge graph construction, including:
- GO term definitions and relationships
- Gene-GO associations from GAF format
- Cross-identifier mappings (Symbol, Entrez, UniProt)
- Alternative GO ID mappings
- Rich ontology definitions from OBO format
"""

import gzip
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set
import logging

# Configure logging
logger = logging.getLogger(__name__)


class GODataParser:
    """Parser for Gene Ontology data (supports GO_BP, GO_CC, GO_MF)."""
    
    def __init__(self, data_dir: str, namespace: str = None):
        """
        Initialize parser with GO data directory.
        
        Args:
            data_dir: Path to GO data directory containing all required files
            namespace: GO namespace ('biological_process', 'cellular_component', 'molecular_function')
                      If None, will auto-detect from directory name
        """
        self.data_dir = Path(data_dir)
        
        # Auto-detect namespace from directory name if not provided
        if namespace is None:
            dir_name = self.data_dir.name
            namespace_map = {
                'GO_BP': 'biological_process',
                'GO_CC': 'cellular_component', 
                'GO_MF': 'molecular_function'
            }
            self.namespace = namespace_map.get(dir_name, 'biological_process')
        else:
            self.namespace = namespace
        
        # Core data structures
        self.go_terms = {}
        self.go_relationships = []
        self.gene_go_associations = []
        
        # Enhanced data structures
        self.go_alt_ids = {}
        self.gene_id_mappings = {}
        self.collapsed_data = {}  # Cache for collapsed_go file data
        
    def parse_go_terms(self) -> Dict[str, Dict]:
        """
        Parse GO term definitions from goID_2_name.tab and goID_2_namespace.tab.
        
        Returns:
            Dictionary mapping GO IDs to term information
        """
        logger.info("Parsing GO term definitions...")
        
        # Parse GO term names
        names_file = self.data_dir / "goID_2_name.tab"
        names_df = pd.read_csv(names_file, sep='\t', header=None, 
                              names=['go_id', 'name'], index_col=False)
        
        # Parse GO term namespaces  
        namespace_file = self.data_dir / "goID_2_namespace.tab"
        namespace_df = pd.read_csv(namespace_file, sep='\t', header=None,
                                  names=['go_id', 'namespace'], index_col=False)
        
        # Merge the data
        go_terms = {}
        for _, row in names_df.iterrows():
            if pd.notna(row['go_id']) and row['go_id'].startswith('GO:'):
                go_terms[row['go_id']] = {
                    'name': row['name'],
                    'namespace': self.namespace
                }
        
        # Add namespace info if available
        for _, row in namespace_df.iterrows():
            if pd.notna(row['go_id']) and row['go_id'] in go_terms:
                go_terms[row['go_id']]['namespace'] = row['namespace']
        
        logger.info(f"Parsed {len(go_terms)} GO terms")
        self.go_terms = go_terms
        return go_terms
    
    def parse_go_relationships(self) -> List[Dict]:
        """
        Parse GO term relationships from go.tab.
        
        Returns:
            List of relationship dictionaries
        """
        logger.info("Parsing GO term relationships...")
        
        relationships_file = self.data_dir / "go.tab"
        relationships_df = pd.read_csv(relationships_file, sep='\t', header=None,
                                     names=['parent_id', 'child_id', 'relationship_type', 'namespace'])
        
        relationships = []
        for _, row in relationships_df.iterrows():
            if (pd.notna(row['parent_id']) and pd.notna(row['child_id']) and
                row['parent_id'].startswith('GO:') and row['child_id'].startswith('GO:')):
                relationships.append({
                    'parent_id': row['parent_id'],
                    'child_id': row['child_id'],
                    'relationship_type': row['relationship_type'],
                    'namespace': row['namespace']
                })
        
        logger.info(f"Parsed {len(relationships)} GO relationships")
        self.go_relationships = relationships
        return relationships
    
    def parse_gene_go_associations_from_gaf(self) -> List[Dict]:
        """
        Parse gene-GO associations from GAF (Gene Association File) format.
        
        Returns:
            List of gene-GO association dictionaries
        """
        logger.info("Parsing gene-GO associations from GAF file...")
        
        gaf_file = self.data_dir / "goa_human.gaf.gz"
        
        associations = []
        
        # Read compressed GAF file
        with gzip.open(gaf_file, 'rt') as f:
            for line in f:
                # Skip comment lines
                if line.startswith('!'):
                    continue
                    
                parts = line.strip().split('\t')
                if len(parts) >= 15:  # GAF 2.2 format has 15+ columns
                    # GAF format columns:
                    # 0: DB, 1: DB_Object_ID, 2: DB_Object_Symbol, 3: Qualifier,
                    # 4: GO_ID, 5: DB_Reference, 6: Evidence_Code, 7: With_From,
                    # 8: Aspect, 9: DB_Object_Name, 10: DB_Object_Synonym,
                    # 11: DB_Object_Type, 12: Taxon, 13: Date, 14: Assigned_By
                    
                    association = {
                        'database': parts[0],
                        'uniprot_id': parts[1], 
                        'gene_symbol': parts[2],
                        'qualifier': parts[3],
                        'go_id': parts[4],
                        'evidence_code': parts[6],
                        'aspect': parts[8],  # P=biological_process, F=molecular_function, C=cellular_component
                        'gene_name': parts[9],
                        'gene_type': parts[11],
                        'taxon': parts[12],
                        'date': parts[13],
                        'assigned_by': parts[14]
                    }
                    
                    # Filter by namespace aspect
                    namespace_to_aspect = {
                        'biological_process': 'P',
                        'cellular_component': 'C',
                        'molecular_function': 'F'
                    }
                    target_aspect = namespace_to_aspect.get(self.namespace, 'P')
                    
                    if association['aspect'] == target_aspect:
                        associations.append(association)
        
        logger.info(f"Parsed {len(associations)} gene-GO {self.namespace} associations")
        self.gene_go_associations = associations
        return associations
    
    def parse_collapsed_go_file(self, identifier_type: str = 'symbol') -> Dict:
        """
        Parse collapsed_go files which contain both GO clustering and gene associations.
        
        Format structure:
        - Lines 1-27,733: GO-GO clustering relationships (GO_ID -> GO_ID -> 'default')
        - Lines 27,734+: Gene-GO associations (GO_ID -> GENE_ID -> 'gene')
        
        Args:
            identifier_type: Type of file to parse ('symbol', 'entrez', 'uniprot')
            
        Returns:
            Dictionary with 'clusters' and 'gene_associations' keys
        """
        # Check cache first to avoid re-parsing
        if identifier_type in self.collapsed_data:
            return self.collapsed_data[identifier_type]
        
        logger.info(f"Parsing collapsed_go.{identifier_type} file...")
        
        file_mapping = {
            'symbol': 'collapsed_go.symbol',
            'entrez': 'collapsed_go.entrez', 
            'uniprot': 'collapsed_go.uniprot'
        }
        
        if identifier_type not in file_mapping:
            raise ValueError(f"Invalid identifier type: {identifier_type}. Must be one of {list(file_mapping.keys())}")
        
        collapsed_file = self.data_dir / file_mapping[identifier_type]
        if not collapsed_file.exists():
            raise FileNotFoundError(f"File not found: {collapsed_file}")
        
        clusters = {}
        gene_associations = []
        
        with open(collapsed_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                
                col1, col2, col3 = parts[0], parts[1], parts[2]
                
                # GO-GO clustering relationships
                if col1.startswith('GO:') and col2.startswith('GO:') and col3 == 'default':
                    if col1 not in clusters:
                        clusters[col1] = []
                    clusters[col1].append({
                        'child_go': col2,
                        'cluster_type': col3
                    })
                
                # Gene-GO associations  
                elif col1.startswith('GO:') and col3 == 'gene':
                    gene_associations.append({
                        'go_id': col1,
                        'gene_id': col2,
                        'identifier_type': identifier_type,
                        'annotation_type': col3
                    })
        
        result = {
            'clusters': clusters,
            'gene_associations': gene_associations
        }
        
        # Cache the result
        self.collapsed_data[identifier_type] = result
        
        logger.info(f"Parsed {len(clusters)} GO clusters and {len(gene_associations)} {identifier_type} gene associations")
        return result
    
    def parse_go_term_clustering(self, identifier_type: str = 'symbol') -> Dict:
        """
        Parse GO term clustering from collapsed_go files (backward compatibility).
        
        Args:
            identifier_type: Type of file to parse ('symbol', 'entrez', 'uniprot')
            
        Returns:
            Dictionary mapping parent GO terms to child GO terms
        """
        result = self.parse_collapsed_go_file(identifier_type)
        return result['clusters']
    
    def parse_go_alternative_ids(self) -> Dict[str, str]:
        """
        Parse GO alternative/obsolete ID mappings from goID_2_alt_id.tab.
        
        Returns:
            Dictionary mapping alternative GO IDs to primary GO IDs
        """
        logger.info("Parsing GO alternative ID mappings...")
        
        alt_ids_file = self.data_dir / "goID_2_alt_id.tab"
        
        alt_ids = {}
        with open(alt_ids_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and parts[0].startswith('GO:') and parts[1].startswith('GO:'):
                    primary_id = parts[0]
                    alt_id = parts[1]
                    alt_ids[alt_id] = primary_id
        
        logger.info(f"Parsed {len(alt_ids)} alternative GO ID mappings")
        self.go_alt_ids = alt_ids
        return alt_ids
    
    def parse_all_gene_associations_from_collapsed_files(self) -> Dict[str, List]:
        """
        Parse gene associations from all collapsed_go files to get comprehensive 
        gene-GO mappings with different identifier types.
        
        Returns:
            Dictionary with gene associations by identifier type
        """
        logger.info("Parsing gene associations from all collapsed_go files...")
        
        all_associations = {}
        
        # Parse each collapsed_go file
        for id_type in ['symbol', 'entrez', 'uniprot']:
            try:
                collapsed_data = self.parse_collapsed_go_file(id_type)
                all_associations[id_type] = collapsed_data['gene_associations']
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Could not parse {id_type} file: {e}")
                all_associations[id_type] = []
        
        logger.info(f"Parsed gene associations: "
                   f"Symbol={len(all_associations['symbol'])}, "
                   f"Entrez={len(all_associations['entrez'])}, "
                   f"UniProt={len(all_associations['uniprot'])}")
        
        return all_associations
    
    def parse_gene_identifier_mappings(self) -> Dict[str, Dict]:
        """
        Parse comprehensive gene identifier mappings from multiple sources.
        
        Returns:
            Dictionary with mappings between different gene identifier systems
        """
        logger.info("Parsing comprehensive gene identifier mappings...")
        
        mappings = {
            'symbol_to_entrez': {},
            'symbol_to_uniprot': {},
            'entrez_to_symbol': {},
            'uniprot_to_symbol': {},
            'entrez_to_uniprot': {},
            'uniprot_to_entrez': {}
        }
        
        # Method 1: Extract from GAF file (Symbol <-> UniProt)
        gaf_file = self.data_dir / "goa_human.gaf.gz"
        with gzip.open(gaf_file, 'rt') as f:
            for line in f:
                if line.startswith('!'):
                    continue
                    
                parts = line.strip().split('\t')
                if len(parts) >= 15:
                    uniprot_id = parts[1]
                    gene_symbol = parts[2]
                    
                    # Store bidirectional mappings
                    mappings['symbol_to_uniprot'][gene_symbol] = uniprot_id
                    mappings['uniprot_to_symbol'][uniprot_id] = gene_symbol
        
        # Method 2: Cross-reference using collapsed_go files
        # Get gene associations from all collapsed files
        all_associations = self.parse_all_gene_associations_from_collapsed_files()
        
        # Build GO-term based cross-references
        go_to_symbols = {}
        go_to_entrez = {}
        go_to_uniprot = {}
        
        # Group genes by GO terms
        for assoc in all_associations['symbol']:
            go_id = assoc['go_id']
            if go_id not in go_to_symbols:
                go_to_symbols[go_id] = set()
            go_to_symbols[go_id].add(assoc['gene_id'])
        
        for assoc in all_associations['entrez']:
            go_id = assoc['go_id']
            if go_id not in go_to_entrez:
                go_to_entrez[go_id] = set()
            go_to_entrez[go_id].add(assoc['gene_id'])
        
        for assoc in all_associations['uniprot']:
            go_id = assoc['go_id']
            if go_id not in go_to_uniprot:
                go_to_uniprot[go_id] = set()
            go_to_uniprot[go_id].add(assoc['gene_id'])
        
        # Find cross-references through shared GO terms (only for unambiguous mappings)
        self._create_cross_references(go_to_symbols, go_to_entrez, mappings, 'symbol', 'entrez')
        self._create_cross_references(go_to_uniprot, go_to_entrez, mappings, 'uniprot', 'entrez')
        
        logger.info(f"Parsed comprehensive gene ID mappings: "
                   f"Symbol-UniProt: {len(mappings['symbol_to_uniprot'])}, "
                   f"Symbol-Entrez: {len(mappings['symbol_to_entrez'])}, "
                   f"UniProt-Entrez: {len(mappings['uniprot_to_entrez'])}")
        
        self.gene_id_mappings = mappings
        return mappings
    
    def _create_cross_references(self, source_dict: Dict, target_dict: Dict, 
                               mappings: Dict, source_type: str, target_type: str) -> None:
        """
        Helper method to create cross-references between identifier types.
        
        Args:
            source_dict: Dictionary mapping GO terms to source identifiers
            target_dict: Dictionary mapping GO terms to target identifiers  
            mappings: Main mappings dictionary to update
            source_type: Source identifier type (e.g., 'symbol')
            target_type: Target identifier type (e.g., 'entrez')
        """
        source_to_target_key = f"{source_type}_to_{target_type}"
        target_to_source_key = f"{target_type}_to_{source_type}"
        
        for go_id in source_dict:
            if go_id in target_dict:
                source_ids = source_dict[go_id]
                target_ids = target_dict[go_id]
                
                # Only create mappings for unambiguous cases (1:1 mapping)
                if len(source_ids) == 1 and len(target_ids) == 1:
                    source_id = list(source_ids)[0]
                    target_id = list(target_ids)[0]
                    
                    mappings[source_to_target_key][source_id] = target_id
                    mappings[target_to_source_key][target_id] = source_id
    
    def parse_obo_ontology(self) -> Dict[str, Dict]:
        """
        Parse rich GO ontology structure from OBO format file.
        
        Returns:
            Dictionary with enhanced GO term information including definitions and synonyms
        """
        logger.info("Parsing OBO ontology file...")
        
        obo_file = self.data_dir / "go-basic-filtered.obo"
        
        terms = {}
        current_term = None
        
        with open(obo_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                if line == "[Term]":
                    current_term = {}
                elif line.startswith("id: GO:"):
                    go_id = line.split(": ")[1]
                    current_term['id'] = go_id
                elif line.startswith("name: "):
                    current_term['name'] = line.split(": ", 1)[1]
                elif line.startswith("def: "):
                    # Extract definition (remove quotes and reference)
                    definition = line.split(": \"", 1)[1]
                    if "\" [" in definition:
                        definition = definition.split("\" [")[0]
                    current_term['definition'] = definition
                elif line.startswith("synonym: "):
                    if 'synonyms' not in current_term:
                        current_term['synonyms'] = []
                    # Extract synonym (remove quotes and type info)
                    synonym = line.split(": \"", 1)[1]
                    if "\" " in synonym:
                        synonym = synonym.split("\" ")[0]
                    current_term['synonyms'].append(synonym)
                elif line.startswith("namespace: "):
                    current_term['namespace'] = line.split(": ")[1]
                elif line.startswith("is_obsolete: true"):
                    current_term['is_obsolete'] = True
                elif line == "" and current_term and 'id' in current_term:
                    # End of term, store it
                    terms[current_term['id']] = current_term
                    current_term = None
        
        # Store last term if file doesn't end with blank line
        if current_term and 'id' in current_term:
            terms[current_term['id']] = current_term
        
        logger.info(f"Parsed {len(terms)} terms from OBO file")
        return terms
    
    def validate_parsed_data(self) -> Dict[str, bool]:
        """
        Validate the integrity of parsed data.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'has_go_terms': len(self.go_terms) > 0,
            'has_go_relationships': len(self.go_relationships) > 0,
            'has_gene_associations': len(self.gene_go_associations) > 0,
            'has_alt_ids': len(self.go_alt_ids) > 0,
            'has_id_mappings': len(self.gene_id_mappings) > 0,
            'relationships_valid': all(
                rel['parent_id'].startswith('GO:') and rel['child_id'].startswith('GO:')
                for rel in self.go_relationships
            ) if self.go_relationships else False,
            'associations_valid': all(
                assoc['go_id'].startswith('GO:') and assoc['gene_symbol']
                for assoc in self.gene_go_associations
            ) if self.gene_go_associations else False
        }
        
        validation['overall_valid'] = all(validation.values())
        return validation
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics of the parsed data.
        
        Returns:
            Dictionary with data summary
        """
        # Count unique genes if associations are loaded
        unique_genes = set()
        if self.gene_go_associations:
            unique_genes = set(assoc['gene_symbol'] for assoc in self.gene_go_associations)
            
        return {
            'num_go_terms': len(self.go_terms),
            'num_go_relationships': len(self.go_relationships),
            'num_gene_associations': len(self.gene_go_associations),
            'num_unique_genes': len(unique_genes),
            'relationship_types': list(set(r['relationship_type'] for r in self.go_relationships)) if self.go_relationships else [],
            'sample_go_terms': list(self.go_terms.keys())[:5]
        }


# Backward compatibility alias
GOBPDataParser = GODataParser


class CombinedGOParser:
    """Parser for multiple GO namespaces (GO_BP + GO_CC + GO_MF)."""
    
    def __init__(self, base_data_dir: str):
        """
        Initialize combined parser for multiple GO namespaces.
        
        Args:
            base_data_dir: Base directory containing GO_BP, GO_CC, GO_MF subdirectories
        """
        self.base_data_dir = Path(base_data_dir)
        self.parsers = {}
        
        # Initialize parsers for each namespace
        namespace_dirs = {
            'biological_process': 'GO_BP',
            'cellular_component': 'GO_CC',
            'molecular_function': 'GO_MF'
        }
        
        for namespace, dir_name in namespace_dirs.items():
            data_dir = self.base_data_dir / dir_name
            if data_dir.exists():
                self.parsers[namespace] = GODataParser(str(data_dir), namespace)
                logger.info(f"Initialized parser for {namespace}")
            else:
                logger.warning(f"Directory not found: {data_dir}")
    
    def parse_all_namespaces(self) -> Dict[str, Dict]:
        """
        Parse data from all available GO namespaces.
        
        Returns:
            Dictionary with parsed data by namespace
        """
        results = {}
        
        for namespace, parser in self.parsers.items():
            logger.info(f"Parsing {namespace} data...")
            
            results[namespace] = {
                'go_terms': parser.parse_go_terms(),
                'go_relationships': parser.parse_go_relationships(), 
                'gene_associations': parser.parse_gene_go_associations_from_gaf(),
                'alt_ids': parser.parse_go_alternative_ids(),
                'id_mappings': parser.parse_gene_identifier_mappings(),
                'obo_terms': parser.parse_obo_ontology(),
                'collapsed_data': {
                    'symbol': parser.parse_collapsed_go_file('symbol'),
                    'entrez': parser.parse_collapsed_go_file('entrez'),
                    'uniprot': parser.parse_collapsed_go_file('uniprot')
                }
            }
            
            logger.info(f"Completed parsing {namespace}: {len(results[namespace]['go_terms'])} terms")
        
        return results
    
    def get_combined_summary(self) -> Dict:
        """
        Get summary statistics across all parsed namespaces.
        
        Returns:
            Combined summary dictionary
        """
        summary = {
            'namespaces': list(self.parsers.keys()),
            'by_namespace': {}
        }
        
        for namespace, parser in self.parsers.items():
            summary['by_namespace'][namespace] = parser.get_data_summary()
        
        return summary


def main():
    """Comprehensive demonstration of GO data parser capabilities (BP + CC support)."""
    
    print("=" * 80)
    print("COMBINED GO DATA PARSER DEMONSTRATION (GO_BP + GO_CC)")
    print("=" * 80)
    
    # Test combined parser first
    base_data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data"
    combined_parser = CombinedGOParser(base_data_dir)
    
    print(f"\nAvailable namespaces: {list(combined_parser.parsers.keys())}")
    
    # Demonstrate individual namespace parsing
    print("\n" + "="*60)
    print("INDIVIDUAL NAMESPACE DEMONSTRATIONS")
    print("="*60)
    
    for namespace in ['biological_process', 'cellular_component']:
        if namespace in combined_parser.parsers:
            print(f"\n{namespace.upper().replace('_', ' ')} DATA PARSING")
            print("-" * 40)
            
            parser = combined_parser.parsers[namespace]
            
            # Parse core data
            go_terms = parser.parse_go_terms()
            relationships = parser.parse_go_relationships()
            associations = parser.parse_gene_go_associations_from_gaf()
            
            print(f"✓ GO terms: {len(go_terms):,}")
            print(f"✓ GO relationships: {len(relationships):,}")
            print(f"✓ Gene associations: {len(associations):,}")
            
            # Show sample terms
            if go_terms:
                sample_terms = list(go_terms.items())[:2]
                for go_id, info in sample_terms:
                    print(f"  Sample: {go_id} - {info['name']}")
    
    # Original single parser demo for backward compatibility
    print(f"\n\n" + "="*60)
    print("BACKWARD COMPATIBILITY - GO_BP PARSER")
    print("="*60)
    
    # Initialize single parser (backward compatible)
    data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data/GO_BP"
    parser = GOBPDataParser(data_dir)
    
    print("=" * 60)
    print("GO_BP DATA PARSER COMPREHENSIVE DEMONSTRATION")
    print("=" * 60)
    
    # Parse core data structures
    print("\n1. CORE DATA PARSING")
    print("-" * 30)
    
    go_terms = parser.parse_go_terms()
    relationships = parser.parse_go_relationships()
    associations = parser.parse_gene_go_associations_from_gaf()
    
    print(f"✓ GO terms: {len(go_terms):,}")
    print(f"✓ GO relationships: {len(relationships):,}")  
    print(f"✓ Gene-GO associations: {len(associations):,}")
    
    # Parse enhanced data structures
    print("\n2. ENHANCED DATA PARSING")
    print("-" * 30)
    
    alt_ids = parser.parse_go_alternative_ids()
    id_mappings = parser.parse_gene_identifier_mappings()
    obo_terms = parser.parse_obo_ontology()
    
    print(f"✓ Alternative GO IDs: {len(alt_ids):,}")
    print(f"✓ Gene ID cross-references: {sum(len(m) for m in id_mappings.values()):,}")
    print(f"✓ OBO enhanced terms: {len(obo_terms):,}")
    
    # Demonstrate collapsed file parsing
    print("\n3. COLLAPSED FILE ANALYSIS")
    print("-" * 30)
    
    for id_type in ['symbol', 'entrez', 'uniprot']:
        collapsed_data = parser.parse_collapsed_go_file(id_type)
        print(f"✓ {id_type.title()}: {len(collapsed_data['clusters']):,} clusters, "
              f"{len(collapsed_data['gene_associations']):,} gene associations")
    
    # Show validation results
    print("\n4. DATA VALIDATION")
    print("-" * 30)
    
    validation = parser.validate_parsed_data()
    for check, result in validation.items():
        status = "✓" if result else "✗"
        print(f"{status} {check.replace('_', ' ').title()}: {result}")
    
    # Show examples
    print("\n5. EXAMPLES")
    print("-" * 30)
    
    # Sample terms
    sample_terms = list(go_terms.items())[:2]
    for go_id, info in sample_terms:
        print(f"GO Term: {go_id} - {info['name']}")
    
    # Sample cross-reference
    if id_mappings['symbol_to_entrez']:
        sample_symbol = list(id_mappings['symbol_to_entrez'].keys())[0]
        entrez_id = id_mappings['symbol_to_entrez'][sample_symbol]
        uniprot_id = id_mappings['symbol_to_uniprot'].get(sample_symbol, 'N/A')
        print(f"Cross-ref: {sample_symbol} = {entrez_id} (Entrez) = {uniprot_id} (UniProt)")
    
    # Enhanced term example
    if obo_terms:
        sample_obo = list(obo_terms.values())[0]
        if 'definition' in sample_obo:
            print(f"Enhanced: {sample_obo['id']} has full definition and metadata")
    
    # Final summary
    summary = parser.get_data_summary()
    print(f"\n6. FINAL SUMMARY")
    print("-" * 30)
    print(f"Total unique genes: {summary['num_unique_genes']:,}")
    print(f"Relationship types: {', '.join(summary['relationship_types'])}")
    print(f"Overall validation: {'PASSED' if validation['overall_valid'] else 'FAILED'}")
    
    print(f"\n{'='*60}")
    print("PARSER DEMONSTRATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()