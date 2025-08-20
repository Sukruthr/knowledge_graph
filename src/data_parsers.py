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

# Import model comparison parser
try:
    from .model_compare_parser import ModelCompareParser
except ImportError:
    try:
        from model_compare_parser import ModelCompareParser
    except ImportError:
        ModelCompareParser = None
        logger.warning("ModelCompareParser not available")


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


class OmicsDataParser:
    """Parser for Omics data including Disease, Drug, and Viral infection associations."""
    
    def __init__(self, omics_data_dir: str, omics_data2_dir: str = None):
        """
        Initialize Omics data parser.
        
        Args:
            omics_data_dir: Path to Omics_data directory containing association files
            omics_data2_dir: Path to Omics_data2 directory containing enhanced semantic data
        """
        self.omics_data_dir = Path(omics_data_dir)
        self.omics_data2_dir = Path(omics_data2_dir) if omics_data2_dir else None
        
        # Core data structures
        self.disease_associations = []
        self.drug_associations = []
        self.viral_associations = []
        self.cluster_relationships = []
        self.disease_expression_matrix = {}
        self.viral_expression_matrix = {}
        
        # Enhanced data structures for Omics_data2
        self.gene_set_annotations = {}
        self.literature_references = {}
        self.go_term_validations = {}
        self.experimental_metadata = {}
        self.functional_enrichments = {}
        
        logger.info(f"Initialized Omics data parser for {omics_data_dir}")
        if self.omics_data2_dir:
            logger.info(f"Enhanced semantic data from {omics_data2_dir}")
    
    def parse_disease_gene_associations(self) -> List[Dict]:
        """
        Parse gene-disease associations from Disease__gene_attribute_edges.txt.
        
        Returns:
            List of gene-disease association dictionaries
        """
        logger.info("Parsing gene-disease associations...")
        
        disease_file = self.omics_data_dir / "Disease__gene_attribute_edges.txt"
        associations = []
        
        with open(disease_file, 'r') as f:
            lines = f.readlines()
            
            # Skip header lines
            for line in lines[2:]:  # Skip first 2 header lines
                parts = line.strip().split('\t')
                if len(parts) >= 7:
                    association = {
                        'gene_symbol': parts[0],
                        'gene_id': parts[2],
                        'disease_condition': parts[3],
                        'disease_name': parts[4],
                        'gse_id': parts[5],
                        'weight': float(parts[6]),
                        'association_type': 'disease'
                    }
                    associations.append(association)
        
        logger.info(f"Parsed {len(associations)} gene-disease associations")
        self.disease_associations = associations
        return associations
    
    def parse_drug_gene_associations(self) -> List[Dict]:
        """
        Parse gene-drug associations from Small_molecule__gene_attribute_edges.txt.
        
        Returns:
            List of gene-drug association dictionaries
        """
        logger.info("Parsing gene-drug associations...")
        
        drug_file = self.omics_data_dir / "Small_molecule __gene_attribute_edges.txt"
        associations = []
        
        with open(drug_file, 'r') as f:
            lines = f.readlines()
            
            # Skip header lines
            for line in lines[2:]:  # Skip first 2 header lines
                parts = line.strip().split('\t')
                if len(parts) >= 7:
                    association = {
                        'gene_symbol': parts[0],
                        'gene_id': parts[2],
                        'drug_condition': parts[3],
                        'drug_name': parts[4],
                        'perturbation_id': parts[5],
                        'weight': float(parts[6]),
                        'association_type': 'drug_perturbation'
                    }
                    associations.append(association)
        
        logger.info(f"Parsed {len(associations)} gene-drug associations")
        self.drug_associations = associations
        return associations
    
    def parse_viral_gene_associations(self) -> List[Dict]:
        """
        Parse gene-viral infection associations from Viral_Infections__gene_attribute_edges.txt.
        
        Returns:
            List of gene-viral association dictionaries
        """
        logger.info("Parsing gene-viral associations...")
        
        viral_file = self.omics_data_dir / "Viral_Infections__gene_attribute_edges.txt"
        associations = []
        
        with open(viral_file, 'r') as f:
            lines = f.readlines()
            
            # Skip header lines
            for line in lines[2:]:  # Skip first 2 header lines
                parts = line.strip().split('\t')
                if len(parts) >= 7:
                    association = {
                        'gene_symbol': parts[0],
                        'gene_id': parts[2],
                        'viral_condition': parts[3],
                        'viral_perturbation': parts[4],
                        'gse_id': parts[5],
                        'weight': float(parts[6]),
                        'association_type': 'viral_response'
                    }
                    associations.append(association)
        
        logger.info(f"Parsed {len(associations)} gene-viral associations")
        self.viral_associations = associations
        return associations
    
    def parse_cluster_relationships(self) -> List[Dict]:
        """
        Parse network cluster relationships from NeST__IAS_clixo_hidef_Nov17.edges.
        
        Returns:
            List of cluster relationship dictionaries
        """
        logger.info("Parsing network cluster relationships...")
        
        cluster_file = self.omics_data_dir / "NeST__IAS_clixo_hidef_Nov17.edges"
        relationships = []
        
        with open(cluster_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    relationship = {
                        'parent_cluster': parts[0],
                        'child_cluster': parts[1],
                        'relationship_type': parts[2],
                        'association_type': 'cluster_hierarchy'
                    }
                    relationships.append(relationship)
        
        logger.info(f"Parsed {len(relationships)} cluster relationships")
        self.cluster_relationships = relationships
        return relationships
    
    def parse_disease_expression_matrix(self) -> Dict:
        """
        Parse disease gene expression matrix from Disease_gene_attribute_matrix_standardized.txt.
        
        Returns:
            Dictionary with expression data structure
        """
        logger.info("Parsing disease expression matrix...")
        
        matrix_file = self.omics_data_dir / "Disease_gene_attribute_matrix_standardized.txt"
        
        # Read the matrix (large file, so we'll parse efficiently)
        matrix_data = {}
        condition_names = []
        
        with open(matrix_file, 'r') as f:
            lines = f.readlines()
            
            # Parse header with condition names (line 1)
            if len(lines) > 0:
                header_parts = lines[0].strip().split('\t')
                condition_names = header_parts[3:]  # Skip first 3 columns
            
            # Parse disease names (line 2)
            disease_names = []
            if len(lines) > 1:
                disease_parts = lines[1].strip().split('\t')
                disease_names = disease_parts[3:]  # Skip first 3 columns
            
            # Parse gene expression data
            for line in lines[2:]:  # Skip first 2 header lines
                parts = line.strip().split('\t')
                if len(parts) > 3:
                    gene_symbol = parts[0]
                    gene_id = parts[2]
                    
                    # Store expression values for non-zero entries
                    expressions = []
                    for i, expr_val in enumerate(parts[3:]):
                        if expr_val and float(expr_val) != 0.0:
                            expressions.append({
                                'condition': condition_names[i] if i < len(condition_names) else f"condition_{i}",
                                'disease': disease_names[i] if i < len(disease_names) else f"disease_{i}",
                                'expression_value': float(expr_val)
                            })
                    
                    if expressions:  # Only store genes with non-zero expressions
                        matrix_data[gene_symbol] = {
                            'gene_id': gene_id,
                            'expressions': expressions
                        }
        
        logger.info(f"Parsed expression matrix for {len(matrix_data)} genes across {len(condition_names)} conditions")
        self.disease_expression_matrix = matrix_data
        return matrix_data
    
    def parse_viral_expression_matrix(self, expression_threshold: float = 0.5) -> Dict:
        """
        Parse viral gene expression matrix from Viral_Infections_gene_attribute_matrix_standardized.txt.
        
        Args:
            expression_threshold: Minimum absolute expression value to include (default: 0.5)
            
        Returns:
            Dictionary with viral expression data structure
        """
        logger.info(f"Parsing viral expression matrix (threshold: {expression_threshold})...")
        
        matrix_file = self.omics_data_dir / "Viral_Infections_gene_attribute_matrix_standardized.txt"
        
        # Read the matrix efficiently
        matrix_data = {}
        condition_names = []
        viral_names = []
        
        with open(matrix_file, 'r') as f:
            lines = f.readlines()
            
            # Parse header with condition names (line 1)
            if len(lines) > 0:
                header_parts = lines[0].strip().split('\t')
                condition_names = header_parts[3:]  # Skip first 3 columns
            
            # Parse viral perturbation names (line 2)
            if len(lines) > 1:
                viral_parts = lines[1].strip().split('\t')
                viral_names = viral_parts[3:]  # Skip first 3 columns
            
            # Parse gene expression data
            for line in lines[2:]:  # Skip first 2 header lines
                parts = line.strip().split('\t')
                if len(parts) > 3:
                    gene_symbol = parts[0]
                    gene_id = parts[2]
                    
                    # Store expression values above threshold
                    expressions = []
                    for i, expr_val in enumerate(parts[3:]):
                        if expr_val and expr_val != '0.000000':
                            try:
                                expr_float = float(expr_val)
                                if abs(expr_float) >= expression_threshold:
                                    expressions.append({
                                        'condition': condition_names[i] if i < len(condition_names) else f"condition_{i}",
                                        'viral_perturbation': viral_names[i] if i < len(viral_names) else f"viral_{i}",
                                        'expression_value': expr_float,
                                        'expression_direction': 'up' if expr_float > 0 else 'down',
                                        'expression_magnitude': abs(expr_float)
                                    })
                            except ValueError:
                                continue  # Skip invalid values
                    
                    if expressions:  # Only store genes with significant expressions
                        matrix_data[gene_symbol] = {
                            'gene_id': gene_id,
                            'expressions': expressions,
                            'num_significant_conditions': len(expressions)
                        }
        
        logger.info(f"Parsed viral expression matrix for {len(matrix_data)} genes across {len(condition_names)} conditions")
        logger.info(f"Found {sum(len(data['expressions']) for data in matrix_data.values())} significant expression events (threshold: {expression_threshold})")
        
        self.viral_expression_matrix = matrix_data
        return matrix_data
    
    def get_unique_entities(self) -> Dict:
        """
        Extract unique entities from parsed data for node creation.
        
        Returns:
            Dictionary with unique diseases, drugs, viral conditions, and clusters
        """
        entities = {
            'diseases': set(),
            'drugs': set(),
            'viral_conditions': set(),
            'clusters': set(),
            'studies': set()
        }
        
        # Extract unique diseases
        for assoc in self.disease_associations:
            entities['diseases'].add(assoc['disease_name'])
            entities['studies'].add(assoc['gse_id'])
        
        # Extract unique drugs
        for assoc in self.drug_associations:
            entities['drugs'].add(assoc['drug_name'])
        
        # Extract unique viral conditions
        for assoc in self.viral_associations:
            entities['viral_conditions'].add(assoc['viral_perturbation'])
            entities['studies'].add(assoc['gse_id'])
        
        # Extract unique clusters
        for rel in self.cluster_relationships:
            entities['clusters'].add(rel['parent_cluster'])
            entities['clusters'].add(rel['child_cluster'])
        
        logger.info(f"Identified unique entities: "
                   f"Diseases={len(entities['diseases'])}, "
                   f"Drugs={len(entities['drugs'])}, "
                   f"Viral={len(entities['viral_conditions'])}, "
                   f"Clusters={len(entities['clusters'])}, "
                   f"Studies={len(entities['studies'])}")
        
        return entities
    
    def validate_omics_data(self) -> Dict[str, bool]:
        """
        Validate the integrity of parsed Omics data.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'has_disease_associations': len(self.disease_associations) > 0,
            'has_drug_associations': len(self.drug_associations) > 0,
            'has_viral_associations': len(self.viral_associations) > 0,
            'has_cluster_relationships': len(self.cluster_relationships) > 0,
            'has_expression_matrix': len(self.disease_expression_matrix) > 0,
            'associations_valid': True,
            'clusters_valid': True
        }
        
        # Validate disease associations
        if self.disease_associations:
            invalid_disease = sum(1 for assoc in self.disease_associations 
                                if not assoc.get('gene_symbol') or not assoc.get('disease_name'))
            validation['associations_valid'] = invalid_disease == 0
        
        # Validate cluster relationships
        if self.cluster_relationships:
            invalid_clusters = sum(1 for rel in self.cluster_relationships 
                                 if not rel.get('parent_cluster') or not rel.get('child_cluster'))
            validation['clusters_valid'] = invalid_clusters == 0
        
        validation['overall_valid'] = all(validation.values())
        
        logger.info(f"Omics data validation: {validation}")
        return validation
    
    def get_omics_summary(self) -> Dict:
        """
        Get summary statistics of the parsed Omics data.
        
        Returns:
            Dictionary with data summary
        """
        unique_entities = self.get_unique_entities()
        
        # Count unique genes across all association types
        all_genes = set()
        for assoc in self.disease_associations:
            all_genes.add(assoc['gene_symbol'])
        for assoc in self.drug_associations:
            all_genes.add(assoc['gene_symbol'])
        for assoc in self.viral_associations:
            all_genes.add(assoc['gene_symbol'])
        
        return {
            'num_disease_associations': len(self.disease_associations),
            'num_drug_associations': len(self.drug_associations),
            'num_viral_associations': len(self.viral_associations),
            'num_cluster_relationships': len(self.cluster_relationships),
            'num_unique_genes': len(all_genes),
            'num_unique_diseases': len(unique_entities['diseases']),
            'num_unique_drugs': len(unique_entities['drugs']),
            'num_unique_viral_conditions': len(unique_entities['viral_conditions']),
            'num_unique_clusters': len(unique_entities['clusters']),
            'num_studies': len(unique_entities['studies']),
            'expression_matrix_genes': len(self.disease_expression_matrix),
            'gene_set_annotations': len(self.gene_set_annotations),
            'literature_references': len(self.literature_references),
            'go_term_validations': len(self.go_term_validations)
        }
    
    def parse_gene_set_annotations(self) -> Dict[str, Dict]:
        """
        Parse LLM-enhanced gene set annotations from omics_revamped_LLM_DF.tsv.
        
        Returns:
            Dictionary mapping gene set IDs to semantic annotations
        """
        if not self.omics_data2_dir:
            logger.warning("No Omics_data2 directory provided, skipping gene set annotations")
            return {}
        
        logger.info("Parsing LLM-enhanced gene set annotations...")
        
        annotation_file = self.omics_data2_dir / "omics_revamped_LLM_DF.tsv"
        if not annotation_file.exists():
            logger.warning(f"Gene set annotation file not found: {annotation_file}")
            return {}
        
        annotations = {}
        try:
            df = pd.read_csv(annotation_file, sep='\t')
            
            for _, row in df.iterrows():
                if pd.notna(row.get('GeneSetID')):
                    gene_set_id = str(row['GeneSetID'])
                    
                    annotations[gene_set_id] = {
                        'source': row.get('Source', ''),
                        'gene_set_name': row.get('GeneSetName', ''),
                        'gene_list': str(row.get('GeneList', '')).split(),
                        'n_genes': int(row.get('n_Genes', 0)) if pd.notna(row.get('n_Genes')) else 0,
                        'llm_name': row.get('LLM Name', ''),
                        'llm_analysis': row.get('LLM Analysis', ''),
                        'score': float(row.get('Score', 0.0)) if pd.notna(row.get('Score')) else 0.0,
                        'supporting_genes': str(row.get('Supporting Genes', '')).split(),
                        'supporting_count': int(row.get('Supporting Count', 0)) if pd.notna(row.get('Supporting Count')) else 0,
                        'llm_support_analysis': row.get('LLM Support Analysis', ''),
                        'genes_mentioned_in_text': int(row.get('GenesMentionedInText', 0)) if pd.notna(row.get('GenesMentionedInText')) else 0,
                        'llm_coverage': float(row.get('LLM_coverage', 0.0)) if pd.notna(row.get('LLM_coverage')) else 0.0
                    }
        
        except Exception as e:
            logger.error(f"Error parsing gene set annotations: {e}")
            return {}
        
        logger.info(f"Parsed {len(annotations)} gene set annotations")
        self.gene_set_annotations = annotations
        return annotations
    
    def parse_literature_references(self) -> Dict[str, List]:
        """
        Parse literature references and keywords from omics_paragraph_keywords_dict.json.
        
        Returns:
            Dictionary mapping gene set IDs to literature references
        """
        if not self.omics_data2_dir:
            logger.warning("No Omics_data2 directory provided, skipping literature references")
            return {}
        
        logger.info("Parsing literature references...")
        
        import json
        literature_file = self.omics_data2_dir / "omics_paragraph_keywords_dict.json"
        if not literature_file.exists():
            logger.warning(f"Literature reference file not found: {literature_file}")
            return {}
        
        references = {}
        try:
            with open(literature_file, 'r') as f:
                data = json.load(f)
            
            for gene_set_id, paragraphs in data.items():
                if paragraphs and isinstance(paragraphs, list):
                    references[gene_set_id] = []
                    
                    for paragraph_data in paragraphs:
                        if paragraph_data and isinstance(paragraph_data, dict):
                            ref_entry = {
                                'paragraph': paragraph_data.get('paragraph', ''),
                                'keyword': paragraph_data.get('keyword', ''),
                                'references': paragraph_data.get('references', [])
                            }
                            references[gene_set_id].append(ref_entry)
        
        except Exception as e:
            logger.error(f"Error parsing literature references: {e}")
            return {}
        
        logger.info(f"Parsed literature references for {len(references)} gene sets")
        self.literature_references = references
        return references
    
    def parse_go_term_validations(self) -> Dict[str, Dict]:
        """
        Parse GO term validations from omics_revamped_LLM_gprofiler_new_gene_name_DF_APV_only.tsv.
        
        Returns:
            Dictionary mapping gene set IDs to GO term validation data
        """
        if not self.omics_data2_dir:
            logger.warning("No Omics_data2 directory provided, skipping GO term validations")
            return {}
        
        logger.info("Parsing GO term validations...")
        
        validation_file = self.omics_data2_dir / "omics_revamped_LLM_gprofiler_new_gene_name_DF_APV_only.tsv"
        if not validation_file.exists():
            logger.warning(f"GO validation file not found: {validation_file}")
            return {}
        
        validations = {}
        try:
            df = pd.read_csv(validation_file, sep='\t')
            
            for _, row in df.iterrows():
                if pd.notna(row.get('GeneSetID')):
                    gene_set_id = str(row['GeneSetID'])
                    
                    validations[gene_set_id] = {
                        'go_term': row.get('Term', ''),
                        'go_id': row.get('GO ID', ''),
                        'adjusted_p_value': float(row.get('Adjusted P-value', 1.0)) if pd.notna(row.get('Adjusted P-value')) else 1.0,
                        'intersection_size': int(row.get('intersection_size', 0)) if pd.notna(row.get('intersection_size')) else 0,
                        'term_size': int(row.get('term_size', 0)) if pd.notna(row.get('term_size')) else 0,
                        'query_size': int(row.get('query_size', 0)) if pd.notna(row.get('query_size')) else 0,
                        'intersections': str(row.get('intersections', '')).split(',') if pd.notna(row.get('intersections')) else [],
                        'gprofiler_ji': float(row.get('gprofiler_JI', 0.0)) if pd.notna(row.get('gprofiler_JI')) else 0.0,
                        'gprofiler_coverage': float(row.get('gprofiler_coverage', 0.0)) if pd.notna(row.get('gprofiler_coverage')) else 0.0,
                        'llm_best_matching_go': row.get('LLM_best_matching_GO', ''),
                        'best_matching_go_id': row.get('best_matching_GO_ID', ''),
                        'llm_ji': float(row.get('LLM_JI', 0.0)) if pd.notna(row.get('LLM_JI')) else 0.0,
                        'llm_success_tf': row.get('LLM_success_TF', False),
                        'gprofiler_success_tf': row.get('gprofiler_success_TF', False)
                    }
        
        except Exception as e:
            logger.error(f"Error parsing GO term validations: {e}")
            return {}
        
        logger.info(f"Parsed GO validations for {len(validations)} gene sets")
        self.go_term_validations = validations
        return validations
    
    def parse_experimental_metadata(self) -> Dict[str, Dict]:
        """
        Parse enhanced experimental metadata from gene count and reference files.
        
        Returns:
            Dictionary mapping gene set IDs to experimental metadata
        """
        if not self.omics_data2_dir:
            logger.warning("No Omics_data2 directory provided, skipping experimental metadata")
            return {}
        
        logger.info("Parsing experimental metadata...")
        
        metadata = {}
        
        # Parse gene count data
        genecounts_file = self.omics_data2_dir / "omics_revamped_LLM_genecounts_DF.tsv"
        if genecounts_file.exists():
            try:
                df = pd.read_csv(genecounts_file, sep='\t')
                for _, row in df.iterrows():
                    if pd.notna(row.get('GeneSetID')):
                        gene_set_id = str(row['GeneSetID'])
                        if gene_set_id not in metadata:
                            metadata[gene_set_id] = {}
                        
                        metadata[gene_set_id].update({
                            'supporting_genes': str(row.get('Supporting Genes', '')).split(),
                            'supporting_count': int(row.get('Supporting Count', 0)) if pd.notna(row.get('Supporting Count')) else 0,
                            'llm_support_analysis': row.get('LLM Support Analysis', ''),
                            'genes_mentioned_in_text': int(row.get('GenesMentionedInText', 0)) if pd.notna(row.get('GenesMentionedInText')) else 0,
                            'llm_coverage': float(row.get('LLM_coverage', 0.0)) if pd.notna(row.get('LLM_coverage')) else 0.0
                        })
            except Exception as e:
                logger.error(f"Error parsing gene counts metadata: {e}")
        
        # Parse reference data
        ref_file = self.omics_data2_dir / "omics_revamped_LLM_ref_DF.tsv"
        if ref_file.exists():
            try:
                df = pd.read_csv(ref_file, sep='\t')
                for _, row in df.iterrows():
                    if pd.notna(row.get('GeneSetID')):
                        gene_set_id = str(row['GeneSetID'])
                        if gene_set_id not in metadata:
                            metadata[gene_set_id] = {}
                        
                        metadata[gene_set_id].update({
                            'referenced_analysis': row.get('referenced_analysis', ''),
                            'overlap': int(row.get('Overlap', 0)) if pd.notna(row.get('Overlap')) else 0,
                            'p_value': float(row.get('P-value', 1.0)) if pd.notna(row.get('P-value')) else 1.0,
                            'adjusted_p_value': float(row.get('Adjusted P-value', 1.0)) if pd.notna(row.get('Adjusted P-value')) else 1.0,
                            'genes': str(row.get('Genes', '')).split(',') if pd.notna(row.get('Genes')) else [],
                            'go_term_genes': str(row.get('GO_term_genes', '')).split() if pd.notna(row.get('GO_term_genes')) else [],
                            'llm_name_go_term_sim': float(row.get('LLM_name_GO_term_sim', 0.0)) if pd.notna(row.get('LLM_name_GO_term_sim')) else 0.0
                        })
            except Exception as e:
                logger.error(f"Error parsing reference metadata: {e}")
        
        logger.info(f"Parsed experimental metadata for {len(metadata)} gene sets")
        self.experimental_metadata = metadata
        return metadata
    
    def parse_all_enhanced_data(self) -> Dict[str, Dict]:
        """
        Parse all enhanced semantic data from Omics_data2.
        
        Returns:
            Dictionary containing all enhanced data structures
        """
        if not self.omics_data2_dir:
            logger.warning("No Omics_data2 directory provided, skipping enhanced data parsing")
            return {}
        
        logger.info("Parsing all enhanced semantic data...")
        
        enhanced_data = {
            'gene_set_annotations': self.parse_gene_set_annotations(),
            'literature_references': self.parse_literature_references(),
            'go_term_validations': self.parse_go_term_validations(),
            'experimental_metadata': self.parse_experimental_metadata()
        }
        
        # Compute integration statistics
        enhanced_data['integration_stats'] = {
            'total_annotated_sets': len(self.gene_set_annotations),
            'sets_with_literature': len(self.literature_references),
            'sets_with_go_validation': len(self.go_term_validations),
            'sets_with_metadata': len(self.experimental_metadata),
            'average_llm_score': sum(ann.get('score', 0) for ann in self.gene_set_annotations.values()) / max(len(self.gene_set_annotations), 1),
            'average_llm_coverage': sum(ann.get('llm_coverage', 0) for ann in self.gene_set_annotations.values()) / max(len(self.gene_set_annotations), 1)
        }
        
        logger.info("Enhanced semantic data parsing complete")
        return enhanced_data


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


class CombinedBiomedicalParser:
    """Parser for comprehensive biomedical data (GO + Omics integration)."""
    
    def __init__(self, base_data_dir: str):
        """
        Initialize combined biomedical parser for GO and Omics data.
        
        Args:
            base_data_dir: Base directory containing GO_BP, GO_CC, GO_MF, Omics_data, Omics_data2, and GO_term_analysis subdirectories
        """
        self.base_data_dir = Path(base_data_dir)
        
        # Initialize GO parser
        self.go_parser = CombinedGOParser(str(base_data_dir))
        
        # Initialize Omics parser with both Omics_data and Omics_data2
        omics_dir = self.base_data_dir / "Omics_data"
        omics_data2_dir = self.base_data_dir / "Omics_data2"
        
        if omics_dir.exists():
            omics_data2_path = str(omics_data2_dir) if omics_data2_dir.exists() else None
            self.omics_parser = OmicsDataParser(str(omics_dir), omics_data2_path)
            logger.info("Initialized Omics data parser")
            if omics_data2_path:
                logger.info("Enhanced semantic data integration enabled")
        else:
            self.omics_parser = None
            logger.warning(f"Omics_data directory not found: {omics_dir}")
        
        # Initialize Model Comparison parser
        model_compare_dir = self.base_data_dir / "GO_term_analysis" / "model_compare"
        if model_compare_dir.exists() and ModelCompareParser is not None:
            self.model_compare_parser = ModelCompareParser(str(model_compare_dir))
            logger.info("Initialized Model Comparison parser")
        else:
            self.model_compare_parser = None
            if model_compare_dir.exists():
                logger.warning("Model comparison data found but parser not available")
            else:
                logger.info("No model comparison data directory found")
        
        self.parsed_data = {}
        
    def parse_all_biomedical_data(self) -> Dict[str, Dict]:
        """
        Parse all biomedical data (GO + Omics).
        
        Returns:
            Dictionary with parsed data from all sources
        """
        logger.info("Starting comprehensive biomedical data parsing...")
        
        # Parse GO data
        go_data = self.go_parser.parse_all_namespaces()
        self.parsed_data['go_data'] = go_data
        
        # Parse Omics data if available
        if self.omics_parser:
            omics_data = {
                'disease_associations': self.omics_parser.parse_disease_gene_associations(),
                'drug_associations': self.omics_parser.parse_drug_gene_associations(),
                'viral_associations': self.omics_parser.parse_viral_gene_associations(),
                'cluster_relationships': self.omics_parser.parse_cluster_relationships(),
                'disease_expression_matrix': self.omics_parser.parse_disease_expression_matrix(),
                'viral_expression_matrix': self.omics_parser.parse_viral_expression_matrix(),
                'unique_entities': self.omics_parser.get_unique_entities(),
                'validation': self.omics_parser.validate_omics_data(),
                'summary': self.omics_parser.get_omics_summary()
            }
            
            # Parse enhanced semantic data from Omics_data2 if available
            if self.omics_parser.omics_data2_dir:
                enhanced_data = self.omics_parser.parse_all_enhanced_data()
                omics_data['enhanced_data'] = enhanced_data
                logger.info("Enhanced semantic data integrated successfully")
            
            self.parsed_data['omics_data'] = omics_data
        
        # Parse Model Comparison data if available
        if self.model_compare_parser:
            model_compare_data = self.model_compare_parser.parse_all_model_data()
            self.parsed_data['model_compare_data'] = model_compare_data
            logger.info("Model comparison data integrated successfully")
        
        logger.info("Comprehensive biomedical data parsing complete")
        return self.parsed_data
    
    def get_comprehensive_summary(self) -> Dict:
        """
        Get comprehensive summary across all parsed data sources.
        
        Returns:
            Combined summary dictionary
        """
        summary = {
            'data_sources': [],
            'go_summary': {},
            'omics_summary': {},
            'integration_stats': {}
        }
        
        # GO summary
        if 'go_data' in self.parsed_data:
            summary['data_sources'].append('GO_ontology')
            summary['go_summary'] = self.go_parser.get_combined_summary()
        
        # Omics summary
        if 'omics_data' in self.parsed_data and self.omics_parser:
            summary['data_sources'].append('Omics_associations')
            summary['omics_summary'] = self.omics_parser.get_omics_summary()
        
        # Integration statistics
        if 'go_data' in self.parsed_data and 'omics_data' in self.parsed_data:
            # Find gene overlaps between GO and Omics data
            go_genes = set()
            for namespace_data in self.parsed_data['go_data'].values():
                for assoc in namespace_data.get('gene_associations', []):
                    go_genes.add(assoc['gene_symbol'])
            
            omics_genes = set()
            for assoc in self.parsed_data['omics_data']['disease_associations']:
                omics_genes.add(assoc['gene_symbol'])
            for assoc in self.parsed_data['omics_data']['drug_associations']:
                omics_genes.add(assoc['gene_symbol'])
            for assoc in self.parsed_data['omics_data']['viral_associations']:
                omics_genes.add(assoc['gene_symbol'])
            
            overlap_genes = go_genes & omics_genes
            
            summary['integration_stats'] = {
                'go_genes': len(go_genes),
                'omics_genes': len(omics_genes),
                'overlapping_genes': len(overlap_genes),
                'integration_coverage': len(overlap_genes) / len(go_genes) if go_genes else 0,
                'can_integrate': len(overlap_genes) > 0
            }
        
        return summary
    
    def validate_comprehensive_data(self) -> Dict[str, bool]:
        """
        Validate all parsed biomedical data.
        
        Returns:
            Comprehensive validation results
        """
        validation = {
            'go_data_valid': False,
            'omics_data_valid': False,
            'integration_possible': False,
            'overall_valid': False
        }
        
        # Validate GO data
        if 'go_data' in self.parsed_data:
            go_valid = True
            for namespace_data in self.parsed_data['go_data'].values():
                if not namespace_data.get('go_terms') or not namespace_data.get('gene_associations'):
                    go_valid = False
                    break
            validation['go_data_valid'] = go_valid
        
        # Validate Omics data
        if 'omics_data' in self.parsed_data and self.omics_parser:
            omics_validation = self.parsed_data['omics_data']['validation']
            validation['omics_data_valid'] = omics_validation.get('overall_valid', False)
        
        # Check integration possibility
        summary = self.get_comprehensive_summary()
        integration_stats = summary.get('integration_stats', {})
        validation['integration_possible'] = integration_stats.get('can_integrate', False)
        
        # Overall validation
        validation['overall_valid'] = (validation['go_data_valid'] and 
                                     validation['omics_data_valid'] and 
                                     validation['integration_possible'])
        
        logger.info(f"Comprehensive biomedical data validation: {validation}")
        return validation


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