"""
Data parsers for Gene Ontology Biological Process (GO_BP) data.

This module provides utilities to parse GO_BP data files and extract
structured information for knowledge graph construction.
"""

import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Set
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GOBPDataParser:
    """Parser for Gene Ontology Biological Process data."""
    
    def __init__(self, data_dir: str):
        """
        Initialize parser with GO_BP data directory.
        
        Args:
            data_dir: Path to GO_BP data directory
        """
        self.data_dir = Path(data_dir)
        self.go_terms = {}
        self.go_relationships = []
        self.gene_go_associations = []
        
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
                    'namespace': 'biological_process'  # This is GO_BP folder
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
        import gzip
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
                    
                    # Filter for biological processes only (aspect = 'P')
                    if association['aspect'] == 'P':
                        associations.append(association)
        
        logger.info(f"Parsed {len(associations)} gene-GO biological process associations")
        self.gene_go_associations = associations
        return associations
    
    def parse_go_term_clustering(self, identifier_type: str = 'symbol') -> Dict:
        """
        Parse GO term clustering/grouping from collapsed_go files.
        These files seem to contain GO-GO relationships for clustering.
        
        Args:
            identifier_type: Type of file to parse ('symbol', 'entrez', 'uniprot')
            
        Returns:
            Dictionary mapping parent GO terms to child GO terms
        """
        logger.info(f"Parsing GO term clustering from {identifier_type} file...")
        
        file_map = {
            'symbol': 'collapsed_go.symbol',
            'entrez': 'collapsed_go.entrez', 
            'uniprot': 'collapsed_go.uniprot'
        }
        
        clustering_file = self.data_dir / file_map[identifier_type]
        
        clusters = {}
        with open(clustering_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3 and parts[0].startswith('GO:') and parts[1].startswith('GO:'):
                    parent_go = parts[0]
                    child_go = parts[1]
                    cluster_type = parts[2] if len(parts) > 2 else 'default'
                    
                    if parent_go not in clusters:
                        clusters[parent_go] = []
                    clusters[parent_go].append({
                        'child_go': child_go,
                        'cluster_type': cluster_type
                    })
        
        logger.info(f"Parsed {len(clusters)} GO term clusters")
        return clusters
    
    def load_pickle_data(self, identifier_type: str = 'symbol') -> Dict:
        """
        Load pre-processed data from pickle files.
        
        Args:
            identifier_type: Type of gene identifier ('symbol', 'entrez', 'uniprot')
            
        Returns:
            Loaded data dictionary
        """
        logger.info(f"Loading pickle data for {identifier_type} identifiers...")
        
        file_map = {
            'symbol': 'collapsed_go.symbol.pkl',
            'entrez': 'collapsed_go.entrez.pkl',
            'uniprot': 'collapsed_go.uniprot.pkl'
        }
        
        pickle_file = self.data_dir / file_map[identifier_type]
        
        try:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded pickle data with {len(data)} entries")
            return data
        except Exception as e:
            logger.error(f"Error loading pickle file: {e}")
            return {}
    
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


def main():
    """Example usage of the GO_BP data parser."""
    
    # Initialize parser
    data_dir = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data/GO_BP"
    parser = GOBPDataParser(data_dir)
    
    # Parse GO terms
    go_terms = parser.parse_go_terms()
    print(f"Sample GO terms:")
    for go_id, info in list(go_terms.items())[:3]:
        print(f"  {go_id}: {info['name']}")
    
    # Parse relationships
    relationships = parser.parse_go_relationships()
    print(f"\nSample relationships:")
    for rel in relationships[:3]:
        print(f"  {rel['child_id']} --{rel['relationship_type']}--> {rel['parent_id']}")
    
    # Parse gene-GO associations from GAF file
    associations = parser.parse_gene_go_associations_from_gaf()
    print(f"\nSample gene-GO associations:")
    for assoc in associations[:3]:
        print(f"  {assoc['gene_symbol']} -> {assoc['go_id']} ({parser.go_terms.get(assoc['go_id'], {}).get('name', 'Unknown')})")
    
    # Parse GO term clustering
    clusters = parser.parse_go_term_clustering('symbol')
    print(f"\nSample GO clusters:")
    for parent_go, children in list(clusters.items())[:2]:
        parent_name = parser.go_terms.get(parent_go, {}).get('name', 'Unknown')
        print(f"  {parent_go} ({parent_name}):")
        for child in children[:3]:
            child_name = parser.go_terms.get(child['child_go'], {}).get('name', 'Unknown')
            print(f"    -> {child['child_go']} ({child_name})")
    
    # Print summary
    summary = parser.get_data_summary()
    print(f"\nData Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()