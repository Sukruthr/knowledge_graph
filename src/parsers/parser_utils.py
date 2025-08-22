"""
Common utilities for biomedical data parsers.

Provides shared functionality for file loading, validation, and data processing
across all parser types.
"""

import gzip
import pandas as pd
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Union

logger = logging.getLogger(__name__)


class ParserUtils:
    """Common utilities for all biomedical data parsers."""
    
    @staticmethod
    def load_file_safe(file_path: Union[str, Path], file_type: str = 'auto') -> Any:
        """
        Safely load files with proper error handling and format detection.
        
        Args:
            file_path: Path to the file to load
            file_type: File type ('auto', 'csv', 'tsv', 'json', 'yaml', 'gzip')
            
        Returns:
            Loaded file content or None if failed
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None
        
        try:
            # Auto-detect file type from extension
            if file_type == 'auto':
                suffix = file_path.suffix.lower()
                if suffix == '.gz':
                    # Handle compressed files
                    if file_path.name.endswith('.gaf.gz'):
                        file_type = 'gaf_gzip'
                    else:
                        file_type = 'gzip'
                elif suffix in ['.csv', '.tsv', '.tab']:
                    file_type = 'csv'
                elif suffix == '.json':
                    file_type = 'json'
                elif suffix in ['.yaml', '.yml']:
                    file_type = 'yaml'
                else:
                    file_type = 'txt'
            
            # Load based on file type
            if file_type == 'csv' or file_type == 'tsv':
                separator = '\t' if file_type == 'tsv' or file_path.suffix == '.tsv' else ','
                return pd.read_csv(file_path, sep=separator)
            
            elif file_type == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            elif file_type == 'yaml':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            
            elif file_type == 'gzip':
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    return f.read()
            
            elif file_type == 'gaf_gzip':
                # Special handling for GAF files
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    return f.readlines()
            
            else:
                # Default text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
                    
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return None
    
    @staticmethod
    def validate_required_columns(df: pd.DataFrame, required_cols: List[str], 
                                file_name: str = "file") -> bool:
        """
        Validate that DataFrame contains required columns.
        
        Args:
            df: DataFrame to validate
            required_cols: List of required column names
            file_name: Name of file for logging purposes
            
        Returns:
            True if all required columns present, False otherwise
        """
        if df is None or df.empty:
            logger.warning(f"{file_name}: DataFrame is empty or None")
            return False
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"{file_name}: Missing required columns: {missing_cols}")
            return False
        
        return True
    
    @staticmethod
    def clean_gene_identifiers(gene_list: List[str]) -> List[str]:
        """
        Standardize gene identifier formatting.
        
        Args:
            gene_list: List of gene identifiers to clean
            
        Returns:
            List of cleaned gene identifiers
        """
        if not gene_list:
            return []
        
        cleaned = []
        for gene in gene_list:
            if gene and isinstance(gene, str):
                # Remove extra whitespace
                gene = gene.strip()
                
                # Skip empty strings
                if not gene:
                    continue
                
                # Convert to uppercase for gene symbols (standard convention)
                if not gene.startswith(('HGNC:', 'ENSG', 'GO:')):
                    gene = gene.upper()
                
                cleaned.append(gene)
        
        return cleaned
    
    @staticmethod
    def extract_metadata(content: Dict, required_fields: List[str], 
                        optional_fields: List[str] = None) -> Dict[str, Any]:
        """
        Extract and validate metadata from content dictionary.
        
        Args:
            content: Dictionary containing metadata
            required_fields: List of required field names
            optional_fields: List of optional field names
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {}
        
        # Extract required fields
        for field in required_fields:
            if field in content and content[field] is not None:
                metadata[field] = content[field]
            else:
                logger.warning(f"Missing required field: {field}")
                metadata[field] = None
        
        # Extract optional fields
        if optional_fields:
            for field in optional_fields:
                if field in content and content[field] is not None:
                    metadata[field] = content[field]
        
        return metadata
    
    @staticmethod
    def validate_go_id(go_id: str) -> bool:
        """
        Validate GO ID format.
        
        Args:
            go_id: GO identifier to validate
            
        Returns:
            True if valid GO ID format, False otherwise
        """
        if not go_id or not isinstance(go_id, str):
            return False
        
        return go_id.startswith('GO:') and len(go_id) == 10
    
    @staticmethod
    def validate_gene_symbol(gene_symbol: str) -> bool:
        """
        Validate gene symbol format.
        
        Args:
            gene_symbol: Gene symbol to validate
            
        Returns:
            True if valid gene symbol, False otherwise
        """
        if not gene_symbol or not isinstance(gene_symbol, str):
            return False
        
        # Gene symbols should be alphanumeric with some special characters
        gene_symbol = gene_symbol.strip()
        return len(gene_symbol) > 0 and gene_symbol.replace('-', '').replace('_', '').isalnum()
    
    @staticmethod
    def extract_unique_values(data_list: List[Dict], key: str) -> Set[str]:
        """
        Extract unique values for a specific key from list of dictionaries.
        
        Args:
            data_list: List of dictionaries
            key: Key to extract values for
            
        Returns:
            Set of unique values
        """
        unique_values = set()
        
        for item in data_list:
            if isinstance(item, dict) and key in item and item[key]:
                unique_values.add(str(item[key]))
        
        return unique_values
    
    @staticmethod
    def create_cross_references(source_dict: Dict, target_dict: Dict, 
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
    
    @staticmethod
    def calculate_statistics(data_list: List[Dict], numeric_fields: List[str] = None) -> Dict[str, Any]:
        """
        Calculate basic statistics for a list of data dictionaries.
        
        Args:
            data_list: List of dictionaries containing data
            numeric_fields: List of fields to calculate numeric statistics for
            
        Returns:
            Dictionary with calculated statistics
        """
        if not data_list:
            return {'count': 0}
        
        stats = {
            'count': len(data_list),
            'unique_keys': set()
        }
        
        # Collect all unique keys
        for item in data_list:
            if isinstance(item, dict):
                stats['unique_keys'].update(item.keys())
        
        stats['unique_keys'] = list(stats['unique_keys'])
        
        # Calculate numeric statistics if requested
        if numeric_fields:
            for field in numeric_fields:
                values = []
                for item in data_list:
                    if isinstance(item, dict) and field in item:
                        try:
                            value = float(item[field])
                            values.append(value)
                        except (ValueError, TypeError):
                            continue
                
                if values:
                    stats[f"{field}_min"] = min(values)
                    stats[f"{field}_max"] = max(values)
                    stats[f"{field}_mean"] = sum(values) / len(values)
                    stats[f"{field}_count"] = len(values)
        
        return stats
    
    @staticmethod
    def log_parsing_progress(current: int, total: int, step: int = 1000) -> None:
        """
        Log parsing progress at regular intervals.
        
        Args:
            current: Current item number
            total: Total number of items
            step: Progress reporting step size
        """
        if current % step == 0 or current == total:
            percentage = (current / total) * 100 if total > 0 else 0
            logger.info(f"Parsing progress: {current:,}/{total:,} ({percentage:.1f}%)")