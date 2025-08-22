#!/usr/bin/env python3
"""
Enhanced Comprehensive Test Suite for parser_utils.py

Tests ALL utility functions with comprehensive edge cases and validation.
Includes tests for create_cross_references which was missing from original test.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
import yaml
import gzip
import logging
from typing import List, Dict, Any
from collections import defaultdict

# Import the parser utilities
from parsers.parser_utils import ParserUtils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedTestParserUtils:
    """Enhanced comprehensive test class for ALL ParserUtils functionality."""
    
    def __init__(self):
        self.test_results = {
            'load_file_safe': {'passed': 0, 'failed': 0, 'details': []},
            'validate_required_columns': {'passed': 0, 'failed': 0, 'details': []},
            'clean_gene_identifiers': {'passed': 0, 'failed': 0, 'details': []},
            'extract_metadata': {'passed': 0, 'failed': 0, 'details': []},
            'validate_go_id': {'passed': 0, 'failed': 0, 'details': []},
            'validate_gene_symbol': {'passed': 0, 'failed': 0, 'details': []},
            'extract_unique_values': {'passed': 0, 'failed': 0, 'details': []},
            'create_cross_references': {'passed': 0, 'failed': 0, 'details': []},
            'calculate_statistics': {'passed': 0, 'failed': 0, 'details': []},
            'log_parsing_progress': {'passed': 0, 'failed': 0, 'details': []}
        }
        self.temp_dir = None
        
    def setup_test_files(self):
        """Create comprehensive temporary test files for testing."""
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {self.temp_dir}")
        
        # Create test CSV file
        test_csv_data = pd.DataFrame({
            'gene_symbol': ['TP53', 'BRCA1', 'MYC', '', 'INVALID'],
            'go_id': ['GO:0008150', 'GO:0003674', 'GO:0005575', 'GO:invalid', 'GO:0008150'],
            'score': [1.5, 2.3, 0.8, 'invalid', None],
            'count': [10, 20, 15, '5', 0],
            'description': ['tumor protein', 'breast cancer', 'oncogene', '', 'invalid']
        })
        self.test_csv_path = os.path.join(self.temp_dir, 'test.csv')
        test_csv_data.to_csv(self.test_csv_path, index=False)
        
        # Create test TSV file
        test_tsv_data = pd.DataFrame({
            'id': ['1', '2', '3'],
            'name': ['Gene1', 'Gene2', 'Gene3'],
            'value': [1.0, 2.0, 3.0]
        })
        self.test_tsv_path = os.path.join(self.temp_dir, 'test.tsv')
        test_tsv_data.to_csv(self.test_tsv_path, sep='\t', index=False)
        
        # Create test JSON file
        test_json_data = {
            'genes': ['TP53', 'BRCA1'], 
            'terms': ['GO:0008150'],
            'metadata': {'version': '1.0', 'source': 'test'}
        }
        self.test_json_path = os.path.join(self.temp_dir, 'test.json')
        with open(self.test_json_path, 'w') as f:
            json.dump(test_json_data, f, indent=2)
        
        # Create test YAML file
        test_yaml_data = {'config': {'threads': 4, 'memory': '8GB'}, 'features': ['parsing', 'validation']}
        self.test_yaml_path = os.path.join(self.temp_dir, 'test.yaml')
        with open(self.test_yaml_path, 'w') as f:
            yaml.dump(test_yaml_data, f)
        
        # Create test TXT file
        self.test_txt_path = os.path.join(self.temp_dir, 'test.txt')
        with open(self.test_txt_path, 'w') as f:
            f.write("gene1\tGO:0008150\n")
            f.write("gene2\tGO:0003674\n")
            f.write("# This is a comment\n")
            f.write("gene3\tGO:0005575\n")
        
        # Create test gzip file
        self.test_gzip_path = os.path.join(self.temp_dir, 'test.txt.gz')
        with gzip.open(self.test_gzip_path, 'wt', encoding='utf-8') as f:
            f.write("compressed gene data\nTP53\tGO:0008150\n")
        
        # Create test GAF.gz file
        self.test_gaf_gz_path = os.path.join(self.temp_dir, 'test.gaf.gz')
        gaf_content = [
            "!gaf-version: 2.2\n",
            "UniProtKB\tP04637\tTP53\t\tGO:0008150\tPMID:123456\tIEA\t\tP\tTumor protein p53\t\tprotein\ttaxon:9606\t20230101\tUniProt\n"
        ]
        with gzip.open(self.test_gaf_gz_path, 'wt', encoding='utf-8') as f:
            f.writelines(gaf_content)
        
        # Create empty file
        self.test_empty_path = os.path.join(self.temp_dir, 'empty.csv')
        with open(self.test_empty_path, 'w') as f:
            pass
        
        # Create corrupted file
        self.test_corrupted_path = os.path.join(self.temp_dir, 'corrupted.json')
        with open(self.test_corrupted_path, 'w') as f:
            f.write('{"invalid": json content}')
            
    def cleanup_test_files(self):
        """Clean up temporary test files."""
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
    
    def test_load_file_safe_comprehensive(self):
        """Comprehensive test of load_file_safe function."""
        logger.info("🧪 Testing load_file_safe function (enhanced)")
        
        test_cases = [
            # CSV file test
            {
                'name': 'CSV file loading',
                'file_path': self.test_csv_path,
                'file_type': 'csv',
                'expected_type': pd.DataFrame,
                'should_pass': True,
                'validation': lambda r: len(r.columns) == 5 and 'gene_symbol' in r.columns
            },
            # TSV file test
            {
                'name': 'TSV file loading',
                'file_path': self.test_tsv_path,
                'file_type': 'tsv',
                'expected_type': pd.DataFrame,
                'should_pass': True,
                'validation': lambda r: len(r.columns) == 3 and 'name' in r.columns
            },
            # JSON file test
            {
                'name': 'JSON file loading',
                'file_path': self.test_json_path,
                'file_type': 'json',
                'expected_type': dict,
                'should_pass': True,
                'validation': lambda r: 'genes' in r and 'metadata' in r
            },
            # YAML file test
            {
                'name': 'YAML file loading',
                'file_path': self.test_yaml_path,
                'file_type': 'yaml',
                'expected_type': dict,
                'should_pass': True,
                'validation': lambda r: 'config' in r and 'features' in r
            },
            # TXT file test
            {
                'name': 'TXT file loading',
                'file_path': self.test_txt_path,
                'file_type': 'txt',
                'expected_type': str,
                'should_pass': True,
                'validation': lambda r: 'gene1' in r and 'GO:0008150' in r
            },
            # GZIP file test
            {
                'name': 'GZIP file loading',
                'file_path': self.test_gzip_path,
                'file_type': 'gzip',
                'expected_type': str,
                'should_pass': True,
                'validation': lambda r: 'compressed gene data' in r
            },
            # GAF.GZ file test
            {
                'name': 'GAF.GZ file loading',
                'file_path': self.test_gaf_gz_path,
                'file_type': 'gaf_gzip',
                'expected_type': list,
                'should_pass': True,
                'validation': lambda r: len(r) == 2 and 'TP53' in r[1]
            },
            # Auto detection tests
            {
                'name': 'Auto detection - CSV',
                'file_path': self.test_csv_path,
                'file_type': 'auto',
                'expected_type': pd.DataFrame,
                'should_pass': True,
                'validation': lambda r: isinstance(r, pd.DataFrame)
            },
            {
                'name': 'Auto detection - JSON',
                'file_path': self.test_json_path,
                'file_type': 'auto',
                'expected_type': dict,
                'should_pass': True,
                'validation': lambda r: isinstance(r, dict)
            },
            {
                'name': 'Auto detection - YAML',
                'file_path': self.test_yaml_path,
                'file_type': 'auto',
                'expected_type': dict,
                'should_pass': True,
                'validation': lambda r: isinstance(r, dict)
            },
            # Error cases
            {
                'name': 'Non-existent file',
                'file_path': '/nonexistent/file.csv',
                'file_type': 'csv',
                'expected_type': None,
                'should_pass': False,
                'validation': lambda r: r is None
            },
            {
                'name': 'Empty file',
                'file_path': self.test_empty_path,
                'file_type': 'csv',
                'expected_type': None,
                'should_pass': False,
                'validation': lambda r: r is None
            },
            {
                'name': 'Corrupted JSON file',
                'file_path': self.test_corrupted_path,
                'file_type': 'json',
                'expected_type': None,
                'should_pass': False,
                'validation': lambda r: r is None
            }
        ]
        
        for case in test_cases:
            try:
                result = ParserUtils.load_file_safe(case['file_path'], case['file_type'])
                
                # Check basic type and pass/fail expectation
                type_match = (result is None and case['expected_type'] is None) or isinstance(result, case['expected_type'])
                
                if case['should_pass'] and type_match and (not case.get('validation') or case['validation'](result)):
                    self.test_results['load_file_safe']['passed'] += 1
                    self.test_results['load_file_safe']['details'].append(f"✅ {case['name']}: PASSED")
                    logger.info(f"  ✅ {case['name']}: PASSED")
                    
                elif not case['should_pass'] and case['validation'](result):
                    self.test_results['load_file_safe']['passed'] += 1
                    self.test_results['load_file_safe']['details'].append(f"✅ {case['name']}: PASSED (correctly failed)")
                    logger.info(f"  ✅ {case['name']}: PASSED (correctly failed)")
                    
                else:
                    self.test_results['load_file_safe']['failed'] += 1
                    self.test_results['load_file_safe']['details'].append(f"❌ {case['name']}: FAILED")
                    logger.error(f"  ❌ {case['name']}: FAILED")
                        
            except Exception as e:
                if case['should_pass']:
                    self.test_results['load_file_safe']['failed'] += 1
                    self.test_results['load_file_safe']['details'].append(f"❌ {case['name']}: Exception - {str(e)}")
                    logger.error(f"  ❌ {case['name']}: Exception - {str(e)}")
                else:
                    # Some errors are expected for negative test cases
                    self.test_results['load_file_safe']['passed'] += 1
                    self.test_results['load_file_safe']['details'].append(f"✅ {case['name']}: PASSED (expected error)")
                    logger.info(f"  ✅ {case['name']}: PASSED (expected error)")

    def test_create_cross_references(self):
        """Test the create_cross_references static method."""
        logger.info("🧪 Testing create_cross_references function")
        
        test_cases = [
            {
                'name': 'Basic cross-reference mapping',
                'source_dict': {
                    'GO:0008150': {'TP53', 'BRCA1'},
                    'GO:0003674': {'MYC'},
                    'GO:0005575': {'EGFR'}
                },
                'target_dict': {
                    'GO:0008150': {'7157', '672'},  # Entrez IDs for TP53, BRCA1
                    'GO:0003674': {'4609'},         # Entrez ID for MYC
                    'GO:0005575': {'1956'}          # Entrez ID for EGFR
                },
                'source_type': 'symbol',
                'target_type': 'entrez',
                'expected_mappings': 0  # No 1:1 mappings (multiple genes per GO term)
            },
            {
                'name': 'One-to-one mapping',
                'source_dict': {
                    'GO:0008150': {'TP53'},
                    'GO:0003674': {'BRCA1'},
                    'GO:0005575': {'MYC'}
                },
                'target_dict': {
                    'GO:0008150': {'7157'},
                    'GO:0003674': {'672'},
                    'GO:0005575': {'4609'}
                },
                'source_type': 'symbol',
                'target_type': 'entrez',
                'expected_mappings': 3  # All should create 1:1 mappings
            },
            {
                'name': 'Partial overlap',
                'source_dict': {
                    'GO:0008150': {'TP53'},
                    'GO:0003674': {'BRCA1'},
                    'GO:0005575': {'MYC'}
                },
                'target_dict': {
                    'GO:0008150': {'7157'},
                    'GO:0003674': {'672'}
                    # GO:0005575 missing from target
                },
                'source_type': 'symbol',
                'target_type': 'entrez',
                'expected_mappings': 2  # Only overlapping GO terms create mappings
            },
            {
                'name': 'Empty dictionaries',
                'source_dict': {},
                'target_dict': {},
                'source_type': 'symbol',
                'target_type': 'entrez',
                'expected_mappings': 0
            }
        ]
        
        for case in test_cases:
            try:
                # Initialize mappings dictionary
                mappings = {
                    f"{case['source_type']}_to_{case['target_type']}": {},
                    f"{case['target_type']}_to_{case['source_type']}": {}
                }
                
                # Call the method
                ParserUtils.create_cross_references(
                    case['source_dict'], 
                    case['target_dict'], 
                    mappings, 
                    case['source_type'], 
                    case['target_type']
                )
                
                # Check results
                forward_key = f"{case['source_type']}_to_{case['target_type']}"
                reverse_key = f"{case['target_type']}_to_{case['source_type']}"
                
                forward_count = len(mappings[forward_key])
                reverse_count = len(mappings[reverse_key])
                
                if forward_count == case['expected_mappings'] and forward_count == reverse_count:
                    self.test_results['create_cross_references']['passed'] += 1
                    self.test_results['create_cross_references']['details'].append(f"✅ {case['name']}: PASSED")
                    logger.info(f"  ✅ {case['name']}: PASSED")
                    logger.info(f"    Forward mappings: {forward_count}, Reverse mappings: {reverse_count}")
                else:
                    self.test_results['create_cross_references']['failed'] += 1
                    self.test_results['create_cross_references']['details'].append(f"❌ {case['name']}: Expected {case['expected_mappings']}, got forward={forward_count}, reverse={reverse_count}")
                    logger.error(f"  ❌ {case['name']}: Expected {case['expected_mappings']}, got forward={forward_count}, reverse={reverse_count}")
                    
            except Exception as e:
                self.test_results['create_cross_references']['failed'] += 1
                self.test_results['create_cross_references']['details'].append(f"❌ {case['name']}: Exception - {str(e)}")
                logger.error(f"  ❌ {case['name']}: Exception - {str(e)}")

    def test_calculate_statistics_comprehensive(self):
        """Comprehensive test of calculate_statistics function."""
        logger.info("🧪 Testing calculate_statistics function (enhanced)")
        
        test_cases = [
            {
                'name': 'Basic statistics with numeric fields',
                'data_list': [
                    {'score': 1.0, 'count': 10, 'name': 'test1'},
                    {'score': 2.0, 'count': 20, 'name': 'test2'},
                    {'score': 3.0, 'count': 30, 'name': 'test3'}
                ],
                'numeric_fields': ['score', 'count'],
                'expected_count': 3,
                'expected_fields': ['score_min', 'score_max', 'score_mean', 'count_min', 'count_max', 'count_mean']
            },
            {
                'name': 'Statistics with mixed data types',
                'data_list': [
                    {'score': 1.5, 'count': 'invalid', 'name': 'test1'},
                    {'score': 'invalid', 'count': 20, 'name': 'test2'},
                    {'score': 3.5, 'count': 30, 'name': 'test3'}
                ],
                'numeric_fields': ['score', 'count'],
                'expected_count': 3,
                'expected_fields': ['score_min', 'score_max', 'score_mean', 'count_min', 'count_max', 'count_mean']
            },
            {
                'name': 'Empty data list',
                'data_list': [],
                'numeric_fields': ['score'],
                'expected_count': 0,
                'expected_fields': []
            },
            {
                'name': 'No numeric fields specified',
                'data_list': [
                    {'name': 'test1', 'type': 'gene'},
                    {'name': 'test2', 'type': 'protein'}
                ],
                'numeric_fields': None,
                'expected_count': 2,
                'expected_fields': ['unique_keys']
            },
            {
                'name': 'Statistics with missing fields',
                'data_list': [
                    {'score': 1.0, 'name': 'test1'},
                    {'count': 20, 'name': 'test2'},
                    {'score': 3.0, 'count': 30, 'name': 'test3'}
                ],
                'numeric_fields': ['score', 'count'],
                'expected_count': 3,
                'expected_fields': ['score_min', 'score_max', 'score_mean', 'count_min', 'count_max', 'count_mean']
            }
        ]
        
        for case in test_cases:
            try:
                result = ParserUtils.calculate_statistics(case['data_list'], case['numeric_fields'])
                
                # Check basic structure
                if not isinstance(result, dict) or 'count' not in result:
                    self.test_results['calculate_statistics']['failed'] += 1
                    self.test_results['calculate_statistics']['details'].append(f"❌ {case['name']}: Invalid result structure")
                    logger.error(f"  ❌ {case['name']}: Invalid result structure")
                    continue
                
                # Check count
                if result['count'] != case['expected_count']:
                    self.test_results['calculate_statistics']['failed'] += 1
                    self.test_results['calculate_statistics']['details'].append(f"❌ {case['name']}: Expected count {case['expected_count']}, got {result['count']}")
                    logger.error(f"  ❌ {case['name']}: Expected count {case['expected_count']}, got {result['count']}")
                    continue
                
                # Check expected fields presence (for non-empty data)
                if case['expected_count'] > 0:
                    if case['numeric_fields']:
                        # Check if numeric statistics are present when data allows
                        has_valid_numeric = any(
                            field in item and isinstance(item[field], (int, float))
                            for item in case['data_list']
                            for field in case['numeric_fields']
                        )
                        if has_valid_numeric:
                            # At least some numeric fields should have statistics
                            has_any_stats = any(
                                f"{field}_min" in result
                                for field in case['numeric_fields']
                            )
                            if not has_any_stats:
                                self.test_results['calculate_statistics']['failed'] += 1
                                self.test_results['calculate_statistics']['details'].append(f"❌ {case['name']}: Missing numeric statistics")
                                logger.error(f"  ❌ {case['name']}: Missing numeric statistics")
                                continue
                    
                    if 'unique_keys' not in result:
                        self.test_results['calculate_statistics']['failed'] += 1
                        self.test_results['calculate_statistics']['details'].append(f"❌ {case['name']}: Missing unique_keys field")
                        logger.error(f"  ❌ {case['name']}: Missing unique_keys field")
                        continue
                
                self.test_results['calculate_statistics']['passed'] += 1
                self.test_results['calculate_statistics']['details'].append(f"✅ {case['name']}: PASSED")
                logger.info(f"  ✅ {case['name']}: PASSED")
                logger.info(f"    Result keys: {list(result.keys())}")
                    
            except Exception as e:
                self.test_results['calculate_statistics']['failed'] += 1
                self.test_results['calculate_statistics']['details'].append(f"❌ {case['name']}: Exception - {str(e)}")
                logger.error(f"  ❌ {case['name']}: Exception - {str(e)}")

    def test_extract_metadata_comprehensive(self):
        """Comprehensive test of extract_metadata function."""
        logger.info("🧪 Testing extract_metadata function (enhanced)")
        
        test_cases = [
            {
                'name': 'Complete metadata extraction',
                'content': {
                    'name': 'TP53',
                    'type': 'gene',
                    'score': 1.5,
                    'description': 'tumor protein',
                    'extra': 'additional_info'
                },
                'required_fields': ['name', 'type'],
                'optional_fields': ['score', 'description'],
                'expected_keys': ['name', 'type', 'score', 'description']
            },
            {
                'name': 'Missing required field',
                'content': {
                    'name': 'TP53',
                    'score': 1.5
                },
                'required_fields': ['name', 'type'],
                'optional_fields': ['score'],
                'expected_keys': ['name', 'type', 'score'],  # type will be None
                'expected_nulls': ['type']
            },
            {
                'name': 'Missing optional fields',
                'content': {
                    'name': 'TP53',
                    'type': 'gene'
                },
                'required_fields': ['name', 'type'],
                'optional_fields': ['score', 'description'],
                'expected_keys': ['name', 'type']  # Optional fields not included if missing
            },
            {
                'name': 'No optional fields specified',
                'content': {
                    'name': 'TP53',
                    'type': 'gene',
                    'extra': 'ignored'
                },
                'required_fields': ['name', 'type'],
                'optional_fields': None,
                'expected_keys': ['name', 'type']
            },
            {
                'name': 'Null values in content',
                'content': {
                    'name': 'TP53',
                    'type': None,
                    'score': 1.5
                },
                'required_fields': ['name', 'type'],
                'optional_fields': ['score'],
                'expected_keys': ['name', 'type', 'score'],
                'expected_nulls': ['type']
            }
        ]
        
        for case in test_cases:
            try:
                result = ParserUtils.extract_metadata(
                    case['content'], 
                    case['required_fields'], 
                    case.get('optional_fields')
                )
                
                # Check result is dictionary
                if not isinstance(result, dict):
                    self.test_results['extract_metadata']['failed'] += 1
                    self.test_results['extract_metadata']['details'].append(f"❌ {case['name']}: Result is not a dictionary")
                    logger.error(f"  ❌ {case['name']}: Result is not a dictionary")
                    continue
                
                # Check required fields are present
                missing_required = [field for field in case['required_fields'] if field not in result]
                if missing_required:
                    self.test_results['extract_metadata']['failed'] += 1
                    self.test_results['extract_metadata']['details'].append(f"❌ {case['name']}: Missing required fields: {missing_required}")
                    logger.error(f"  ❌ {case['name']}: Missing required fields: {missing_required}")
                    continue
                
                # Check expected keys
                if set(result.keys()) != set(case['expected_keys']):
                    self.test_results['extract_metadata']['failed'] += 1
                    self.test_results['extract_metadata']['details'].append(f"❌ {case['name']}: Expected keys {case['expected_keys']}, got {list(result.keys())}")
                    logger.error(f"  ❌ {case['name']}: Expected keys {case['expected_keys']}, got {list(result.keys())}")
                    continue
                
                # Check expected null values
                if case.get('expected_nulls'):
                    for null_field in case['expected_nulls']:
                        if result.get(null_field) is not None:
                            self.test_results['extract_metadata']['failed'] += 1
                            self.test_results['extract_metadata']['details'].append(f"❌ {case['name']}: Expected {null_field} to be None, got {result[null_field]}")
                            logger.error(f"  ❌ {case['name']}: Expected {null_field} to be None, got {result[null_field]}")
                            continue
                
                self.test_results['extract_metadata']['passed'] += 1
                self.test_results['extract_metadata']['details'].append(f"✅ {case['name']}: PASSED")
                logger.info(f"  ✅ {case['name']}: PASSED")
                logger.info(f"    Extracted: {result}")
                    
            except Exception as e:
                self.test_results['extract_metadata']['failed'] += 1
                self.test_results['extract_metadata']['details'].append(f"❌ {case['name']}: Exception - {str(e)}")
                logger.error(f"  ❌ {case['name']}: Exception - {str(e)}")

    def run_enhanced_tests(self):
        """Run all enhanced test suites."""
        logger.info("=" * 70)
        logger.info("🚀 STARTING ENHANCED COMPREHENSIVE PARSER_UTILS TESTING")
        logger.info("=" * 70)
        
        self.setup_test_files()
        
        try:
            # Run original comprehensive tests first
            self.test_load_file_safe_comprehensive()
            
            # Run basic tests for other functions (reuse from original test)
            self._run_basic_validation_tests()
            self._run_basic_utility_tests()
            
            # Run enhanced tests for missing/complex functions
            self.test_create_cross_references()
            self.test_calculate_statistics_comprehensive()
            self.test_extract_metadata_comprehensive()
            
        finally:
            self.cleanup_test_files()
        
        return self.generate_enhanced_test_report()
    
    def _run_basic_validation_tests(self):
        """Run basic validation tests for common functions."""
        # Test validate_required_columns
        test_df = ParserUtils.load_file_safe(self.test_csv_path, 'csv')
        
        # Test cases
        cases = [
            (test_df, ['gene_symbol', 'go_id'], True),
            (test_df, ['gene_symbol', 'missing_col'], False),
            (None, ['gene_symbol'], False),
            (test_df, [], True)
        ]
        
        for df, cols, expected in cases:
            try:
                result = ParserUtils.validate_required_columns(df, cols, "test.csv")
                if result == expected:
                    self.test_results['validate_required_columns']['passed'] += 1
                    self.test_results['validate_required_columns']['details'].append(f"✅ validate_required_columns test: PASSED")
                else:
                    self.test_results['validate_required_columns']['failed'] += 1
                    self.test_results['validate_required_columns']['details'].append(f"❌ validate_required_columns test: FAILED")
            except Exception as e:
                self.test_results['validate_required_columns']['failed'] += 1
                self.test_results['validate_required_columns']['details'].append(f"❌ validate_required_columns test: Exception - {str(e)}")

    def _run_basic_utility_tests(self):
        """Run basic tests for utility functions."""
        # Test clean_gene_identifiers
        test_cases = [
            (['TP53', 'BRCA1'], ['TP53', 'BRCA1']),
            (['tp53', 'brca1'], ['TP53', 'BRCA1']),
            (['TP53', '', 'BRCA1'], ['TP53', 'BRCA1']),
            ([], [])
        ]
        
        for input_genes, expected in test_cases:
            try:
                result = ParserUtils.clean_gene_identifiers(input_genes)
                if result == expected:
                    self.test_results['clean_gene_identifiers']['passed'] += 1
                    self.test_results['clean_gene_identifiers']['details'].append(f"✅ clean_gene_identifiers test: PASSED")
                else:
                    self.test_results['clean_gene_identifiers']['failed'] += 1
                    self.test_results['clean_gene_identifiers']['details'].append(f"❌ clean_gene_identifiers test: FAILED")
            except Exception as e:
                self.test_results['clean_gene_identifiers']['failed'] += 1
                self.test_results['clean_gene_identifiers']['details'].append(f"❌ clean_gene_identifiers test: Exception - {str(e)}")
        
        # Test validate_go_id
        go_test_cases = [
            ('GO:0008150', True),
            ('GO:1234567', True),
            ('GG:0008150', False),
            ('GO:123', False),
            ('', False),
            (None, False)
        ]
        
        for go_id, expected in go_test_cases:
            try:
                result = ParserUtils.validate_go_id(go_id)
                if result == expected:
                    self.test_results['validate_go_id']['passed'] += 1
                    self.test_results['validate_go_id']['details'].append(f"✅ validate_go_id test: PASSED")
                else:
                    self.test_results['validate_go_id']['failed'] += 1
                    self.test_results['validate_go_id']['details'].append(f"❌ validate_go_id test: FAILED")
            except Exception as e:
                self.test_results['validate_go_id']['failed'] += 1
                self.test_results['validate_go_id']['details'].append(f"❌ validate_go_id test: Exception - {str(e)}")
        
        # Test validate_gene_symbol
        gene_test_cases = [
            ('TP53', True),
            ('BRCA1', True),
            ('GENE-1', True),
            ('GENE_1', True),
            ('', False),
            (None, False)
        ]
        
        for gene, expected in gene_test_cases:
            try:
                result = ParserUtils.validate_gene_symbol(gene)
                if result == expected:
                    self.test_results['validate_gene_symbol']['passed'] += 1
                    self.test_results['validate_gene_symbol']['details'].append(f"✅ validate_gene_symbol test: PASSED")
                else:
                    self.test_results['validate_gene_symbol']['failed'] += 1
                    self.test_results['validate_gene_symbol']['details'].append(f"❌ validate_gene_symbol test: FAILED")
            except Exception as e:
                self.test_results['validate_gene_symbol']['failed'] += 1
                self.test_results['validate_gene_symbol']['details'].append(f"❌ validate_gene_symbol test: Exception - {str(e)}")
        
        # Test extract_unique_values
        test_data = [
            {'gene': 'TP53', 'go_id': 'GO:0008150'},
            {'gene': 'BRCA1', 'go_id': 'GO:0003674'},
            {'gene': 'TP53', 'go_id': 'GO:0005575'}
        ]
        
        try:
            result = ParserUtils.extract_unique_values(test_data, 'gene')
            expected = {'TP53', 'BRCA1'}
            if result == expected:
                self.test_results['extract_unique_values']['passed'] += 1
                self.test_results['extract_unique_values']['details'].append(f"✅ extract_unique_values test: PASSED")
            else:
                self.test_results['extract_unique_values']['failed'] += 1
                self.test_results['extract_unique_values']['details'].append(f"❌ extract_unique_values test: Expected {expected}, got {result}")
        except Exception as e:
            self.test_results['extract_unique_values']['failed'] += 1
            self.test_results['extract_unique_values']['details'].append(f"❌ extract_unique_values test: Exception - {str(e)}")
        
        # Test log_parsing_progress (just check it doesn't crash)
        try:
            ParserUtils.log_parsing_progress(500, 1000, 100)
            self.test_results['log_parsing_progress']['passed'] += 1
            self.test_results['log_parsing_progress']['details'].append(f"✅ log_parsing_progress test: PASSED")
        except Exception as e:
            self.test_results['log_parsing_progress']['failed'] += 1
            self.test_results['log_parsing_progress']['details'].append(f"❌ log_parsing_progress test: Exception - {str(e)}")

    def generate_enhanced_test_report(self):
        """Generate comprehensive enhanced test report."""
        logger.info("\n" + "=" * 70)
        logger.info("📊 ENHANCED PARSER_UTILS TEST RESULTS SUMMARY")
        logger.info("=" * 70)
        
        total_passed = 0
        total_failed = 0
        
        for func_name, results in self.test_results.items():
            passed = results['passed']
            failed = results['failed']
            total_passed += passed
            total_failed += failed
            
            if passed + failed > 0:
                success_rate = (passed / (passed + failed)) * 100
                status = "✅ PASS" if failed == 0 else "⚠️ PARTIAL" if passed > 0 else "❌ FAIL"
                logger.info(f"{status} {func_name}: {passed} passed, {failed} failed ({success_rate:.1f}%)")
        
        overall_success_rate = (total_passed / (total_passed + total_failed)) * 100 if (total_passed + total_failed) > 0 else 0
        
        logger.info("\n" + "-" * 70)
        logger.info(f"📈 ENHANCED OVERALL RESULTS:")
        logger.info(f"   Total Tests: {total_passed + total_failed}")
        logger.info(f"   Passed: {total_passed}")
        logger.info(f"   Failed: {total_failed}")
        logger.info(f"   Success Rate: {overall_success_rate:.1f}%")
        
        final_status = "🎉 ALL TESTS PASSED" if total_failed == 0 else f"⚠️ {total_failed} TESTS FAILED"
        logger.info(f"   Final Status: {final_status}")
        logger.info("=" * 70)
        
        return {
            'total_tests': total_passed + total_failed,
            'passed': total_passed,
            'failed': total_failed,
            'success_rate': overall_success_rate,
            'detailed_results': self.test_results
        }


def main():
    """Main enhanced test execution function."""
    tester = EnhancedTestParserUtils()
    results = tester.run_enhanced_tests()
    
    # Save results to file
    results_file = os.path.join(os.path.dirname(__file__), 'enhanced_parser_utils_test_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n💾 Enhanced test results saved to: {results_file}")
    
    return results['success_rate'] >= 95.0  # Higher bar for enhanced tests


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)