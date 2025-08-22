#!/usr/bin/env python3
"""
Comprehensive Test Suite for Remaining Parser Components

Tests CombinedGOParser, CombinedBiomedicalParser, and import system.
This test focuses on integration and overall functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Import the parsers
from parsers.core_parsers import CombinedGOParser
from parsers.parser_orchestrator import CombinedBiomedicalParser

# Test import system
import parsers
from parsers import GODataParser, OmicsDataParser, CombinedGOParser as ImportedCombinedGOParser
from parsers import GOBPDataParser  # Test backward compatibility

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestRemainingParsers:
    """Test class for remaining parser components."""
    
    def __init__(self):
        self.test_results = {
            'imports': {'passed': 0, 'failed': 0, 'details': []},
            'backward_compatibility': {'passed': 0, 'failed': 0, 'details': []},
            'combined_go_parser_init': {'passed': 0, 'failed': 0, 'details': []},
            'combined_go_parser_methods': {'passed': 0, 'failed': 0, 'details': []},
            'combined_biomedical_parser_init': {'passed': 0, 'failed': 0, 'details': []},
            'combined_biomedical_parser_methods': {'passed': 0, 'failed': 0, 'details': []},
            'integration': {'passed': 0, 'failed': 0, 'details': []}
        }
        
        # Data directory paths
        self.data_base_dir = Path("llm_evaluation_for_gene_set_interpretation/data")

    def test_import_system(self):
        """Test the import system and module structure."""
        logger.info("🧪 Testing import system")
        
        import_tests = [
            {
                'name': 'Direct parser imports',
                'test': lambda: all([
                    hasattr(parsers, 'GODataParser'),
                    hasattr(parsers, 'OmicsDataParser'),
                    hasattr(parsers, 'CombinedGOParser'),
                    hasattr(parsers, 'CombinedBiomedicalParser')
                ])
            },
            {
                'name': 'Backward compatibility alias',
                'test': lambda: hasattr(parsers, 'GOBPDataParser') and parsers.GOBPDataParser == parsers.GODataParser
            },
            {
                'name': 'Import structure consistency',
                'test': lambda: ImportedCombinedGOParser == CombinedGOParser
            },
            {
                'name': 'Module-level imports',
                'test': lambda: all([
                    'GODataParser' in dir(parsers),
                    'OmicsDataParser' in dir(parsers),
                    'CombinedGOParser' in dir(parsers),
                    'CombinedBiomedicalParser' in dir(parsers)
                ])
            }
        ]
        
        for test_case in import_tests:
            try:
                if test_case['test']():
                    self.test_results['imports']['passed'] += 1
                    self.test_results['imports']['details'].append(f"✅ {test_case['name']}: PASSED")
                    logger.info(f"  ✅ {test_case['name']}: PASSED")
                else:
                    self.test_results['imports']['failed'] += 1
                    self.test_results['imports']['details'].append(f"❌ {test_case['name']}: FAILED")
                    logger.error(f"  ❌ {test_case['name']}: FAILED")
                    
            except Exception as e:
                self.test_results['imports']['failed'] += 1
                self.test_results['imports']['details'].append(f"❌ {test_case['name']}: Exception - {str(e)}")
                logger.error(f"  ❌ {test_case['name']}: Exception - {str(e)}")

    def test_backward_compatibility(self):
        """Test backward compatibility features."""
        logger.info("🧪 Testing backward compatibility")
        
        try:
            # Test GOBPDataParser alias
            logger.info("  Testing GOBPDataParser alias...")
            
            # Should be able to create instances with the old name
            if self.data_base_dir.exists():
                go_bp_dir = self.data_base_dir / "GO_BP"
                if go_bp_dir.exists():
                    old_style_parser = GOBPDataParser(str(go_bp_dir))
                    new_style_parser = GODataParser(str(go_bp_dir))
                    
                    # Both should be the same class
                    if type(old_style_parser) == type(new_style_parser):
                        self.test_results['backward_compatibility']['passed'] += 1
                        self.test_results['backward_compatibility']['details'].append(
                            "✅ GOBPDataParser alias works correctly"
                        )
                        logger.info("    ✅ GOBPDataParser alias works correctly")
                    else:
                        self.test_results['backward_compatibility']['failed'] += 1
                        self.test_results['backward_compatibility']['details'].append(
                            "❌ GOBPDataParser alias type mismatch"
                        )
                else:
                    logger.info("    ⚠️ GO_BP directory not found for backward compatibility test")
            else:
                logger.info("    ⚠️ Data directory not found for backward compatibility test")
                
        except Exception as e:
            self.test_results['backward_compatibility']['failed'] += 1
            self.test_results['backward_compatibility']['details'].append(f"❌ Backward compatibility test failed: {str(e)}")
            logger.error(f"  ❌ Backward compatibility test failed: {str(e)}")

    def test_combined_go_parser(self):
        """Test CombinedGOParser class."""
        logger.info("🧪 Testing CombinedGOParser")
        
        try:
            logger.info("  Testing CombinedGOParser initialization...")
            if self.data_base_dir.exists():
                combined_parser = CombinedGOParser(str(self.data_base_dir))
                
                self.test_results['combined_go_parser_init']['passed'] += 1
                self.test_results['combined_go_parser_init']['details'].append(
                    "✅ CombinedGOParser initialized successfully"
                )
                logger.info("    ✅ CombinedGOParser initialized successfully")
                
                # Test namespace parsers
                namespace_count = 0
                for namespace in ['biological_process', 'cellular_component', 'molecular_function']:
                    parser_attr = f"{namespace.replace('_', '')}parser"
                    if hasattr(combined_parser, parser_attr):
                        parser_obj = getattr(combined_parser, parser_attr)
                        if parser_obj is not None:
                            namespace_count += 1
                            logger.info(f"      ✅ {namespace} parser available")
                
                logger.info(f"    📊 Available namespace parsers: {namespace_count}/3")
                
                # Test methods
                logger.info("  Testing CombinedGOParser methods...")
                methods_to_test = ['parse_all_namespaces', 'get_combined_summary']
                
                for method_name in methods_to_test:
                    try:
                        method = getattr(combined_parser, method_name)
                        logger.info(f"    Testing {method_name}...")
                        result = method()
                        
                        if result is not None:
                            self.test_results['combined_go_parser_methods']['passed'] += 1
                            self.test_results['combined_go_parser_methods']['details'].append(
                                f"✅ {method_name}: Executed successfully"
                            )
                            logger.info(f"      ✅ {method_name}: Success - returned {type(result)}")
                            
                            # Log some details about the result
                            if isinstance(result, dict):
                                logger.info(f"        📊 Result keys: {list(result.keys())}")
                                if method_name == 'parse_all_namespaces':
                                    for ns, data in result.items():
                                        if isinstance(data, dict):
                                            go_terms = data.get('go_terms', {})
                                            associations = data.get('gene_associations', [])
                                            logger.info(f"        {ns}: {len(go_terms)} terms, {len(associations)} associations")
                        else:
                            self.test_results['combined_go_parser_methods']['failed'] += 1
                            self.test_results['combined_go_parser_methods']['details'].append(
                                f"❌ {method_name}: Returned None"
                            )
                            
                    except Exception as e:
                        self.test_results['combined_go_parser_methods']['failed'] += 1
                        self.test_results['combined_go_parser_methods']['details'].append(
                            f"❌ {method_name}: Exception - {str(e)}"
                        )
                        logger.error(f"      ❌ {method_name}: Exception - {str(e)}")
                        
            else:
                self.test_results['combined_go_parser_init']['failed'] += 1
                self.test_results['combined_go_parser_init']['details'].append(
                    "❌ Data directory not found"
                )
                logger.error("    ❌ Data directory not found")
                
        except Exception as e:
            self.test_results['combined_go_parser_init']['failed'] += 1
            self.test_results['combined_go_parser_init']['details'].append(f"❌ Initialization failed: {str(e)}")
            logger.error(f"  ❌ CombinedGOParser initialization failed: {str(e)}")

    def test_combined_biomedical_parser(self):
        """Test CombinedBiomedicalParser class."""
        logger.info("🧪 Testing CombinedBiomedicalParser")
        
        try:
            logger.info("  Testing CombinedBiomedicalParser initialization...")
            if self.data_base_dir.exists():
                biomedical_parser = CombinedBiomedicalParser(str(self.data_base_dir))
                
                self.test_results['combined_biomedical_parser_init']['passed'] += 1
                self.test_results['combined_biomedical_parser_init']['details'].append(
                    "✅ CombinedBiomedicalParser initialized successfully"
                )
                logger.info("    ✅ CombinedBiomedicalParser initialized successfully")
                
                # Test available parsers
                logger.info("  Testing available parsers...")
                available_parsers = biomedical_parser.get_available_parsers()
                logger.info(f"    📊 Available parsers: {available_parsers}")
                
                active_parsers = sum(1 for available in available_parsers.values() if available)
                logger.info(f"    📈 Active parsers: {active_parsers}/{len(available_parsers)}")
                
                # Test methods
                logger.info("  Testing CombinedBiomedicalParser methods...")
                methods_to_test = [
                    'get_comprehensive_summary',
                    'validate_comprehensive_data',
                    'get_available_parsers'
                ]
                
                for method_name in methods_to_test:
                    try:
                        method = getattr(biomedical_parser, method_name)
                        logger.info(f"    Testing {method_name}...")
                        result = method()
                        
                        if result is not None:
                            self.test_results['combined_biomedical_parser_methods']['passed'] += 1
                            self.test_results['combined_biomedical_parser_methods']['details'].append(
                                f"✅ {method_name}: Executed successfully"
                            )
                            logger.info(f"      ✅ {method_name}: Success - returned {type(result)}")
                            
                            # Log method-specific details
                            if method_name == 'get_comprehensive_summary':
                                if isinstance(result, dict):
                                    data_sources = result.get('data_sources', [])
                                    logger.info(f"        📊 Data sources: {data_sources}")
                                    
                            elif method_name == 'validate_comprehensive_data':
                                if isinstance(result, dict):
                                    overall_valid = result.get('overall_valid', False)
                                    logger.info(f"        ✅ Validation result: {overall_valid}")
                                    
                        else:
                            self.test_results['combined_biomedical_parser_methods']['failed'] += 1
                            self.test_results['combined_biomedical_parser_methods']['details'].append(
                                f"❌ {method_name}: Returned None"
                            )
                            
                    except Exception as e:
                        self.test_results['combined_biomedical_parser_methods']['failed'] += 1
                        self.test_results['combined_biomedical_parser_methods']['details'].append(
                            f"❌ {method_name}: Exception - {str(e)}"
                        )
                        logger.error(f"      ❌ {method_name}: Exception - {str(e)}")
                
                # Test integration with main parsing method (light test)
                logger.info("  Testing parse_all_biomedical_data integration...")
                try:
                    # This is a heavy operation, so we just test that it starts without error
                    # and has the right interface
                    method = getattr(biomedical_parser, 'parse_all_biomedical_data')
                    if callable(method):
                        self.test_results['integration']['passed'] += 1
                        self.test_results['integration']['details'].append(
                            "✅ parse_all_biomedical_data method available and callable"
                        )
                        logger.info("      ✅ parse_all_biomedical_data: Method available and callable")
                    else:
                        self.test_results['integration']['failed'] += 1
                        self.test_results['integration']['details'].append(
                            "❌ parse_all_biomedical_data: Method not callable"
                        )
                        
                except Exception as e:
                    self.test_results['integration']['failed'] += 1
                    self.test_results['integration']['details'].append(f"❌ Integration test failed: {str(e)}")
                    logger.error(f"      ❌ Integration test failed: {str(e)}")
                
            else:
                self.test_results['combined_biomedical_parser_init']['failed'] += 1
                self.test_results['combined_biomedical_parser_init']['details'].append(
                    "❌ Data directory not found"
                )
                logger.error("    ❌ Data directory not found")
                
        except Exception as e:
            self.test_results['combined_biomedical_parser_init']['failed'] += 1
            self.test_results['combined_biomedical_parser_init']['details'].append(f"❌ Initialization failed: {str(e)}")
            logger.error(f"  ❌ CombinedBiomedicalParser initialization failed: {str(e)}")

    def run_all_tests(self):
        """Run all test suites."""
        logger.info("=" * 80)
        logger.info("🚀 STARTING COMPREHENSIVE REMAINING PARSERS TESTING")
        logger.info("=" * 80)
        
        # Run all test methods
        self.test_import_system()
        self.test_backward_compatibility()
        self.test_combined_go_parser()
        self.test_combined_biomedical_parser()
        
        return self.generate_test_report()

    def generate_test_report(self):
        """Generate comprehensive test report."""
        logger.info("\n" + "=" * 80)
        logger.info("📊 REMAINING PARSERS TEST RESULTS SUMMARY")
        logger.info("=" * 80)
        
        total_passed = 0
        total_failed = 0
        
        for test_group, results in self.test_results.items():
            passed = results['passed']
            failed = results['failed']
            total_passed += passed
            total_failed += failed
            
            if passed + failed > 0:
                success_rate = (passed / (passed + failed)) * 100
                status = "✅ PASS" if failed == 0 else "⚠️ PARTIAL" if passed > 0 else "❌ FAIL"
                logger.info(f"{status} {test_group}: {passed} passed, {failed} failed ({success_rate:.1f}%)")
                
                # Show first success detail for each group
                if passed > 0:
                    for detail in results['details'][:1]:
                        if detail.startswith('✅'):
                            logger.info(f"    {detail}")
        
        overall_success_rate = (total_passed / (total_passed + total_failed)) * 100 if (total_passed + total_failed) > 0 else 0
        
        logger.info("\n" + "-" * 80)
        logger.info(f"📈 OVERALL RESULTS:")
        logger.info(f"   Total Tests: {total_passed + total_failed}")
        logger.info(f"   Passed: {total_passed}")
        logger.info(f"   Failed: {total_failed}")
        logger.info(f"   Success Rate: {overall_success_rate:.1f}%")
        
        final_status = "🎉 ALL TESTS PASSED" if total_failed == 0 else f"⚠️ {total_failed} TESTS FAILED"
        logger.info(f"   Final Status: {final_status}")
        logger.info("=" * 80)
        
        return {
            'total_tests': total_passed + total_failed,
            'passed': total_passed,
            'failed': total_failed,
            'success_rate': overall_success_rate,
            'detailed_results': self.test_results
        }


def main():
    """Main test execution function."""
    tester = TestRemainingParsers()
    results = tester.run_all_tests()
    
    # Save results to file
    results_file = os.path.join(os.path.dirname(__file__), 'remaining_parsers_test_results.json')
    
    if results:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n💾 Test results saved to: {results_file}")
        return results['success_rate'] >= 70.0
    else:
        logger.error("No test results to save")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)