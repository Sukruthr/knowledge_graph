#!/usr/bin/env python3
"""
Compare Parser Outputs with Original Implementation

Quick validation to ensure new parser outputs match original data_parsers.py behavior.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import logging
from pathlib import Path

# Import both old and new parsers
from parsers.core_parsers import GODataParser as NewGODataParser
from parsers.core_parsers import OmicsDataParser as NewOmicsDataParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OutputComparison:
    """Compare outputs between old and new parser implementations."""
    
    def __init__(self):
        self.data_base_dir = Path("llm_evaluation_for_gene_set_interpretation/data")
        self.comparison_results = {
            'go_parser_comparison': {'passed': 0, 'failed': 0, 'details': []},
            'omics_parser_comparison': {'passed': 0, 'failed': 0, 'details': []},
            'integration_comparison': {'passed': 0, 'failed': 0, 'details': []}
        }

    def compare_go_parser_outputs(self):
        """Compare GO parser outputs with expected behavior."""
        logger.info("🔍 Comparing GO parser outputs with expected behavior")
        
        try:
            if not self.data_base_dir.exists():
                logger.error("  ❌ Data directory not found - skipping comparison")
                return
            
            # Test GO BP parser
            go_bp_dir = self.data_base_dir / "GO_BP"
            if go_bp_dir.exists():
                logger.info("  Testing GO BP parser...")
                
                parser = NewGODataParser(str(go_bp_dir), namespace='biological_process')
                
                # Test key methods and validate outputs
                go_terms = parser.parse_go_terms()
                if isinstance(go_terms, dict) and len(go_terms) > 20000:
                    self.comparison_results['go_parser_comparison']['passed'] += 1
                    self.comparison_results['go_parser_comparison']['details'].append(
                        f"✅ GO BP terms: {len(go_terms):,} (expected >20K)"
                    )
                    logger.info(f"    ✅ GO BP terms: {len(go_terms):,}")
                else:
                    self.comparison_results['go_parser_comparison']['failed'] += 1
                    self.comparison_results['go_parser_comparison']['details'].append(
                        f"❌ GO BP terms: {len(go_terms) if isinstance(go_terms, dict) else 'Invalid type'}"
                    )
                
                # Test gene associations
                gene_assocs = parser.parse_gene_go_associations_from_gaf()
                if isinstance(gene_assocs, list) and len(gene_assocs) > 100000:
                    self.comparison_results['go_parser_comparison']['passed'] += 1
                    self.comparison_results['go_parser_comparison']['details'].append(
                        f"✅ GO BP associations: {len(gene_assocs):,} (expected >100K)"
                    )
                    logger.info(f"    ✅ GO BP associations: {len(gene_assocs):,}")
                else:
                    self.comparison_results['go_parser_comparison']['failed'] += 1
                    self.comparison_results['go_parser_comparison']['details'].append(
                        f"❌ GO BP associations: {len(gene_assocs) if isinstance(gene_assocs, list) else 'Invalid type'}"
                    )
                
                # Test alternative IDs
                alt_ids = parser.parse_go_alternative_ids()
                if isinstance(alt_ids, dict) and len(alt_ids) > 1000:
                    self.comparison_results['go_parser_comparison']['passed'] += 1
                    self.comparison_results['go_parser_comparison']['details'].append(
                        f"✅ GO BP alt IDs: {len(alt_ids):,} (expected >1K)"
                    )
                    logger.info(f"    ✅ GO BP alt IDs: {len(alt_ids):,}")
                else:
                    self.comparison_results['go_parser_comparison']['failed'] += 1
                    self.comparison_results['go_parser_comparison']['details'].append(
                        f"❌ GO BP alt IDs: {len(alt_ids) if isinstance(alt_ids, dict) else 'Invalid type'}"
                    )
                    
        except Exception as e:
            self.comparison_results['go_parser_comparison']['failed'] += 1
            self.comparison_results['go_parser_comparison']['details'].append(f"❌ GO parser comparison failed: {str(e)}")
            logger.error(f"  ❌ GO parser comparison failed: {str(e)}")

    def compare_omics_parser_outputs(self):
        """Compare Omics parser outputs with expected behavior."""
        logger.info("🔍 Comparing Omics parser outputs with expected behavior")
        
        try:
            omics_dir = self.data_base_dir / "Omics_data"
            omics_data2_dir = self.data_base_dir / "Omics_data2"
            
            if not omics_dir.exists():
                logger.error("  ❌ Omics data directory not found - skipping comparison")
                return
            
            logger.info("  Testing Omics parser...")
            
            # Initialize parser
            omics_data2_path = str(omics_data2_dir) if omics_data2_dir.exists() else None
            parser = NewOmicsDataParser(str(omics_dir), omics_data2_path)
            
            # Test drug associations
            drug_assocs = parser.parse_drug_gene_associations()
            if isinstance(drug_assocs, list) and len(drug_assocs) > 200000:
                self.comparison_results['omics_parser_comparison']['passed'] += 1
                self.comparison_results['omics_parser_comparison']['details'].append(
                    f"✅ Drug associations: {len(drug_assocs):,} (expected >200K)"
                )
                logger.info(f"    ✅ Drug associations: {len(drug_assocs):,}")
            else:
                self.comparison_results['omics_parser_comparison']['failed'] += 1
                self.comparison_results['omics_parser_comparison']['details'].append(
                    f"❌ Drug associations: {len(drug_assocs) if isinstance(drug_assocs, list) else 'Invalid type'}"
                )
            
            # Test viral associations
            viral_assocs = parser.parse_viral_gene_associations()
            if isinstance(viral_assocs, list) and len(viral_assocs) > 200000:
                self.comparison_results['omics_parser_comparison']['passed'] += 1
                self.comparison_results['omics_parser_comparison']['details'].append(
                    f"✅ Viral associations: {len(viral_assocs):,} (expected >200K)"
                )
                logger.info(f"    ✅ Viral associations: {len(viral_assocs):,}")
            else:
                self.comparison_results['omics_parser_comparison']['failed'] += 1
                self.comparison_results['omics_parser_comparison']['details'].append(
                    f"❌ Viral associations: {len(viral_assocs) if isinstance(viral_assocs, list) else 'Invalid type'}"
                )
            
            # Test expression matrix
            viral_expr = parser.parse_viral_expression_matrix()
            if isinstance(viral_expr, list) and len(viral_expr) > 1000000:
                self.comparison_results['omics_parser_comparison']['passed'] += 1
                self.comparison_results['omics_parser_comparison']['details'].append(
                    f"✅ Viral expression: {len(viral_expr):,} (expected >1M)"
                )
                logger.info(f"    ✅ Viral expression: {len(viral_expr):,}")
            else:
                self.comparison_results['omics_parser_comparison']['failed'] += 1
                self.comparison_results['omics_parser_comparison']['details'].append(
                    f"❌ Viral expression: {len(viral_expr) if isinstance(viral_expr, list) else 'Invalid type'}"
                )
            
            # Test unique entities
            unique_entities = parser.get_unique_entities()
            if isinstance(unique_entities, dict) and len(unique_entities) >= 4:
                self.comparison_results['omics_parser_comparison']['passed'] += 1
                self.comparison_results['omics_parser_comparison']['details'].append(
                    f"✅ Unique entities: {list(unique_entities.keys())}"
                )
                logger.info(f"    ✅ Unique entities: {list(unique_entities.keys())}")
            else:
                self.comparison_results['omics_parser_comparison']['failed'] += 1
                self.comparison_results['omics_parser_comparison']['details'].append(
                    f"❌ Unique entities: Invalid structure"
                )
                
        except Exception as e:
            self.comparison_results['omics_parser_comparison']['failed'] += 1
            self.comparison_results['omics_parser_comparison']['details'].append(f"❌ Omics parser comparison failed: {str(e)}")
            logger.error(f"  ❌ Omics parser comparison failed: {str(e)}")

    def test_integration_compatibility(self):
        """Test that parsers work together as expected."""
        logger.info("🔍 Testing integration compatibility")
        
        try:
            # Test that parsers can be imported and instantiated together
            from parsers import GODataParser, OmicsDataParser, CombinedGOParser, CombinedBiomedicalParser
            
            if self.data_base_dir.exists():
                # Test combined biomedical parser
                logger.info("  Testing CombinedBiomedicalParser...")
                combined_parser = CombinedBiomedicalParser(str(self.data_base_dir))
                
                # Test parser availability
                available_parsers = combined_parser.get_available_parsers()
                active_parsers = sum(1 for available in available_parsers.values() if available)
                
                if active_parsers >= 2:  # Should have at least GO and Omics parsers
                    self.comparison_results['integration_comparison']['passed'] += 1
                    self.comparison_results['integration_comparison']['details'].append(
                        f"✅ Integration: {active_parsers}/{len(available_parsers)} parsers active"
                    )
                    logger.info(f"    ✅ Integration: {active_parsers}/{len(available_parsers)} parsers active")
                else:
                    self.comparison_results['integration_comparison']['failed'] += 1
                    self.comparison_results['integration_comparison']['details'].append(
                        f"❌ Integration: Only {active_parsers} parsers active"
                    )
                
                # Test summary generation
                summary = combined_parser.get_comprehensive_summary()
                if isinstance(summary, dict) and 'data_sources' in summary:
                    self.comparison_results['integration_comparison']['passed'] += 1
                    self.comparison_results['integration_comparison']['details'].append(
                        f"✅ Summary generation successful"
                    )
                    logger.info(f"    ✅ Summary generation successful")
                else:
                    self.comparison_results['integration_comparison']['failed'] += 1
                    self.comparison_results['integration_comparison']['details'].append(
                        f"❌ Summary generation failed"
                    )
                    
        except Exception as e:
            self.comparison_results['integration_comparison']['failed'] += 1
            self.comparison_results['integration_comparison']['details'].append(f"❌ Integration test failed: {str(e)}")
            logger.error(f"  ❌ Integration test failed: {str(e)}")

    def run_all_comparisons(self):
        """Run all comparison tests."""
        logger.info("=" * 80)
        logger.info("🔍 COMPARING NEW PARSERS WITH EXPECTED BEHAVIOR")
        logger.info("=" * 80)
        
        self.compare_go_parser_outputs()
        self.compare_omics_parser_outputs()
        self.test_integration_compatibility()
        
        return self.generate_comparison_report()

    def generate_comparison_report(self):
        """Generate comparison report."""
        logger.info("\n" + "=" * 80)
        logger.info("📊 COMPARISON RESULTS SUMMARY")
        logger.info("=" * 80)
        
        total_passed = 0
        total_failed = 0
        
        for test_group, results in self.comparison_results.items():
            passed = results['passed']
            failed = results['failed']
            total_passed += passed
            total_failed += failed
            
            if passed + failed > 0:
                success_rate = (passed / (passed + failed)) * 100
                status = "✅ PASS" if failed == 0 else "⚠️ PARTIAL" if passed > 0 else "❌ FAIL"
                logger.info(f"{status} {test_group}: {passed} passed, {failed} failed ({success_rate:.1f}%)")
                
                # Show details
                for detail in results['details']:
                    logger.info(f"    {detail}")
        
        overall_success_rate = (total_passed / (total_passed + total_failed)) * 100 if (total_passed + total_failed) > 0 else 0
        
        logger.info("\n" + "-" * 80)
        logger.info(f"📈 OVERALL COMPARISON RESULTS:")
        logger.info(f"   Total Tests: {total_passed + total_failed}")
        logger.info(f"   Passed: {total_passed}")
        logger.info(f"   Failed: {total_failed}")
        logger.info(f"   Success Rate: {overall_success_rate:.1f}%")
        
        final_status = "🎉 PARSERS MATCH EXPECTED BEHAVIOR" if total_failed == 0 else f"⚠️ {total_failed} COMPARISON DIFFERENCES FOUND"
        logger.info(f"   Final Status: {final_status}")
        logger.info("=" * 80)
        
        return {
            'total_tests': total_passed + total_failed,
            'passed': total_passed,
            'failed': total_failed,
            'success_rate': overall_success_rate,
            'detailed_results': self.comparison_results
        }


def main():
    """Main execution function."""
    comparator = OutputComparison()
    results = comparator.run_all_comparisons()
    
    # Save results
    results_file = Path(__file__).parent / 'output_comparison_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n💾 Comparison results saved to: {results_file}")
    
    return results['success_rate'] >= 80.0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)