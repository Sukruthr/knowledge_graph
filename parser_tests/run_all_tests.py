#!/usr/bin/env python3
"""
Master Test Runner for Comprehensive Parser Testing

Runs all parser tests and generates a comprehensive summary report.
"""

import sys
import os
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MasterTestRunner:
    """Master test runner for all parser tests."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.results = {}
        self.test_scripts = [
            'test_parser_utils.py',
            'test_go_data_parser.py', 
            'test_omics_data_parser.py',
            'test_remaining_parsers.py'
        ]
        
    def run_single_test(self, test_script):
        """Run a single test script and capture results."""
        logger.info(f"🧪 Running {test_script}...")
        
        try:
            # Run the test script
            result = subprocess.run(
                [sys.executable, str(self.test_dir / test_script)],
                capture_output=True,
                text=True,
                cwd=str(self.test_dir.parent),  # Run from project root
                timeout=300  # 5 minute timeout
            )
            
            # Load results from JSON file if available
            json_file = test_script.replace('.py', '_test_results.json')
            json_path = self.test_dir / json_file
            
            if json_path.exists():
                with open(json_path, 'r') as f:
                    test_data = json.load(f)
                    
                success = result.returncode == 0
                self.results[test_script] = {
                    'success': success,
                    'return_code': result.returncode,
                    'detailed_results': test_data,
                    'stdout_lines': len(result.stdout.split('\n')),
                    'stderr_lines': len(result.stderr.split('\n')) if result.stderr else 0
                }
                
                logger.info(f"  ✅ {test_script}: {'PASSED' if success else 'FAILED'}")
                logger.info(f"    📊 Tests: {test_data.get('total_tests', 0)}, "
                           f"Passed: {test_data.get('passed', 0)}, "
                           f"Failed: {test_data.get('failed', 0)}, "
                           f"Success Rate: {test_data.get('success_rate', 0):.1f}%")
            else:
                self.results[test_script] = {
                    'success': result.returncode == 0,
                    'return_code': result.returncode,
                    'error': 'No JSON results file found',
                    'stdout_lines': len(result.stdout.split('\n')),
                    'stderr_lines': len(result.stderr.split('\n')) if result.stderr else 0
                }
                
            # Log any errors
            if result.stderr and result.stderr.strip():
                logger.warning(f"  ⚠️ {test_script} stderr: {result.stderr.strip()[:200]}...")
                
        except subprocess.TimeoutExpired:
            logger.error(f"  ❌ {test_script}: Timeout after 5 minutes")
            self.results[test_script] = {
                'success': False,
                'error': 'Timeout after 5 minutes',
                'return_code': -1
            }
            
        except Exception as e:
            logger.error(f"  ❌ {test_script}: Exception - {str(e)}")
            self.results[test_script] = {
                'success': False,
                'error': str(e),
                'return_code': -1
            }

    def run_all_tests(self):
        """Run all test scripts."""
        logger.info("=" * 80)
        logger.info("🚀 STARTING COMPREHENSIVE PARSER TEST SUITE")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        for test_script in self.test_scripts:
            self.run_single_test(test_script)
            
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"\n⏱️ Total test runtime: {duration:.1f} seconds")
        
        return self.generate_summary_report()

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        logger.info("\n" + "=" * 80)
        logger.info("📊 COMPREHENSIVE PARSER TEST SUMMARY")
        logger.info("=" * 80)
        
        # Calculate overall statistics
        total_tests = 0
        total_passed = 0
        total_failed = 0
        successful_scripts = 0
        failed_scripts = 0
        
        for script, result in self.results.items():
            if result['success']:
                successful_scripts += 1
            else:
                failed_scripts += 1
                
            if 'detailed_results' in result:
                detailed = result['detailed_results']
                total_tests += detailed.get('total_tests', 0)
                total_passed += detailed.get('passed', 0)
                total_failed += detailed.get('failed', 0)
        
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Print summary
        logger.info(f"📋 Test Scripts Summary:")
        logger.info(f"   Total Scripts: {len(self.test_scripts)}")
        logger.info(f"   Successful: {successful_scripts}")
        logger.info(f"   Failed: {failed_scripts}")
        logger.info(f"   Script Success Rate: {successful_scripts / len(self.test_scripts) * 100:.1f}%")
        
        logger.info(f"\n📊 Individual Test Cases Summary:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Passed: {total_passed}")
        logger.info(f"   Failed: {total_failed}")
        logger.info(f"   Overall Success Rate: {overall_success_rate:.1f}%")
        
        # Detailed breakdown
        logger.info(f"\n📋 Detailed Breakdown:")
        for script, result in self.results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"  {status} {script}")
            
            if 'detailed_results' in result:
                detailed = result['detailed_results']
                logger.info(f"    Tests: {detailed.get('total_tests', 0)}, "
                           f"Passed: {detailed.get('passed', 0)}, "
                           f"Failed: {detailed.get('failed', 0)}, "
                           f"Rate: {detailed.get('success_rate', 0):.1f}%")
            elif 'error' in result:
                logger.info(f"    Error: {result['error']}")
        
        # Final assessment
        if overall_success_rate >= 90.0:
            final_status = f"🎉 EXCELLENT - {overall_success_rate:.1f}% SUCCESS RATE"
        elif overall_success_rate >= 80.0:
            final_status = f"✅ GOOD - {overall_success_rate:.1f}% SUCCESS RATE"
        elif overall_success_rate >= 70.0:
            final_status = f"⚠️ ACCEPTABLE - {overall_success_rate:.1f}% SUCCESS RATE"
        else:
            final_status = f"❌ NEEDS ATTENTION - {overall_success_rate:.1f}% SUCCESS RATE"
            
        logger.info(f"\n🏆 Final Assessment: {final_status}")
        logger.info("=" * 80)
        
        # Save comprehensive results
        summary_report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_scripts': len(self.test_scripts),
                'successful_scripts': successful_scripts,
                'failed_scripts': failed_scripts,
                'script_success_rate': successful_scripts / len(self.test_scripts) * 100,
                'total_individual_tests': total_tests,
                'total_passed': total_passed,
                'total_failed': total_failed,
                'overall_success_rate': overall_success_rate,
                'final_status': final_status
            },
            'detailed_results': self.results
        }
        
        # Save to file
        summary_file = self.test_dir / 'comprehensive_test_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        logger.info(f"💾 Comprehensive test summary saved to: {summary_file}")
        
        return summary_report


def main():
    """Main execution function."""
    runner = MasterTestRunner()
    summary = runner.run_all_tests()
    
    # Return success based on overall performance
    overall_success_rate = summary['summary']['overall_success_rate']
    return overall_success_rate >= 80.0  # 80% threshold for success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)