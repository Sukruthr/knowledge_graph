#!/usr/bin/env python3
"""
Phase 8: Regression Testing

This script performs backward compatibility validation and regression testing
to ensure system stability and compatibility.
"""

import sys
import os
import time
import logging
import json
from datetime import datetime

# Add project root to path
sys.path.append('/home/mreddy1/knowledge_graph/src')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RegressionTester:
    """Regression testing for biomedical knowledge graph."""
    
    def __init__(self):
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'validation_summary': {},
            'quality_metrics': {},
            'recommendations': []
        }
    
    def test_import_compatibility(self):
        """Test import compatibility."""
        logger.info("📦 TESTING IMPORT COMPATIBILITY")
        
        try:
            compatibility_results = {
                'old_style_imports': [],
                'new_style_imports': [],
                'import_errors': []
            }
            
            # Test old-style imports (should work with deprecation warnings)
            try:
                from kg_builder import ComprehensiveBiomedicalKnowledgeGraph as OldKG
                compatibility_results['old_style_imports'].append('kg_builder import successful')
                logger.info("✅ Old-style import works")
            except Exception as e:
                compatibility_results['import_errors'].append(f'Old-style import failed: {str(e)}')
                logger.warning(f"⚠️ Old-style import failed: {str(e)}")
            
            # Test new-style imports
            try:
                from kg_builders import ComprehensiveBiomedicalKnowledgeGraph as NewKG
                compatibility_results['new_style_imports'].append('kg_builders import successful')
                logger.info("✅ New-style import works")
            except Exception as e:
                compatibility_results['import_errors'].append(f'New-style import failed: {str(e)}')
                logger.error(f"❌ New-style import failed: {str(e)}")
            
            # Test method compatibility
            try:
                kg = NewKG()
                methods = [method for method in dir(kg) if not method.startswith('_') and callable(getattr(kg, method))]
                compatibility_results['available_methods'] = len(methods)
                logger.info(f"✅ {len(methods)} methods available")
            except Exception as e:
                compatibility_results['import_errors'].append(f'Method inspection failed: {str(e)}')
            
            compatibility_score = 100 - (len(compatibility_results['import_errors']) * 25)
            compatibility_results['compatibility_score'] = max(compatibility_score, 0)
            
            self.test_results['detailed_results'] = {'import_compatibility': compatibility_results}
            logger.info(f"Import compatibility score: {compatibility_results['compatibility_score']:.1f}%")
            
            return len(compatibility_results['import_errors']) == 0
            
        except Exception as e:
            logger.error(f"❌ Import compatibility test failed: {str(e)}")
            return False
    
    def test_method_preservation(self):
        """Test method preservation."""
        logger.info("🔍 TESTING METHOD PRESERVATION")
        
        try:
            from kg_builders import ComprehensiveBiomedicalKnowledgeGraph
            kg = ComprehensiveBiomedicalKnowledgeGraph()
            
            # Check for key methods
            key_methods = [
                'load_data', 'build_comprehensive_graph', 'get_comprehensive_stats',
                'query_gene_comprehensive', 'query_go_term'
            ]
            
            preserved_methods = []
            missing_methods = []
            
            for method_name in key_methods:
                if hasattr(kg, method_name) and callable(getattr(kg, method_name)):
                    preserved_methods.append(method_name)
                else:
                    missing_methods.append(method_name)
            
            preservation_rate = (len(preserved_methods) / len(key_methods)) * 100
            
            method_results = {
                'total_key_methods': len(key_methods),
                'preserved_methods': len(preserved_methods),
                'missing_methods': missing_methods,
                'preservation_rate': preservation_rate
            }
            
            self.test_results['detailed_results']['method_preservation'] = method_results
            logger.info(f"Method preservation rate: {preservation_rate:.1f}%")
            
            return preservation_rate >= 95
            
        except Exception as e:
            logger.error(f"❌ Method preservation test failed: {str(e)}")
            return False
    
    def generate_quality_metrics(self):
        """Generate regression testing quality metrics."""
        logger.info("📊 GENERATING REGRESSION QUALITY METRICS")
        
        try:
            metrics = {}
            
            # Import compatibility score
            import_results = self.test_results['detailed_results'].get('import_compatibility', {})
            import_score = import_results.get('compatibility_score', 0)
            metrics['import_compatibility_score'] = import_score
            
            # Method preservation score
            method_results = self.test_results['detailed_results'].get('method_preservation', {})
            preservation_score = method_results.get('preservation_rate', 0)
            metrics['method_preservation_score'] = preservation_score
            
            # Overall regression score
            overall_regression = (import_score + preservation_score) / 2
            metrics['overall_regression_score'] = overall_regression
            
            # Regression grade
            if overall_regression >= 95:
                grade = 'A+'
            elif overall_regression >= 90:
                grade = 'A'
            elif overall_regression >= 85:
                grade = 'B+'
            elif overall_regression >= 80:
                grade = 'B'
            else:
                grade = 'C'
            
            metrics['regression_grade'] = grade
            
            self.test_results['quality_metrics'] = metrics
            
            logger.info(f"Regression Score: {overall_regression:.1f}/100 (Grade: {grade})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Quality metrics generation failed: {str(e)}")
            return False
    
    def save_results(self):
        """Save regression test results."""
        try:
            output_path = '/home/mreddy1/knowledge_graph/quality_control/8_regression_testing/regression_testing_results.json'
            with open(output_path, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            logger.info(f"📄 Results saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save results: {str(e)}")
            return False
    
    def run_regression_tests(self):
        """Run regression tests."""
        logger.info("🔄 REGRESSION TESTING")
        
        steps = [
            ('Test Import Compatibility', self.test_import_compatibility),
            ('Test Method Preservation', self.test_method_preservation),
            ('Generate Quality Metrics', self.generate_quality_metrics),
            ('Save Results', self.save_results)
        ]
        
        passed = 0
        for name, func in steps:
            logger.info(f"Executing: {name}")
            if func():
                passed += 1
                logger.info(f"✅ {name} completed")
            else:
                logger.error(f"❌ {name} failed")
        
        success_rate = (passed / len(steps)) * 100
        self.test_results['validation_summary'] = {
            'total_steps': len(steps),
            'passed_steps': passed,
            'success_rate': success_rate,
            'overall_status': 'PASSED' if passed == len(steps) else 'FAILED'
        }
        
        return success_rate == 100

def main():
    """Main execution function."""
    try:
        tester = RegressionTester()
        success = tester.run_regression_tests()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())