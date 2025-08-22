#!/usr/bin/env python3
"""
Backward Compatibility Testing Suite for kg_builders

This script tests that all original imports and functionality from kg_builder.py
continue to work with the new modular kg_builders structure.
"""

import sys
import os
import warnings
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, '/home/mreddy1/knowledge_graph/src')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BackwardCompatibilityTestSuite:
    """Test suite for backward compatibility."""
    
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = {}
        
    def run_test(self, test_name, test_function):
        """Run a single test and record results."""
        try:
            logger.info(f"Running test: {test_name}")
            result = test_function()
            if result:
                self.passed_tests += 1
                self.test_results[test_name] = {'status': 'PASSED', 'details': result}
                logger.info(f"✅ {test_name} - PASSED")
            else:
                self.failed_tests += 1
                self.test_results[test_name] = {'status': 'FAILED', 'details': 'Test returned False'}
                logger.error(f"❌ {test_name} - FAILED")
        except Exception as e:
            self.failed_tests += 1
            self.test_results[test_name] = {'status': 'ERROR', 'details': str(e)}
            logger.error(f"❌ {test_name} - ERROR: {e}")
    
    def test_old_import_style_with_warnings(self):
        """Test that old import style works but shows deprecation warning."""
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # This should work but show warning
            from kg_builder import ComprehensiveBiomedicalKnowledgeGraph
            
            # Check that warning was issued
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0, "Should issue deprecation warning"
            
            warning_message = str(deprecation_warnings[0].message)
            assert "deprecated" in warning_message.lower(), "Warning should mention deprecation"
            assert "kg_builders" in warning_message, "Warning should mention new import path"
            
            # Test that the class works
            kg = ComprehensiveBiomedicalKnowledgeGraph()
            assert kg is not None, "Should create instance successfully"
            assert hasattr(kg, 'graph'), "Should have graph attribute"
        
        return f"Old import works with {len(deprecation_warnings)} deprecation warning(s)"
    
    def test_old_import_all_classes(self):
        """Test that all original classes can be imported from kg_builder."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            
            # Test all three original classes
            from kg_builder import GOKnowledgeGraph
            from kg_builder import CombinedGOKnowledgeGraph
            from kg_builder import ComprehensiveBiomedicalKnowledgeGraph
            
            # Test instantiation
            go_kg = GOKnowledgeGraph()
            combined_kg = CombinedGOKnowledgeGraph()
            comprehensive_kg = ComprehensiveBiomedicalKnowledgeGraph()
            
            assert all([go_kg, combined_kg, comprehensive_kg]), "All classes should instantiate"
            
        return "All original classes imported and instantiated successfully"
    
    def test_new_import_style(self):
        """Test that new import style works without warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # This should work without warnings
            from kg_builders import ComprehensiveBiomedicalKnowledgeGraph
            from kg_builders import GOKnowledgeGraph
            from kg_builders import CombinedGOKnowledgeGraph
            
            # Check that no deprecation warnings were issued
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0, f"Should not issue deprecation warnings, got: {[str(w.message) for w in deprecation_warnings]}"
            
            # Test that classes work
            kg = ComprehensiveBiomedicalKnowledgeGraph()
            go_kg = GOKnowledgeGraph()
            combined_kg = CombinedGOKnowledgeGraph()
            
            assert all([kg, go_kg, combined_kg]), "All classes should instantiate"
        
        return "New import style works without warnings"
    
    def test_direct_module_imports(self):
        """Test that direct module imports work."""
        # Test direct imports
        from kg_builders.go_knowledge_graph import GOKnowledgeGraph
        from kg_builders.combined_go_graph import CombinedGOKnowledgeGraph
        from kg_builders.comprehensive_graph import ComprehensiveBiomedicalKnowledgeGraph
        from kg_builders.shared_utils import save_graph_to_file, load_graph_from_file
        
        # Test instantiation
        go_kg = GOKnowledgeGraph()
        combined_kg = CombinedGOKnowledgeGraph()
        comprehensive_kg = ComprehensiveBiomedicalKnowledgeGraph()
        
        assert all([go_kg, combined_kg, comprehensive_kg]), "All classes should instantiate"
        assert callable(save_graph_to_file), "Utility functions should be callable"
        assert callable(load_graph_from_file), "Utility functions should be callable"
        
        return "Direct module imports work correctly"
    
    def test_migration_helper(self):
        """Test that migration helper function works."""
        import io
        import contextlib
        
        # Capture output from migration helper
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            import kg_builder
            kg_builder.migrate_imports()
        
        output = f.getvalue()
        
        # Check that migration guide was printed
        assert "MIGRATION GUIDE" in output, "Should print migration guide"
        assert "OLD IMPORTS" in output, "Should show old import examples"
        assert "NEW IMPORTS" in output, "Should show new import examples"
        assert "kg_builders" in output, "Should mention new package name"
        
        return "Migration helper function works and provides guidance"
    
    def test_functionality_equivalence(self):
        """Test that old and new imports provide equivalent functionality."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            
            # Import using old style
            from kg_builder import ComprehensiveBiomedicalKnowledgeGraph as OldKG
            
        # Import using new style
        from kg_builders import ComprehensiveBiomedicalKnowledgeGraph as NewKG
        
        # Create instances
        old_kg = OldKG()
        new_kg = NewKG()
        
        # Test that they have the same type
        assert type(old_kg) == type(new_kg), "Should be the same class type"
        
        # Test that they have the same methods
        old_methods = set(dir(old_kg))
        new_methods = set(dir(new_kg))
        
        # Should have identical methods
        assert old_methods == new_methods, f"Method sets should be identical. Missing in new: {old_methods - new_methods}, Extra in new: {new_methods - old_methods}"
        
        # Test some key methods exist
        key_methods = [
            'load_data', 'build_comprehensive_graph', 'get_comprehensive_stats',
            'query_gene_comprehensive', 'save_comprehensive_graph'
        ]
        
        for method in key_methods:
            assert hasattr(old_kg, method), f"Old KG should have {method}"
            assert hasattr(new_kg, method), f"New KG should have {method}"
            assert callable(getattr(old_kg, method)), f"Old KG {method} should be callable"
            assert callable(getattr(new_kg, method)), f"New KG {method} should be callable"
        
        return f"Old and new imports provide equivalent functionality with {len(key_methods)} key methods verified"
    
    def test_package_structure(self):
        """Test that package structure is correct."""
        # Test that kg_builders package exists and has expected modules
        import kg_builders
        
        # Check package attributes
        expected_classes = ['GOKnowledgeGraph', 'CombinedGOKnowledgeGraph', 'ComprehensiveBiomedicalKnowledgeGraph']
        for class_name in expected_classes:
            assert hasattr(kg_builders, class_name), f"kg_builders should have {class_name}"
            assert callable(getattr(kg_builders, class_name)), f"{class_name} should be callable"
        
        # Check that __all__ is defined
        assert hasattr(kg_builders, '__all__'), "kg_builders should have __all__ defined"
        assert set(kg_builders.__all__) == set(expected_classes), "__all__ should contain all expected classes"
        
        # Check version info
        assert hasattr(kg_builders, '__version__'), "Should have version info"
        
        return f"Package structure verified with {len(expected_classes)} classes and version {kg_builders.__version__}"
    
    def test_original_kg_builder_file(self):
        """Test that original kg_builder.py file works as expected."""
        # Test that kg_builder.py exists and is importable
        import kg_builder
        
        # Check that it has the expected classes
        expected_classes = ['GOKnowledgeGraph', 'CombinedGOKnowledgeGraph', 'ComprehensiveBiomedicalKnowledgeGraph']
        for class_name in expected_classes:
            assert hasattr(kg_builder, class_name), f"kg_builder should have {class_name}"
        
        # Check that __all__ is defined
        assert hasattr(kg_builder, '__all__'), "kg_builder should have __all__ defined"
        assert set(kg_builder.__all__) == set(expected_classes), "__all__ should contain all expected classes"
        
        # Check that migration helper exists
        assert hasattr(kg_builder, 'migrate_imports'), "Should have migrate_imports function"
        assert callable(kg_builder.migrate_imports), "migrate_imports should be callable"
        
        return "Original kg_builder.py file maintains expected interface"
    
    def run_all_tests(self):
        """Run all backward compatibility tests."""
        logger.info("=" * 80)
        logger.info("🧪 BACKWARD COMPATIBILITY COMPREHENSIVE TEST SUITE")
        logger.info("=" * 80)
        
        # Define all tests
        tests = [
            ("Old Import Style With Warnings", self.test_old_import_style_with_warnings),
            ("Old Import All Classes", self.test_old_import_all_classes),
            ("New Import Style", self.test_new_import_style),
            ("Direct Module Imports", self.test_direct_module_imports),
            ("Migration Helper", self.test_migration_helper),
            ("Functionality Equivalence", self.test_functionality_equivalence),
            ("Package Structure", self.test_package_structure),
            ("Original kg_builder File", self.test_original_kg_builder_file)
        ]
        
        # Run all tests
        for test_name, test_function in tests:
            self.run_test(test_name, test_function)
        
        # Generate summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 BACKWARD COMPATIBILITY TEST RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Tests: {self.passed_tests + self.failed_tests}")
        logger.info(f"Passed: {self.passed_tests}")
        logger.info(f"Failed: {self.failed_tests}")
        
        if self.passed_tests + self.failed_tests > 0:
            success_rate = (self.passed_tests / (self.passed_tests + self.failed_tests)) * 100
            logger.info(f"Success Rate: {success_rate:.1f}%")
        
        if self.failed_tests > 0:
            logger.info("\n❌ FAILED TESTS:")
            for test_name, result in self.test_results.items():
                if result['status'] in ['FAILED', 'ERROR']:
                    logger.info(f"   {test_name}: {result['details']}")
        
        logger.info("=" * 80)
        
        return self.failed_tests == 0


def main():
    """Main execution function."""
    test_suite = BackwardCompatibilityTestSuite()
    success = test_suite.run_all_tests()
    
    # Save results
    results_file = '/home/mreddy1/knowledge_graph/kg_testing/backward_compatibility_test_results.json'
    import json
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': str(Path(__file__).stat().st_mtime),
            'total_tests': test_suite.passed_tests + test_suite.failed_tests,
            'passed': test_suite.passed_tests,
            'failed': test_suite.failed_tests,
            'success_rate': (test_suite.passed_tests / (test_suite.passed_tests + test_suite.failed_tests)) * 100 if (test_suite.passed_tests + test_suite.failed_tests) > 0 else 0,
            'results': test_suite.test_results
        }, f, indent=2)
    
    logger.info(f"📄 Detailed results saved to: {results_file}")
    
    if success:
        print("\n🎉 All backward compatibility tests passed!")
        return True
    else:
        print("\n⚠️ Some backward compatibility tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)