#!/usr/bin/env python3
"""
Comprehensive Code Migration Verification

This script performs detailed verification to ensure all code and functions
from the original data_parsers.py have been properly migrated to the new
reorganized parser structure.
"""

import ast
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CodeVerifier:
    """Comprehensive code migration verifier."""
    
    def __init__(self):
        self.original_file = "src/data_parsers.py.backup"
        self.new_files = {
            "core_parsers": "src/parsers/core_parsers.py",
            "orchestrator": "src/parsers/parser_orchestrator.py",
            "utils": "src/parsers/parser_utils.py"
        }
        
        self.verification_results = {
            'classes': {'found': 0, 'missing': 0, 'details': {}},
            'methods': {'found': 0, 'missing': 0, 'details': {}},
            'functions': {'found': 0, 'missing': 0, 'details': {}},
            'imports': {'found': 0, 'missing': 0, 'details': {}},
            'constants': {'found': 0, 'missing': 0, 'details': {}},
            'overall': {'success': False, 'score': 0.0}
        }

    def parse_python_file(self, file_path: str) -> Dict:
        """Parse a Python file and extract all code elements."""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return {}
        
        extractor = CodeElementExtractor()
        extractor.visit(tree)
        
        return {
            'classes': extractor.classes,
            'methods': extractor.methods,
            'functions': extractor.functions,
            'imports': extractor.imports,
            'constants': extractor.constants,
            'docstrings': extractor.docstrings
        }

    def verify_class_migration(self, original_classes: Dict, new_files_data: Dict) -> Dict:
        """Verify that all classes have been migrated correctly."""
        logger.info("Verifying class migration...")
        
        results = {'found': {}, 'missing': [], 'partial': {}}
        
        for class_name, class_info in original_classes.items():
            found = False
            found_in = []
            
            # Check each new file
            for file_name, file_data in new_files_data.items():
                if class_name in file_data.get('classes', {}):
                    found = True
                    found_in.append(file_name)
                    
                    # Compare class methods
                    original_methods = set(class_info.get('methods', []))
                    new_methods = set(file_data['classes'][class_name].get('methods', []))
                    
                    missing_methods = original_methods - new_methods
                    extra_methods = new_methods - original_methods
                    
                    results['found'][class_name] = {
                        'file': file_name,
                        'methods_match': len(missing_methods) == 0,
                        'missing_methods': list(missing_methods),
                        'extra_methods': list(extra_methods),
                        'original_method_count': len(original_methods),
                        'new_method_count': len(new_methods)
                    }
            
            if not found:
                results['missing'].append(class_name)
            elif len(found_in) > 1:
                logger.warning(f"Class {class_name} found in multiple files: {found_in}")
        
        return results

    def verify_method_migration(self, original_methods: Dict, new_files_data: Dict) -> Dict:
        """Verify that all methods have been migrated correctly."""
        logger.info("Verifying method migration...")
        
        results = {'found': {}, 'missing': [], 'signature_changes': {}}
        
        for method_key, method_info in original_methods.items():
            class_name, method_name = method_key.split('.', 1)
            found = False
            
            for file_name, file_data in new_files_data.items():
                if class_name in file_data.get('classes', {}):
                    class_methods = file_data['classes'][class_name].get('methods', [])
                    if method_name in class_methods:
                        found = True
                        results['found'][method_key] = file_name
                        break
            
            if not found:
                results['missing'].append(method_key)
        
        return results

    def verify_function_migration(self, original_functions: Dict, new_files_data: Dict) -> Dict:
        """Verify that all standalone functions have been migrated correctly."""
        logger.info("Verifying function migration...")
        
        results = {'found': {}, 'missing': []}
        
        for func_name, func_info in original_functions.items():
            found = False
            
            for file_name, file_data in new_files_data.items():
                if func_name in file_data.get('functions', {}):
                    found = True
                    results['found'][func_name] = file_name
                    break
            
            if not found:
                results['missing'].append(func_name)
        
        return results

    def verify_import_migration(self, original_imports: List, new_files_data: Dict) -> Dict:
        """Verify that necessary imports are preserved."""
        logger.info("Verifying import migration...")
        
        results = {'found': {}, 'missing': [], 'relocated': {}}
        
        # Combine all imports from new files
        all_new_imports = set()
        new_imports_by_file = {}
        
        for file_name, file_data in new_files_data.items():
            file_imports = set(file_data.get('imports', []))
            all_new_imports.update(file_imports)
            new_imports_by_file[file_name] = file_imports
        
        for import_stmt in original_imports:
            if import_stmt in all_new_imports:
                # Find which file has this import
                for file_name, file_imports in new_imports_by_file.items():
                    if import_stmt in file_imports:
                        results['found'][import_stmt] = file_name
                        break
            else:
                # Check if it's a variation (e.g., relative vs absolute import)
                found_variation = False
                for new_import in all_new_imports:
                    if self._imports_equivalent(import_stmt, new_import):
                        results['relocated'][import_stmt] = new_import
                        found_variation = True
                        break
                
                if not found_variation:
                    results['missing'].append(import_stmt)
        
        return results

    def _imports_equivalent(self, import1: str, import2: str) -> bool:
        """Check if two import statements are equivalent."""
        # Remove leading dots and normalize
        normalized1 = import1.lstrip('.').replace('from ', '').replace('import ', '')
        normalized2 = import2.lstrip('.').replace('from ', '').replace('import ', '')
        
        # Check if they reference the same module
        return normalized1.split()[0] == normalized2.split()[0]

    def count_lines_of_code(self, file_path: str) -> Dict:
        """Count lines of code, comments, and docstrings."""
        if not os.path.exists(file_path):
            return {'total': 0, 'code': 0, 'comments': 0, 'blank': 0}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        stats = {'total': len(lines), 'code': 0, 'comments': 0, 'blank': 0}
        
        in_docstring = False
        docstring_char = None
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                stats['blank'] += 1
            elif stripped.startswith('#'):
                stats['comments'] += 1
            elif '"""' in stripped or "'''" in stripped:
                # Handle docstrings (simplified)
                if not in_docstring:
                    in_docstring = True
                    docstring_char = '"""' if '"""' in stripped else "'''"
                else:
                    in_docstring = False
                stats['comments'] += 1
            elif in_docstring:
                stats['comments'] += 1
            else:
                stats['code'] += 1
        
        return stats

    def run_comprehensive_verification(self) -> Dict:
        """Run the complete verification process."""
        logger.info("🔍 Starting comprehensive code migration verification...")
        logger.info("=" * 80)
        
        # Parse original file
        logger.info("Parsing original data_parsers.py.backup...")
        original_data = self.parse_python_file(self.original_file)
        
        if not original_data:
            logger.error("Failed to parse original file")
            return self.verification_results
        
        # Parse new files
        logger.info("Parsing new parser files...")
        new_files_data = {}
        for file_type, file_path in self.new_files.items():
            logger.info(f"  Parsing {file_path}...")
            new_files_data[file_type] = self.parse_python_file(file_path)
        
        # Verify each component
        logger.info("\n" + "=" * 50)
        logger.info("VERIFICATION ANALYSIS")
        logger.info("=" * 50)
        
        # 1. Class verification
        class_results = self.verify_class_migration(
            original_data.get('classes', {}), 
            new_files_data
        )
        self.verification_results['classes'] = {
            'found': len(class_results['found']),
            'missing': len(class_results['missing']),
            'details': class_results
        }
        
        # 2. Method verification
        method_results = self.verify_method_migration(
            original_data.get('methods', {}), 
            new_files_data
        )
        self.verification_results['methods'] = {
            'found': len(method_results['found']),
            'missing': len(method_results['missing']),
            'details': method_results
        }
        
        # 3. Function verification
        function_results = self.verify_function_migration(
            original_data.get('functions', {}), 
            new_files_data
        )
        self.verification_results['functions'] = {
            'found': len(function_results['found']),
            'missing': len(function_results['missing']),
            'details': function_results
        }
        
        # 4. Import verification
        import_results = self.verify_import_migration(
            original_data.get('imports', []), 
            new_files_data
        )
        self.verification_results['imports'] = {
            'found': len(import_results['found']),
            'missing': len(import_results['missing']),
            'details': import_results
        }
        
        # 5. Line count comparison
        original_stats = self.count_lines_of_code(self.original_file)
        new_stats = {}
        total_new_lines = 0
        
        for file_type, file_path in self.new_files.items():
            stats = self.count_lines_of_code(file_path)
            new_stats[file_type] = stats
            total_new_lines += stats['code']
        
        # Calculate overall score
        total_original = (len(original_data.get('classes', {})) + 
                         len(original_data.get('methods', {})) + 
                         len(original_data.get('functions', {})))
        
        total_found = (self.verification_results['classes']['found'] + 
                      self.verification_results['methods']['found'] + 
                      self.verification_results['functions']['found'])
        
        if total_original > 0:
            score = (total_found / total_original) * 100
            self.verification_results['overall']['score'] = score
            self.verification_results['overall']['success'] = score >= 95.0
        
        # Store line count comparison
        self.verification_results['line_counts'] = {
            'original': original_stats,
            'new_files': new_stats,
            'total_new_code_lines': total_new_lines,
            'code_line_ratio': total_new_lines / max(original_stats['code'], 1)
        }
        
        return self.verification_results

    def generate_report(self, results: Dict) -> str:
        """Generate a detailed verification report."""
        report = []
        report.append("📋 COMPREHENSIVE CODE MIGRATION VERIFICATION REPORT")
        report.append("=" * 80)
        
        # Overall score
        score = results['overall']['score']
        status = "✅ PASSED" if results['overall']['success'] else "❌ FAILED"
        report.append(f"\nOVERALL VERIFICATION: {status}")
        report.append(f"Migration Score: {score:.1f}%")
        
        # Classes
        report.append(f"\n📦 CLASSES:")
        report.append(f"  Found: {results['classes']['found']}")
        report.append(f"  Missing: {results['classes']['missing']}")
        
        if results['classes']['details']['missing']:
            report.append(f"  ❌ Missing classes: {results['classes']['details']['missing']}")
        
        for class_name, details in results['classes']['details']['found'].items():
            status = "✅" if details['methods_match'] else "⚠️"
            report.append(f"  {status} {class_name} → {details['file']}")
            if details['missing_methods']:
                report.append(f"    Missing methods: {details['missing_methods']}")
        
        # Methods
        report.append(f"\n🔧 METHODS:")
        report.append(f"  Found: {results['methods']['found']}")
        report.append(f"  Missing: {results['methods']['missing']}")
        
        if results['methods']['details']['missing']:
            report.append(f"  ❌ Missing methods:")
            for method in results['methods']['details']['missing'][:10]:  # Show first 10
                report.append(f"    - {method}")
            if len(results['methods']['details']['missing']) > 10:
                report.append(f"    ... and {len(results['methods']['details']['missing']) - 10} more")
        
        # Functions
        report.append(f"\n⚙️ FUNCTIONS:")
        report.append(f"  Found: {results['functions']['found']}")
        report.append(f"  Missing: {results['functions']['missing']}")
        
        if results['functions']['details']['missing']:
            report.append(f"  ❌ Missing functions: {results['functions']['details']['missing']}")
        
        # Imports
        report.append(f"\n📥 IMPORTS:")
        report.append(f"  Found: {results['imports']['found']}")
        report.append(f"  Missing: {results['imports']['missing']}")
        
        if results['imports']['details']['missing']:
            report.append(f"  ❌ Missing imports:")
            for imp in results['imports']['details']['missing'][:5]:
                report.append(f"    - {imp}")
        
        # Line counts
        if 'line_counts' in results:
            lc = results['line_counts']
            report.append(f"\n📊 LINE COUNT ANALYSIS:")
            report.append(f"  Original code lines: {lc['original']['code']}")
            report.append(f"  New total code lines: {lc['total_new_code_lines']}")
            report.append(f"  Code preservation ratio: {lc['code_line_ratio']:.2f}")
            
            report.append(f"\n  New file breakdown:")
            for file_type, stats in lc['new_files'].items():
                report.append(f"    {file_type}: {stats['code']} lines")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS:")
        if score >= 95.0:
            report.append("  ✅ Migration appears complete and successful")
            report.append("  ✅ All major components have been preserved")
        else:
            report.append("  ⚠️ Migration needs attention - some components missing")
            report.append("  🔍 Review missing items listed above")
        
        return "\n".join(report)


class CodeElementExtractor(ast.NodeVisitor):
    """Extract code elements from AST."""
    
    def __init__(self):
        self.classes = {}
        self.methods = {}
        self.functions = {}
        self.imports = []
        self.constants = {}
        self.docstrings = []
        self.current_class = None

    def visit_ClassDef(self, node):
        """Visit class definitions."""
        self.current_class = node.name
        self.classes[node.name] = {
            'methods': [],
            'lineno': node.lineno,
            'docstring': ast.get_docstring(node)
        }
        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
        """Visit function definitions."""
        if self.current_class:
            # This is a method
            self.classes[self.current_class]['methods'].append(node.name)
            method_key = f"{self.current_class}.{node.name}"
            self.methods[method_key] = {
                'args': [arg.arg for arg in node.args.args],
                'lineno': node.lineno,
                'docstring': ast.get_docstring(node)
            }
        else:
            # This is a standalone function
            self.functions[node.name] = {
                'args': [arg.arg for arg in node.args.args],
                'lineno': node.lineno,
                'docstring': ast.get_docstring(node)
            }
        
        self.generic_visit(node)

    def visit_Import(self, node):
        """Visit import statements."""
        for alias in node.names:
            self.imports.append(f"import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Visit from...import statements."""
        module = node.module or ''
        for alias in node.names:
            self.imports.append(f"from {module} import {alias.name}")
        self.generic_visit(node)

    def visit_Assign(self, node):
        """Visit assignments (for constants)."""
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                self.constants[target.id] = node.lineno
        self.generic_visit(node)


def main():
    """Run comprehensive verification."""
    verifier = CodeVerifier()
    results = verifier.run_comprehensive_verification()
    
    # Generate and display report
    report = verifier.generate_report(results)
    print(report)
    
    # Save report to file
    with open('code_migration_verification_report.txt', 'w') as f:
        f.write(report)
    
    logger.info(f"\n📄 Detailed report saved to: code_migration_verification_report.txt")
    
    # Return success status
    return results['overall']['success']


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)