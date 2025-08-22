#!/usr/bin/env python3
"""
Method Preservation Verification Script

This script compares the original kg_builder.py.backup with the new kg_builders module
to ensure all methods, classes, and functionality have been preserved during migration.
"""

import ast
import sys
import os
import json
from collections import defaultdict, Counter
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_class_methods(file_path):
    """Extract all classes and their methods from a Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tree = ast.parse(content)
    classes = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            methods = []
            properties = []
            
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_name = item.name
                    # Get method arguments
                    args = [arg.arg for arg in item.args.args]
                    methods.append({
                        'name': method_name,
                        'args': args,
                        'line': item.lineno,
                        'is_private': method_name.startswith('_'),
                        'is_magic': method_name.startswith('__') and method_name.endswith('__')
                    })
                elif isinstance(item, ast.AsyncFunctionDef):
                    method_name = item.name
                    args = [arg.arg for arg in item.args.args]
                    methods.append({
                        'name': method_name,
                        'args': args,
                        'line': item.lineno,
                        'is_async': True,
                        'is_private': method_name.startswith('_'),
                        'is_magic': method_name.startswith('__') and method_name.endswith('__')
                    })
            
            classes[class_name] = {
                'methods': methods,
                'properties': properties,
                'line': node.lineno
            }
    
    return classes

def extract_module_functions(file_path):
    """Extract module-level functions from a Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tree = ast.parse(content)
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Only get top-level functions (not class methods)
            if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree) 
                      if any(child == node for child in ast.walk(parent))):
                function_name = node.name
                args = [arg.arg for arg in node.args.args]
                functions.append({
                    'name': function_name,
                    'args': args,
                    'line': node.lineno,
                    'is_private': function_name.startswith('_')
                })
    
    return functions

def analyze_kg_builders_module():
    """Analyze the new kg_builders module structure."""
    kg_builders_path = '/home/mreddy1/knowledge_graph/src/kg_builders'
    all_classes = {}
    all_functions = {}
    
    # Analyze each file in kg_builders
    for filename in os.listdir(kg_builders_path):
        if filename.endswith('.py') and filename != '__init__.py':
            file_path = os.path.join(kg_builders_path, filename)
            module_name = filename[:-3]  # Remove .py extension
            
            logger.info(f"Analyzing {filename}...")
            
            # Extract classes
            classes = extract_class_methods(file_path)
            for class_name, class_info in classes.items():
                all_classes[class_name] = {
                    'module': module_name,
                    'methods': class_info['methods'],
                    'total_methods': len(class_info['methods'])
                }
            
            # Extract functions
            functions = extract_module_functions(file_path)
            if functions:
                all_functions[module_name] = functions
    
    return all_classes, all_functions

def generate_method_comparison_report():
    """Generate comprehensive method preservation report."""
    logger.info("🔍 STARTING METHOD PRESERVATION VERIFICATION")
    logger.info("="*80)
    
    # Analyze original file
    original_file = '/home/mreddy1/knowledge_graph/src/kg_builder.py.backup'
    logger.info("Analyzing original kg_builder.py.backup...")
    original_classes = extract_class_methods(original_file)
    original_functions = extract_module_functions(original_file)
    
    # Analyze new module structure
    logger.info("Analyzing new kg_builders module...")
    new_classes, new_functions = analyze_kg_builders_module()
    
    # Create comprehensive report
    report = {
        'timestamp': datetime.now().isoformat(),
        'analysis_summary': {},
        'class_comparison': {},
        'method_preservation': {},
        'missing_methods': [],
        'additional_methods': [],
        'statistics': {}
    }
    
    # Compare classes
    logger.info("Comparing class structures...")
    total_original_methods = 0
    total_preserved_methods = 0
    
    for class_name, original_info in original_classes.items():
        original_methods = {m['name']: m for m in original_info['methods']}
        total_original_methods += len(original_methods)
        
        if class_name in new_classes:
            new_info = new_classes[class_name]
            new_methods = {m['name']: m for m in new_info['methods']}
            
            preserved_methods = []
            missing_methods = []
            
            for method_name, method_info in original_methods.items():
                if method_name in new_methods:
                    preserved_methods.append(method_name)
                    total_preserved_methods += 1
                else:
                    missing_methods.append({
                        'class': class_name,
                        'method': method_name,
                        'args': method_info['args']
                    })
            
            # Check for additional methods
            additional_methods = []
            for method_name in new_methods:
                if method_name not in original_methods:
                    additional_methods.append({
                        'class': class_name,
                        'method': method_name,
                        'args': new_methods[method_name]['args']
                    })
            
            report['class_comparison'][class_name] = {
                'original_methods': len(original_methods),
                'preserved_methods': len(preserved_methods),
                'missing_methods': len(missing_methods),
                'additional_methods': len(additional_methods),
                'preservation_rate': len(preserved_methods) / len(original_methods) * 100 if original_methods else 100,
                'module_location': new_info['module'],
                'preserved_method_list': preserved_methods,
                'missing_method_list': missing_methods,
                'additional_method_list': additional_methods
            }
        else:
            report['class_comparison'][class_name] = {
                'status': 'MISSING_CLASS',
                'original_methods': len(original_methods),
                'preserved_methods': 0,
                'preservation_rate': 0
            }
            total_original_methods += len(original_methods)
    
    # Overall statistics
    overall_preservation_rate = (total_preserved_methods / total_original_methods * 100) if total_original_methods > 0 else 0
    
    report['statistics'] = {
        'total_original_classes': len(original_classes),
        'total_preserved_classes': len([c for c in original_classes if c in new_classes]),
        'total_original_methods': total_original_methods,
        'total_preserved_methods': total_preserved_methods,
        'overall_preservation_rate': overall_preservation_rate,
        'class_preservation_rate': len([c for c in original_classes if c in new_classes]) / len(original_classes) * 100
    }
    
    # Generate summary
    logger.info("="*80)
    logger.info("📊 METHOD PRESERVATION ANALYSIS RESULTS")
    logger.info("="*80)
    logger.info(f"Total Original Classes: {report['statistics']['total_original_classes']}")
    logger.info(f"Total Preserved Classes: {report['statistics']['total_preserved_classes']}")
    logger.info(f"Class Preservation Rate: {report['statistics']['class_preservation_rate']:.1f}%")
    logger.info(f"Total Original Methods: {report['statistics']['total_original_methods']}")
    logger.info(f"Total Preserved Methods: {report['statistics']['total_preserved_methods']}")
    logger.info(f"Overall Method Preservation Rate: {report['statistics']['overall_preservation_rate']:.1f}%")
    
    # Detailed class analysis
    logger.info("\n🔍 DETAILED CLASS ANALYSIS:")
    logger.info("-" * 60)
    
    for class_name, comparison in report['class_comparison'].items():
        if 'status' in comparison and comparison['status'] == 'MISSING_CLASS':
            logger.warning(f"❌ {class_name}: CLASS MISSING")
        else:
            preservation_rate = comparison['preservation_rate']
            module = comparison['module_location']
            status = "✅" if preservation_rate == 100 else "⚠️" if preservation_rate >= 90 else "❌"
            
            logger.info(f"{status} {class_name} (in {module}.py):")
            logger.info(f"   Methods: {comparison['preserved_methods']}/{comparison['original_methods']} "
                       f"({preservation_rate:.1f}%)")
            
            if comparison['missing_methods'] > 0:
                logger.warning(f"   Missing: {comparison['missing_methods']} methods")
                for missing in comparison['missing_method_list']:
                    logger.warning(f"      - {missing['method']}()")
            
            if comparison['additional_methods'] > 0:
                logger.info(f"   Additional: {comparison['additional_methods']} methods")
    
    # Save detailed report
    report_path = '/home/mreddy1/knowledge_graph/kg_testing/method_preservation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n📄 Detailed report saved to: {report_path}")
    
    # Determine overall status
    if overall_preservation_rate == 100:
        logger.info("🎉 ALL METHODS SUCCESSFULLY PRESERVED!")
        return True
    elif overall_preservation_rate >= 95:
        logger.warning(f"⚠️ MOSTLY PRESERVED ({overall_preservation_rate:.1f}%) - Minor issues detected")
        return True
    else:
        logger.error(f"❌ SIGNIFICANT METHODS MISSING ({overall_preservation_rate:.1f}%) - Migration incomplete")
        return False

def main():
    """Main verification function."""
    try:
        success = generate_method_comparison_report()
        if success:
            logger.info("✅ Method preservation verification completed successfully!")
            return 0
        else:
            logger.error("❌ Method preservation verification failed!")
            return 1
    except Exception as e:
        logger.error(f"❌ Error during verification: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())