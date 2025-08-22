#!/usr/bin/env python3
"""
Detailed Method Verification

Compare original vs new method signatures and implementations.
"""

import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_method_signatures(file_path):
    """Extract method signatures from a Python file."""
    methods = {}
    current_class = None
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # Class definition
        if line.strip().startswith('class '):
            class_match = re.match(r'\s*class\s+(\w+)', line)
            if class_match:
                current_class = class_match.group(1)
        
        # Method definition
        if line.strip().startswith('def '):
            method_match = re.match(r'\s*def\s+(\w+)\s*\((.*?)\)', line)
            if method_match:
                method_name = method_match.group(1)
                method_args = method_match.group(2)
                
                if current_class:
                    full_name = f"{current_class}.{method_name}"
                else:
                    full_name = method_name
                
                methods[full_name] = {
                    'line': i + 1,
                    'signature': line.strip(),
                    'args': method_args
                }
    
    return methods

def main():
    """Compare method signatures between original and new files."""
    logger.info("🔍 DETAILED METHOD VERIFICATION")
    logger.info("=" * 60)
    
    # Extract methods from original file
    original_methods = extract_method_signatures("src/data_parsers.py.backup")
    
    # Extract methods from new files
    core_methods = extract_method_signatures("src/parsers/core_parsers.py")
    orchestrator_methods = extract_method_signatures("src/parsers/parser_orchestrator.py")
    
    # Combine new methods
    new_methods = {**core_methods, **orchestrator_methods}
    
    logger.info(f"Original methods: {len(original_methods)}")
    logger.info(f"New methods: {len(new_methods)}")
    
    # Check each original method
    missing_methods = []
    signature_changes = []
    found_methods = []
    
    for method_name, method_info in original_methods.items():
        if method_name == 'main':  # Skip standalone main function
            continue
            
        if method_name in new_methods:
            found_methods.append(method_name)
            
            # Compare signatures (simplified)
            orig_sig = method_info['signature']
            new_sig = new_methods[method_name]['signature']
            
            if orig_sig != new_sig:
                signature_changes.append({
                    'method': method_name,
                    'original': orig_sig,
                    'new': new_sig
                })
        else:
            missing_methods.append(method_name)
    
    # Results
    logger.info(f"\n✅ Found methods: {len(found_methods)}")
    logger.info(f"❌ Missing methods: {len(missing_methods)}")
    logger.info(f"⚠️ Signature changes: {len(signature_changes)}")
    
    if missing_methods:
        logger.info(f"\nMissing methods:")
        for method in missing_methods:
            logger.info(f"  - {method}")
    
    if signature_changes:
        logger.info(f"\nSignature changes:")
        for change in signature_changes[:5]:  # Show first 5
            logger.info(f"  - {change['method']}")
            logger.info(f"    Original: {change['original']}")
            logger.info(f"    New:      {change['new']}")
    
    # Calculate success rate
    total_methods = len(original_methods) - 1  # Exclude main function
    found_count = len(found_methods)
    success_rate = (found_count / total_methods) * 100 if total_methods > 0 else 0
    
    logger.info(f"\n📊 SUCCESS RATE: {success_rate:.1f}% ({found_count}/{total_methods})")
    
    if success_rate >= 100.0:
        logger.info("🎉 ALL METHODS SUCCESSFULLY MIGRATED!")
        return True
    else:
        logger.info("⚠️ Some methods missing - review needed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)