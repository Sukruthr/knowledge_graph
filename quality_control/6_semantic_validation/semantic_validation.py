#!/usr/bin/env python3
"""
Phase 6: Semantic Validation

This script uses the comprehensive semantic validation framework
to ensure biological logic and semantic correctness.
"""
import sys
import os

# Import the comprehensive semantic validator
sys.path.append(os.path.dirname(__file__))
from comprehensive_semantic_validation import main

if __name__ == "__main__":
    sys.exit(main())