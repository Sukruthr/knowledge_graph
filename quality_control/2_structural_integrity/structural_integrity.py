#!/usr/bin/env python3
"""Alias for structural_validation.py for naming consistency"""
import sys
import os
sys.path.append(os.path.dirname(__file__))
from structural_validation import main

if __name__ == "__main__":
    sys.exit(main())