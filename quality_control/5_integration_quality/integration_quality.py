#!/usr/bin/env python3
"""Alias for integration_quality_validation.py for naming consistency"""
import sys
import os
sys.path.append(os.path.dirname(__file__))
from integration_quality_validation import main

if __name__ == "__main__":
    sys.exit(main())