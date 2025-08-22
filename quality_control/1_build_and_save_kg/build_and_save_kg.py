#!/usr/bin/env python3
"""Alias for build_complete_kg.py for naming consistency"""
import sys
import os
sys.path.append(os.path.dirname(__file__))
from build_complete_kg import main

if __name__ == "__main__":
    sys.exit(main())