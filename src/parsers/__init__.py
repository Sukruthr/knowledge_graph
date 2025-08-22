"""
Biomedical Data Parsers Package

This package contains all parsers for the comprehensive biomedical knowledge graph system.
Organized into core parsers, specialized parsers, and orchestration components.
"""

# Import common utilities
from .parser_utils import ParserUtils

# Core parser classes
from .core_parsers import GODataParser, OmicsDataParser, CombinedGOParser

# Orchestrator
from .parser_orchestrator import CombinedBiomedicalParser

# Specialized parsers (with error handling for optional imports)
try:
    from .model_compare_parser import ModelCompareParser
except ImportError:
    ModelCompareParser = None

try:
    from .cc_mf_branch_parser import CCMFBranchParser
except ImportError:
    CCMFBranchParser = None

try:
    from .llm_processed_parser import LLMProcessedParser
except ImportError:
    LLMProcessedParser = None

try:
    from .go_analysis_data_parser import GOAnalysisDataParser
except ImportError:
    GOAnalysisDataParser = None

try:
    from .remaining_data_parser import RemainingDataParser
except ImportError:
    RemainingDataParser = None

try:
    from .talisman_gene_sets_parser import TalismanGeneSetsParser
except ImportError:
    TalismanGeneSetsParser = None

# Backward compatibility aliases
GOBPDataParser = GODataParser

__all__ = [
    'ParserUtils',
    'GODataParser',
    'OmicsDataParser',
    'CombinedGOParser',
    'CombinedBiomedicalParser',
    'GOBPDataParser',  # Backward compatibility
    'ModelCompareParser',
    'CCMFBranchParser',
    'LLMProcessedParser',
    'GOAnalysisDataParser',
    'RemainingDataParser',
    'TalismanGeneSetsParser'
]

__version__ = "1.0.0"
__author__ = "Biomedical Knowledge Graph Team"