"""
Knowledge Graph Builder - DEPRECATED

⚠️ DEPRECATION NOTICE ⚠️
This module has been refactored into smaller, more maintainable components.

Please update your imports:
    OLD: from kg_builder import ComprehensiveBiomedicalKnowledgeGraph
    NEW: from kg_builders import ComprehensiveBiomedicalKnowledgeGraph

The original 3,783-line file has been split into focused modules:
    - kg_builders/go_knowledge_graph.py (GOKnowledgeGraph)
    - kg_builders/combined_go_graph.py (CombinedGOKnowledgeGraph)  
    - kg_builders/comprehensive_graph.py (ComprehensiveBiomedicalKnowledgeGraph)
    - kg_builders/shared_utils.py (Common utilities)

This file will be removed in a future version.
"""

import warnings

# Issue deprecation warning when this module is imported
warnings.warn(
    "kg_builder.py is deprecated and will be removed in a future version. "
    "Please use 'from kg_builders import ...' instead. "
    "See kg_builders/__init__.py for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

# Import all classes from the new modular structure for backward compatibility
try:
    from .kg_builders import (
        GOKnowledgeGraph,
        CombinedGOKnowledgeGraph,
        ComprehensiveBiomedicalKnowledgeGraph
    )
except ImportError:
    # Fallback for direct execution
    from kg_builders import (
        GOKnowledgeGraph,
        CombinedGOKnowledgeGraph,
        ComprehensiveBiomedicalKnowledgeGraph
    )

# Maintain all original exports
__all__ = [
    'GOKnowledgeGraph',
    'CombinedGOKnowledgeGraph',
    'ComprehensiveBiomedicalKnowledgeGraph'
]

# Migration helper
def migrate_imports():
    """
    Print migration guide for updating imports.
    """
    print("=" * 80)
    print("KG_BUILDER MIGRATION GUIDE")
    print("=" * 80)
    print()
    print("The kg_builder.py file has been refactored for better maintainability.")
    print("Please update your imports as follows:")
    print()
    print("OLD IMPORTS:")
    print("    from kg_builder import GOKnowledgeGraph")
    print("    from kg_builder import CombinedGOKnowledgeGraph") 
    print("    from kg_builder import ComprehensiveBiomedicalKnowledgeGraph")
    print()
    print("NEW IMPORTS:")
    print("    from kg_builders import GOKnowledgeGraph")
    print("    from kg_builders import CombinedGOKnowledgeGraph")
    print("    from kg_builders import ComprehensiveBiomedicalKnowledgeGraph")
    print()
    print("DIRECT MODULE IMPORTS (optional):")
    print("    from kg_builders.go_knowledge_graph import GOKnowledgeGraph")
    print("    from kg_builders.combined_go_graph import CombinedGOKnowledgeGraph")
    print("    from kg_builders.comprehensive_graph import ComprehensiveBiomedicalKnowledgeGraph")
    print()
    print("All functionality remains identical - only the import paths have changed.")
    print("=" * 80)

if __name__ == "__main__":
    migrate_imports()