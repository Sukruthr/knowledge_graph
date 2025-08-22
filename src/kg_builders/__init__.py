"""
Knowledge Graph Builders Package

This package provides modular knowledge graph builders for biomedical data.
All classes maintain backward compatibility with the original kg_builder.py.

Classes:
    GOKnowledgeGraph: Generic GO knowledge graph (supports GO_BP, GO_CC, GO_MF)
    CombinedGOKnowledgeGraph: Multi-namespace GO knowledge graph  
    ComprehensiveBiomedicalKnowledgeGraph: Full biomedical knowledge graph with Omics integration

Usage:
    # Original import style still works
    from kg_builder import ComprehensiveBiomedicalKnowledgeGraph
    
    # New modular import style
    from kg_builders import ComprehensiveBiomedicalKnowledgeGraph
    from kg_builders.comprehensive_graph import ComprehensiveBiomedicalKnowledgeGraph
"""

# Import all classes for backward compatibility
from .go_knowledge_graph import GOKnowledgeGraph
from .combined_go_graph import CombinedGOKnowledgeGraph
from .comprehensive_graph import ComprehensiveBiomedicalKnowledgeGraph

# Make all classes available at package level
__all__ = [
    'GOKnowledgeGraph',
    'CombinedGOKnowledgeGraph', 
    'ComprehensiveBiomedicalKnowledgeGraph'
]

# Version info
__version__ = "2.0.0"
__author__ = "Biomedical Knowledge Graph Team"