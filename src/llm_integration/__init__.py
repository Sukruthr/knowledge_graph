"""
LLM Integration Package for Knowledge Graph Q&A System

This package provides intelligent question-answering capabilities by combining
the comprehensive biomedical knowledge graph with Ollama LLM models.

Components:
    - KnowledgeGraphService: Interface to the pre-built knowledge graph
    - OllamaClient: Interface to local Ollama API
    - QueryPlanner: Natural language to KG query translation
    - ResponseSynthesizer: Combines KG results with LLM reasoning
    - QASystem: Main orchestrator for interactive Q&A
"""

from .kg_service import KnowledgeGraphService
from .ollama_client import OllamaClient  
from .query_planner import QueryPlanner
from .response_synthesizer import ResponseSynthesizer
from .qa_system import QASystem

__all__ = [
    'KnowledgeGraphService',
    'OllamaClient', 
    'QueryPlanner',
    'ResponseSynthesizer',
    'QASystem'
]

__version__ = "1.0.0"
__author__ = "Biomedical Knowledge Graph Team"