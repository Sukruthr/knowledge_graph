"""
Main Q&A System Orchestrator

This is the primary interface for the intelligent biomedical knowledge graph Q&A system.
It orchestrates all components to provide natural language query capabilities.
"""

import logging
import time
import json
import yaml
from typing import Dict, List, Optional, Any, Iterator
from pathlib import Path
from dataclasses import asdict
import argparse
import sys

from .kg_service import KnowledgeGraphService
from .ollama_client import OllamaClient, OllamaModelType
from .query_planner import QueryPlanner, QueryPlan
from .response_synthesizer import ResponseSynthesizer, SynthesizedResponse

logger = logging.getLogger(__name__)

class QASessionManager:
    """Manage conversation sessions and context."""
    
    def __init__(self, max_context_length: int = 10):
        self.max_context_length = max_context_length
        self.conversation_history = []
        self.session_metadata = {
            'session_id': str(int(time.time())),
            'start_time': time.time(),
            'questions_asked': 0,
            'avg_response_time': 0.0
        }
    
    def add_exchange(self, question: str, answer: str, metadata: Dict[str, Any]):
        """Add a Q&A exchange to the session."""
        exchange = {
            'timestamp': time.time(),
            'question': question,
            'answer': answer,
            'metadata': metadata
        }
        
        self.conversation_history.append(exchange)
        self.session_metadata['questions_asked'] += 1
        
        # Maintain context window
        if len(self.conversation_history) > self.max_context_length:
            self.conversation_history.pop(0)
    
    def get_conversation_context(self) -> List[Dict[str, str]]:
        """Get conversation context for LLM."""
        context = []
        for exchange in self.conversation_history[-5:]:  # Last 5 exchanges
            context.extend([
                {'role': 'user', 'content': exchange['question']},
                {'role': 'assistant', 'content': exchange['answer'][:500]}  # Truncate long answers
            ])
        return context
    
    def export_session(self, filepath: str):
        """Export session to file."""
        session_data = {
            'metadata': self.session_metadata,
            'conversation_history': self.conversation_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)

class QASystem:
    """
    Main Q&A System that orchestrates all components for intelligent
    biomedical knowledge graph question answering.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Q&A system.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.kg_service = None
        self.ollama_client = None
        self.query_planner = None
        self.response_synthesizer = None
        self.session_manager = None
        
        # System state
        self.is_initialized = False
        self.performance_stats = {
            'queries_processed': 0,
            'avg_query_time': 0.0,
            'avg_kg_query_time': 0.0,
            'avg_synthesis_time': 0.0,
            'success_rate': 0.0
        }
        
        # Initialize logging
        self._setup_logging()
        
        logger.info("Q&A System initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load system configuration."""
        default_config = {
            'ollama': {
                'base_url': 'http://localhost:11434',
                'default_model': 'llama3.2:1b',
                'timeout': 60,
                'max_retries': 3
            },
            'knowledge_graph': {
                'graph_path': 'quality_control/1_build_and_save_kg/saved_graphs/biomedical_graph.gpickle',
                'cache_size': 1000,
                'query_timeout': 30
            },
            'conversation': {
                'max_context_length': 10,
                'enable_follow_ups': True,
                'confidence_threshold': 0.3
            },
            'logging': {
                'level': 'INFO',
                'file': 'qa_system.log'
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        loaded_config = yaml.safe_load(f)
                    else:
                        loaded_config = json.load(f)
                
                # Merge with defaults
                default_config.update(loaded_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config['logging']['level'].upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.config['logging']['file'])
            ]
        )
    
    def initialize(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing Q&A system components...")
            
            # Initialize Knowledge Graph Service
            kg_config = self.config['knowledge_graph']
            graph_path = kg_config['graph_path']
            
            # Handle relative paths
            if not Path(graph_path).is_absolute():
                graph_path = Path.cwd() / graph_path
            
            logger.info(f"Loading knowledge graph from: {graph_path}")
            self.kg_service = KnowledgeGraphService(
                graph_path=str(graph_path),
                enable_caching=True,
                cache_size=kg_config['cache_size']
            )
            
            # Initialize Ollama Client
            ollama_config = self.config['ollama']
            logger.info("Initializing Ollama client...")
            self.ollama_client = OllamaClient(
                base_url=ollama_config['base_url'],
                model=ollama_config['default_model'],
                timeout=ollama_config['timeout'],
                max_retries=ollama_config['max_retries']
            )
            
            # Set system prompt for biomedical context
            system_prompt = """You are an expert biomedical researcher with extensive knowledge of genetics, molecular biology, disease mechanisms, and drug interactions. Your role is to provide accurate, evidence-based answers to biomedical questions using information from a comprehensive knowledge graph containing genes, pathways, diseases, and drug data.

Always:
1. Base your answers on the provided evidence
2. Use specific citations [1], [2], etc. to reference evidence
3. Explain mechanisms when possible
4. Acknowledge limitations in the data
5. Maintain scientific accuracy and precision"""
            
            self.ollama_client.set_system_prompt(system_prompt)
            
            # Initialize Query Planner
            logger.info("Initializing query planner...")
            self.query_planner = QueryPlanner()
            
            # Initialize Response Synthesizer
            logger.info("Initializing response synthesizer...")
            self.response_synthesizer = ResponseSynthesizer(self.ollama_client)
            
            # Initialize Session Manager
            conversation_config = self.config['conversation']
            self.session_manager = QASessionManager(
                max_context_length=conversation_config['max_context_length']
            )
            
            self.is_initialized = True
            logger.info("Q&A system initialization completed successfully")
            
            # Print system info
            self._print_system_info()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Q&A system: {e}")
            return False
    
    def ask_question(self, question: str, stream: bool = False) -> SynthesizedResponse:
        """
        Process a question and return a comprehensive answer.
        
        Args:
            question: Natural language question
            stream: Whether to stream the response (not yet implemented)
            
        Returns:
            SynthesizedResponse with answer and evidence
        """
        if not self.is_initialized:
            raise RuntimeError("Q&A system not initialized. Call initialize() first.")
        
        start_time = time.time()
        logger.info(f"Processing question: {question[:100]}...")
        
        try:
            # Step 1: Plan the query
            logger.info("Step 1: Planning query strategy...")
            query_plan = self.query_planner.plan_query(question)
            
            # Step 2: Execute knowledge graph queries
            logger.info("Step 2: Executing knowledge graph queries...")
            kg_query_start = time.time()
            kg_results = self._execute_kg_queries(query_plan)
            kg_query_time = time.time() - kg_query_start
            
            # Step 3: Synthesize response
            logger.info("Step 3: Synthesizing response...")
            synthesis_start = time.time()
            conversation_context = self.session_manager.get_conversation_context()
            
            response = self.response_synthesizer.synthesize_response(
                question=question,
                kg_results=kg_results,
                query_plan=query_plan,
                conversation_history=conversation_context
            )
            synthesis_time = time.time() - synthesis_start
            
            # Step 4: Update session and stats
            total_time = time.time() - start_time
            
            metadata = {
                'query_plan': asdict(query_plan),
                'kg_results_summary': self._summarize_kg_results(kg_results),
                'timing': {
                    'total_time': total_time,
                    'kg_query_time': kg_query_time,
                    'synthesis_time': synthesis_time
                }
            }
            
            self.session_manager.add_exchange(question, response.answer, metadata)
            self._update_performance_stats(total_time, kg_query_time, synthesis_time, response.confidence > 0.3)
            
            logger.info(f"Question processed successfully in {total_time:.2f}s (confidence: {response.confidence:.2f})")
            return response
            
        except Exception as e:
            logger.error(f"Failed to process question: {e}")
            # Return error response
            return SynthesizedResponse(
                answer=f"I encountered an error while processing your question: {str(e)}. Please try rephrasing or ask a different question.",
                confidence=0.1,
                evidence_sources=[],
                follow_up_questions=["Could you rephrase the question?", "Would you like to try a different topic?"],
                query_coverage={},
                synthesis_metadata={'error': str(e)}
            )
    
    def _execute_kg_queries(self, query_plan: QueryPlan) -> Dict[str, Any]:
        """Execute knowledge graph queries according to the plan."""
        results = {}
        
        for step in query_plan.query_strategy:
            method = step.get('method')
            params = step.get('params', {})
            description = step.get('description', f"Execute {method}")
            
            logger.info(f"Executing: {description}")
            
            try:
                if method == 'query_gene_information':
                    results[method] = self.kg_service.query_gene_information(params['gene'])
                elif method == 'query_pathway_information':
                    results[method] = self.kg_service.query_pathway_information(params['pathway'])
                elif method == 'query_disease_associations':
                    results[method] = self.kg_service.query_disease_associations(params['disease'])
                elif method == 'query_drug_interactions':
                    results[method] = self.kg_service.query_drug_interactions(params['drug'])
                elif method == 'search_by_keywords':
                    results[method] = self.kg_service.search_by_keywords(params['keywords'])
                elif method == 'get_related_entities':
                    results[method] = self.kg_service.get_related_entities(
                        params['entity'], 
                        params.get('relation_types'),
                        params.get('max_depth', 2)
                    )
                elif method == 'query_viral_expression':
                    results[method] = self.kg_service.query_viral_expression(params.get('limit', 50))
                else:
                    logger.warning(f"Unknown query method: {method}")
                    
            except Exception as e:
                logger.warning(f"Query method {method} failed: {e}")
                results[method] = {'error': str(e), 'found': False}
        
        return results
    
    def _summarize_kg_results(self, kg_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of KG query results."""
        summary = {
            'methods_executed': len(kg_results),
            'successful_queries': 0,
            'total_entities_found': 0
        }
        
        for method, result in kg_results.items():
            if isinstance(result, dict):
                if result.get('found', False):
                    summary['successful_queries'] += 1
                
                # Count entities in various result structures
                for key in ['genes', 'go_terms', 'diseases', 'drugs', 'pathways', 'neighbors']:
                    if key in result and isinstance(result[key], list):
                        summary['total_entities_found'] += len(result[key])
        
        return summary
    
    def _update_performance_stats(self, total_time: float, kg_time: float, synthesis_time: float, success: bool):
        """Update system performance statistics."""
        stats = self.performance_stats
        
        # Update counts
        stats['queries_processed'] += 1
        
        # Update running averages
        n = stats['queries_processed']
        stats['avg_query_time'] = ((stats['avg_query_time'] * (n-1)) + total_time) / n
        stats['avg_kg_query_time'] = ((stats['avg_kg_query_time'] * (n-1)) + kg_time) / n
        stats['avg_synthesis_time'] = ((stats['avg_synthesis_time'] * (n-1)) + synthesis_time) / n
        
        # Update success rate
        current_successes = stats['success_rate'] * (n-1) + (1 if success else 0)
        stats['success_rate'] = current_successes / n
    
    def _print_system_info(self):
        """Print system information."""
        kg_stats = self.kg_service.get_graph_stats()
        ollama_stats = self.ollama_client.get_client_stats()
        
        print("\n" + "="*60)
        print("🧬 BIOMEDICAL KNOWLEDGE GRAPH Q&A SYSTEM")
        print("="*60)
        print(f"📊 Knowledge Graph: {kg_stats['total_nodes']:,} nodes, {kg_stats['total_edges']:,} edges")
        print(f"🤖 LLM Model: {ollama_stats['current_model']}")
        print(f"🔗 Ollama Status: {'✅ Connected' if ollama_stats['connection_status'] else '❌ Disconnected'}")
        print(f"💾 Graph Load Time: {kg_stats['load_time_seconds']:.1f}s")
        print("="*60)
        print("Ready for questions! Type 'help' for commands or 'quit' to exit.")
        print("="*60 + "\n")
    
    def interactive_session(self):
        """Run an interactive Q&A session."""
        if not self.is_initialized:
            print("System not initialized. Initializing...")
            if not self.initialize():
                print("Failed to initialize system. Exiting.")
                return
        
        print("Starting interactive session...")
        
        while True:
            try:
                # Get user input
                question = input("\n🤔 Your question: ").strip()
                
                if not question:
                    continue
                
                # Handle special commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                elif question.lower() == 'help':
                    self._print_help()
                    continue
                elif question.lower() == 'stats':
                    self._print_stats()
                    continue
                elif question.lower().startswith('config'):
                    self._print_config()
                    continue
                elif question.lower() == 'clear':
                    self.session_manager.conversation_history.clear()
                    print("Conversation history cleared.")
                    continue
                
                # Process the question
                print("\n🔍 Processing your question...")
                response = self.ask_question(question)
                
                # Display the answer
                print(f"\n🤖 Answer (Confidence: {response.confidence:.1%}):")
                print("-" * 50)
                print(response.answer)
                
                # Show follow-up questions if available
                if response.follow_up_questions:
                    print(f"\n💡 Follow-up questions:")
                    for i, follow_up in enumerate(response.follow_up_questions, 1):
                        print(f"   {i}. {follow_up}")
                
                # Show evidence count
                if response.evidence_sources:
                    print(f"\n📚 Based on {len(response.evidence_sources)} pieces of evidence from the knowledge graph")
                
            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                logger.error(f"Interactive session error: {e}")
    
    def _print_help(self):
        """Print help information."""
        help_text = """
Available Commands:
• help          - Show this help message
• stats         - Show system performance statistics  
• config        - Show current configuration
• clear         - Clear conversation history
• quit/exit/q   - Exit the system

Example Questions:
• "What is the function of the TP53 gene?"
• "Which genes are associated with breast cancer?"
• "What pathways involve insulin signaling?"
• "How does aspirin interact with genetic factors?"
• "What genes influence bird feather color?"
        """
        print(help_text)
    
    def _print_stats(self):
        """Print system statistics."""
        stats = self.performance_stats
        kg_stats = self.kg_service.get_graph_stats()
        cache_stats = self.kg_service.get_cache_stats()
        
        print("\n📈 System Statistics:")
        print(f"  Queries Processed: {stats['queries_processed']}")
        print(f"  Success Rate: {stats['success_rate']:.1%}")
        print(f"  Avg Query Time: {stats['avg_query_time']:.2f}s")
        print(f"  Avg KG Query Time: {stats['avg_kg_query_time']:.2f}s")
        print(f"  Avg Synthesis Time: {stats['avg_synthesis_time']:.2f}s")
        print(f"  Session Questions: {self.session_manager.session_metadata['questions_asked']}")
        print(f"  Cache Usage: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
    
    def _print_config(self):
        """Print current configuration."""
        print("\n⚙️ Current Configuration:")
        print(json.dumps(self.config, indent=2))
    
    def export_session(self, filepath: str):
        """Export current session."""
        if self.session_manager:
            self.session_manager.export_session(filepath)
            print(f"Session exported to: {filepath}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'initialized': self.is_initialized,
            'performance_stats': self.performance_stats,
            'kg_stats': self.kg_service.get_graph_stats() if self.kg_service else {},
            'ollama_stats': self.ollama_client.get_client_stats() if self.ollama_client else {},
            'session_info': self.session_manager.session_metadata if self.session_manager else {}
        }

def main():
    """Main entry point for the Q&A system."""
    parser = argparse.ArgumentParser(description="Biomedical Knowledge Graph Q&A System")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--question", "-q", help="Single question to ask")
    parser.add_argument("--model", "-m", help="Ollama model to use", default="llama3.2:1b")
    parser.add_argument("--graph", "-g", help="Knowledge graph file path")
    parser.add_argument("--export", "-e", help="Export session to file")
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    config_overrides = {}
    if args.model:
        config_overrides['ollama'] = {'default_model': args.model}
    if args.graph:
        config_overrides['knowledge_graph'] = {'graph_path': args.graph}
    
    # Initialize system
    qa_system = QASystem(args.config)
    
    # Apply command line overrides
    if config_overrides:
        for key, value in config_overrides.items():
            qa_system.config[key].update(value)
    
    # Initialize components
    if not qa_system.initialize():
        print("Failed to initialize Q&A system")
        sys.exit(1)
    
    try:
        if args.question:
            # Single question mode
            response = qa_system.ask_question(args.question)
            print(f"\nQuestion: {args.question}")
            print(f"\nAnswer: {response.answer}")
            print(f"\nConfidence: {response.confidence:.1%}")
        else:
            # Interactive mode
            qa_system.interactive_session()
        
        # Export session if requested
        if args.export:
            qa_system.export_session(args.export)
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()