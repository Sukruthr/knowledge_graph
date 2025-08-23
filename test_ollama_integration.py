#!/usr/bin/env python3
"""
Test script for Ollama LLM Integration with Knowledge Graph Q&A System

This script tests the complete integration including:
1. Knowledge Graph Service functionality
2. Ollama Client connectivity and response generation
3. Query Planning Agent
4. Response Synthesizer
5. Main Q&A System orchestrator

Run this test to verify the complete system works before interactive use.
"""

import sys
import logging
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

def test_knowledge_graph_service():
    """Test Knowledge Graph Service functionality."""
    print("🧬 Testing Knowledge Graph Service...")
    
    try:
        from llm_integration.kg_service import KnowledgeGraphService
        
        # Initialize service
        graph_path = "quality_control/1_build_and_save_kg/saved_graphs/biomedical_graph.gpickle"
        kg_service = KnowledgeGraphService(graph_path)
        
        # Test basic stats
        stats = kg_service.get_graph_stats()
        print(f"   ✅ Graph loaded: {stats['total_nodes']:,} nodes, {stats['total_edges']:,} edges")
        
        # Test gene query
        gene_result = kg_service.query_gene_information("TP53")
        if gene_result['found']:
            print(f"   ✅ Gene query successful: Found TP53 with {len(gene_result.get('go_annotations', []))} GO annotations")
        else:
            print("   ⚠️  Gene query: TP53 not found (may be expected)")
        
        # Test search functionality
        search_result = kg_service.search_by_keywords(["cancer", "tumor"])
        total_matches = search_result.get('total_matches', 0)
        print(f"   ✅ Keyword search successful: {total_matches} matches for cancer/tumor")
        
        return kg_service
        
    except Exception as e:
        print(f"   ❌ Knowledge Graph Service test failed: {e}")
        traceback.print_exc()
        return None

def test_ollama_client():
    """Test Ollama Client connectivity and functionality."""
    print("🤖 Testing Ollama Client...")
    
    try:
        from llm_integration.ollama_client import OllamaClient
        
        # Initialize client with correct model
        ollama_client = OllamaClient(model="llama3.2:1b")
        
        # Test connection
        stats = ollama_client.get_client_stats()
        if stats['connection_status']:
            print(f"   ✅ Ollama connection successful")
            print(f"   ✅ Available models: {stats['available_models']}")
            print(f"   ✅ Current model: {stats['current_model']}")
        else:
            print("   ❌ Ollama connection failed - is Ollama running?")
            return None
        
        # Test simple generation
        test_prompt = "What is the role of genes in biological systems? Please provide a brief scientific explanation."
        print("   🔄 Testing response generation...")
        
        response = ollama_client.generate_response(test_prompt)
        if response and response.content:
            print(f"   ✅ Response generation successful ({len(response.content)} characters)")
            print(f"   📝 Sample response: {response.content[:200]}...")
        else:
            print("   ❌ Response generation failed")
            return None
        
        return ollama_client
        
    except Exception as e:
        print(f"   ❌ Ollama Client test failed: {e}")
        traceback.print_exc()
        return None

def test_query_planner():
    """Test Query Planning Agent."""
    print("🎯 Testing Query Planning Agent...")
    
    try:
        from llm_integration.query_planner import QueryPlanner
        
        # Initialize planner
        planner = QueryPlanner()
        
        # Test entity extraction
        test_question = "What is the function of the TP53 gene in cancer?"
        query_plan = planner.plan_query(test_question)
        
        print(f"   ✅ Query planning successful")
        print(f"   📊 Intent: {query_plan.intent.value}")
        print(f"   🔍 Entities found: {[e.text for e in query_plan.entities]}")
        print(f"   📋 Query steps: {len(query_plan.query_strategy)}")
        print(f"   🎯 Confidence: {query_plan.confidence:.2f}")
        
        return planner
        
    except Exception as e:
        print(f"   ❌ Query Planner test failed: {e}")
        traceback.print_exc()
        return None

def test_response_synthesizer(ollama_client):
    """Test Response Synthesizer."""
    print("🔬 Testing Response Synthesizer...")
    
    try:
        from llm_integration.response_synthesizer import ResponseSynthesizer
        from llm_integration.query_planner import QueryPlanner
        
        # Initialize synthesizer
        synthesizer = ResponseSynthesizer(ollama_client)
        planner = QueryPlanner()
        
        # Create test data
        test_question = "What is the function of TP53?"
        query_plan = planner.plan_query(test_question)
        
        # Mock KG results for testing
        mock_kg_results = {
            "query_gene_information": {
                "gene": "TP53",
                "found": True,
                "go_annotations": [
                    {
                        "go_id": "GO:0006915",
                        "name": "apoptotic process",
                        "namespace": "biological_process",
                        "evidence": "IDA"
                    }
                ],
                "associations": [
                    {
                        "entity": "cancer",
                        "type": "disease",
                        "name": "various cancers",
                        "relationship_type": "associated_with"
                    }
                ]
            }
        }
        
        # Test synthesis
        print("   🔄 Testing response synthesis...")
        response = synthesizer.synthesize_response(
            question=test_question,
            kg_results=mock_kg_results,
            query_plan=query_plan
        )
        
        if response and response.answer:
            print(f"   ✅ Response synthesis successful")
            print(f"   📝 Answer length: {len(response.answer)} characters")
            print(f"   📊 Confidence: {response.confidence:.2f}")
            print(f"   🧾 Evidence sources: {len(response.evidence_sources)}")
            print(f"   ❓ Follow-up questions: {len(response.follow_up_questions)}")
        else:
            print("   ❌ Response synthesis failed")
            return None
        
        return synthesizer
        
    except Exception as e:
        print(f"   ❌ Response Synthesizer test failed: {e}")
        traceback.print_exc()
        return None

def test_qa_system():
    """Test complete Q&A System integration."""
    print("🎪 Testing Complete Q&A System...")
    
    try:
        from llm_integration.qa_system import QASystem
        
        # Initialize Q&A system
        qa_system = QASystem()
        
        # Test initialization
        print("   🔄 Initializing Q&A system...")
        if qa_system.initialize():
            print("   ✅ Q&A system initialization successful")
        else:
            print("   ❌ Q&A system initialization failed")
            return False
        
        # Test simple question
        test_question = "What is the function of the TP53 gene?"
        print(f"   🔄 Testing question: {test_question}")
        
        response = qa_system.ask_question(test_question)
        
        if response and response.answer:
            print("   ✅ Complete Q&A workflow successful!")
            print(f"   📝 Answer preview: {response.answer[:300]}...")
            print(f"   📊 Confidence: {response.confidence:.2f}")
            print(f"   🧾 Evidence sources: {len(response.evidence_sources)}")
            
            # Show follow-up questions
            if response.follow_up_questions:
                print("   💡 Follow-up questions:")
                for i, question in enumerate(response.follow_up_questions[:3], 1):
                    print(f"      {i}. {question}")
        else:
            print("   ❌ Complete Q&A workflow failed")
            return False
        
        # Test system stats
        stats = qa_system.get_system_status()
        print(f"   📊 System stats: {stats['performance_stats']['queries_processed']} queries processed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Q&A System test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("🧪 OLLAMA LLM INTEGRATION TESTS")
    print("=" * 60)
    print()
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    # Check prerequisites  
    print("📋 Checking prerequisites...")
    
    # Check if knowledge graph exists
    graph_path = Path("quality_control/1_build_and_save_kg/saved_graphs/biomedical_graph.gpickle")
    if not graph_path.exists():
        print(f"   ❌ Knowledge graph not found at: {graph_path}")
        print("      Please ensure the knowledge graph has been built and saved.")
        return False
    else:
        print(f"   ✅ Knowledge graph found: {graph_path}")
    
    print()
    
    # Test each component
    test_results = []
    
    # Test 1: Knowledge Graph Service
    kg_service = test_knowledge_graph_service()
    test_results.append(kg_service is not None)
    print()
    
    # Test 2: Ollama Client
    ollama_client = test_ollama_client()
    test_results.append(ollama_client is not None)
    print()
    
    # Test 3: Query Planner
    query_planner = test_query_planner()
    test_results.append(query_planner is not None)
    print()
    
    # Test 4: Response Synthesizer (requires Ollama)
    if ollama_client:
        synthesizer = test_response_synthesizer(ollama_client)
        test_results.append(synthesizer is not None)
    else:
        print("🔬 Skipping Response Synthesizer test (Ollama not available)")
        test_results.append(False)
    print()
    
    # Test 5: Complete Q&A System
    if all(test_results[:3]):  # Only if basic components work
        qa_success = test_qa_system()
        test_results.append(qa_success)
    else:
        print("🎪 Skipping Q&A System test (prerequisites failed)")
        test_results.append(False)
    print()
    
    # Final results
    print("=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    test_names = [
        "Knowledge Graph Service",
        "Ollama Client", 
        "Query Planning Agent",
        "Response Synthesizer",
        "Complete Q&A System"
    ]
    
    for name, result in zip(test_names, test_results):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {name:<25}: {status}")
    
    total_passed = sum(test_results)
    total_tests = len(test_results)
    
    print()
    print(f"🎯 Overall Result: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.0f}%)")
    
    if total_passed == total_tests:
        print()
        print("🎉 ALL TESTS PASSED! The Ollama LLM integration is fully functional.")
        print("   You can now run the interactive Q&A system using:")
        print("   python -m src.llm_integration.qa_system")
        print()
        return True
    else:
        print()
        print("⚠️  Some tests failed. Please check the error messages above.")
        
        if not test_results[1]:  # Ollama client failed
            print("   💡 Make sure Ollama is installed and running:")
            print("      curl -fsSL https://ollama.ai/install.sh | sh")
            print("      ollama serve")
            print("      ollama pull llama2")
        
        print()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)