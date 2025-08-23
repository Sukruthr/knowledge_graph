# Ollama LLM Integration Plan for Knowledge Graph Q&A System

## Project Overview

This document outlines the comprehensive plan for integrating Ollama LLM with our biomedical knowledge graph to create an intelligent Q&A system. The system will enable users to ask complex biomedical questions in natural language and receive answers backed by structured knowledge from our graph containing 174,210 nodes and 5,060,768 edges.

## Current System Analysis

### Knowledge Graph Capabilities
- **Scale**: 174,210 nodes, 5,060,768 edges
- **Data Types**: GO ontology (BP/CC/MF), Omics data, disease associations, drug perturbations, viral responses, model predictions, LLM interpretations
- **Query Methods**: 15+ specialized query functions for comprehensive gene profiling, model comparisons, GO analysis, contamination studies
- **Storage**: NetworkX-based graph saved as pickle file (biomedical_graph.gpickle)
- **Performance**: ~14s load time, extensive cross-modal connectivity

### Available Query Functions
1. `query_gene_comprehensive()` - Complete gene profiles with all associations
2. `query_model_predictions()` - LLM model prediction analysis
3. `query_cc_mf_terms()` - Cellular component and molecular function terms
4. `query_llm_interpretations()` - Multi-model LLM interpretations
5. `query_go_core_analysis()` - GO enrichment analysis
6. Additional specialized queries for contamination analysis, similarity rankings, confidence evaluations

## Architecture Design

### 1. System Components

#### A. Knowledge Graph Service (`src/llm_integration/kg_service.py`)
- Load and manage the pre-built knowledge graph
- Provide high-level query interface for LLM system
- Handle query optimization and result formatting
- Cache frequently accessed data

#### B. Ollama LLM Client (`src/llm_integration/ollama_client.py`)
- Interface with local Ollama API
- Manage conversation context and history
- Handle prompt engineering and response processing
- Support multiple models (llama2, codellama, etc.)

#### C. Query Planning Agent (`src/llm_integration/query_planner.py`)
- Parse natural language questions into structured queries
- Determine which KG query methods to use
- Plan multi-step reasoning workflows
- Generate appropriate search strategies

#### D. Response Synthesis Agent (`src/llm_integration/response_synthesizer.py`)
- Combine KG query results with LLM reasoning
- Generate coherent explanations from structured data
- Maintain conversation context and follow-up capabilities
- Format responses with evidence citations

#### E. Interactive Q&A System (`src/llm_integration/qa_system.py`)
- Main orchestrator for user interactions
- Handle conversation flow and state management
- Coordinate between all system components
- Provide CLI and potential web interface

### 2. Workflow Architecture

```
User Question → Query Planner → KG Service → Response Synthesizer → Final Answer
      ↑              ↓             ↓              ↓                    ↓
   Feedback    Ollama LLM    Query Results   LLM Reasoning      Ollama LLM
```

## Detailed Implementation Plan

### Phase 1: Core Infrastructure Setup

#### 1.1 Knowledge Graph Service Implementation
**File**: `src/llm_integration/kg_service.py`

**Responsibilities**:
- Load pre-built knowledge graph from pickle file
- Provide unified query interface
- Handle query result formatting for LLM consumption
- Implement query optimization and caching

**Key Methods**:
```python
class KnowledgeGraphService:
    def __init__(self, graph_path: str)
    def query_gene_information(self, gene: str) -> Dict
    def query_pathway_information(self, pathway: str) -> Dict
    def query_disease_associations(self, disease: str) -> Dict
    def query_drug_interactions(self, drug: str) -> Dict
    def search_by_keywords(self, keywords: List[str]) -> Dict
    def get_related_entities(self, entity: str, relation_types: List[str]) -> Dict
```

#### 1.2 Ollama Client Implementation
**File**: `src/llm_integration/ollama_client.py`

**Responsibilities**:
- Connect to local Ollama API
- Manage model selection and configuration
- Handle streaming responses
- Maintain conversation context

**Key Methods**:
```python
class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2")
    def generate_response(self, prompt: str, context: List[str] = None) -> str
    def stream_response(self, prompt: str, context: List[str] = None) -> Iterator[str]
    def set_system_prompt(self, system_prompt: str)
    def get_available_models(self) -> List[str]
```

### Phase 2: Intelligent Query Processing

#### 2.1 Query Planning Agent
**File**: `src/llm_integration/query_planner.py`

**Responsibilities**:
- Parse natural language questions
- Map questions to appropriate KG query strategies
- Handle complex multi-step queries
- Generate query execution plans

**Key Components**:
- **Intent Classification**: Determine query type (gene info, pathway analysis, drug discovery, etc.)
- **Entity Extraction**: Identify biological entities mentioned in questions
- **Query Strategy Selection**: Choose appropriate KG query methods
- **Multi-step Planning**: Break complex questions into sub-queries

**Example Query Plans**:
```python
# Question: "Which genes influence bird feather color?"
query_plan = {
    "intent": "gene_function_search",
    "entities": ["feather", "color", "bird"],
    "strategy": [
        {"method": "search_by_keywords", "params": {"keywords": ["feather", "pigmentation", "color"]}},
        {"method": "query_pathway_information", "params": {"pathway": "melanin_biosynthesis"}},
        {"method": "filter_by_organism", "params": {"organism": "bird"}}
    ],
    "synthesis_instructions": "Focus on genes involved in pigmentation pathways relevant to avian biology"
}
```

#### 2.2 Advanced Query Processing
**File**: `src/llm_integration/advanced_queries.py`

**Responsibilities**:
- Handle complex biological reasoning
- Cross-reference multiple data types
- Perform graph traversals and relationship analysis
- Generate evidence-based insights

### Phase 3: Response Generation and Synthesis

#### 3.1 Response Synthesizer
**File**: `src/llm_integration/response_synthesizer.py`

**Responsibilities**:
- Combine structured KG data with natural language generation
- Maintain scientific accuracy while ensuring readability
- Provide evidence citations and confidence scores
- Handle follow-up question generation

**Key Methods**:
```python
class ResponseSynthesizer:
    def synthesize_answer(self, question: str, kg_results: Dict, conversation_history: List[Dict]) -> Dict
    def generate_evidence_summary(self, kg_results: Dict) -> str
    def create_follow_up_questions(self, answer: str, kg_results: Dict) -> List[str]
    def format_scientific_response(self, raw_answer: str, evidence: Dict) -> str
```

### Phase 4: Interactive Q&A System

#### 4.1 Main Q&A Orchestrator
**File**: `src/llm_integration/qa_system.py`

**Responsibilities**:
- Coordinate all system components
- Manage conversation state and context
- Handle user interaction flow
- Provide both CLI and potential API interfaces

**Conversation Flow**:
1. **Question Reception**: Parse and validate user input
2. **Query Planning**: Generate KG query strategy
3. **Knowledge Retrieval**: Execute queries and collect results
4. **LLM Reasoning**: Apply Ollama model for analysis and synthesis
5. **Response Generation**: Create final answer with evidence
6. **Follow-up Handling**: Manage conversation continuity

### Phase 5: Advanced Features

#### 5.1 Multi-turn Conversation Support
- Maintain conversation context across multiple questions
- Reference previous answers and build upon them
- Handle clarification requests and refinements

#### 5.2 Evidence-based Reasoning
- Provide confidence scores for answers
- Show knowledge graph paths used in reasoning
- Enable users to explore supporting evidence

#### 5.3 Interactive Code Generation
- Generate Python code for KG queries based on user requests
- Provide executable scripts for complex analyses
- Support data export and visualization

## Prompt Engineering Strategy

### System Prompts for Different Roles

#### 1. Query Planning Prompt
```
You are a biomedical knowledge graph query planner. Your task is to analyze natural language questions about biology, genetics, diseases, and drugs, then determine the best strategy to query our comprehensive biomedical knowledge graph.

Our knowledge graph contains:
- 174,210 nodes including genes, GO terms, diseases, drugs, viral conditions
- 5,060,768 relationships including annotations, associations, expressions
- Specialized data on model predictions, LLM interpretations, contamination analysis

Available query methods: [list of methods]

For each question, provide:
1. Intent classification
2. Key entities mentioned
3. Step-by-step query strategy
4. Expected result format
```

#### 2. Response Synthesis Prompt
```
You are a biomedical expert tasked with synthesizing knowledge graph data into comprehensive, accurate answers. You have access to structured biological data and must:

1. Analyze the provided knowledge graph results
2. Combine multiple data sources coherently
3. Provide evidence-based explanations
4. Maintain scientific accuracy
5. Generate follow-up questions for deeper exploration

Always cite your sources from the knowledge graph data and indicate confidence levels where appropriate.
```

## Example Workflow: "Which genes influence bird feather color?"

### Step-by-Step Process:

1. **Question Analysis**:
   - Intent: Gene function identification
   - Entities: feather, color, bird
   - Domain: Genetics, development biology

2. **Query Planning**:
   ```python
   query_plan = [
       "search_by_keywords(['feather', 'pigmentation', 'melanin'])",
       "query_pathway_information('pigment_biosynthesis')",
       "query_gene_comprehensive('TYRP1', 'TYR', 'MC1R')",
       "filter_by_biological_process('pigmentation')"
   ]
   ```

3. **Knowledge Retrieval**:
   - Execute KG queries
   - Collect gene associations, pathway data, GO annotations
   - Retrieve model predictions and LLM interpretations

4. **LLM Analysis**:
   ```
   Prompt: "Based on this knowledge graph data about pigmentation genes [data], 
   explain which genes influence bird feather color and provide the mechanisms."
   ```

5. **Response Synthesis**:
   - Combine KG results with LLM reasoning
   - Generate comprehensive explanation with evidence
   - Create follow-up questions

6. **Final Answer**:
   ```
   Several genes are known to influence bird feather color through pigmentation pathways:
   
   1. **MC1R (Melanocortin 1 Receptor)**: Controls production of eumelanin vs pheomelanin...
   2. **TYR (Tyrosinase)**: Rate-limiting enzyme in melanin synthesis...
   3. **TYRP1 (Tyrosinase Related Protein 1)**: Involved in eumelanin production...
   
   [Generated code for further exploration]
   
   Follow-up questions:
   - How do these genes interact in specific bird species?
   - What mutations in these genes cause specific color variations?
   ```

## Implementation Timeline

### Week 1-2: Infrastructure Setup
- Implement KnowledgeGraphService
- Set up Ollama client integration
- Create basic query interfaces
- Test knowledge graph loading and basic queries

### Week 3-4: Query Processing
- Develop QueryPlanner with intent classification
- Implement entity extraction and query strategy selection
- Create advanced query processing capabilities
- Test complex multi-step queries

### Week 5-6: Response Generation
- Build ResponseSynthesizer
- Implement evidence-based reasoning
- Create conversation context management
- Develop prompt engineering strategies

### Week 7-8: Integration and Testing
- Integrate all components into main Q&A system
- Comprehensive testing with various question types
- Performance optimization and caching
- User interface refinement

## Testing Strategy

### 1. Unit Testing
- Test each component independently
- Validate KG query results
- Test Ollama client connectivity
- Verify response synthesis quality

### 2. Integration Testing
- End-to-end workflow testing
- Multi-turn conversation testing
- Error handling and edge cases
- Performance benchmarking

### 3. Domain Expert Validation
- Scientific accuracy validation
- Biological reasoning correctness
- Evidence citation verification
- Use case coverage assessment

## Configuration Files

### 1. System Configuration (`config/qa_system_config.yaml`)
```yaml
ollama:
  base_url: "http://localhost:11434"
  default_model: "llama2"
  timeout: 30
  max_tokens: 4096

knowledge_graph:
  graph_path: "quality_control/1_build_and_save_kg/saved_graphs/biomedical_graph.gpickle"
  cache_size: 1000
  query_timeout: 10

conversation:
  max_context_length: 10
  enable_follow_ups: true
  confidence_threshold: 0.7
```

### 2. Prompt Templates (`config/prompts/`)
- `query_planning_prompts.yaml`
- `synthesis_prompts.yaml`
- `domain_specific_prompts.yaml`

## Dependencies and Requirements

### Python Packages
```python
# Core dependencies
networkx>=3.0
requests>=2.28
pydantic>=2.0
pyyaml>=6.0

# LLM integration
ollama>=0.1.0  # If available, otherwise requests for API calls

# Optional enhancements
langchain>=0.1.0  # For advanced prompt management
sentence-transformers>=2.2  # For semantic similarity
```

### System Requirements
- Python 3.9+
- Ollama installed and running locally
- Minimum 8GB RAM (16GB recommended for full KG)
- SSD storage for optimal KG loading performance

## Success Metrics

### 1. Functionality Metrics
- **Query Success Rate**: >95% of valid questions should receive relevant answers
- **Response Accuracy**: Domain expert validation of >90% accuracy
- **Context Preservation**: Maintain conversation context across 5+ turns

### 2. Performance Metrics
- **Response Time**: <30 seconds for complex multi-step queries
- **KG Loading Time**: <15 seconds (already achieved)
- **Memory Usage**: <8GB total system memory

### 3. User Experience Metrics
- **Question Coverage**: Support for 10+ biological question types
- **Follow-up Quality**: Generate meaningful follow-up questions >80% of the time
- **Code Generation**: Provide executable KG query code when requested

## Risk Assessment and Mitigation

### 1. Technical Risks
- **Ollama Connectivity**: Implement fallback modes and retry logic
- **Knowledge Graph Loading**: Optimize loading and implement lazy loading
- **Memory Management**: Implement proper caching and garbage collection

### 2. Scientific Accuracy Risks
- **Hallucination Control**: Use structured KG data to ground responses
- **Evidence Validation**: Require citations for all claims
- **Uncertainty Communication**: Clearly communicate confidence levels

### 3. Performance Risks
- **Query Complexity**: Implement query optimization and timeouts
- **Response Generation**: Use streaming for long responses
- **Scalability**: Design for potential multi-user scenarios

## Future Enhancements

### 1. Advanced Capabilities
- **Multi-modal Integration**: Support for images, structures, sequences
- **Real-time Updates**: Integration with live databases
- **Collaborative Features**: Multi-user conversation support

### 2. Expanded Coverage
- **Domain Expansion**: Additional biological domains
- **Model Diversity**: Support for multiple LLM backends
- **Export Capabilities**: Results to various formats (PDF, Excel, etc.)

---

## Approval Required

This plan provides a comprehensive roadmap for integrating Ollama LLM with your biomedical knowledge graph. The implementation will create a sophisticated Q&A system capable of:

1. **Understanding complex biological questions** in natural language
2. **Querying your comprehensive knowledge graph** strategically  
3. **Synthesizing evidence-based answers** with proper citations
4. **Maintaining conversational context** for follow-up questions
5. **Generating executable code** for further exploration

**Please review this plan and provide approval to proceed with implementation. Once approved, I will follow this document precisely to build the complete system.**