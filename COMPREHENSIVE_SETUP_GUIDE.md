# Comprehensive Setup Guide: Biomedical Knowledge Graph Q&A System

## 🎯 Project Overview

This is a comprehensive biomedical knowledge graph system that integrates Gene Ontology (GO) data with multi-modal Omics data, LLM model comparison analysis, viral expression matrices, and advanced Q&A capabilities powered by Ollama LLM. The system creates a unified knowledge graph containing **174K+ nodes** and **5M+ edges** for biomedical research and AI-powered question answering.

### System Capabilities
- **Knowledge Graph**: 174,210 nodes, 5,060,768 edges
- **Gene Coverage**: 3,175 genes with comprehensive annotations
- **Data Types**: GO ontology, disease associations, drug perturbations, viral responses, LLM interpretations
- **AI Integration**: Natural language Q&A system powered by Ollama
- **Query Methods**: 15+ specialized query functions for comprehensive analysis

---

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 18.04+ recommended)
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 10GB free space (SSD recommended)
- **Python**: 3.10+
- **Internet**: Required for Ollama model downloads

### Required Software
1. **Conda/Miniconda** - For environment management
2. **Ollama** - For local LLM inference
3. **Git** - For version control (if needed)

---

## 🚀 Step 1: Environment Setup

### 1.1 Install Miniconda (if not already installed)

```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install Miniconda
bash Miniconda3-latest-Linux-x86_64.sh

# Reload shell
source ~/.bashrc
```

### 1.2 Create Conda Environment

```bash
# Navigate to project directory
cd /home/mreddy1/knowledge_graph

# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate knowledge_graph
```

### 1.3 Verify Environment Setup

```bash
# Check Python version
python --version  # Should show Python 3.10.x

# Check key packages
python -c "import networkx, pandas, numpy, sklearn; print('Core packages installed successfully')"

# Check Neo4j driver (for advanced features)
python -c "import neo4j; print('Neo4j driver available')"
```

---

## 🤖 Step 2: Ollama Installation and Setup

### 2.1 Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve &

# Wait a few seconds for startup
sleep 5
```

### 2.2 Download Required Models

```bash
# Download primary model (lightweight, fast)
ollama pull llama3.2:1b

# Optional: Download additional models
ollama pull llama2:7b        # More capable but slower
ollama pull codellama        # For code-related queries
ollama pull mistral          # Alternative model

# Verify models
ollama list
```

### 2.3 Set Up Local Ollama Installation

The project includes a local Ollama setup at `~/ollama-local/`:

```bash
# Check local Ollama installation
ls -la ~/ollama-local/

# The startup script will use this if available
# Local installation path: ~/ollama-local/bin/ollama
```

---

## 📊 Step 3: Knowledge Graph Setup

### 3.1 Verify Data Availability

```bash
# Check if knowledge graph is built
ls -la quality_control/1_build_and_save_kg/saved_graphs/

# Should show:
# - biomedical_graph.gpickle (main graph file)
# - graph_statistics.json
# - validation_results.json
```

### 3.2 Build Knowledge Graph (if needed)

If the knowledge graph doesn't exist, build it:

```bash
# Navigate to build directory
cd quality_control/1_build_and_save_kg/

# Build and save knowledge graph
python build_and_save_kg.py

# This will:
# - Load all data sources (GO, Omics, LLM data)
# - Construct unified knowledge graph
# - Save as pickle file (~14 second load time)
# - Generate statistics and validation reports
```

### 3.3 Verify Knowledge Graph

```bash
# Quick verification
python -c "
import pickle
graph = pickle.load(open('quality_control/1_build_and_save_kg/saved_graphs/biomedical_graph.gpickle', 'rb'))
print(f'Graph loaded: {len(graph.nodes):,} nodes, {len(graph.edges):,} edges')
"
```

---

## 🔧 Step 4: System Configuration

### 4.1 Configuration Files

The system uses `src/llm_integration/config/qa_system_config.yaml`:

```yaml
# Key configuration settings:
ollama:
  base_url: "http://localhost:11434"
  default_model: "llama3.2:1b"
  timeout: 60

knowledge_graph:
  graph_path: "quality_control/1_build_and_save_kg/saved_graphs/biomedical_graph.gpickle"
  cache_size: 1000

conversation:
  max_context_length: 10
  enable_follow_ups: true
  confidence_threshold: 0.3
```

### 4.2 Customize Configuration (Optional)

```bash
# Edit configuration file
nano src/llm_integration/config/qa_system_config.yaml

# Key settings to customize:
# - ollama.default_model: Change model
# - knowledge_graph.graph_path: Different graph path
# - conversation.max_context_length: Conversation memory
```

---

## ✅ Step 5: System Testing and Validation

### 5.1 Run Integration Tests

```bash
# Run comprehensive test suite
python test_ollama_integration.py

# Expected output:
# 🧬 Testing Knowledge Graph Service... ✅
# 🤖 Testing Ollama Client... ✅
# 🎯 Testing Query Planning Agent... ✅
# 🔬 Testing Response Synthesizer... ✅
# 🎪 Testing Complete Q&A System... ✅
```

### 5.2 Validate Knowledge Graph

```bash
# Run knowledge graph validation
python validation/comprehensive_omics_validation.py

# Should show:
# - 7/7 validation criteria passed
# - Performance metrics
# - Data quality statistics
```

### 5.3 Test Core Components

```bash
# Test parser functionality
python -m pytest parser_tests/

# Test knowledge graph builders
python -m pytest kg_testing/

# Run quality control checks
python quality_control/run_comprehensive_qc.py
```

---

## 🎮 Step 6: Running the Q&A System

### 6.1 Quick Start with Startup Script

```bash
# Use the provided startup script
chmod +x start_qa_system.sh
./start_qa_system.sh

# This will:
# - Activate conda environment
# - Start Ollama if needed
# - Launch Q&A system
```

### 6.2 Manual Startup

```bash
# Activate environment
conda activate knowledge_graph

# Ensure Ollama is running
ollama serve &
sleep 5

# Start Q&A system
python -m src.llm_integration.qa_system
```

### 6.3 Command Line Options

```bash
# Single question mode
python -m src.llm_integration.qa_system -q "What is the function of TP53?"

# Use different model
python -m src.llm_integration.qa_system -m llama2:7b

# Use custom config
python -m src.llm_integration.qa_system -c my_config.yaml

# Export session
python -m src.llm_integration.qa_system -e session_export.json
```

---

## 📁 File Structure and Purposes

### Core Directory Structure

```
/home/mreddy1/knowledge_graph/
├── src/                                    # Main source code
│   ├── llm_integration/                   # Q&A system components
│   │   ├── qa_system.py                   # Main orchestrator
│   │   ├── kg_service.py                  # Knowledge graph interface
│   │   ├── ollama_client.py               # Ollama LLM client
│   │   ├── query_planner.py               # Query planning agent
│   │   ├── response_synthesizer.py        # Response generation
│   │   └── config/                        # Configuration files
│   ├── kg_builder.py                      # Knowledge graph construction
│   ├── parsers/                           # Data parsing modules
│   └── kg_builders/                       # Graph builder classes
├── llm_evaluation_for_gene_set_interpretation/  # Raw data
│   └── data/                              # All biological data sources
├── quality_control/                       # QC and validation
│   └── 1_build_and_save_kg/
│       └── saved_graphs/                  # Built knowledge graphs
├── validation/                            # Validation scripts
├── parser_tests/                          # Parser test suites
├── kg_testing/                           # Graph builder tests
├── environment.yml                        # Conda environment
├── CLAUDE.md                             # Development instructions
├── start_qa_system.sh                    # Startup script
└── test_ollama_integration.py            # Integration tests
```

### Key Component Files

#### 🧬 Knowledge Graph Components
- **`src/kg_builder.py`**: Main knowledge graph construction
- **`src/parsers/`**: Modular data parsing (GO, Omics, LLM data)
- **`quality_control/1_build_and_save_kg/saved_graphs/biomedical_graph.gpickle`**: Built knowledge graph

#### 🤖 LLM Integration Components
- **`src/llm_integration/qa_system.py`**: Main Q&A orchestrator
- **`src/llm_integration/kg_service.py`**: Knowledge graph query interface
- **`src/llm_integration/ollama_client.py`**: Ollama LLM client
- **`src/llm_integration/query_planner.py`**: Natural language query planning
- **`src/llm_integration/response_synthesizer.py`**: Evidence-based response generation

#### 📊 Data Sources
- **`llm_evaluation_for_gene_set_interpretation/data/`**: Complete biological datasets
  - `GO_BP/`, `GO_CC/`, `GO_MF/`: Gene Ontology data
  - `Omics_data/`: Disease, drug, viral association data
  - `GO_term_analysis/`: LLM interpretations and model comparisons
  - `remaining_data_files/`: GMT gene sets, L1000 perturbations, embeddings

#### 🧪 Testing and Validation
- **`test_ollama_integration.py`**: Complete integration testing
- **`validation/comprehensive_omics_validation.py`**: Primary validation script
- **`parser_tests/`**: Individual component testing
- **`quality_control/`**: Comprehensive QC framework

---

## 🎯 Sample Questions and Usage Examples

### Example Questions for the Q&A System

#### Basic Gene Queries
```
"What is the function of the TP53 gene?"
"Which genes are associated with breast cancer?"
"What pathways involve insulin signaling?"
```

#### Disease and Drug Interactions  
```
"How does aspirin interact with genetic factors?"
"Which genes show significant expression changes in viral infections?"
"What diseases are connected to highly expressed genes?"
```

#### Advanced Multi-Modal Queries
```
"How do LLM interpretations compare with experimental data for TP53?"
"What gene sets show coordinated expression patterns in viral conditions?"
"Which cellular components contain the most highly connected proteins?"
```

### Interactive Commands

Once in the Q&A system:
- `help` - Show available commands
- `stats` - Display system performance statistics
- `config` - Show current configuration
- `clear` - Clear conversation history
- `quit` - Exit the system

---

## 🔍 Testing and Validation Procedures

### 1. Pre-Flight Checks

```bash
# Check environment
conda list | grep -E "(networkx|pandas|numpy|scikit-learn)"

# Check Ollama
ollama list

# Check knowledge graph
ls -la quality_control/1_build_and_save_kg/saved_graphs/biomedical_graph.gpickle
```

### 2. Component Testing

```bash
# Test knowledge graph service
python -c "
from src.llm_integration.kg_service import KnowledgeGraphService
kg = KnowledgeGraphService('quality_control/1_build_and_save_kg/saved_graphs/biomedical_graph.gpickle')
print(f'KG loaded: {kg.get_graph_stats()}')
"

# Test Ollama connectivity
python -c "
from src.llm_integration.ollama_client import OllamaClient
client = OllamaClient()
print(f'Ollama status: {client.get_client_stats()}')
"
```

### 3. Integration Testing

```bash
# Run complete integration test
python test_ollama_integration.py

# Expected results:
# - Knowledge Graph Service: ✅ PASSED
# - Ollama Client: ✅ PASSED  
# - Query Planning Agent: ✅ PASSED
# - Response Synthesizer: ✅ PASSED
# - Complete Q&A System: ✅ PASSED
```

### 4. Performance Validation

```bash
# Run knowledge graph validation
python validation/comprehensive_omics_validation.py

# Check for:
# - 7/7 validation criteria passed
# - ~37 second construction time
# - 90K+ nodes, 3M+ edges
# - 89%+ gene integration rate
```

### 5. Quality Control

```bash
# Run comprehensive QC framework
cd quality_control
python run_comprehensive_qc.py

# This tests:
# - Structural integrity
# - Data quality
# - Functional correctness
# - Integration quality
# - Semantic validation
# - Performance benchmarks
```

---

## 🛠 Troubleshooting Common Issues

### Issue 1: Ollama Connection Failed
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama service
ollama serve &

# Check available models
ollama list

# If no models, download one
ollama pull llama3.2:1b
```

### Issue 2: Knowledge Graph Not Found
```bash
# Check if graph exists
ls -la quality_control/1_build_and_save_kg/saved_graphs/

# If missing, build it
cd quality_control/1_build_and_save_kg/
python build_and_save_kg.py
```

### Issue 3: Import Errors
```bash
# Check if in correct environment
conda info --envs

# Activate correct environment
conda activate knowledge_graph

# Verify Python path
python -c "import sys; print(sys.path)"
```

### Issue 4: Memory Issues
```bash
# Check memory usage
free -h

# Reduce cache size in config
nano src/llm_integration/config/qa_system_config.yaml
# Set knowledge_graph.cache_size: 500
```

### Issue 5: Slow Performance
```bash
# Check if using SSD for graph file
# Move graph to SSD if needed
mv quality_control/1_build_and_save_kg/saved_graphs/ /path/to/ssd/

# Update config with new path
nano src/llm_integration/config/qa_system_config.yaml
```

---

## 🎯 Quick Start Summary

For immediate setup and testing:

```bash
# 1. Setup environment
cd /home/mreddy1/knowledge_graph
conda activate knowledge_graph

# 2. Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull llama3.2:1b

# 3. Test integration
python test_ollama_integration.py

# 4. Start Q&A system
./start_qa_system.sh
```

---

## 📊 System Performance Expectations

### Startup Performance
- **Environment Activation**: <2 seconds
- **Knowledge Graph Loading**: ~14 seconds
- **Ollama Model Loading**: ~5-10 seconds
- **Total System Startup**: ~20-25 seconds

### Query Performance
- **Simple Gene Queries**: <5 seconds
- **Complex Multi-Modal Queries**: <15 seconds
- **Response Generation**: <10 seconds
- **Total Question Processing**: <30 seconds

### Resource Usage
- **Memory**: 4-6GB during operation
- **Storage**: ~2GB for graph data
- **CPU**: Moderate during query processing

---

## 🎉 Success Criteria

Your system is properly set up when:

1. ✅ **Environment**: Conda environment activates successfully
2. ✅ **Ollama**: Can connect to Ollama and generate responses  
3. ✅ **Knowledge Graph**: Graph loads with 174K+ nodes and 5M+ edges
4. ✅ **Integration Tests**: All 5 integration tests pass
5. ✅ **Q&A System**: Can ask questions and receive evidence-based answers
6. ✅ **Validation**: System passes 7/7 validation criteria

Once these criteria are met, you have a fully functional biomedical knowledge graph Q&A system ready for interactive use!

---

## 📞 Support and Resources

### Documentation Files
- **`CLAUDE.md`**: Development instructions and system overview
- **`OLLAMA_LLM_INTEGRATION_PLAN.md`**: Detailed integration architecture
- **`SAMPLE_QUESTIONS_FOR_KG.md`**: Example questions and expected outputs

### Log Files
- **`qa_system.log`**: Q&A system operational logs
- **`~/ollama-local/server.log`**: Ollama server logs
- **`quality_control/comprehensive_qc_log.txt`**: Quality control logs

### Test and Validation
- **`test_ollama_integration.py`**: Primary integration test
- **`validation/comprehensive_omics_validation.py`**: System validation
- **`quality_control/run_comprehensive_qc.py`**: Quality control framework

This comprehensive setup guide should enable you to successfully deploy and use the biomedical knowledge graph Q&A system with Ollama integration.