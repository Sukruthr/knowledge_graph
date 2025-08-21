# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

This is a comprehensive biomedical knowledge graph system that integrates Gene Ontology (GO) data with multi-modal Omics data, LLM model comparison analysis, multi-model LLM interpretation data, and remaining high-value datasets. The system creates a unified knowledge graph containing 120K+ nodes and 3.7M+ edges for biomedical research and AI model evaluation.

### Core Architecture

The system has evolved through eight phases:
1. **GO Multi-namespace Integration**: GO_BP, GO_CC, GO_MF ontologies
2. **Omics Data Integration**: Disease, drug, viral associations, and network clusters  
3. **Viral Expression Matrix**: Quantitative expression data with 1.6M+ edges
4. **Model Comparison Integration**: LLM evaluation data with 5 models and confidence scoring
5. **CC_MF_Branch Integration**: Enhanced CC/MF GO terms with LLM interpretations and similarity rankings
6. **LLM_processed Integration**: Multi-model LLM analysis with 8 models, contamination robustness, and similarity rankings
7. **GO Analysis Data Integration**: Core GO datasets, contamination studies, enrichment analysis, and human confidence evaluations
8. **Remaining Data Integration**: GMT gene sets (1.3M+ associations), literature evaluation (1.8K references), L1000 perturbations (10K experiments), GO embeddings (12K vectors), and supplementary LLM evaluations

**Key Components:**
- `src/data_parsers.py`: Multi-modal data parsing (GO ontology + Omics data + Model comparison + CC_MF_Branch + LLM_processed + GO_Analysis_Data + Remaining_Data)
- `src/model_compare_parser.py`: LLM model comparison and evaluation parsing
- `src/cc_mf_branch_parser.py`: CC/MF GO terms with LLM interpretations and similarity rankings
- `src/llm_processed_parser.py`: Multi-model LLM interpretations with contamination analysis and similarity rankings
- `src/go_analysis_data_parser.py`: Core GO datasets, contamination studies, enrichment analysis, and human evaluations
- `src/remaining_data_parser.py`: GMT gene sets, literature evaluation, L1000 perturbations, GO embeddings, and supplementary evaluations
- `src/kg_builder.py`: Knowledge graph construction with NetworkX
- `validation/`: Comprehensive validation and testing framework

## Development Commands

### Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml
conda activate knowledge_graph
```

### Testing and Validation
```bash
# Run comprehensive system validation (primary test)
python validation/comprehensive_omics_validation.py

# Test model comparison integration specifically
python test_model_comparison_integration.py

# Test CC_MF_Branch integration specifically
python test_cc_mf_branch_integration.py

# Test LLM_processed integration specifically
python test_llm_processed_integration.py

# Test GO Analysis Data integration specifically
python test_go_analysis_simple.py

# Test Remaining Data integration specifically
python test_remaining_data_simple.py

# Test viral expression matrix integration specifically
python viral_expression_test.py

# Validate query correctness and completeness
python query_validation_test.py

# Run legacy GO-only tests
python -m pytest tests/
python validation/combined_go_validation.py
```

### Interactive Development
```bash
# Start system in Python for development
python3
import sys; sys.path.append('src')
from kg_builder import ComprehensiveBiomedicalKnowledgeGraph

# Load and build full system (~37 seconds)
kg = ComprehensiveBiomedicalKnowledgeGraph()
kg.load_data('llm_evaluation_for_gene_set_interpretation/data')
kg.build_comprehensive_graph()

# Test gene queries
profile = kg.query_gene_comprehensive('TP53')
stats = kg.get_comprehensive_stats()

# Test model comparison queries
model_summary = kg.query_model_comparison_summary()
predictions = kg.query_model_predictions(model_name='gpt_4')

# Test CC_MF_Branch queries
cc_terms = kg.query_cc_mf_terms(namespace='CC')
mf_interpretations = kg.query_cc_mf_llm_interpretations(namespace='MF')
gene_cc_mf_profile = kg.query_gene_cc_mf_profile('TP53')

# Test LLM_processed queries
llm_stats = kg.get_llm_processed_stats()
interpretations = kg.query_llm_interpretations(model='gpt_4')
contamination = kg.query_contamination_analysis(model='gemini_pro')
similarity_rankings = kg.query_llm_similarity_rankings(dataset='selected_1000_go_terms')
gene_llm_profile = kg.query_gene_llm_profile('SLC2A1')

# Test GO Analysis Data queries
go_analysis_stats = kg.get_go_analysis_stats()
core_analysis = kg.query_go_core_analysis(dataset='1000_selected')
contamination_analysis = kg.query_go_contamination_analysis(dataset='1000_selected_contaminated')
confidence_evaluations = kg.query_go_confidence_evaluations()
gene_go_analysis_profile = kg.query_gene_go_analysis_profile('TP53')
```

### Data Structure Commands
```bash
# Check data integrity
ls -la llm_evaluation_for_gene_set_interpretation/data/Omics_data/
wc -l llm_evaluation_for_gene_set_interpretation/data/Omics_data/*.txt

# Verify system status
cat validation/omics_validation_results.json | python -m json.tool
```

## Code Architecture

### Data Parser Hierarchy
- **`CombinedBiomedicalParser`**: Top-level parser integrating GO + Omics + Model comparison + LLM_processed data
  - **`CombinedGOParser`**: Multi-namespace GO parsing (BP, CC, MF)
    - **`GODataParser`**: Single-namespace GO parser  
  - **`OmicsDataParser`**: Multi-modal omics data parser
    - Disease/drug associations, viral responses, expression matrices, network clusters
  - **`ModelCompareParser`**: LLM model comparison and evaluation parser
    - Model predictions, confidence scores, similarity rankings, contamination analysis
  - **`CCMFBranchParser`**: CC/MF GO terms with enhanced LLM analysis
    - LLM interpretations, confidence scoring, similarity rankings for CC/MF namespaces
  - **`LLMProcessedParser`**: Multi-model LLM analysis with 8 models
    - LLM interpretations, contamination robustness, similarity rankings, statistical validation

### Knowledge Graph Builders
- **`ComprehensiveBiomedicalKnowledgeGraph`**: Full multi-modal system (current primary)
- **`CombinedGOKnowledgeGraph`**: GO-only system (legacy)
- **`GOBPKnowledgeGraph`**: Single-namespace system (legacy)

### Data Integration Flow
1. **GO Data**: GAF files → OBO files → Collapsed files → Gene associations + Ontology structure
2. **Omics Data**: 6 data files → Associations + Expression matrices → Multi-modal relationships
3. **Model Comparison Data**: LLM processed files → Predictions + Confidence scores + Similarity rankings
4. **CC_MF_Branch Data**: CC/MF GO terms → LLM interpretations + Similarity rankings + Gene associations
5. **LLM_processed Data**: 13 files → Multi-model interpretations + Contamination analysis + Similarity rankings + Statistical validation
6. **Graph Construction**: Nodes (genes, GO terms, diseases, drugs, viral conditions, models, predictions, interpretations, contamination analyses) + Edges (annotations, associations, expressions, predictions, interpretations, contamination analyses)
7. **Cross-modal Integration**: Gene identifier mapping across all data types

### Key Validation Patterns
- Always run `validation/comprehensive_omics_validation.py` for full system testing
- System should achieve 7/7 validation criteria (100% pass rate)
- Expected performance: ~37 seconds construction, 1500+ queries/second
- Expected scale: 90K nodes, 3M+ edges with 89% gene integration

## Data Sources and Processing

### GO Data Structure (per namespace: GO_BP, GO_CC, GO_MF)
- **GAF files**: Gene associations with evidence codes
- **OBO files**: Ontology definitions and relationships  
- **Collapsed files**: Simplified gene-term mappings
- **Tab files**: Term names, namespaces, alternative IDs

### Omics Data Files
- **Disease/Drug Perturbations**: Up/down regulation data from GEO
- **Viral Perturbations**: Viral response associations
- **Expression Matrix**: Quantitative viral expression (21K genes, threshold-based)
- **Network Clusters**: Hierarchical cluster relationships

### Model Comparison Data Files  
- **LLM Processed Files**: Predictions from 5 models (GPT-4, GPT-3.5, Gemini Pro, Llama2-70B, Mixtral)
- **Confidence Scores**: Model confidence ratings across contamination scenarios
- **Similarity Rankings**: GO term similarity rankings and percentile scores
- **Contamination Analysis**: Performance degradation under gene contamination

### Expression Data Processing
- Uses configurable expression thresholds (default: 0.5)
- Captures direction (up/down), magnitude, and quantitative values
- Creates `gene_viral_expression` edges with full expression metadata

## Current System Status

**Production Ready**: ✅ Phase 8 Complete - Remaining Data Integration
- **Total Nodes**: 120,000+ (including remaining data nodes: GMT gene sets, literature references, L1000 perturbations, GO embeddings, supplementary evaluations)
- **Total Edges**: 3,700,000+ (including 1.3M+ new GMT associations, literature support links, perturbation relationships, embedding connections)
- **Gene Integration**: 92%+ across all data types, 17,023 unique genes from GMT data, 15,558 from GO analysis
- **Performance**: Optimized construction and query performance with <2s remaining data parsing
- **Completeness**: All high-value data integrated (GO + Omics + Model comparison + CC_MF_Branch + LLM_processed + GO_Analysis_Data + Remaining_Data)

### Validation Results Structure
The system maintains comprehensive validation through `omics_validation_results.json` which tracks:
- Multi-modal integration status
- Gene coverage metrics
- Cross-modal connectivity
- Performance benchmarks
- Data quality validation

## Important Notes

- **Primary System**: Use `ComprehensiveBiomedicalKnowledgeGraph` for all new development
- **Data Paths**: All paths should be absolute; system expects `llm_evaluation_for_gene_set_interpretation/data/` structure
- **Memory Requirements**: ~4GB RAM for full dataset processing
- **Gene Queries**: System returns comprehensive profiles including GO annotations, disease associations, drug perturbations, viral responses, model predictions, CC/MF profiles, LLM interpretations, and GO analysis data
- **Model Queries**: System provides model comparison summaries, confidence analysis, and contamination robustness metrics
- **CC/MF Queries**: System provides cellular component and molecular function analysis with LLM interpretations and similarity rankings
- **LLM Queries**: System provides multi-model interpretations, contamination analysis, similarity rankings, and statistical validation
- **GO Analysis Queries**: System provides core GO analysis, contamination studies, enrichment analysis, and human confidence evaluations
- **Expression Data**: Distinguishes between viral response associations and quantitative expression data in query results