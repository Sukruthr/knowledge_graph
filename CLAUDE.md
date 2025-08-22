# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

This is a comprehensive biomedical knowledge graph system that integrates Gene Ontology (GO) data with multi-modal Omics data, LLM model comparison analysis, multi-model LLM interpretation data, and remaining high-value datasets. The system creates a unified knowledge graph containing 120K+ nodes and 3.7M+ edges for biomedical research and AI model evaluation.

### Core Architecture

The system has evolved through nine phases:
1. **GO Multi-namespace Integration**: GO_BP, GO_CC, GO_MF ontologies
2. **Omics Data Integration**: Disease, drug, viral associations, and network clusters  
3. **Viral Expression Matrix**: Quantitative expression data with 1.6M+ edges
4. **Model Comparison Integration**: LLM evaluation data with 5 models and confidence scoring
5. **CC_MF_Branch Integration**: Enhanced CC/MF GO terms with LLM interpretations and similarity rankings
6. **LLM_processed Integration**: Multi-model LLM analysis with 8 models, contamination robustness, and similarity rankings
7. **GO Analysis Data Integration**: Core GO datasets, contamination studies, enrichment analysis, and human confidence evaluations
8. **Remaining Data Integration**: GMT gene sets (1.3M+ associations), literature evaluation (1.8K references), L1000 perturbations (10K experiments), GO embeddings (12K vectors), and supplementary LLM evaluations
9. **Talisman Gene Sets Integration**: HALLMARK pathways (49 sets), bicluster gene sets (3 sets), custom pathways, GO custom sets, disease sets, and specialized gene functions (71 total)

**Key Components:**
- `src/data_parsers.py`: Multi-modal data parsing (GO ontology + Omics data + Model comparison + CC_MF_Branch + LLM_processed + GO_Analysis_Data + Remaining_Data + Talisman_Gene_Sets)
- `src/model_compare_parser.py`: LLM model comparison and evaluation parsing
- `src/cc_mf_branch_parser.py`: CC/MF GO terms with LLM interpretations and similarity rankings
- `src/llm_processed_parser.py`: Multi-model LLM interpretations with contamination analysis and similarity rankings
- `src/go_analysis_data_parser.py`: Core GO datasets, contamination studies, enrichment analysis, and human evaluations
- `src/remaining_data_parser.py`: GMT gene sets, literature evaluation, L1000 perturbations, GO embeddings, and supplementary evaluations
- `src/talisman_gene_sets_parser.py`: HALLMARK pathways, bicluster gene sets, custom pathways, GO custom sets, disease sets, and specialized functions
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

# Test Talisman Gene Sets integration specifically
python test_talisman_gene_sets_simple.py

# Test parser reorganization 
python test_parser_reorganization.py

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

# Alternative: Import parsers directly (new organized structure)
from parsers import CombinedBiomedicalParser, GODataParser, OmicsDataParser

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

### Reorganized Parser Structure (Clean & Modular)
```
src/parsers/
├── __init__.py                    # Clean imports and backward compatibility
├── parser_utils.py                # Common utilities (file loading, validation, gene cleaning)
├── core_parsers.py                # Core GO & Omics parsers
│   ├── GODataParser              # Single-namespace GO parser (BP, CC, MF)
│   ├── OmicsDataParser           # Multi-modal omics data parser
│   └── CombinedGOParser          # Multi-namespace GO coordinator
├── parser_orchestrator.py         # Clean orchestration without messy imports
│   └── CombinedBiomedicalParser  # Top-level parser with clean dependency management
└── specialized parsers/           # Individual data type parsers
    ├── model_compare_parser.py    # LLM model comparison and evaluation
    ├── cc_mf_branch_parser.py     # CC/MF GO terms with enhanced LLM analysis
    ├── llm_processed_parser.py    # Multi-model LLM analysis with 8 models
    ├── go_analysis_data_parser.py # Core GO datasets and enrichment analysis
    ├── remaining_data_parser.py   # GMT files, L1000, embeddings, references
    └── talisman_gene_sets_parser.py # HALLMARK pathways and gene sets
```

### Parser Organization Benefits
- **Single Responsibility**: Each parser handles one clear data type
- **Clean Imports**: No more try/except import blocks
- **Shared Utilities**: Common functionality in `parser_utils.py`
- **Logical Grouping**: Core parsers separate from specialized parsers
- **Easy Extension**: Simple to add new parsers following established patterns
- **Backward Compatibility**: All existing code continues to work

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

**Production Ready**: ✅ Phase 9 Complete - Talisman Gene Sets Integration
- **Total Nodes**: 135,000+ (including talisman gene sets: 49 HALLMARK pathways, 3 bicluster sets, 2 custom pathways, 2 GO custom sets, 4 disease sets, 11 specialized sets)
- **Total Edges**: 3,800,000+ (including 8K+ new talisman gene-set associations, HALLMARK pathway relationships, bicluster connections)
- **Gene Integration**: 93%+ across all data types, 4,928 unique genes from talisman data, plus existing 17,023 from GMT and 15,558 from GO analysis
- **Performance**: Optimized construction and query performance with <0.3s talisman parsing, <0.03s KG integration
- **Completeness**: All high-value data integrated (GO + Omics + Model comparison + CC_MF_Branch + LLM_processed + GO_Analysis_Data + Remaining_Data + Talisman_Gene_Sets)

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