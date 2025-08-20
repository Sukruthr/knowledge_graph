# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

This is a comprehensive biomedical knowledge graph system that integrates Gene Ontology (GO) data with multi-modal Omics data. The system creates a unified knowledge graph containing 88K+ nodes and 3M+ edges for biomedical research and analysis.

### Core Architecture

The system has evolved through three phases:
1. **GO Multi-namespace Integration**: GO_BP, GO_CC, GO_MF ontologies
2. **Omics Data Integration**: Disease, drug, viral associations, and network clusters  
3. **Viral Expression Matrix**: Quantitative expression data with 1.6M+ edges

**Key Components:**
- `src/data_parsers.py`: Multi-modal data parsing (GO ontology + Omics data)
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
- **`CombinedBiomedicalParser`**: Top-level parser integrating GO + Omics data
  - **`CombinedGOParser`**: Multi-namespace GO parsing (BP, CC, MF)
    - **`GODataParser`**: Single-namespace GO parser  
  - **`OmicsDataParser`**: Multi-modal omics data parser
    - Disease/drug associations, viral responses, expression matrices, network clusters

### Knowledge Graph Builders
- **`ComprehensiveBiomedicalKnowledgeGraph`**: Full multi-modal system (current primary)
- **`CombinedGOKnowledgeGraph`**: GO-only system (legacy)
- **`GOBPKnowledgeGraph`**: Single-namespace system (legacy)

### Data Integration Flow
1. **GO Data**: GAF files → OBO files → Collapsed files → Gene associations + Ontology structure
2. **Omics Data**: 6 data files → Associations + Expression matrices → Multi-modal relationships
3. **Graph Construction**: Nodes (genes, GO terms, diseases, drugs, viral conditions) + Edges (annotations, associations, expressions)
4. **Cross-modal Integration**: Gene identifier mapping across all data types

### Key Validation Patterns
- Always run `validation/comprehensive_omics_validation.py` for full system testing
- System should achieve 7/7 validation criteria (100% pass rate)
- Expected performance: ~37 seconds construction, 1500+ queries/second
- Expected scale: 88K nodes, 3M+ edges with 89% gene integration

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

### Expression Data Processing
- Uses configurable expression thresholds (default: 0.5)
- Captures direction (up/down), magnitude, and quantitative values
- Creates `gene_viral_expression` edges with full expression metadata

## Current System Status

**Production Ready**: ✅ 7/7 validation criteria passed
- **Total Edges**: 3,011,416 (including 1.6M viral expression)
- **Gene Integration**: 89.3% across all data types
- **Performance**: <40s construction, >1500 queries/second
- **Completeness**: All requested Omics data integrated including viral expression matrix

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
- **Gene Queries**: System returns comprehensive profiles including GO annotations, disease associations, drug perturbations, and viral responses (both association and expression types)
- **Expression Data**: Distinguishes between viral response associations and quantitative expression data in query results