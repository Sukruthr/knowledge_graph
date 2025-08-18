# GO_BP Knowledge Graph

A comprehensive knowledge graph system for Gene Ontology Biological Process (GO_BP) data, designed for gene set interpretation and biological analysis.

## Features

- **Comprehensive Data Parsing**: Supports all 9 GO_BP file formats including GAF, OBO, and collapsed_go files
- **Rich Knowledge Graph**: 66K+ nodes, 520K+ edges with GO terms, genes, and cross-references  
- **Multiple Identifier Support**: Gene symbols, Entrez IDs, and UniProt IDs with cross-references
- **Advanced Querying**: Gene function lookup, GO term search, hierarchy traversal
- **Data Validation**: Built-in integrity checking and semantic validation
- **Flexible Backend**: NetworkX for analysis, Neo4j-ready for production

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd knowledge_graph
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate knowledge_graph
```

### Basic Usage

```python
from src.kg_builder import GOBPKnowledgeGraph

# Initialize and build knowledge graph
kg = GOBPKnowledgeGraph()
kg.load_data("path/to/GO_BP/data")
kg.build_graph()

# Query gene functions
functions = kg.query_gene_functions("TP53")
print(f"TP53 has {len(functions)} GO annotations")

# Search GO terms
dna_repair_terms = kg.search_go_terms_by_definition("DNA repair")
print(f"Found {len(dna_repair_terms)} DNA repair related terms")

# Get comprehensive statistics
stats = kg.get_stats()
print(f"Knowledge graph contains {stats['total_nodes']:,} nodes and {stats['total_edges']:,} edges")
```

## Project Structure

```
knowledge_graph/
├── src/                    # Source code
│   ├── data_parsers.py     # Comprehensive GO_BP data parsing
│   └── kg_builder.py       # Knowledge graph construction
├── tests/                  # Test suite
├── examples/               # Usage examples
├── docs/                   # Documentation
├── validation/             # Validation scripts
├── reports/                # Analysis reports
├── data/                   # Generated knowledge graphs
└── environment.yml         # Conda environment
```

## Capabilities

### Data Sources Supported
- **GO Terms**: 29,602 biological process terms with rich metadata
- **GO Relationships**: 63,195 hierarchical relationships (is_a, part_of, regulates)
- **GO Clustering**: 27,733 clustering relationships from collapsed_go files
- **Gene Associations**: 408,135 gene-GO annotations from GAF and collapsed files
- **Gene Cross-References**: 19,861 identifier mappings (Symbol ↔ Entrez ↔ UniProt)
- **Alternative IDs**: 1,434 obsolete/alternative GO ID mappings
- **OBO Enhancement**: Rich definitions and synonyms for 27,473 terms

### Query Capabilities
- **Gene Function Analysis**: Find all GO terms associated with a gene
- **GO Term Search**: Search by definition, name, or synonyms with relevance scoring
- **Hierarchy Traversal**: Navigate parent-child relationships in GO hierarchy
- **Cross-Reference Lookup**: Map between different gene identifier systems
- **Alternative ID Resolution**: Handle obsolete GO IDs automatically

### Validation & Quality
- **100% Validation Success Rate**: Comprehensive integrity checking
- **Semantic Validation**: Ensures biological relationships are meaningful
- **Cross-File Consistency**: Validates references between data sources
- **Graph Integrity**: Checks node and edge validity

## Examples

See `examples/basic_usage.py` for a complete working example.

### Advanced Queries

```python
# Find genes involved in DNA repair
dna_repair_terms = kg.search_go_terms_by_definition("DNA repair")
repair_genes = set()
for term in dna_repair_terms:
    genes = kg.query_go_term_genes(term['go_id'])
    repair_genes.update(gene['gene_symbol'] for gene in genes)

print(f"Found {len(repair_genes)} genes involved in DNA repair")

# Explore GO hierarchy
go_term = "GO:0006281"  # DNA repair
parents = kg.query_go_hierarchy(go_term, 'parents')
children = kg.query_go_hierarchy(go_term, 'children')

print(f"GO:{go_term} has {len(parents)} parent terms and {len(children)} child terms")

# Get cross-references for a gene
cross_refs = kg.get_gene_cross_references("BRCA1")
print(f"BRCA1 identifiers: {cross_refs}")
```

## Documentation

- **[API Reference](docs/API_REFERENCE.md)**: Complete API documentation
- **[Parser Verification Report](reports/PARSER_VERIFICATION_REPORT.md)**: Data parser validation results
- **[KG Builder Update Report](reports/KG_BUILDER_UPDATE_REPORT.md)**: Knowledge graph enhancements

## Testing & Validation

Run the test suite:
```bash
python -m pytest tests/
```

Run validation scripts:
```bash
python validation/comprehensive_kg_validation_fixed.py
```

## Performance

- **Loading Time**: ~30-60 seconds for complete GO_BP dataset
- **Memory Usage**: ~2-4 GB for full knowledge graph
- **Query Performance**: Sub-second response for most queries
- **Graph Size**: 66,397 nodes, 520,358 edges

## Requirements

- Python 3.8+
- NetworkX 3.x
- Pandas 2.x
- NumPy 1.x
- See `environment.yml` for complete dependencies

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this knowledge graph in your research, please cite:

```
GO_BP Knowledge Graph
A comprehensive knowledge graph for Gene Ontology Biological Process data
https://github.com/your-username/knowledge_graph
```