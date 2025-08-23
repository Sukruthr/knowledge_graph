# API Reference

## GOBPDataParser

### Overview
The `GOBPDataParser` class provides comprehensive parsing capabilities for Gene Ontology Biological Process (GO_BP) data files.

### Initialization
```python
from src.data_parsers import GOBPDataParser

parser = GOBPDataParser(data_dir)
```

**Parameters:**
- `data_dir` (str): Path to GO_BP data directory containing all required files

### Core Methods

#### `parse_go_terms() -> Dict[str, Dict]`
Parse GO term definitions from goID_2_name.tab and goID_2_namespace.tab.

**Returns:** Dictionary mapping GO IDs to term information

#### `parse_go_relationships() -> List[Dict]`
Parse GO term relationships from go.tab.

**Returns:** List of relationship dictionaries with parent_id, child_id, relationship_type

#### `parse_gene_go_associations_from_gaf() -> List[Dict]`
Parse gene-GO associations from GAF (Gene Association File) format.

**Returns:** List of gene-GO association dictionaries

#### `parse_collapsed_go_file(identifier_type: str = 'symbol') -> Dict`
Parse collapsed_go files containing both GO clustering and gene associations.

**Parameters:**
- `identifier_type` (str): Type of file to parse ('symbol', 'entrez', 'uniprot')

**Returns:** Dictionary with 'clusters' and 'gene_associations' keys

#### `parse_go_alternative_ids() -> Dict[str, str]`
Parse GO alternative/obsolete ID mappings from goID_2_alt_id.tab.

**Returns:** Dictionary mapping alternative GO IDs to primary GO IDs

#### `parse_gene_identifier_mappings() -> Dict[str, Dict]`
Parse comprehensive gene identifier mappings from multiple sources.

**Returns:** Dictionary with mappings between different gene identifier systems

#### `parse_obo_ontology() -> Dict[str, Dict]`
Parse rich GO ontology structure from OBO format file.

**Returns:** Dictionary with enhanced GO term information including definitions and synonyms

#### `validate_parsed_data() -> Dict[str, bool]`
Validate the integrity of parsed data.

**Returns:** Dictionary with validation results

---

## GOBPKnowledgeGraph

### Overview
The `GOBPKnowledgeGraph` class builds and manages a comprehensive knowledge graph from GO_BP data.

### Initialization
```python
from src.kg_builder import GOBPKnowledgeGraph

kg = GOBPKnowledgeGraph(use_neo4j=False)
```

**Parameters:**
- `use_neo4j` (bool): Whether to use Neo4j database or NetworkX (default: False)

### Core Methods

#### `load_data(data_dir: str)`
Load and parse GO_BP data comprehensively.

**Parameters:**
- `data_dir` (str): Path to GO_BP data directory

#### `build_graph()`
Build the comprehensive knowledge graph from parsed data.

#### `get_stats() -> Dict`
Get comprehensive graph statistics.

**Returns:** Dictionary with detailed graph metrics

#### `save_graph(filepath: str)`
Save the NetworkX graph to disk.

**Parameters:**
- `filepath` (str): Path to save the graph (.pkl or .graphml)

#### `load_graph(filepath: str)`
Load a NetworkX graph from disk.

**Parameters:**
- `filepath` (str): Path to load the graph from

### Query Methods

#### `query_gene_functions(gene_symbol: str) -> List[Dict]`
Query GO terms associated with a gene.

**Parameters:**
- `gene_symbol` (str): Gene symbol to query

**Returns:** List of GO terms with details

#### `query_go_term_genes(go_id: str) -> List[Dict]`
Query genes associated with a GO term.

**Parameters:**
- `go_id` (str): GO term ID to query

**Returns:** List of genes with details

#### `query_go_hierarchy(go_id: str, direction: str = 'children') -> List[Dict]`
Query GO term hierarchy.

**Parameters:**
- `go_id` (str): GO term ID to query
- `direction` (str): 'children' for child terms, 'parents' for parent terms

**Returns:** List of related GO terms

#### `search_go_terms_by_definition(search_term: str) -> List[Dict]`
Search GO terms by definition or synonym using enhanced OBO data.

**Parameters:**
- `search_term` (str): Term to search for in definitions and synonyms

**Returns:** List of matching GO terms with relevance score

#### `resolve_alternative_go_id(go_id: str) -> str`
Resolve alternative/obsolete GO ID to primary ID.

**Parameters:**
- `go_id` (str): GO ID to resolve

**Returns:** Primary GO ID or original ID if no mapping exists

#### `get_gene_cross_references(gene_symbol: str) -> Dict`
Get cross-reference information for a gene.

**Parameters:**
- `gene_symbol` (str): Gene symbol to look up

**Returns:** Dictionary with cross-reference IDs

#### `validate_graph_integrity() -> Dict[str, bool]`
Validate the integrity and consistency of the knowledge graph.

**Returns:** Dictionary with validation results

### Graph Statistics

The `get_stats()` method returns a dictionary with the following keys:

- `total_nodes`: Total number of nodes in the graph
- `total_edges`: Total number of edges in the graph
- `go_terms`: Number of GO term nodes
- `genes`: Number of gene nodes
- `gene_identifiers`: Number of gene identifier nodes
- `enhanced_go_terms`: Number of GO terms with OBO enhancement
- `alternative_go_ids`: Number of alternative GO ID mappings
- `go_relationships`: Number of GO-GO hierarchical relationships
- `go_clusters`: Number of GO clustering relationships
- `gene_associations`: Total number of gene-GO associations
- `gaf_associations`: Number of GAF-sourced associations
- `collapsed_associations`: Number of collapsed file associations
- `gene_cross_references`: Number of gene cross-reference edges
- `alternative_id_mappings`: Number of alternative GO ID mappings
- `total_gene_id_mappings`: Total gene identifier mappings

### Node Types

The knowledge graph contains the following node types:

1. **GO Terms** (`node_type='go_term'`)
   - Attributes: name, namespace, definition, synonyms, is_obsolete
   
2. **Genes** (`node_type='gene'`)
   - Attributes: gene_symbol, uniprot_id, gene_name, gene_type, taxon, entrez_id
   
3. **Gene Identifiers** (`node_type='gene_identifier'`)
   - Attributes: identifier_type, entrez_id/uniprot_id, gene_symbol

### Edge Types

The knowledge graph contains the following edge types:

1. **GO Hierarchy** (`edge_type='go_hierarchy'`)
   - Parent-child relationships between GO terms
   - Attributes: relationship_type (is_a, part_of, regulates, etc.)

2. **GO Clustering** (`edge_type='go_clustering'`)
   - Clustering relationships from collapsed_go files
   - Attributes: cluster_type, relationship_type

3. **Gene Annotation** (`edge_type='gene_annotation'`)
   - Gene-GO associations
   - Attributes: source, evidence_code, qualifier, assigned_by, date

4. **Gene Cross-Reference** (`edge_type='gene_cross_reference'`)
   - Cross-references between different gene identifier types
   - Attributes: reference_type

5. **Alternative ID Mapping** (`edge_type='alternative_id_mapping'`)
   - Mappings from alternative to primary GO IDs
   - Attributes: relationship_type


EXCELLENT! ALL TESTS PASSED!

  The end-to-end integration test was successful! Here's what we achieved:

  🎯 Complete End-to-End Test Results: 5/5 PASSED (100%)

  ✅ Successfully Tested Components:

  1. Knowledge Graph Service: ✅ PASSED
    - Loaded 174,210 nodes and 5,060,768 edges
    - Gene queries working (TP53 found)
    - Keyword searches working (50 cancer/tumor matches)
  2. Ollama Client: ✅ PASSED
    - Connected to local Ollama server
    - Model llama3.2:1b successfully loaded
    - Response generation working (2,130 character response)
  3. Query Planning Agent: ✅ PASSED
    - Successfully parsed "What is the function of TP53 gene?"
    - Classified intent as gene_function
    - Extracted entities: TP53, cancer
    - Generated 4-step query plan
  4. Response Synthesizer: ✅ PASSED
    - Synthesized response successfully
    - Generated 2,475 character answer
    - Produced 5 follow-up questions
    - Evidence sources processed
  5. Complete Q&A System: ✅ PASSED
    - Full system initialization successful
    - Interactive workflow functional
    - End-to-end question processing working

  🚀 The System is Now Fully Operational!

  You can now run the interactive Q&A system using:

  source /opt/miniconda3/etc/profile.d/conda.sh && conda activate knowledge_graph
  export PATH=~/ollama-local/bin:$PATH
  python -m src.llm_integration.qa_system

  The system successfully demonstrates:
  - ✅ Real-time biomedical question answering
  - ✅ Knowledge graph integration (174K+ nodes, 5M+ edges)
  - ✅ Natural language processing with Ollama LLM
  - ✅ Evidence-based response synthesis
  - ✅ Follow-up question generation
  - ✅ Interactive conversation support

  This is a complete, working biomedical AI assistant powered by your knowledge graph and Ollama LLM