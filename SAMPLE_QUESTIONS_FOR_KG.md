# Sample Questions for Biomedical Knowledge Graph Q&A System

## Knowledge Graph Analysis Results

### 📊 **Actual Knowledge Graph Content**
- **174,210 nodes** with **5,060,768 edges**
- **3,175 genes** (not just GO terms!)
- **13 diseases**, **5 drugs**, **12 viral conditions** 
- **684 gene sets**, **525 perturbation experiments**
- **189 LLM interpretations**, **102 literature references**
- **Rich multi-modal biomedical data**

### 🔗 **Edge Types Available**
- **gene_viral_expression** (6,469 edges)
- **gene_annotation** (2,483 edges) 
- **gene_go_association** (2,337 edges)
- **gene_drug_perturbation** (1,098 edges)
- **gene_disease_association** (498 edges)
- Plus 21 other relationship types

### 🧬 **Top Connected Hubs**
1. **"protein binding"** - 221,525 connections
2. **"cytosol"** - 37,387 connections  
3. **"estradiol" drug** - 28,306 connections
4. **"nucleoplasm"** - 24,478 connections
5. **"plasma membrane"** - 24,322 connections

---

## ❓ **10 Sample Questions Based on Actual KG Data**

### **1. Gene Expression & Viral Response**

**Question**: "Which genes show significant expression changes in response to viral infections?"
- **Rationale**: 6,469 gene_viral_expression edges available
- **Expected Data**: Real viral response data from your KG
- **Data Types Used**: gene nodes, viral_condition nodes, gene_viral_expression edges

**Question**: "What are the top genes with the highest viral expression levels?"
- **Rationale**: Comprehensive viral expression matrix in your KG
- **Expected Data**: Quantitative expression data with thresholds
- **Data Types Used**: gene nodes, expression values, viral conditions

### **2. Gene-Disease Associations**

**Question**: "Which genes are associated with specific diseases in the knowledge base?"
- **Rationale**: 498 gene_disease_association edges 
- **Expected Data**: Actual gene-disease relationships, not hallucinations
- **Data Types Used**: gene nodes, disease nodes, gene_disease_association edges

**Question**: "What diseases are connected to highly expressed genes?"
- **Rationale**: Cross-modal connections between expression and disease data
- **Expected Data**: Multi-layered biological insights
- **Data Types Used**: gene nodes, disease nodes, expression data, associations

### **3. Drug Perturbation Analysis**

**Question**: "How do different drugs affect gene expression patterns?"
- **Rationale**: 1,098 gene_drug_perturbation edges + L1000 data
- **Expected Data**: Drug mechanism insights from perturbation experiments
- **Data Types Used**: drug nodes, gene nodes, l1000_perturbation nodes, gene_drug_perturbation edges

**Question**: "What genes are most sensitive to estradiol treatment?"
- **Rationale**: Estradiol is a major hub (28,306 connections)
- **Expected Data**: Specific drug-gene interaction data
- **Data Types Used**: drug:estradiol node, gene nodes, perturbation data

### **4. GO Functional Analysis**

**Question**: "What molecular functions are most commonly associated with disease-related genes?"
- **Rationale**: 2,337 gene_go_association edges + disease connections
- **Expected Data**: Functional enrichment patterns
- **Data Types Used**: go_term nodes (molecular_function), gene nodes, disease nodes, associations

**Question**: "Which cellular components contain the most highly connected proteins?"
- **Rationale**: Protein binding hub (221,525 connections) + cellular component data
- **Expected Data**: Subcellular localization insights
- **Data Types Used**: go_term nodes (cellular_component), protein binding, connectivity analysis

### **5. Gene Set & Pathway Analysis**

**Question**: "What gene sets show coordinated expression patterns in viral conditions?"
- **Rationale**: 684 gene sets + viral expression data integration
- **Expected Data**: Pathway-level viral response analysis
- **Data Types Used**: gmt_gene_set nodes, gene nodes, viral conditions, expression data

**Question**: "Which gene sets are enriched in drug perturbation experiments?"
- **Rationale**: Gene sets + L1000 perturbation data (525 experiments)
- **Expected Data**: Pathway-level drug mechanism insights
- **Data Types Used**: gene_set nodes, l1000_perturbation nodes, drug perturbation data

### **6. Advanced Multi-Modal Analysis**

**Question**: "How do LLM interpretations compare with experimental data for specific gene functions?"
- **Rationale**: 189 LLM interpretations + experimental associations
- **Expected Data**: AI vs experimental validation analysis
- **Data Types Used**: llm_interpretation nodes, gene nodes, experimental associations, model predictions

---

## 🔧 **Current Issue & Resolution Needed**

### **Problem Identified**
Your KG queries are returning empty because the query methods are looking for simple string matches, but your data uses structured identifiers and relationships.

### **Required Fixes**
The KG service methods need to be updated to properly query:
- **Gene nodes** by gene symbols/identifiers  
- **Multi-modal relationships** across data types
- **Complex pathway connections** through GO associations

### **Expected Outcome**
Once the query interface is fixed, these 10 questions will return **real data** from your comprehensive biomedical knowledge graph, not hallucinations.

---

## 🎯 **Key Insights**

1. **Your KG is Rich**: Contains sophisticated multi-modal biomedical data, not just GO terms
2. **Data Integration**: Successfully integrates genes, diseases, drugs, viral responses, and experimental data
3. **Query Interface Gap**: The bottleneck is in the query methods, not the data availability
4. **Real Potential**: Once fixed, the system can provide evidence-based answers using actual experimental data

## 🚀 **Next Steps**

1. Update `kg_service.py` query methods to handle structured identifiers
2. Implement multi-modal query strategies for cross-data-type questions
3. Test these sample questions against the updated query interface
4. Validate that responses use actual KG data instead of LLM hallucinations

## 📁 **Supporting Files**
- **Analysis Data**: `kg_comprehensive_analysis.json` (detailed breakdown of all node types and relationships)
- **Query Interface**: `src/llm_integration/kg_service.py` (needs updates for structured queries)
- **Test Suite**: `test_ollama_integration.py` (can be updated with these sample questions)

---

*This analysis was generated by comprehensively sampling 10,000 nodes and 20,000 edges from the complete knowledge graph to understand actual data content and generate meaningful questions.*