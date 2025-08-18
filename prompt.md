**Prompt:**

You are an expert bioinformatician and data scientist specializing in knowledge graphs (KGs) and semantic modeling. I am a developer with no background in biology or KG construction. My goal is to build a knowledge graph from two specific data directories and then learn how to query it and evaluate its quality.

Your task is to act as my guide. Please provide a detailed, step-by-step project plan that I can follow. Assume I know nothing about the biological concepts involved.

---

#### **Primary Objective**

To construct a Knowledge Graph that models the relationships between genes, gene sets, biological pathways, and functions. The ultimate goal is to use this KG to perform gene set interpretation—that is, to understand the biological significance of a given set of genes.

---

#### **Data Sources**

The data is located in two directories. Since you cannot access my file system, I will describe the likely structure and content based on the directory names. Please base your plan on this description:

**Directory 1:** `/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data`

**Directory 2:** `/home/mreddy1/knowledge_graph/talisman-paper/genesets/human`


#### **Required Project Plan**

Please structure your response into the following four phases. For each phase, explain the concepts, propose specific actions, and suggest tools where appropriate. Remember to define biological jargon using simple analogies.

**Phase 1: KG Schema Design (The Blueprint)**
1.  **Identify Entities (Nodes):** Based on the data sources, what are the primary "things" or concepts we should represent? (e.g., Gene, Gene Set, Pathway). Please list them out.
2.  **Identify Relationships (Edges):** How do these entities connect to each other? (e.g., a Gene `IS_PART_OF` a Gene Set). Please list the relationships and specify their direction (e.g., `(Gene) -> [IS_PART_OF] -> (GeneSet)`).
3.  **Define Properties:** What attributes or metadata should each node and relationship have? (e.g., a `Gene` node might have a `symbol` property like "TP53" and a `full_name` property).
4.  **Visualize:** Provide a simple text-based or Mermaid diagram of this proposed schema.

**Phase 2: KG Construction Plan (The Build)**
1.  **Technology Stack:** Recommend a beginner-friendly technology stack. For example, should I use a graph database like **Neo4j**, an RDF triple store, or just Python libraries like `NetworkX`? Explain the pros and cons of your recommendation.
2.  **Data Ingestion Pipeline:** Provide a step-by-step process for parsing the `.csv`, `.tsv`, and `.gmt` files and loading them into the chosen technology, following the schema from Phase 1. Provide pseudocode or Python snippets for key steps.

**Phase 3: Information Extraction (Using the KG)**
1.  **Formulate Sample Queries:** Propose 3-5 sample questions we could ask our KG. These should range from simple to complex to demonstrate the KG's utility.
    * *Example (Simple):* "Which genes are in the 'HALLMARK_APOPTOSIS' gene set?"
    * *Example (Complex):* "Find all biological pathways that share at least 5 genes with the 'apoptotic process' (GO:0006915) gene set."
2.  **Query Language:** For your recommended technology (e.g., Neo4j), provide the actual query code (e.g., Cypher) for these sample questions.

**Phase 4: Evaluation Framework (Quality Control)**
1.  **Defining "Good":** How do we determine if our KG is "good"? Propose 2-3 specific criteria for evaluation. For example:
    * **Correctness:** Are the facts in the KG accurate?
    * **Completeness:** How much of the domain knowledge does our KG capture?
    * **Utility:** Can the KG effectively answer the types of questions we care about?
2.  **Evaluation Methods:** For each criterion, describe a method to measure it.
    * For **Correctness**, you might suggest manually checking a random sample of facts against a trusted source like NCBI or GO Consortium.
    * For **Utility**, you could propose a set of "benchmark questions" and measure how easily and accurately the KG can answer them.
    * Introduce the concept of **Link Prediction** as an automated method for assessing completeness and coherence.

Please present this as a comprehensive project plan. Use Markdown for clear formatting. I will review this plan and give you the go-ahead before we proceed.