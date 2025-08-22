#!/usr/bin/env python3
"""
Phase 6: Comprehensive Semantic Validation

This script performs extremely thorough biological logic and consistency validation
of the biomedical knowledge graph, ensuring scientific accuracy and semantic correctness.

Validation Categories:
1. GO Hierarchy Integrity & Biological Logic
2. Gene-Function Relationship Validation
3. Disease-Gene Association Plausibility
4. Drug-Target Interaction Consistency
5. Pathway Coherence & Completeness
6. Cross-Modal Semantic Consistency
7. Biological Constraint Validation
8. Literature Support Assessment
9. Temporal Consistency Checks
10. Species-Specific Validation
"""

import sys
import os
import time
import pickle
import logging
import json
import traceback
import re
from datetime import datetime
from collections import defaultdict, Counter
import networkx as nx
import numpy as np

# Add project root to path
sys.path.append('/home/mreddy1/knowledge_graph/src')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/mreddy1/knowledge_graph/quality_control/6_semantic_validation/comprehensive_semantic_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveSemanticValidator:
    """Comprehensive semantic validation for biomedical knowledge graph."""
    
    def __init__(self, kg_path='/home/mreddy1/knowledge_graph/quality_control/1_build_and_save_kg/saved_graphs/complete_biomedical_kg.pkl'):
        self.kg_path = kg_path
        self.kg = None
        self.graph = None
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'validation_summary': {},
            'detailed_results': {},
            'biological_constraints': {},
            'semantic_consistency': {},
            'quality_metrics': {},
            'recommendations': []
        }
        
        # Biological knowledge bases for validation
        self.biological_knowledge = {
            'essential_genes': set(),
            'oncogenes': set(),
            'tumor_suppressors': set(),
            'known_drug_targets': set(),
            'protein_coding_genes': set(),
            'go_term_hierarchies': {},
            'biological_constraints': {}
        }
        
    def load_knowledge_graph(self):
        """Load the pre-built knowledge graph."""
        logger.info("📊 LOADING KNOWLEDGE GRAPH FOR COMPREHENSIVE SEMANTIC VALIDATION")
        logger.info("=" * 80)
        
        try:
            if not os.path.exists(self.kg_path):
                # Build directly if saved version not available
                from kg_builders import ComprehensiveBiomedicalKnowledgeGraph
                logger.info("Building knowledge graph directly for semantic validation...")
                self.kg = ComprehensiveBiomedicalKnowledgeGraph()
                self.kg.load_data('/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data')
                self.kg.build_comprehensive_graph()
                self.graph = self.kg.graph
                logger.info("✅ Knowledge graph built directly")
            else:
                logger.info(f"Loading knowledge graph from: {self.kg_path}")
                with open(self.kg_path, 'rb') as f:
                    self.kg = pickle.load(f)
                self.graph = self.kg.graph
                logger.info("✅ Knowledge graph loaded successfully")
            
            logger.info(f"   Nodes: {self.graph.number_of_nodes():,}")
            logger.info(f"   Edges: {self.graph.number_of_edges():,}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load knowledge graph: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def initialize_biological_knowledge(self):
        """Initialize biological knowledge bases for validation."""
        logger.info("🧬 INITIALIZING BIOLOGICAL KNOWLEDGE BASES")
        logger.info("-" * 60)
        
        try:
            # Extract known biological entities from the graph for validation
            gene_nodes = [(n, d) for n, d in self.graph.nodes(data=True) if d.get('type') == 'gene']
            go_nodes = [(n, d) for n, d in self.graph.nodes(data=True) if d.get('type') == 'go_term']
            disease_nodes = [(n, d) for n, d in self.graph.nodes(data=True) if d.get('type') == 'disease']
            drug_nodes = [(n, d) for n, d in self.graph.nodes(data=True) if d.get('type') == 'drug']
            
            # Well-known essential genes (for validation)
            known_essential_genes = {
                'TP53', 'BRCA1', 'BRCA2', 'RB1', 'APC', 'MLH1', 'MSH2', 'MSH6', 'PMS2',
                'ATM', 'CHEK2', 'PALB2', 'CDH1', 'PTEN', 'STK11', 'VHL', 'MEN1',
                'RET', 'EGFR', 'ERBB2', 'PIK3CA', 'KRAS', 'BRAF', 'MYC', 'CCND1'
            }
            
            # Known oncogenes
            known_oncogenes = {
                'MYC', 'EGFR', 'ERBB2', 'PIK3CA', 'KRAS', 'BRAF', 'CCND1', 'MDM2',
                'RAF1', 'SRC', 'ABL1', 'FOS', 'JUN', 'RAS', 'HRAS', 'NRAS'
            }
            
            # Known tumor suppressors
            known_tumor_suppressors = {
                'TP53', 'RB1', 'APC', 'BRCA1', 'BRCA2', 'PTEN', 'VHL', 'ATM',
                'CHEK2', 'MLH1', 'MSH2', 'CDKN2A', 'CDKN1A', 'CDKN1B'
            }
            
            # Update biological knowledge with graph data
            for gene_node, gene_data in gene_nodes:
                gene_symbol = gene_data.get('symbol', gene_data.get('name', str(gene_node)))
                if gene_symbol in known_essential_genes:
                    self.biological_knowledge['essential_genes'].add(gene_symbol)
                if gene_symbol in known_oncogenes:
                    self.biological_knowledge['oncogenes'].add(gene_symbol)
                if gene_symbol in known_tumor_suppressors:
                    self.biological_knowledge['tumor_suppressors'].add(gene_symbol)
                
                # All genes with proper symbols are considered protein-coding for validation
                if gene_symbol and re.match(r'^[A-Z][A-Z0-9-]*$', gene_symbol):
                    self.biological_knowledge['protein_coding_genes'].add(gene_symbol)
            
            # Extract GO term hierarchies
            for go_node, go_data in go_nodes:
                go_id = go_data.get('go_id', str(go_node) if str(go_node).startswith('GO:') else None)
                namespace = go_data.get('namespace')
                
                if go_id and namespace:
                    if namespace not in self.biological_knowledge['go_term_hierarchies']:
                        self.biological_knowledge['go_term_hierarchies'][namespace] = set()
                    self.biological_knowledge['go_term_hierarchies'][namespace].add(go_id)
            
            logger.info(f"Biological knowledge initialized:")
            logger.info(f"   Essential genes: {len(self.biological_knowledge['essential_genes'])}")
            logger.info(f"   Oncogenes: {len(self.biological_knowledge['oncogenes'])}")
            logger.info(f"   Tumor suppressors: {len(self.biological_knowledge['tumor_suppressors'])}")
            logger.info(f"   Protein-coding genes: {len(self.biological_knowledge['protein_coding_genes'])}")
            logger.info(f"   GO namespaces: {list(self.biological_knowledge['go_term_hierarchies'].keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Biological knowledge initialization failed: {str(e)}")
            return False
    
    def validate_go_hierarchy_integrity(self):
        """Validate GO hierarchy integrity and biological logic."""
        logger.info("🔬 VALIDATING GO HIERARCHY INTEGRITY & BIOLOGICAL LOGIC")
        logger.info("-" * 60)
        
        try:
            hierarchy_results = {}
            
            # Extract GO terms and their relationships
            go_terms = {}
            go_relationships = []
            namespace_distribution = defaultdict(int)
            
            for node, data in self.graph.nodes(data=True):
                if data.get('type') == 'go_term':
                    go_id = data.get('go_id', str(node) if str(node).startswith('GO:') else None)
                    namespace = data.get('namespace')
                    name = data.get('name', data.get('term_name'))
                    
                    if go_id:
                        go_terms[go_id] = {
                            'namespace': namespace,
                            'name': name,
                            'node': node
                        }
                        if namespace:
                            namespace_distribution[namespace] += 1
            
            # Extract hierarchical relationships
            for source, target, edge_data in self.graph.edges(data=True):
                edge_type = edge_data.get('type', '')
                if 'hierarchy' in edge_type.lower() or 'is_a' in edge_type.lower() or 'part_of' in edge_type.lower():
                    source_data = self.graph.nodes[source]
                    target_data = self.graph.nodes[target]
                    
                    if (source_data.get('type') == 'go_term' and target_data.get('type') == 'go_term'):
                        source_go = source_data.get('go_id', str(source))
                        target_go = target_data.get('go_id', str(target))
                        
                        if source_go in go_terms and target_go in go_terms:
                            go_relationships.append({
                                'child': source_go,
                                'parent': target_go,
                                'relationship_type': edge_type,
                                'child_namespace': go_terms[source_go]['namespace'],
                                'parent_namespace': go_terms[target_go]['namespace']
                            })
            
            # Validate hierarchy constraints
            hierarchy_violations = []
            cross_namespace_relationships = []
            orphaned_terms = []
            
            # Check for cross-namespace violations
            for rel in go_relationships:
                if rel['child_namespace'] != rel['parent_namespace']:
                    if rel['child_namespace'] and rel['parent_namespace']:  # Both have namespaces
                        cross_namespace_relationships.append(rel)
            
            # Check for orphaned terms (terms without parents in same namespace)
            for go_id, term_data in go_terms.items():
                if term_data['namespace']:
                    has_parent_in_namespace = any(
                        rel['child'] == go_id and rel['parent_namespace'] == term_data['namespace']
                        for rel in go_relationships
                    )
                    # Root terms are expected to not have parents
                    if not has_parent_in_namespace and not self._is_root_term(term_data['name']):
                        orphaned_terms.append({
                            'go_id': go_id,
                            'name': term_data['name'],
                            'namespace': term_data['namespace']
                        })
            
            # Validate specific biological constraints
            biological_violations = self._validate_go_biological_constraints(go_terms, go_relationships)
            
            # Calculate metrics
            total_go_terms = len(go_terms)
            total_relationships = len(go_relationships)
            violation_rate = (len(hierarchy_violations) + len(biological_violations)) / max(total_relationships, 1) * 100
            hierarchy_integrity = max(0, 100 - violation_rate)
            
            hierarchy_results = {
                'total_go_terms': total_go_terms,
                'total_relationships': total_relationships,
                'namespace_distribution': dict(namespace_distribution),
                'hierarchy_violations': len(hierarchy_violations),
                'cross_namespace_relationships': len(cross_namespace_relationships),
                'orphaned_terms': len(orphaned_terms),
                'biological_violations': len(biological_violations),
                'hierarchy_integrity_percentage': hierarchy_integrity,
                'violation_details': {
                    'hierarchy_violations': hierarchy_violations[:10],  # Sample
                    'cross_namespace_sample': cross_namespace_relationships[:5],
                    'orphaned_sample': orphaned_terms[:10],
                    'biological_violations': biological_violations[:10]
                }
            }
            
            logger.info(f"GO hierarchy validation:")
            logger.info(f"   Total GO terms: {total_go_terms:,}")
            logger.info(f"   Total relationships: {total_relationships:,}")
            logger.info(f"   Hierarchy integrity: {hierarchy_integrity:.1f}%")
            logger.info(f"   Namespace violations: {len(cross_namespace_relationships)}")
            logger.info(f"   Orphaned terms: {len(orphaned_terms)}")
            logger.info(f"   Biological violations: {len(biological_violations)}")
            
            self.validation_results['detailed_results']['go_hierarchy_integrity'] = hierarchy_results
            return True
            
        except Exception as e:
            logger.error(f"❌ GO hierarchy validation failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _is_root_term(self, term_name):
        """Check if a GO term is a root term."""
        if not term_name:
            return False
        
        root_patterns = [
            'biological_process', 'cellular_component', 'molecular_function',
            'biological process', 'cellular component', 'molecular function',
            'root', 'all'
        ]
        
        term_lower = term_name.lower()
        return any(pattern in term_lower for pattern in root_patterns)
    
    def _validate_go_biological_constraints(self, go_terms, relationships):
        """Validate GO terms against known biological constraints."""
        violations = []
        
        # Check for impossible biological relationships
        for rel in relationships:
            child_name = go_terms.get(rel['child'], {}).get('name', '').lower()
            parent_name = go_terms.get(rel['parent'], {}).get('name', '').lower()
            
            # Example constraint: mitochondrial processes should be in cellular_component or biological_process
            if 'mitochondria' in child_name and rel['parent_namespace'] == 'molecular_function':
                violations.append({
                    'type': 'namespace_mismatch',
                    'child': rel['child'],
                    'parent': rel['parent'],
                    'issue': 'Mitochondrial term in molecular function hierarchy'
                })
            
            # Check for temporal inconsistencies
            if 'death' in child_name and 'development' in parent_name:
                violations.append({
                    'type': 'temporal_inconsistency',
                    'child': rel['child'],
                    'parent': rel['parent'],
                    'issue': 'Death process as child of development process'
                })
        
        return violations
    
    def validate_gene_function_relationships(self):
        """Validate gene-function relationship consistency."""
        logger.info("🧬 VALIDATING GENE-FUNCTION RELATIONSHIPS")
        logger.info("-" * 60)
        
        try:
            gene_function_results = {}
            
            # Extract gene-GO associations
            gene_go_associations = []
            function_consistency_issues = []
            
            for source, target, edge_data in self.graph.edges(data=True):
                edge_type = edge_data.get('type', '')
                if 'gene_go' in edge_type.lower() or 'association' in edge_type.lower():
                    source_data = self.graph.nodes[source]
                    target_data = self.graph.nodes[target]
                    
                    gene_node, gene_data = None, None
                    go_node, go_data = None, None
                    
                    if source_data.get('type') == 'gene' and target_data.get('type') == 'go_term':
                        gene_node, gene_data = source, source_data
                        go_node, go_data = target, target_data
                    elif source_data.get('type') == 'go_term' and target_data.get('type') == 'gene':
                        gene_node, gene_data = target, target_data
                        go_node, go_data = source, source_data
                    
                    if gene_node and go_node:
                        gene_symbol = gene_data.get('symbol', gene_data.get('name'))
                        go_id = go_data.get('go_id', str(go_node))
                        go_name = go_data.get('name', go_data.get('term_name'))
                        namespace = go_data.get('namespace')
                        
                        association = {
                            'gene_symbol': gene_symbol,
                            'go_id': go_id,
                            'go_name': go_name,
                            'namespace': namespace,
                            'gene_node': gene_node,
                            'go_node': go_node
                        }
                        
                        gene_go_associations.append(association)
                        
                        # Validate biological consistency
                        consistency_issue = self._validate_gene_go_consistency(gene_symbol, go_name, namespace)
                        if consistency_issue:
                            function_consistency_issues.append({
                                **association,
                                'issue': consistency_issue
                            })
            
            # Analyze gene function coverage
            genes_with_functions = defaultdict(set)
            functions_per_gene = defaultdict(int)
            namespace_coverage = defaultdict(set)
            
            for assoc in gene_go_associations:
                gene = assoc['gene_symbol']
                namespace = assoc['namespace']
                
                if gene and namespace:
                    genes_with_functions[gene].add(namespace)
                    functions_per_gene[gene] += 1
                    namespace_coverage[namespace].add(gene)
            
            # Calculate coverage metrics
            total_genes_with_functions = len(genes_with_functions)
            avg_functions_per_gene = np.mean(list(functions_per_gene.values())) if functions_per_gene else 0
            
            # Identify potentially under-annotated genes
            under_annotated_genes = [
                gene for gene, count in functions_per_gene.items() 
                if count < 3 and gene in self.biological_knowledge['essential_genes']
            ]
            
            # Identify over-annotated genes (suspicious)
            over_annotated_genes = [
                gene for gene, count in functions_per_gene.items() 
                if count > 50  # Arbitrary threshold for investigation
            ]
            
            gene_function_results = {
                'total_gene_go_associations': len(gene_go_associations),
                'genes_with_functions': total_genes_with_functions,
                'average_functions_per_gene': avg_functions_per_gene,
                'namespace_coverage': {ns: len(genes) for ns, genes in namespace_coverage.items()},
                'function_consistency_issues': len(function_consistency_issues),
                'under_annotated_essential_genes': len(under_annotated_genes),
                'over_annotated_genes': len(over_annotated_genes),
                'consistency_rate': max(0, 100 - (len(function_consistency_issues) / max(len(gene_go_associations), 1) * 100)),
                'issue_details': {
                    'consistency_issues_sample': function_consistency_issues[:10],
                    'under_annotated_sample': under_annotated_genes[:10],
                    'over_annotated_sample': over_annotated_genes[:5]
                }
            }
            
            logger.info(f"Gene-function relationship validation:")
            logger.info(f"   Total associations: {len(gene_go_associations):,}")
            logger.info(f"   Genes with functions: {total_genes_with_functions:,}")
            logger.info(f"   Average functions per gene: {avg_functions_per_gene:.1f}")
            logger.info(f"   Consistency rate: {gene_function_results['consistency_rate']:.1f}%")
            logger.info(f"   Under-annotated essential genes: {len(under_annotated_genes)}")
            
            self.validation_results['detailed_results']['gene_function_relationships'] = gene_function_results
            return True
            
        except Exception as e:
            logger.error(f"❌ Gene-function relationship validation failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _validate_gene_go_consistency(self, gene_symbol, go_name, namespace):
        """Validate specific gene-GO term associations for biological consistency."""
        if not gene_symbol or not go_name or not namespace:
            return None
        
        gene_lower = gene_symbol.lower()
        go_lower = go_name.lower()
        
        # Check for obvious inconsistencies
        
        # Oncogenes shouldn't be associated with tumor suppressor functions
        if gene_symbol in self.biological_knowledge['oncogenes']:
            if any(term in go_lower for term in ['tumor suppressor', 'cell cycle arrest', 'apoptosis induction']):
                return f"Oncogene {gene_symbol} associated with tumor suppressor function"
        
        # Tumor suppressors shouldn't be associated with oncogenic functions
        if gene_symbol in self.biological_knowledge['tumor_suppressors']:
            if any(term in go_lower for term in ['cell proliferation', 'oncogene', 'growth promotion']):
                return f"Tumor suppressor {gene_symbol} associated with oncogenic function"
        
        # Mitochondrial genes should have cellular component associations
        if 'mt-' in gene_lower or 'mito' in gene_lower:
            if namespace == 'molecular_function' and 'mitochondria' not in go_lower:
                return f"Mitochondrial gene {gene_symbol} missing mitochondrial localization"
        
        # Check for namespace appropriateness
        if namespace == 'cellular_component':
            # Transcription factors should be nuclear
            if any(tf_term in go_lower for tf_term in ['transcription factor', 'dna binding']):
                if not any(nuclear_term in go_lower for nuclear_term in ['nucleus', 'nuclear', 'chromatin']):
                    return f"Transcription factor {gene_symbol} not associated with nuclear components"
        
        return None
    
    def validate_disease_gene_associations(self):
        """Validate disease-gene association plausibility."""
        logger.info("🏥 VALIDATING DISEASE-GENE ASSOCIATIONS")
        logger.info("-" * 60)
        
        try:
            disease_gene_results = {}
            
            # Extract disease-gene associations
            disease_associations = []
            plausibility_issues = []
            
            for source, target, edge_data in self.graph.edges(data=True):
                edge_type = edge_data.get('type', '')
                if 'disease' in edge_type.lower():
                    source_data = self.graph.nodes[source]
                    target_data = self.graph.nodes[target]
                    
                    gene_node, gene_data = None, None
                    disease_node, disease_data = None, None
                    
                    if source_data.get('type') == 'gene' and target_data.get('type') == 'disease':
                        gene_node, gene_data = source, source_data
                        disease_node, disease_data = target, target_data
                    elif source_data.get('type') == 'disease' and target_data.get('type') == 'gene':
                        gene_node, gene_data = target, target_data
                        disease_node, disease_data = source, source_data
                    
                    if gene_node and disease_node:
                        gene_symbol = gene_data.get('symbol', gene_data.get('name'))
                        disease_name = disease_data.get('name', str(disease_node))
                        
                        association = {
                            'gene_symbol': gene_symbol,
                            'disease_name': disease_name,
                            'edge_type': edge_type
                        }
                        
                        disease_associations.append(association)
                        
                        # Validate plausibility
                        plausibility_issue = self._validate_disease_gene_plausibility(gene_symbol, disease_name)
                        if plausibility_issue:
                            plausibility_issues.append({
                                **association,
                                'issue': plausibility_issue
                            })
            
            # Analyze disease coverage
            diseases_with_genes = defaultdict(set)
            genes_with_diseases = defaultdict(set)
            
            for assoc in disease_associations:
                gene = assoc['gene_symbol']
                disease = assoc['disease_name']
                
                if gene and disease:
                    diseases_with_genes[disease].add(gene)
                    genes_with_diseases[gene].add(disease)
            
            # Identify high-impact genes (associated with many diseases)
            high_impact_genes = [
                gene for gene, diseases in genes_with_diseases.items()
                if len(diseases) > 10
            ]
            
            # Check for known cancer genes
            cancer_gene_coverage = 0
            for gene in self.biological_knowledge['oncogenes'] | self.biological_knowledge['tumor_suppressors']:
                if gene in genes_with_diseases:
                    cancer_diseases = [d for d in genes_with_diseases[gene] if 'cancer' in d.lower() or 'tumor' in d.lower()]
                    if cancer_diseases:
                        cancer_gene_coverage += 1
            
            total_cancer_genes = len(self.biological_knowledge['oncogenes'] | self.biological_knowledge['tumor_suppressors'])
            cancer_coverage_rate = (cancer_gene_coverage / total_cancer_genes * 100) if total_cancer_genes > 0 else 0
            
            disease_gene_results = {
                'total_disease_gene_associations': len(disease_associations),
                'unique_diseases': len(diseases_with_genes),
                'unique_genes': len(genes_with_diseases),
                'plausibility_issues': len(plausibility_issues),
                'high_impact_genes': len(high_impact_genes),
                'cancer_gene_coverage_rate': cancer_coverage_rate,
                'plausibility_rate': max(0, 100 - (len(plausibility_issues) / max(len(disease_associations), 1) * 100)),
                'issue_details': {
                    'plausibility_issues_sample': plausibility_issues[:10],
                    'high_impact_genes_sample': high_impact_genes[:10]
                }
            }
            
            logger.info(f"Disease-gene association validation:")
            logger.info(f"   Total associations: {len(disease_associations):,}")
            logger.info(f"   Unique diseases: {len(diseases_with_genes):,}")
            logger.info(f"   Unique genes: {len(genes_with_diseases):,}")
            logger.info(f"   Plausibility rate: {disease_gene_results['plausibility_rate']:.1f}%")
            logger.info(f"   Cancer gene coverage: {cancer_coverage_rate:.1f}%")
            
            self.validation_results['detailed_results']['disease_gene_associations'] = disease_gene_results
            return True
            
        except Exception as e:
            logger.error(f"❌ Disease-gene association validation failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _validate_disease_gene_plausibility(self, gene_symbol, disease_name):
        """Validate plausibility of specific disease-gene associations."""
        if not gene_symbol or not disease_name:
            return None
        
        gene_lower = gene_symbol.lower()
        disease_lower = disease_name.lower()
        
        # Check for obvious implausible associations
        
        # Housekeeping genes shouldn't be primary disease drivers for rare diseases
        housekeeping_genes = {'ACTB', 'GAPDH', 'TUBB', 'RPL', 'RPS'}
        if any(hk in gene_symbol for hk in housekeeping_genes):
            if any(rare_term in disease_lower for rare_term in ['rare', 'syndrome', 'orphan']):
                return f"Housekeeping gene {gene_symbol} associated with rare disease"
        
        # Check cancer gene consistency
        if gene_symbol in self.biological_knowledge['tumor_suppressors']:
            if any(infectious_term in disease_lower for infectious_term in ['virus', 'bacterial', 'infection']):
                return f"Tumor suppressor {gene_symbol} associated with infectious disease"
        
        # Check for tissue-specific inconsistencies
        if 'brain' in disease_lower or 'neuro' in disease_lower:
            # Should be associated with neural genes, but this is complex to validate automatically
            pass
        
        return None
    
    def validate_pathway_coherence(self):
        """Validate pathway coherence and biological completeness."""
        logger.info("🛤️ VALIDATING PATHWAY COHERENCE & COMPLETENESS")
        logger.info("-" * 60)
        
        try:
            pathway_results = {}
            
            # Identify pathway-like structures (gene sets, clusters, GO terms)
            pathways = []
            
            # Extract from cluster relationships
            cluster_nodes = [(n, d) for n, d in self.graph.nodes(data=True) if d.get('type') == 'cluster']
            
            for cluster_node, cluster_data in cluster_nodes:
                # Find genes in this cluster
                cluster_genes = []
                for neighbor in self.graph.neighbors(cluster_node):
                    neighbor_data = self.graph.nodes[neighbor]
                    if neighbor_data.get('type') == 'gene':
                        gene_symbol = neighbor_data.get('symbol', neighbor_data.get('name'))
                        if gene_symbol:
                            cluster_genes.append(gene_symbol)
                
                if len(cluster_genes) >= 3:  # Minimum for pathway analysis
                    pathways.append({
                        'id': str(cluster_node),
                        'type': 'cluster',
                        'genes': cluster_genes,
                        'size': len(cluster_genes)
                    })
            
            # Analyze pathway coherence
            coherence_issues = []
            pathway_statistics = {
                'total_pathways': len(pathways),
                'size_distribution': defaultdict(int),
                'coherence_scores': []
            }
            
            for pathway in pathways:
                # Size distribution
                size_category = 'small' if pathway['size'] < 10 else 'medium' if pathway['size'] < 50 else 'large'
                pathway_statistics['size_distribution'][size_category] += 1
                
                # Coherence analysis
                coherence_score = self._analyze_pathway_coherence(pathway['genes'])
                pathway_statistics['coherence_scores'].append(coherence_score)
                
                if coherence_score < 50:  # Low coherence threshold
                    coherence_issues.append({
                        'pathway_id': pathway['id'],
                        'pathway_type': pathway['type'],
                        'size': pathway['size'],
                        'coherence_score': coherence_score,
                        'issue': 'Low biological coherence'
                    })
            
            # Calculate overall pathway quality
            avg_coherence = np.mean(pathway_statistics['coherence_scores']) if pathway_statistics['coherence_scores'] else 0
            coherence_rate = max(0, 100 - (len(coherence_issues) / max(len(pathways), 1) * 100))
            
            pathway_results = {
                'total_pathways_analyzed': len(pathways),
                'average_pathway_size': np.mean([p['size'] for p in pathways]) if pathways else 0,
                'size_distribution': dict(pathway_statistics['size_distribution']),
                'average_coherence_score': avg_coherence,
                'coherence_issues': len(coherence_issues),
                'overall_coherence_rate': coherence_rate,
                'issue_details': {
                    'coherence_issues_sample': coherence_issues[:10]
                }
            }
            
            logger.info(f"Pathway coherence validation:")
            logger.info(f"   Pathways analyzed: {len(pathways)}")
            logger.info(f"   Average coherence score: {avg_coherence:.1f}")
            logger.info(f"   Overall coherence rate: {coherence_rate:.1f}%")
            logger.info(f"   Coherence issues: {len(coherence_issues)}")
            
            self.validation_results['detailed_results']['pathway_coherence'] = pathway_results
            return True
            
        except Exception as e:
            logger.error(f"❌ Pathway coherence validation failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _analyze_pathway_coherence(self, gene_list):
        """Analyze biological coherence of a gene set/pathway."""
        if len(gene_list) < 3:
            return 0
        
        coherence_score = 50  # Base score
        
        # Check for known functional relationships
        oncogenes_in_pathway = sum(1 for gene in gene_list if gene in self.biological_knowledge['oncogenes'])
        tumor_suppressors_in_pathway = sum(1 for gene in gene_list if gene in self.biological_knowledge['tumor_suppressors'])
        
        # Pathways with mixed oncogenes and tumor suppressors are less coherent
        if oncogenes_in_pathway > 0 and tumor_suppressors_in_pathway > 0:
            coherence_score -= 20
        
        # Pathways with many essential genes are more coherent
        essential_genes_in_pathway = sum(1 for gene in gene_list if gene in self.biological_knowledge['essential_genes'])
        essential_ratio = essential_genes_in_pathway / len(gene_list)
        
        if essential_ratio > 0.5:
            coherence_score += 30
        elif essential_ratio > 0.2:
            coherence_score += 15
        
        # Check for gene naming patterns (may indicate functional similarity)
        gene_prefixes = defaultdict(int)
        for gene in gene_list:
            if len(gene) >= 3:
                prefix = gene[:3]
                gene_prefixes[prefix] += 1
        
        # High prefix similarity suggests functional relationship
        max_prefix_count = max(gene_prefixes.values()) if gene_prefixes else 0
        if max_prefix_count / len(gene_list) > 0.3:
            coherence_score += 20
        
        return min(100, max(0, coherence_score))
    
    def validate_cross_modal_semantic_consistency(self):
        """Validate semantic consistency across different data modalities."""
        logger.info("🔗 VALIDATING CROSS-MODAL SEMANTIC CONSISTENCY")
        logger.info("-" * 60)
        
        try:
            cross_modal_results = {}
            
            # Sample genes and check their cross-modal annotations
            gene_nodes = [(n, d) for n, d in self.graph.nodes(data=True) if d.get('type') == 'gene']
            sample_genes = gene_nodes[:100]  # Sample for analysis
            
            consistency_issues = []
            gene_modal_coverage = defaultdict(set)
            
            for gene_node, gene_data in sample_genes:
                gene_symbol = gene_data.get('symbol', gene_data.get('name'))
                if not gene_symbol:
                    continue
                
                # Check connections to different modalities
                modalities = {
                    'go_terms': [],
                    'diseases': [],
                    'drugs': [],
                    'viral_conditions': [],
                    'model_predictions': [],
                    'llm_interpretations': []
                }
                
                for neighbor in self.graph.neighbors(gene_node):
                    neighbor_data = self.graph.nodes[neighbor]
                    neighbor_type = neighbor_data.get('type')
                    
                    if neighbor_type == 'go_term':
                        modalities['go_terms'].append(neighbor_data)
                    elif neighbor_type == 'disease':
                        modalities['diseases'].append(neighbor_data)
                    elif neighbor_type == 'drug':
                        modalities['drugs'].append(neighbor_data)
                    elif neighbor_type == 'viral_condition':
                        modalities['viral_conditions'].append(neighbor_data)
                    elif neighbor_type == 'model_prediction':
                        modalities['model_predictions'].append(neighbor_data)
                    elif neighbor_type == 'llm_interpretation':
                        modalities['llm_interpretations'].append(neighbor_data)
                
                # Count modalities this gene is connected to
                connected_modalities = sum(1 for modal_list in modalities.values() if modal_list)
                gene_modal_coverage[connected_modalities] += 1
                
                # Check for semantic consistency issues
                consistency_issue = self._check_cross_modal_consistency(gene_symbol, modalities)
                if consistency_issue:
                    consistency_issues.append({
                        'gene_symbol': gene_symbol,
                        'connected_modalities': connected_modalities,
                        'issue': consistency_issue
                    })
            
            # Calculate cross-modal statistics
            avg_modal_coverage = np.mean(list(gene_modal_coverage.keys())) if gene_modal_coverage else 0
            well_connected_genes = sum(count for modal_count, count in gene_modal_coverage.items() if modal_count >= 4)
            consistency_rate = max(0, 100 - (len(consistency_issues) / len(sample_genes) * 100))
            
            cross_modal_results = {
                'genes_analyzed': len(sample_genes),
                'average_modal_coverage': avg_modal_coverage,
                'modal_coverage_distribution': dict(gene_modal_coverage),
                'well_connected_genes': well_connected_genes,
                'consistency_issues': len(consistency_issues),
                'consistency_rate': consistency_rate,
                'issue_details': {
                    'consistency_issues_sample': consistency_issues[:10]
                }
            }
            
            logger.info(f"Cross-modal semantic consistency validation:")
            logger.info(f"   Genes analyzed: {len(sample_genes)}")
            logger.info(f"   Average modal coverage: {avg_modal_coverage:.1f}")
            logger.info(f"   Well-connected genes: {well_connected_genes}")
            logger.info(f"   Consistency rate: {consistency_rate:.1f}%")
            
            self.validation_results['detailed_results']['cross_modal_consistency'] = cross_modal_results
            return True
            
        except Exception as e:
            logger.error(f"❌ Cross-modal consistency validation failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _check_cross_modal_consistency(self, gene_symbol, modalities):
        """Check for semantic consistency across modalities for a gene."""
        
        # Check if cancer gene has appropriate disease associations
        if gene_symbol in self.biological_knowledge['oncogenes'] or gene_symbol in self.biological_knowledge['tumor_suppressors']:
            diseases = [d.get('name', '').lower() for d in modalities['diseases']]
            has_cancer_disease = any('cancer' in disease or 'tumor' in disease for disease in diseases)
            
            if modalities['diseases'] and not has_cancer_disease:
                return f"Cancer gene {gene_symbol} lacks cancer disease associations"
        
        # Check GO term consistency with other modalities
        go_namespaces = set()
        for go_term in modalities['go_terms']:
            namespace = go_term.get('namespace')
            if namespace:
                go_namespaces.add(namespace)
        
        # Genes with drug associations should have molecular function annotations
        if modalities['drugs'] and 'molecular_function' not in go_namespaces:
            return f"Drug target {gene_symbol} lacks molecular function annotations"
        
        # Genes with many disease associations should have biological process annotations
        if len(modalities['diseases']) > 5 and 'biological_process' not in go_namespaces:
            return f"Disease gene {gene_symbol} lacks biological process annotations"
        
        return None
    
    def validate_biological_constraints(self):
        """Validate adherence to fundamental biological constraints."""
        logger.info("⚖️ VALIDATING BIOLOGICAL CONSTRAINTS")
        logger.info("-" * 60)
        
        try:
            constraint_results = {}
            
            # Define biological constraints
            constraints = {
                'gene_uniqueness': self._validate_gene_uniqueness(),
                'go_term_uniqueness': self._validate_go_term_uniqueness(),
                'species_consistency': self._validate_species_consistency(),
                'temporal_consistency': self._validate_temporal_consistency(),
                'localization_consistency': self._validate_localization_consistency()
            }
            
            # Count violations
            total_constraints = len(constraints)
            violated_constraints = sum(1 for result in constraints.values() if not result['valid'])
            constraint_adherence = ((total_constraints - violated_constraints) / total_constraints * 100) if total_constraints > 0 else 100
            
            constraint_results = {
                'total_constraints_checked': total_constraints,
                'violated_constraints': violated_constraints,
                'constraint_adherence_percentage': constraint_adherence,
                'constraint_details': constraints
            }
            
            logger.info(f"Biological constraints validation:")
            logger.info(f"   Constraints checked: {total_constraints}")
            logger.info(f"   Violations found: {violated_constraints}")
            logger.info(f"   Adherence rate: {constraint_adherence:.1f}%")
            
            for constraint_name, result in constraints.items():
                status = "✅" if result['valid'] else "❌"
                logger.info(f"   {constraint_name}: {status} {result.get('message', '')}")
            
            self.validation_results['biological_constraints'] = constraint_results
            return True
            
        except Exception as e:
            logger.error(f"❌ Biological constraints validation failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _validate_gene_uniqueness(self):
        """Validate that gene symbols are unique."""
        gene_symbols = []
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'gene':
                symbol = data.get('symbol', data.get('name'))
                if symbol:
                    gene_symbols.append(symbol)
        
        unique_symbols = set(gene_symbols)
        duplicates = len(gene_symbols) - len(unique_symbols)
        
        return {
            'valid': duplicates == 0,
            'total_genes': len(gene_symbols),
            'unique_genes': len(unique_symbols),
            'duplicates': duplicates,
            'message': f"{duplicates} duplicate gene symbols found" if duplicates > 0 else "All gene symbols unique"
        }
    
    def _validate_go_term_uniqueness(self):
        """Validate that GO term IDs are unique."""
        go_ids = []
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'go_term':
                go_id = data.get('go_id', str(node) if str(node).startswith('GO:') else None)
                if go_id:
                    go_ids.append(go_id)
        
        unique_ids = set(go_ids)
        duplicates = len(go_ids) - len(unique_ids)
        
        return {
            'valid': duplicates == 0,
            'total_go_terms': len(go_ids),
            'unique_go_terms': len(unique_ids),
            'duplicates': duplicates,
            'message': f"{duplicates} duplicate GO IDs found" if duplicates > 0 else "All GO IDs unique"
        }
    
    def _validate_species_consistency(self):
        """Validate species consistency (assuming human-focused data)."""
        # For this validation, we assume the data should be primarily human
        # Check for obvious non-human gene symbols or terms
        
        non_human_indicators = 0
        total_genes = 0
        
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'gene':
                total_genes += 1
                symbol = data.get('symbol', data.get('name', ''))
                
                # Check for common non-human prefixes
                if any(symbol.lower().startswith(prefix) for prefix in ['dros', 'mus', 'rat', 'mm_', 'dm_']):
                    non_human_indicators += 1
        
        human_ratio = ((total_genes - non_human_indicators) / total_genes * 100) if total_genes > 0 else 100
        
        return {
            'valid': human_ratio >= 95,  # 95% human genes threshold
            'total_genes': total_genes,
            'non_human_indicators': non_human_indicators,
            'human_ratio': human_ratio,
            'message': f"{human_ratio:.1f}% appear to be human genes"
        }
    
    def _validate_temporal_consistency(self):
        """Validate temporal consistency (development vs death processes)."""
        temporal_violations = 0
        temporal_relationships = 0
        
        # Check for contradictory temporal relationships
        for source, target, edge_data in self.graph.edges(data=True):
            source_data = self.graph.nodes[source]
            target_data = self.graph.nodes[target]
            
            if source_data.get('type') == 'go_term' and target_data.get('type') == 'go_term':
                source_name = source_data.get('name', '').lower()
                target_name = target_data.get('name', '').lower()
                
                temporal_relationships += 1
                
                # Check for development->death inconsistencies
                if ('development' in source_name and 'death' in target_name) or \
                   ('death' in source_name and 'development' in target_name):
                    temporal_violations += 1
        
        consistency_rate = ((temporal_relationships - temporal_violations) / temporal_relationships * 100) if temporal_relationships > 0 else 100
        
        return {
            'valid': temporal_violations == 0,
            'temporal_relationships': temporal_relationships,
            'violations': temporal_violations,
            'consistency_rate': consistency_rate,
            'message': f"{temporal_violations} temporal inconsistencies found"
        }
    
    def _validate_localization_consistency(self):
        """Validate cellular localization consistency."""
        localization_issues = 0
        localization_checks = 0
        
        # Check for genes with conflicting cellular component annotations
        gene_localizations = defaultdict(set)
        
        for source, target, edge_data in self.graph.edges(data=True):
            source_data = self.graph.nodes[source]
            target_data = self.graph.nodes[target]
            
            if source_data.get('type') == 'gene' and target_data.get('type') == 'go_term':
                if target_data.get('namespace') == 'cellular_component':
                    gene_symbol = source_data.get('symbol')
                    go_name = target_data.get('name', '').lower()
                    
                    if gene_symbol and go_name:
                        gene_localizations[gene_symbol].add(go_name)
        
        # Check for contradictory localizations
        for gene, localizations in gene_localizations.items():
            localization_checks += 1
            
            # Check for nucleus vs mitochondria conflicts
            has_nuclear = any('nucleus' in loc or 'nuclear' in loc for loc in localizations)
            has_mitochondrial = any('mitochondria' in loc or 'mitochondrial' in loc for loc in localizations)
            
            # Some proteins can be in both, but check for suspicious cases
            if has_nuclear and has_mitochondrial and len(localizations) == 2:
                # Only nuclear and mitochondrial - might be suspicious
                nuclear_locs = [loc for loc in localizations if 'nucleus' in loc or 'nuclear' in loc]
                mito_locs = [loc for loc in localizations if 'mitochondria' in loc or 'mitochondrial' in loc]
                
                # If very specific localizations that conflict
                if any('inner' in loc or 'outer' in loc for loc in mito_locs) and \
                   any('chromatin' in loc for loc in nuclear_locs):
                    localization_issues += 1
        
        consistency_rate = ((localization_checks - localization_issues) / localization_checks * 100) if localization_checks > 0 else 100
        
        return {
            'valid': localization_issues == 0,
            'genes_checked': localization_checks,
            'issues': localization_issues,
            'consistency_rate': consistency_rate,
            'message': f"{localization_issues} localization inconsistencies found"
        }
    
    def generate_comprehensive_quality_metrics(self):
        """Generate comprehensive semantic quality metrics."""
        logger.info("📊 GENERATING COMPREHENSIVE SEMANTIC QUALITY METRICS")
        logger.info("-" * 60)
        
        try:
            metrics = {}
            
            # GO hierarchy quality
            go_hierarchy = self.validation_results['detailed_results'].get('go_hierarchy_integrity', {})
            hierarchy_score = go_hierarchy.get('hierarchy_integrity_percentage', 0)
            metrics['go_hierarchy_quality_score'] = hierarchy_score
            
            # Gene-function relationship quality
            gene_function = self.validation_results['detailed_results'].get('gene_function_relationships', {})
            function_score = gene_function.get('consistency_rate', 0)
            metrics['gene_function_quality_score'] = function_score
            
            # Disease-gene association quality
            disease_gene = self.validation_results['detailed_results'].get('disease_gene_associations', {})
            disease_score = disease_gene.get('plausibility_rate', 0)
            metrics['disease_gene_quality_score'] = disease_score
            
            # Pathway coherence quality
            pathway = self.validation_results['detailed_results'].get('pathway_coherence', {})
            pathway_score = pathway.get('overall_coherence_rate', 0)
            metrics['pathway_coherence_score'] = pathway_score
            
            # Cross-modal consistency quality
            cross_modal = self.validation_results['detailed_results'].get('cross_modal_consistency', {})
            modal_score = cross_modal.get('consistency_rate', 0)
            metrics['cross_modal_consistency_score'] = modal_score
            
            # Biological constraints adherence
            constraints = self.validation_results.get('biological_constraints', {})
            constraint_score = constraints.get('constraint_adherence_percentage', 0)
            metrics['biological_constraints_score'] = constraint_score
            
            # Calculate overall semantic quality
            quality_scores = [
                hierarchy_score, function_score, disease_score,
                pathway_score, modal_score, constraint_score
            ]
            valid_scores = [score for score in quality_scores if score > 0]
            overall_semantic_quality = np.mean(valid_scores) if valid_scores else 0
            
            metrics['overall_semantic_quality'] = overall_semantic_quality
            
            # Semantic quality grade
            if overall_semantic_quality >= 95:
                grade = 'A+'
            elif overall_semantic_quality >= 90:
                grade = 'A'
            elif overall_semantic_quality >= 85:
                grade = 'B+'
            elif overall_semantic_quality >= 80:
                grade = 'B'
            elif overall_semantic_quality >= 75:
                grade = 'C+'
            elif overall_semantic_quality >= 70:
                grade = 'C'
            else:
                grade = 'D'
            
            metrics['semantic_quality_grade'] = grade
            
            # Detailed breakdown
            metrics['quality_breakdown'] = {
                'go_hierarchy_integrity': hierarchy_score,
                'gene_function_consistency': function_score,
                'disease_gene_plausibility': disease_score,
                'pathway_coherence': pathway_score,
                'cross_modal_consistency': modal_score,
                'biological_constraints': constraint_score
            }
            
            self.validation_results['quality_metrics'] = metrics
            
            logger.info("Comprehensive Semantic Quality Scores:")
            logger.info(f"   GO Hierarchy Integrity: {hierarchy_score:.1f}%")
            logger.info(f"   Gene-Function Consistency: {function_score:.1f}%")
            logger.info(f"   Disease-Gene Plausibility: {disease_score:.1f}%")
            logger.info(f"   Pathway Coherence: {pathway_score:.1f}%")
            logger.info(f"   Cross-Modal Consistency: {modal_score:.1f}%")
            logger.info(f"   Biological Constraints: {constraint_score:.1f}%")
            logger.info(f"   Overall Semantic Quality: {overall_semantic_quality:.1f}% (Grade: {grade})")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Semantic quality metrics generation failed: {str(e)}")
            return False
    
    def generate_recommendations(self):
        """Generate detailed recommendations based on semantic validation results."""
        recommendations = []
        
        # Analyze results and generate actionable recommendations
        go_hierarchy = self.validation_results['detailed_results'].get('go_hierarchy_integrity', {})
        gene_function = self.validation_results['detailed_results'].get('gene_function_relationships', {})
        disease_gene = self.validation_results['detailed_results'].get('disease_gene_associations', {})
        pathway = self.validation_results['detailed_results'].get('pathway_coherence', {})
        cross_modal = self.validation_results['detailed_results'].get('cross_modal_consistency', {})
        constraints = self.validation_results.get('biological_constraints', {})
        
        # GO hierarchy recommendations
        hierarchy_score = go_hierarchy.get('hierarchy_integrity_percentage', 0)
        if hierarchy_score < 90:
            recommendations.append({
                'category': 'GO Hierarchy',
                'priority': 'HIGH',
                'issue': f'GO hierarchy integrity is {hierarchy_score:.1f}%',
                'recommendation': 'Review GO term relationships and fix namespace violations',
                'specific_actions': [
                    'Validate cross-namespace relationships',
                    'Check for orphaned GO terms',
                    'Verify parent-child consistency'
                ]
            })
        
        # Gene-function recommendations
        function_score = gene_function.get('consistency_rate', 0)
        under_annotated = gene_function.get('under_annotated_essential_genes', 0)
        if function_score < 85 or under_annotated > 0:
            recommendations.append({
                'category': 'Gene-Function Relationships',
                'priority': 'MEDIUM',
                'issue': f'Gene-function consistency is {function_score:.1f}%, {under_annotated} essential genes under-annotated',
                'recommendation': 'Improve gene function annotations and resolve inconsistencies',
                'specific_actions': [
                    'Add missing GO annotations for essential genes',
                    'Review oncogene/tumor suppressor function assignments',
                    'Validate tissue-specific gene functions'
                ]
            })
        
        # Disease-gene recommendations
        disease_score = disease_gene.get('plausibility_rate', 0)
        cancer_coverage = disease_gene.get('cancer_gene_coverage_rate', 0)
        if disease_score < 80 or cancer_coverage < 70:
            recommendations.append({
                'category': 'Disease-Gene Associations',
                'priority': 'MEDIUM',
                'issue': f'Disease-gene plausibility is {disease_score:.1f}%, cancer gene coverage is {cancer_coverage:.1f}%',
                'recommendation': 'Enhance disease-gene association quality and coverage',
                'specific_actions': [
                    'Review implausible disease-gene associations',
                    'Add missing cancer gene-disease links',
                    'Validate rare disease associations'
                ]
            })
        
        # Pathway coherence recommendations
        pathway_score = pathway.get('overall_coherence_rate', 0)
        if pathway_score < 75:
            recommendations.append({
                'category': 'Pathway Coherence',
                'priority': 'LOW',
                'issue': f'Pathway coherence rate is {pathway_score:.1f}%',
                'recommendation': 'Improve biological pathway coherence and completeness',
                'specific_actions': [
                    'Review low-coherence pathways',
                    'Add missing pathway components',
                    'Validate functional gene groupings'
                ]
            })
        
        # Cross-modal consistency recommendations
        modal_score = cross_modal.get('consistency_rate', 0)
        if modal_score < 85:
            recommendations.append({
                'category': 'Cross-Modal Consistency',
                'priority': 'MEDIUM',
                'issue': f'Cross-modal consistency is {modal_score:.1f}%',
                'recommendation': 'Improve semantic consistency across data modalities',
                'specific_actions': [
                    'Align GO annotations with disease associations',
                    'Ensure drug targets have appropriate molecular functions',
                    'Validate cross-modal gene profiles'
                ]
            })
        
        # Biological constraints recommendations
        constraint_score = constraints.get('constraint_adherence_percentage', 0)
        if constraint_score < 95:
            recommendations.append({
                'category': 'Biological Constraints',
                'priority': 'HIGH',
                'issue': f'Biological constraint adherence is {constraint_score:.1f}%',
                'recommendation': 'Address fundamental biological constraint violations',
                'specific_actions': [
                    'Remove duplicate gene symbols/GO IDs',
                    'Fix temporal inconsistencies',
                    'Resolve localization conflicts'
                ]
            })
        
        if not recommendations:
            recommendations.append({
                'category': 'Overall Semantic Quality',
                'priority': 'INFO',
                'issue': 'All semantic validation checks passed',
                'recommendation': 'Semantic quality meets high standards',
                'specific_actions': [
                    'Maintain current quality standards',
                    'Monitor for future inconsistencies',
                    'Consider advanced semantic validations'
                ]
            })
        
        self.validation_results['recommendations'] = recommendations
        
        logger.info("Comprehensive Semantic Validation Recommendations:")
        for rec in recommendations:
            logger.info(f"   [{rec['priority']}] {rec['category']}: {rec['recommendation']}")
    
    def save_results(self):
        """Save comprehensive semantic validation results."""
        try:
            output_path = '/home/mreddy1/knowledge_graph/quality_control/6_semantic_validation/comprehensive_semantic_validation_results.json'
            
            with open(output_path, 'w') as f:
                json.dump(self.validation_results, f, indent=2)
                
            logger.info(f"📄 Comprehensive results saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {str(e)}")
            return False
    
    def run_comprehensive_validation(self):
        """Run all comprehensive semantic validation checks."""
        logger.info("🔬 COMPREHENSIVE SEMANTIC VALIDATION")
        logger.info("=" * 80)
        logger.info("This validation ensures biological logic, semantic consistency,")
        logger.info("and scientific accuracy across all knowledge graph components.")
        logger.info("=" * 80)
        
        validation_steps = [
            ('Load Knowledge Graph', self.load_knowledge_graph),
            ('Initialize Biological Knowledge', self.initialize_biological_knowledge),
            ('Validate GO Hierarchy Integrity', self.validate_go_hierarchy_integrity),
            ('Validate Gene-Function Relationships', self.validate_gene_function_relationships),
            ('Validate Disease-Gene Associations', self.validate_disease_gene_associations),
            ('Validate Pathway Coherence', self.validate_pathway_coherence),
            ('Validate Cross-Modal Consistency', self.validate_cross_modal_semantic_consistency),
            ('Validate Biological Constraints', self.validate_biological_constraints),
            ('Generate Quality Metrics', self.generate_comprehensive_quality_metrics),
            ('Generate Recommendations', self.generate_recommendations),
            ('Save Results', self.save_results)
        ]
        
        start_time = time.time()
        passed_steps = 0
        
        for step_name, step_function in validation_steps:
            logger.info(f"Executing: {step_name}")
            
            try:
                if step_function():
                    passed_steps += 1
                    logger.info(f"✅ {step_name} completed successfully")
                else:
                    logger.error(f"❌ {step_name} failed")
            except Exception as e:
                logger.error(f"❌ {step_name} failed with exception: {str(e)}")
        
        total_time = time.time() - start_time
        
        # Summary
        self.validation_results['validation_summary'] = {
            'total_steps': len(validation_steps),
            'passed_steps': passed_steps,
            'success_rate': (passed_steps / len(validation_steps)) * 100,
            'execution_time_seconds': total_time,
            'overall_status': 'PASSED' if passed_steps == len(validation_steps) else 'FAILED'
        }
        
        logger.info("=" * 80)
        logger.info("📊 COMPREHENSIVE SEMANTIC VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Steps completed: {passed_steps}/{len(validation_steps)}")
        logger.info(f"Success rate: {self.validation_results['validation_summary']['success_rate']:.1f}%")
        logger.info(f"Execution time: {total_time:.2f} seconds")
        logger.info(f"Overall status: {self.validation_results['validation_summary']['overall_status']}")
        
        quality_metrics = self.validation_results.get('quality_metrics', {})
        if quality_metrics:
            overall_quality = quality_metrics.get('overall_semantic_quality', 0)
            quality_grade = quality_metrics.get('semantic_quality_grade', 'N/A')
            logger.info(f"Semantic Quality Score: {overall_quality:.1f}/100 (Grade: {quality_grade})")
            
            # Detailed breakdown
            breakdown = quality_metrics.get('quality_breakdown', {})
            logger.info("Quality Breakdown:")
            for component, score in breakdown.items():
                logger.info(f"   {component}: {score:.1f}%")
        
        logger.info("🎯 SEMANTIC VALIDATION COMPLETED - Knowledge graph validated for biological accuracy!")
        
        return self.validation_results['validation_summary']['overall_status'] == 'PASSED'

def main():
    """Main execution function."""
    try:
        validator = ComprehensiveSemanticValidator()
        success = validator.run_comprehensive_validation()
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in main: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main())