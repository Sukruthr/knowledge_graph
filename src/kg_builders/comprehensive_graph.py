"""
Comprehensive Knowledge Graph integrating GO ontology with Omics data and all specialized parsers.

Extracted from kg_builder.py for better modularity and maintainability.
This is the primary knowledge graph implementation for the biomedical system.
"""

import networkx as nx
from typing import Dict, List, Optional, Any
import logging

try:
    from ..parsers import CombinedBiomedicalParser
except ImportError:
    from parsers import CombinedBiomedicalParser

from .shared_utils import save_graph_to_file

logger = logging.getLogger(__name__)

class ComprehensiveBiomedicalKnowledgeGraph:
    """Comprehensive Knowledge Graph integrating GO ontology with Omics data."""
    
    def __init__(self, use_neo4j: bool = False):
        """
        Initialize the comprehensive biomedical knowledge graph.
        
        Args:
            use_neo4j: Whether to use Neo4j database or NetworkX
        """
        self.use_neo4j = use_neo4j
        self.graph = nx.MultiDiGraph()
        self.parser = None
        self.parsed_data = {}
        self.stats = {}
        
        if use_neo4j:
            try:
                from neo4j import GraphDatabase
                self.neo4j_driver = None
                logger.info("Neo4j driver available")
            except ImportError:
                logger.warning("Neo4j driver not available, falling back to NetworkX")
                self.use_neo4j = False
    
    def load_data(self, base_data_dir: str):
        """
        Load and parse comprehensive biomedical data (GO + Omics).
        
        Args:
            base_data_dir: Base directory containing GO_BP, GO_CC, GO_MF, and Omics_data subdirectories
        """
        logger.info(f"Loading comprehensive biomedical data from {base_data_dir}")
        
        # Initialize comprehensive parser
        self.parser = CombinedBiomedicalParser(base_data_dir)
        
        # Parse all biomedical data
        self.parsed_data = self.parser.parse_all_biomedical_data()
        
        logger.info("Comprehensive biomedical data loading complete")
    
    def build_comprehensive_graph(self):
        """Build the comprehensive biomedical knowledge graph."""
        logger.info("Building comprehensive biomedical knowledge graph...")
        
        # Build base GO knowledge graph
        self._build_go_component()
        
        # Add Omics data components
        self._add_omics_nodes()
        self._add_omics_associations()
        self._add_cluster_relationships()
        
        # Add enhanced semantic data from Omics_data2
        self._add_semantic_enhancements()
        
        # Add model comparison data integration
        self._add_model_comparison_data()
        
        # Add CC_MF_Branch data integration
        self._add_cc_mf_branch_data()
        
        # Add LLM_processed data integration
        self._add_llm_processed_data()
        
        # Add GO Analysis Data integration
        self._add_go_analysis_data()
        
        # Add Remaining Data integration
        self._add_remaining_data()
        
        # Add Talisman Gene Sets integration
        self._add_talisman_gene_sets()
        
        # Calculate comprehensive statistics
        self._calculate_comprehensive_stats()
        
        # Validate comprehensive graph
        self._validate_comprehensive_graph()
        
        logger.info("Comprehensive biomedical knowledge graph construction complete")
    
    def _build_go_component(self):
        """Build the GO component of the knowledge graph."""
        logger.info("Building GO component...")
        
        if 'go_data' not in self.parsed_data:
            logger.warning("No GO data available for graph construction")
            return
        
        go_data = self.parsed_data['go_data']
        
        # Add GO term nodes from all namespaces
        for namespace, namespace_data in go_data.items():
            logger.info(f"Adding {namespace} GO terms...")
            
            # Add GO term nodes
            for go_id, go_info in namespace_data.get('go_terms', {}).items():
                node_attrs = {
                    'node_type': 'go_term',
                    'name': go_info['name'],
                    'namespace': go_info['namespace'],
                    'description': go_info.get('description', '')
                }
                
                # Enhance with OBO data if available
                obo_terms = namespace_data.get('obo_terms', {})
                if go_id in obo_terms:
                    obo_info = obo_terms[go_id]
                    node_attrs.update({
                        'definition': obo_info.get('definition', ''),
                        'synonyms': obo_info.get('synonyms', []),
                        'is_obsolete': obo_info.get('is_obsolete', False)
                    })
                
                self.graph.add_node(go_id, **node_attrs)
            
            # Add GO relationships
            for rel in namespace_data.get('go_relationships', []):
                parent_id = rel['parent_id']
                child_id = rel['child_id']
                
                if parent_id in self.graph and child_id in self.graph:
                    self.graph.add_edge(
                        child_id,
                        parent_id,
                        edge_type='go_hierarchy',
                        relationship_type=rel['relationship_type'],
                        namespace=rel['namespace']
                    )
            
            # Add gene nodes and associations
            genes_added = set()
            for assoc in namespace_data.get('gene_associations', []):
                gene_symbol = assoc['gene_symbol']
                go_id = assoc['go_id']
                
                # Add gene node if not already added
                if gene_symbol not in genes_added:
                    gene_attrs = {
                        'node_type': 'gene',
                        'gene_symbol': gene_symbol,
                        'uniprot_id': assoc.get('uniprot_id', ''),
                        'gene_name': assoc.get('gene_name', ''),
                        'gene_type': assoc.get('gene_type', ''),
                        'taxon': assoc.get('taxon', ''),
                        'sources': ['gaf']
                    }
                    self.graph.add_node(gene_symbol, **gene_attrs)
                    genes_added.add(gene_symbol)
                
                # Add gene-GO association
                if gene_symbol in self.graph and go_id in self.graph:
                    self.graph.add_edge(
                        gene_symbol,
                        go_id,
                        edge_type='gene_annotation',
                        source='gaf',
                        evidence_code=assoc.get('evidence_code', ''),
                        qualifier=assoc.get('qualifier', ''),
                        namespace=namespace
                    )
    
    def _add_omics_nodes(self):
        """Add Omics-specific nodes (diseases, drugs, viral conditions, clusters)."""
        logger.info("Adding Omics nodes...")
        
        if 'omics_data' not in self.parsed_data:
            logger.warning("No Omics data available")
            return
        
        omics_data = self.parsed_data['omics_data']
        unique_entities = omics_data.get('unique_entities', {})
        
        # Add disease nodes
        for disease in unique_entities.get('diseases', []):
            self.graph.add_node(
                f"DISEASE:{disease}",
                node_type='disease',
                disease_name=disease,
                entity_type='disease'
            )
        
        # Add drug nodes
        for drug in unique_entities.get('drugs', []):
            self.graph.add_node(
                f"DRUG:{drug}",
                node_type='drug',
                drug_name=drug,
                entity_type='small_molecule'
            )
        
        # Add viral condition nodes
        for viral_condition in unique_entities.get('viral_conditions', []):
            self.graph.add_node(
                f"VIRAL:{viral_condition}",
                node_type='viral_condition',
                viral_condition=viral_condition,
                entity_type='viral_perturbation'
            )
        
        # Add cluster nodes
        for cluster in unique_entities.get('clusters', []):
            self.graph.add_node(
                f"CLUSTER:{cluster}",
                node_type='network_cluster',
                cluster_id=cluster,
                entity_type='network_module'
            )
        
        # Add study context nodes
        for study in unique_entities.get('studies', []):
            if study != '-666':  # Filter out placeholder IDs
                self.graph.add_node(
                    f"STUDY:GSE{study}",
                    node_type='study',
                    gse_id=study,
                    entity_type='expression_study'
                )
        
        logger.info(f"Added Omics nodes: "
                   f"Diseases={len(unique_entities.get('diseases', []))}, "
                   f"Drugs={len(unique_entities.get('drugs', []))}, "
                   f"Viral={len(unique_entities.get('viral_conditions', []))}, "
                   f"Clusters={len(unique_entities.get('clusters', []))}")
    
    def _add_omics_associations(self):
        """Add gene-omics associations (disease, drug, viral)."""
        logger.info("Adding Omics associations...")
        
        if 'omics_data' not in self.parsed_data:
            return
        
        omics_data = self.parsed_data['omics_data']
        association_counts = {'disease': 0, 'drug': 0, 'viral': 0}
        
        # Add gene-disease associations
        for assoc in omics_data.get('disease_associations', []):
            gene_symbol = assoc['gene_symbol']
            disease_node = f"DISEASE:{assoc['disease_name']}"
            
            if gene_symbol in self.graph and disease_node in self.graph:
                self.graph.add_edge(
                    gene_symbol,
                    disease_node,
                    edge_type='gene_disease_association',
                    disease_condition=assoc['disease_condition'],
                    gse_id=assoc['gse_id'],
                    weight=assoc['weight'],
                    source='omics_disease'
                )
                association_counts['disease'] += 1
        
        # Add gene-drug associations
        for assoc in omics_data.get('drug_associations', []):
            gene_symbol = assoc['gene_symbol']
            drug_node = f"DRUG:{assoc['drug_name']}"
            
            if gene_symbol in self.graph and drug_node in self.graph:
                self.graph.add_edge(
                    gene_symbol,
                    drug_node,
                    edge_type='gene_drug_perturbation',
                    drug_condition=assoc['drug_condition'],
                    perturbation_id=assoc['perturbation_id'],
                    weight=assoc['weight'],
                    source='omics_drug'
                )
                association_counts['drug'] += 1
        
        # Add gene-viral associations
        for assoc in omics_data.get('viral_associations', []):
            gene_symbol = assoc['gene_symbol']
            viral_node = f"VIRAL:{assoc['viral_perturbation']}"
            
            if gene_symbol in self.graph and viral_node in self.graph:
                self.graph.add_edge(
                    gene_symbol,
                    viral_node,
                    edge_type='gene_viral_response',
                    viral_condition=assoc['viral_condition'],
                    gse_id=assoc['gse_id'],
                    weight=assoc['weight'],
                    source='omics_viral'
                )
                association_counts['viral'] += 1
        
        # Add viral expression matrix data
        viral_expression_count = 0
        viral_expression_matrix = omics_data.get('viral_expression_matrix', {})
        for gene_symbol, expression_data in viral_expression_matrix.items():
            if gene_symbol in self.graph:
                for expr in expression_data['expressions']:
                    viral_condition = expr['viral_perturbation']
                    viral_node = f"VIRAL:{viral_condition}"
                    
                    if viral_node in self.graph:
                        self.graph.add_edge(
                            gene_symbol,
                            viral_node,
                            edge_type='gene_viral_expression',
                            condition=expr['condition'],
                            expression_value=expr['expression_value'],
                            expression_direction=expr['expression_direction'],
                            expression_magnitude=expr['expression_magnitude'],
                            weight=abs(expr['expression_value']),
                            source='viral_expression_matrix'
                        )
                        viral_expression_count += 1
        
        association_counts['viral'] += viral_expression_count
        
        logger.info(f"Added Omics associations: "
                   f"Disease={association_counts['disease']}, "
                   f"Drug={association_counts['drug']}, "
                   f"Viral={association_counts['viral']}")
    
    def _add_cluster_relationships(self):
        """Add network cluster hierarchy relationships."""
        logger.info("Adding cluster relationships...")
        
        if 'omics_data' not in self.parsed_data:
            return
        
        omics_data = self.parsed_data['omics_data']
        cluster_count = 0
        
        for rel in omics_data.get('cluster_relationships', []):
            parent_cluster = f"CLUSTER:{rel['parent_cluster']}"
            child_cluster = f"CLUSTER:{rel['child_cluster']}"
            
            if parent_cluster in self.graph and child_cluster in self.graph:
                self.graph.add_edge(
                    child_cluster,
                    parent_cluster,
                    edge_type='cluster_hierarchy',
                    relationship_type=rel['relationship_type'],
                    source='nest_network'
                )
                cluster_count += 1
        
        logger.info(f"Added {cluster_count} cluster relationships")
    
    def _add_semantic_enhancements(self):
        """Add enhanced semantic data from Omics_data2."""
        logger.info("Adding semantic enhancements...")
        
        if 'omics_data' not in self.parsed_data:
            logger.warning("No Omics data available for semantic enhancements")
            return
            
        omics_data = self.parsed_data['omics_data']
        enhanced_data = omics_data.get('enhanced_data', {})
        
        if not enhanced_data:
            logger.info("No enhanced semantic data available, skipping")
            return
        
        # Add gene set annotations
        self._add_gene_set_annotations(enhanced_data)
        
        # Add literature references
        self._add_literature_references(enhanced_data)
        
        # Add GO term validations
        self._add_go_term_validations(enhanced_data)
        
        # Add experimental metadata
        self._add_experimental_metadata(enhanced_data)
        
        logger.info("Semantic enhancements integration complete")
    
    def _add_gene_set_annotations(self, enhanced_data):
        """Add LLM-enhanced gene set annotations to the graph."""
        annotations = enhanced_data.get('gene_set_annotations', {})
        if not annotations:
            return
        
        logger.info(f"Adding {len(annotations)} gene set annotations...")
        
        annotation_count = 0
        for gene_set_id, annotation in annotations.items():
            # Create or update gene set node
            gene_set_node = f"GENESET:{gene_set_id}"
            
            # Add enhanced attributes to the gene set node
            node_attrs = {
                'node_type': 'gene_set',
                'gene_set_id': gene_set_id,
                'source': annotation.get('source', ''),
                'gene_set_name': annotation.get('gene_set_name', ''),
                'llm_name': annotation.get('llm_name', ''),
                'llm_analysis': annotation.get('llm_analysis', ''),
                'llm_score': annotation.get('score', 0.0),
                'n_genes': annotation.get('n_genes', 0),
                'supporting_count': annotation.get('supporting_count', 0),
                'llm_coverage': annotation.get('llm_coverage', 0.0)
            }
            
            if gene_set_node in self.graph:
                # Update existing node
                self.graph.nodes[gene_set_node].update(node_attrs)
            else:
                # Add new node
                self.graph.add_node(gene_set_node, **node_attrs)
            
            # Add edges to genes in the set
            for gene_symbol in annotation.get('gene_list', []):
                if gene_symbol in self.graph:
                    self.graph.add_edge(
                        gene_symbol,
                        gene_set_node,
                        edge_type='gene_in_set',
                        support_level='annotated',
                        source='llm_annotation'
                    )
            
            # Add edges to supporting genes with higher confidence
            for gene_symbol in annotation.get('supporting_genes', []):
                if gene_symbol in self.graph:
                    self.graph.add_edge(
                        gene_symbol,
                        gene_set_node,
                        edge_type='gene_supports_set',
                        support_level='high_confidence',
                        source='llm_validation'
                    )
            
            annotation_count += 1
        
        logger.info(f"Added {annotation_count} gene set annotations")
    
    def _add_literature_references(self, enhanced_data):
        """Add literature references to gene sets."""
        references = enhanced_data.get('literature_references', {})
        if not references:
            return
        
        logger.info(f"Adding literature references for {len(references)} gene sets...")
        
        ref_count = 0
        for gene_set_id, ref_list in references.items():
            gene_set_node = f"GENESET:{gene_set_id}"
            
            if gene_set_node in self.graph:
                # Add literature metadata to gene set node
                literature_data = {
                    'has_literature': True,
                    'num_references': len([ref for ref_entry in ref_list 
                                         for ref in ref_entry.get('references', [])]),
                    'keywords': [ref_entry.get('keyword', '') for ref_entry in ref_list],
                    'paragraphs': [ref_entry.get('paragraph', '') for ref_entry in ref_list]
                }
                
                self.graph.nodes[gene_set_node].update(literature_data)
                ref_count += 1
        
        logger.info(f"Added literature references for {ref_count} gene sets")
    
    def _add_go_term_validations(self, enhanced_data):
        """Add GO term validation data."""
        validations = enhanced_data.get('go_term_validations', {})
        if not validations:
            return
        
        logger.info(f"Adding GO term validations for {len(validations)} gene sets...")
        
        validation_count = 0
        for gene_set_id, validation in validations.items():
            gene_set_node = f"GENESET:{gene_set_id}"
            
            if gene_set_node in self.graph:
                # Add GO validation metadata
                go_validation = {
                    'validated_go_term': validation.get('go_term', ''),
                    'validated_go_id': validation.get('go_id', ''),
                    'go_p_value': validation.get('adjusted_p_value', 1.0),
                    'go_intersection_size': validation.get('intersection_size', 0),
                    'go_term_size': validation.get('term_size', 0),
                    'gprofiler_ji': validation.get('gprofiler_ji', 0.0),
                    'llm_go_match': validation.get('llm_best_matching_go', ''),
                    'llm_ji': validation.get('llm_ji', 0.0),
                    'llm_validation_success': validation.get('llm_success_tf', False),
                    'gprofiler_validation_success': validation.get('gprofiler_success_tf', False)
                }
                
                self.graph.nodes[gene_set_node].update(go_validation)
                
                # If there's a validated GO term, create connection
                go_id = validation.get('go_id', '')
                if go_id and go_id in self.graph:
                    self.graph.add_edge(
                        gene_set_node,
                        go_id,
                        edge_type='validated_by_go_term',
                        p_value=validation.get('adjusted_p_value', 1.0),
                        intersection_size=validation.get('intersection_size', 0),
                        validation_source='gprofiler'
                    )
                
                validation_count += 1
        
        logger.info(f"Added GO validations for {validation_count} gene sets")
    
    def _add_experimental_metadata(self, enhanced_data):
        """Add experimental metadata to gene sets."""
        metadata = enhanced_data.get('experimental_metadata', {})
        if not metadata:
            return
        
        logger.info(f"Adding experimental metadata for {len(metadata)} gene sets...")
        
        metadata_count = 0
        for gene_set_id, meta in metadata.items():
            gene_set_node = f"GENESET:{gene_set_id}"
            
            if gene_set_node in self.graph:
                # Add experimental metadata
                experimental_data = {
                    'experimental_overlap': meta.get('overlap', 0),
                    'experimental_p_value': meta.get('p_value', 1.0),
                    'experimental_genes': meta.get('genes', []),
                    'go_term_similarity': meta.get('llm_name_go_term_sim', 0.0),
                    'referenced_analysis': meta.get('referenced_analysis', '')
                }
                
                self.graph.nodes[gene_set_node].update(experimental_data)
                metadata_count += 1
        
        logger.info(f"Added experimental metadata for {metadata_count} gene sets")
    
    def _add_model_comparison_data(self):
        """Add model comparison data and LLM evaluation results to the knowledge graph."""
        logger.info("Adding model comparison data...")
        
        if 'model_compare_data' not in self.parsed_data:
            logger.info("No model comparison data available")
            return
        
        model_data = self.parsed_data['model_compare_data']
        
        # Add model nodes
        self._add_model_nodes(model_data)
        
        # Add model predictions and confidence scores
        self._add_model_predictions(model_data)
        
        # Add similarity rankings
        self._add_similarity_rankings(model_data)
        
        # Add contamination analysis
        self._add_contamination_analysis(model_data)
        
        logger.info("Model comparison data integration complete")
    
    def _add_model_nodes(self, model_data):
        """Add LLM model nodes to the graph."""
        available_models = model_data.get('available_models', [])
        
        for model_name in available_models:
            model_node_id = f"LLM_MODEL:{model_name}"
            
            # Get model statistics from evaluation metrics
            model_metrics = model_data.get('evaluation_metrics', {}).get(model_name, {})
            confidence_stats = model_metrics.get('confidence_stats', {})
            similarity_stats = model_metrics.get('similarity_stats', {})
            
            model_attrs = {
                'node_type': 'llm_model',
                'model_name': model_name,
                'entity_type': 'language_model',
                'mean_confidence': confidence_stats.get('mean_confidence', 0.0),
                'median_confidence': confidence_stats.get('median_confidence', 0.0),
                'high_confidence_count': confidence_stats.get('high_confidence_count', 0),
                'low_confidence_count': confidence_stats.get('low_confidence_count', 0),
                'mean_similarity': similarity_stats.get('mean_similarity', 0.0),
                'mean_percentile': similarity_stats.get('mean_percentile', 0.0),
                'mean_rank': similarity_stats.get('mean_rank', 0),
                'top_10_percent_count': similarity_stats.get('top_10_percent_count', 0),
                'bottom_50_percent_count': similarity_stats.get('bottom_50_percent_count', 0)
            }
            
            self.graph.add_node(model_node_id, **model_attrs)
    
    def _add_model_predictions(self, model_data):
        """Add model prediction nodes and relationships."""
        model_predictions = model_data.get('model_predictions', {})
        prediction_count = 0
        
        for model_name, model_info in model_predictions.items():
            model_node_id = f"LLM_MODEL:{model_name}"
            go_predictions = model_info.get('go_predictions', {})
            
            for go_id, prediction_data in go_predictions.items():
                if go_id in self.graph:  # Only process if GO term exists in graph
                    # Add prediction node for each scenario
                    scenarios = prediction_data.get('scenarios', {})
                    
                    for scenario, scenario_data in scenarios.items():
                        prediction_node_id = f"PREDICTION:{model_name}:{go_id}:{scenario}"
                        
                        prediction_attrs = {
                            'node_type': 'model_prediction',
                            'model_name': model_name,
                            'go_id': go_id,
                            'scenario': scenario,
                            'predicted_name': scenario_data.get('predicted_name', ''),
                            'analysis': scenario_data.get('analysis', ''),
                            'confidence_score': scenario_data.get('confidence_score', 0.0),
                            'confidence_bin': scenario_data.get('confidence_bin', ''),
                            'genes_used': scenario_data.get('genes_used', []),
                            'true_description': prediction_data.get('true_description', ''),
                            'gene_count': prediction_data.get('gene_count', 0)
                        }
                        
                        self.graph.add_node(prediction_node_id, **prediction_attrs)
                        
                        # Add edges: model -> prediction
                        self.graph.add_edge(
                            model_node_id,
                            prediction_node_id,
                            edge_type='model_predicts',
                            confidence_score=scenario_data.get('confidence_score', 0.0),
                            scenario=scenario,
                            prediction_type='go_term_prediction'
                        )
                        
                        # Add edges: prediction -> GO term
                        self.graph.add_edge(
                            prediction_node_id,
                            go_id,
                            edge_type='predicts_go_term',
                            confidence_score=scenario_data.get('confidence_score', 0.0),
                            scenario=scenario,
                            prediction_accuracy='evaluated'
                        )
                        
                        # Add edges: prediction -> genes (if genes exist in graph)
                        for gene_symbol in scenario_data.get('genes_used', []):
                            if gene_symbol in self.graph:
                                self.graph.add_edge(
                                    prediction_node_id,
                                    gene_symbol,
                                    edge_type='prediction_uses_gene',
                                    scenario=scenario,
                                    usage_context='llm_analysis'
                                )
                        
                        prediction_count += 1
        
        logger.info(f"Added {prediction_count} model predictions")
    
    def _add_similarity_rankings(self, model_data):
        """Add similarity ranking data."""
        similarity_rankings = model_data.get('similarity_rankings', {})
        ranking_count = 0
        
        for model_name, ranking_info in similarity_rankings.items():
            model_node_id = f"LLM_MODEL:{model_name}"
            similarity_metrics = ranking_info.get('similarity_metrics', {})
            
            for go_id, similarity_data in similarity_metrics.items():
                if go_id in self.graph:  # Only process if GO term exists in graph
                    ranking_node_id = f"SIMILARITY:{model_name}:{go_id}"
                    
                    ranking_attrs = {
                        'node_type': 'similarity_ranking',
                        'model_name': model_name,
                        'go_id': go_id,
                        'llm_go_similarity': similarity_data.get('llm_go_similarity', 0.0),
                        'similarity_rank': similarity_data.get('similarity_rank', 0),
                        'true_percentile': similarity_data.get('true_percentile', 0.0),
                        'random_go_name': similarity_data.get('random_comparison', {}).get('random_go_name', ''),
                        'random_similarity': similarity_data.get('random_comparison', {}).get('random_similarity', 0.0),
                        'random_rank': similarity_data.get('random_comparison', {}).get('random_rank', 0),
                        'random_percentile': similarity_data.get('random_comparison', {}).get('random_percentile', 0.0),
                        'top_3_hits': similarity_data.get('top_matches', {}).get('top_3_hits', []),
                        'top_3_similarities': similarity_data.get('top_matches', {}).get('top_3_similarities', [])
                    }
                    
                    self.graph.add_node(ranking_node_id, **ranking_attrs)
                    
                    # Add edges: model -> ranking
                    self.graph.add_edge(
                        model_node_id,
                        ranking_node_id,
                        edge_type='model_similarity_ranking',
                        similarity_score=similarity_data.get('llm_go_similarity', 0.0),
                        ranking_percentile=similarity_data.get('true_percentile', 0.0)
                    )
                    
                    # Add edges: ranking -> GO term
                    self.graph.add_edge(
                        ranking_node_id,
                        go_id,
                        edge_type='similarity_evaluated_for',
                        similarity_score=similarity_data.get('llm_go_similarity', 0.0),
                        ranking_position=similarity_data.get('similarity_rank', 0),
                        percentile_rank=similarity_data.get('true_percentile', 0.0)
                    )
                    
                    ranking_count += 1
        
        logger.info(f"Added {ranking_count} similarity rankings")
    
    def _add_contamination_analysis(self, model_data):
        """Add contamination analysis results."""
        contamination_results = model_data.get('contamination_results', {})
        
        for model_name, contamination_info in contamination_results.items():
            model_node_id = f"LLM_MODEL:{model_name}"
            
            if model_node_id in self.graph:
                # Add contamination analysis metadata to model node
                contamination_attrs = {
                    'contamination_robustness_score': contamination_info.get('robustness_score', 0.0),
                    'severe_drops': contamination_info.get('performance_degradation', {}).get('severe_drop', 0),
                    'moderate_drops': contamination_info.get('performance_degradation', {}).get('moderate_drop', 0),
                    'stable_performance': contamination_info.get('performance_degradation', {}).get('stable', 0),
                    'performance_improvements': contamination_info.get('performance_degradation', {}).get('improvement', 0)
                }
                
                self.graph.nodes[model_node_id].update(contamination_attrs)
        
        logger.info("Added contamination analysis results")
    
    def _add_cc_mf_branch_data(self):
        """Add CC_MF_Branch data (CC and MF GO terms with LLM interpretations) to the knowledge graph."""
        logger.info("Adding CC_MF_Branch data...")
        
        if 'cc_mf_branch_data' not in self.parsed_data:
            logger.info("No CC_MF_Branch data available")
            return
        
        cc_mf_data = self.parsed_data['cc_mf_branch_data']
        
        # Add CC and MF GO term nodes
        self._add_cc_mf_go_term_nodes(cc_mf_data)
        
        # Add gene-GO term associations for CC and MF
        self._add_cc_mf_gene_associations(cc_mf_data)
        
        # Add LLM interpretations and confidence scores
        self._add_cc_mf_llm_interpretations(cc_mf_data)
        
        # Add similarity rankings for CC and MF terms
        self._add_cc_mf_similarity_rankings(cc_mf_data)
        
        logger.info("CC_MF_Branch data integration complete")
    
    def _add_cc_mf_go_term_nodes(self, cc_mf_data):
        """Add CC and MF GO term nodes to the graph."""
        cc_terms = cc_mf_data.get('cc_go_terms', {})
        mf_terms = cc_mf_data.get('mf_go_terms', {})
        
        # Add CC GO terms
        for go_id, term_data in cc_terms.items():
            node_attrs = {
                'node_type': 'go_term',
                'name': term_data.get('term_description', ''),
                'namespace': 'CC',
                'description': term_data.get('term_description', ''),
                'gene_count': term_data.get('gene_count', 0),
                'source': 'CC_MF_branch'
            }
            self.graph.add_node(go_id, **node_attrs)
        
        # Add MF GO terms  
        for go_id, term_data in mf_terms.items():
            node_attrs = {
                'node_type': 'go_term',
                'name': term_data.get('term_description', ''),
                'namespace': 'MF',
                'description': term_data.get('term_description', ''),
                'gene_count': term_data.get('gene_count', 0),
                'source': 'CC_MF_branch'
            }
            self.graph.add_node(go_id, **node_attrs)
        
        logger.info(f"Added {len(cc_terms)} CC and {len(mf_terms)} MF GO terms")
    
    def _add_cc_mf_gene_associations(self, cc_mf_data):
        """Add gene-GO term associations for CC and MF terms."""
        cc_terms = cc_mf_data.get('cc_go_terms', {})
        mf_terms = cc_mf_data.get('mf_go_terms', {})
        
        gene_association_count = 0
        
        # Process CC terms
        for go_id, term_data in cc_terms.items():
            genes = term_data.get('genes', [])
            for gene_symbol in genes:
                # Ensure gene node exists
                gene_node_id = f"GENE:{gene_symbol}"
                if gene_node_id not in self.graph:
                    self.graph.add_node(gene_node_id, 
                                      node_type='gene',
                                      gene_symbol=gene_symbol,
                                      entity_type='gene')
                
                # Add association edge
                self.graph.add_edge(
                    gene_node_id,
                    go_id,
                    edge_type='gene_go_association',
                    namespace='CC',
                    association_type='annotated_to',
                    source='CC_MF_branch'
                )
                gene_association_count += 1
        
        # Process MF terms
        for go_id, term_data in mf_terms.items():
            genes = term_data.get('genes', [])
            for gene_symbol in genes:
                # Ensure gene node exists
                gene_node_id = f"GENE:{gene_symbol}"
                if gene_node_id not in self.graph:
                    self.graph.add_node(gene_node_id,
                                      node_type='gene', 
                                      gene_symbol=gene_symbol,
                                      entity_type='gene')
                
                # Add association edge
                self.graph.add_edge(
                    gene_node_id,
                    go_id,
                    edge_type='gene_go_association',
                    namespace='MF',
                    association_type='annotated_to',
                    source='CC_MF_branch'
                )
                gene_association_count += 1
        
        logger.info(f"Added {gene_association_count} CC/MF gene-GO associations")
    
    def _add_cc_mf_llm_interpretations(self, cc_mf_data):
        """Add LLM interpretations and confidence scores for CC and MF terms."""
        cc_interpretations = cc_mf_data.get('cc_llm_interpretations', {})
        mf_interpretations = cc_mf_data.get('mf_llm_interpretations', {})
        
        interpretation_count = 0
        
        # Process CC interpretations
        for go_id, interp_data in cc_interpretations.items():
            interpretation_node_id = f"CC_INTERPRETATION:{go_id}"
            
            interp_attrs = {
                'node_type': 'llm_interpretation',
                'go_term_id': go_id,
                'namespace': 'CC',
                'llm_name': interp_data.get('llm_name', ''),
                'llm_analysis': interp_data.get('llm_analysis', ''),
                'llm_score': interp_data.get('llm_score', 0.0),
                'term_description': interp_data.get('term_description', ''),
                'gene_count': interp_data.get('gene_count', 0),
                'source': 'CC_MF_branch'
            }
            
            self.graph.add_node(interpretation_node_id, **interp_attrs)
            
            # Link interpretation to GO term
            if go_id in self.graph:
                self.graph.add_edge(
                    interpretation_node_id,
                    go_id,
                    edge_type='interprets_go_term',
                    confidence_score=interp_data.get('llm_score', 0.0),
                    namespace='CC'
                )
            
            interpretation_count += 1
        
        # Process MF interpretations
        for go_id, interp_data in mf_interpretations.items():
            interpretation_node_id = f"MF_INTERPRETATION:{go_id}"
            
            interp_attrs = {
                'node_type': 'llm_interpretation',
                'go_term_id': go_id,
                'namespace': 'MF',
                'llm_name': interp_data.get('llm_name', ''),
                'llm_analysis': interp_data.get('llm_analysis', ''),
                'llm_score': interp_data.get('llm_score', 0.0),
                'term_description': interp_data.get('term_description', ''),
                'gene_count': interp_data.get('gene_count', 0),
                'source': 'CC_MF_branch'
            }
            
            self.graph.add_node(interpretation_node_id, **interp_attrs)
            
            # Link interpretation to GO term
            if go_id in self.graph:
                self.graph.add_edge(
                    interpretation_node_id,
                    go_id,
                    edge_type='interprets_go_term',
                    confidence_score=interp_data.get('llm_score', 0.0),
                    namespace='MF'
                )
            
            interpretation_count += 1
        
        logger.info(f"Added {interpretation_count} CC/MF LLM interpretations")
    
    def _add_cc_mf_similarity_rankings(self, cc_mf_data):
        """Add similarity rankings for CC and MF terms."""
        cc_rankings = cc_mf_data.get('cc_similarity_rankings', {})
        mf_rankings = cc_mf_data.get('mf_similarity_rankings', {})
        
        ranking_count = 0
        
        # Process CC similarity rankings
        for go_id, ranking_data in cc_rankings.items():
            ranking_node_id = f"CC_SIMILARITY_RANKING:{go_id}"
            
            ranking_attrs = {
                'node_type': 'similarity_ranking',
                'go_term_id': go_id,
                'namespace': 'CC',
                'llm_name_go_term_sim': ranking_data.get('llm_name_go_term_sim', 0.0),
                'sim_rank': ranking_data.get('sim_rank', 0),
                'true_go_term_sim_percentile': ranking_data.get('true_go_term_sim_percentile', 0.0),
                'random_go_name': ranking_data.get('random_go_name', ''),
                'random_go_llm_sim': ranking_data.get('random_go_llm_sim', 0.0),
                'random_sim_rank': ranking_data.get('random_sim_rank', 0),
                'random_sim_percentile': ranking_data.get('random_sim_percentile', 0.0),
                'top_3_hits': ranking_data.get('top_3_hits', []),
                'top_3_similarities': ranking_data.get('top_3_similarities', []),
                'source': 'CC_MF_branch'
            }
            
            self.graph.add_node(ranking_node_id, **ranking_attrs)
            
            # Link ranking to GO term
            if go_id in self.graph:
                self.graph.add_edge(
                    ranking_node_id,
                    go_id,
                    edge_type='similarity_ranking_for',
                    similarity_score=ranking_data.get('llm_name_go_term_sim', 0.0),
                    percentile_rank=ranking_data.get('true_go_term_sim_percentile', 0.0),
                    namespace='CC'
                )
            
            ranking_count += 1
        
        # Process MF similarity rankings
        for go_id, ranking_data in mf_rankings.items():
            ranking_node_id = f"MF_SIMILARITY_RANKING:{go_id}"
            
            ranking_attrs = {
                'node_type': 'similarity_ranking',
                'go_term_id': go_id,
                'namespace': 'MF',
                'llm_name_go_term_sim': ranking_data.get('llm_name_go_term_sim', 0.0),
                'sim_rank': ranking_data.get('sim_rank', 0),
                'true_go_term_sim_percentile': ranking_data.get('true_go_term_sim_percentile', 0.0),
                'random_go_name': ranking_data.get('random_go_name', ''),
                'random_go_llm_sim': ranking_data.get('random_go_llm_sim', 0.0),
                'random_sim_rank': ranking_data.get('random_sim_rank', 0),
                'random_sim_percentile': ranking_data.get('random_sim_percentile', 0.0),
                'top_3_hits': ranking_data.get('top_3_hits', []),
                'top_3_similarities': ranking_data.get('top_3_similarities', []),
                'source': 'CC_MF_branch'
            }
            
            self.graph.add_node(ranking_node_id, **ranking_attrs)
            
            # Link ranking to GO term
            if go_id in self.graph:
                self.graph.add_edge(
                    ranking_node_id,
                    go_id,
                    edge_type='similarity_ranking_for',
                    similarity_score=ranking_data.get('llm_name_go_term_sim', 0.0),
                    percentile_rank=ranking_data.get('true_go_term_sim_percentile', 0.0),
                    namespace='MF'
                )
            
            ranking_count += 1
        
        logger.info(f"Added {ranking_count} CC/MF similarity rankings")
    
    def _calculate_comprehensive_stats(self):
        """Calculate comprehensive statistics for the integrated graph."""
        logger.info("Calculating comprehensive statistics...")
        
        # Count nodes by type
        node_counts = {}
        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data.get('node_type', 'unknown')
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
        
        # Count edges by type
        edge_counts = {}
        for source, target, edge_data in self.graph.edges(data=True):
            edge_type = edge_data.get('edge_type', 'unknown')
            edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1
        
        # Calculate integration metrics
        go_genes = set()
        omics_genes = set()
        
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('node_type') == 'gene':
                # Check if connected to GO terms
                for neighbor in self.graph.neighbors(node_id):
                    neighbor_data = self.graph.nodes.get(neighbor, {})
                    if neighbor_data.get('node_type') == 'go_term':
                        go_genes.add(node_id)
                    elif neighbor_data.get('node_type') in ['disease', 'drug', 'viral_condition']:
                        omics_genes.add(node_id)
        
        integrated_genes = go_genes & omics_genes
        
        self.stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_counts': node_counts,
            'edge_counts': edge_counts,
            'integration_metrics': {
                'go_connected_genes': len(go_genes),
                'omics_connected_genes': len(omics_genes),
                'integrated_genes': len(integrated_genes),
                'integration_ratio': len(integrated_genes) / len(go_genes) if go_genes else 0
            }
        }
        
        logger.info(f"Comprehensive graph statistics calculated: {self.stats}")
    
    def _validate_comprehensive_graph(self):
        """Validate the comprehensive biomedical knowledge graph."""
        logger.info("Validating comprehensive graph...")
        
        validation = {
            'has_nodes': self.graph.number_of_nodes() > 0,
            'has_edges': self.graph.number_of_edges() > 0,
            'has_go_component': False,
            'has_omics_component': False,
            'has_model_comparison_component': False,
            'integration_successful': False
        }
        
        # Check for GO component
        go_terms = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'go_term']
        validation['has_go_component'] = len(go_terms) > 0
        
        # Check for Omics component
        omics_nodes = [n for n, d in self.graph.nodes(data=True) 
                      if d.get('node_type') in ['disease', 'drug', 'viral_condition', 'network_cluster']]
        validation['has_omics_component'] = len(omics_nodes) > 0
        
        # Check for Model Comparison component
        model_nodes = [n for n, d in self.graph.nodes(data=True) 
                      if d.get('node_type') in ['llm_model', 'model_prediction', 'similarity_ranking']]
        validation['has_model_comparison_component'] = len(model_nodes) > 0
        
        # Check integration
        integration_metrics = self.stats.get('integration_metrics', {})
        validation['integration_successful'] = integration_metrics.get('integrated_genes', 0) > 0
        
        validation['overall_valid'] = all(validation.values())
        
        if validation['overall_valid']:
            logger.info("✅ Comprehensive graph validation passed")
        else:
            logger.warning(f"⚠️ Comprehensive graph validation issues: {validation}")
        
        return validation
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive graph statistics."""
        return self.stats.copy()
    
    def query_gene_comprehensive(self, gene_symbol: str) -> Dict:
        """
        Query all associations for a gene across GO and Omics data.
        
        Args:
            gene_symbol: Gene symbol to query
            
        Returns:
            Dictionary with comprehensive gene information
        """
        if gene_symbol not in self.graph:
            return {}
        
        results = {
            'gene_symbol': gene_symbol,
            'go_annotations': [],
            'disease_associations': [],
            'drug_perturbations': [],
            'viral_responses': [],
            'cluster_memberships': [],
            'gene_set_memberships': [],
            'semantic_annotations': [],
            'model_predictions': []
        }
        
        # Get all neighbors and their relationships
        for neighbor in self.graph.neighbors(gene_symbol):
            edge_data = self.graph[gene_symbol][neighbor]
            neighbor_data = self.graph.nodes[neighbor]
            
            # Handle multiple edges between same nodes
            for edge_key, edge_attrs in edge_data.items():
                edge_type = edge_attrs.get('edge_type')
                
                if edge_type == 'gene_annotation':
                    results['go_annotations'].append({
                        'go_id': neighbor,
                        'go_name': neighbor_data.get('name', ''),
                        'namespace': neighbor_data.get('namespace', ''),
                        'evidence_code': edge_attrs.get('evidence_code', '')
                    })
                
                elif edge_type == 'gene_disease_association':
                    results['disease_associations'].append({
                        'disease': neighbor_data.get('disease_name', ''),
                        'condition': edge_attrs.get('disease_condition', ''),
                        'gse_id': edge_attrs.get('gse_id', '')
                    })
                
                elif edge_type == 'gene_drug_perturbation':
                    results['drug_perturbations'].append({
                        'drug': neighbor_data.get('drug_name', ''),
                        'condition': edge_attrs.get('drug_condition', ''),
                        'perturbation_id': edge_attrs.get('perturbation_id', '')
                    })
                
                elif edge_type == 'gene_viral_response':
                    results['viral_responses'].append({
                        'viral_condition': neighbor_data.get('viral_condition', ''),
                        'condition': edge_attrs.get('viral_condition', ''),
                        'gse_id': edge_attrs.get('gse_id', ''),
                        'type': 'response'
                    })
                
                elif edge_type == 'gene_viral_expression':
                    results['viral_responses'].append({
                        'viral_condition': neighbor_data.get('viral_condition', ''),
                        'condition': edge_attrs.get('condition', ''),
                        'expression_value': edge_attrs.get('expression_value', 0),
                        'expression_direction': edge_attrs.get('expression_direction', ''),
                        'expression_magnitude': edge_attrs.get('expression_magnitude', 0),
                        'type': 'expression'
                    })
                
                elif edge_type == 'gene_in_set':
                    results['gene_set_memberships'].append({
                        'gene_set_id': neighbor_data.get('gene_set_id', ''),
                        'gene_set_name': neighbor_data.get('gene_set_name', ''),
                        'llm_name': neighbor_data.get('llm_name', ''),
                        'llm_score': neighbor_data.get('llm_score', 0.0),
                        'support_level': edge_attrs.get('support_level', ''),
                        'membership_type': 'annotated'
                    })
                
                elif edge_type == 'gene_supports_set':
                    results['gene_set_memberships'].append({
                        'gene_set_id': neighbor_data.get('gene_set_id', ''),
                        'gene_set_name': neighbor_data.get('gene_set_name', ''),
                        'llm_name': neighbor_data.get('llm_name', ''),
                        'llm_score': neighbor_data.get('llm_score', 0.0),
                        'support_level': edge_attrs.get('support_level', ''),
                        'membership_type': 'high_confidence_support'
                    })
                
                elif edge_type == 'prediction_uses_gene':
                    results['model_predictions'].append({
                        'prediction_id': neighbor,
                        'model_name': neighbor_data.get('model_name', ''),
                        'go_id': neighbor_data.get('go_id', ''),
                        'scenario': neighbor_data.get('scenario', ''),
                        'predicted_name': neighbor_data.get('predicted_name', ''),
                        'confidence_score': neighbor_data.get('confidence_score', 0.0),
                        'confidence_bin': neighbor_data.get('confidence_bin', ''),
                        'usage_context': edge_attrs.get('usage_context', ''),
                        'true_description': neighbor_data.get('true_description', '')
                    })
        
        return results
    
    def query_model_predictions(self, go_id: str = None, model_name: str = None) -> List[Dict]:
        """
        Query model predictions with optional filtering.
        
        Args:
            go_id: Optional GO term ID to filter predictions
            model_name: Optional model name to filter predictions
            
        Returns:
            List of model predictions with metadata
        """
        results = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('node_type') == 'model_prediction':
                # Apply filters if provided
                if go_id and node_data.get('go_id') != go_id:
                    continue
                if model_name and node_data.get('model_name') != model_name:
                    continue
                
                prediction_info = {
                    'prediction_id': node_id,
                    'model_name': node_data.get('model_name', ''),
                    'go_id': node_data.get('go_id', ''),
                    'scenario': node_data.get('scenario', ''),
                    'predicted_name': node_data.get('predicted_name', ''),
                    'analysis': node_data.get('analysis', ''),
                    'confidence_score': node_data.get('confidence_score', 0.0),
                    'confidence_bin': node_data.get('confidence_bin', ''),
                    'genes_used': node_data.get('genes_used', []),
                    'true_description': node_data.get('true_description', ''),
                    'gene_count': node_data.get('gene_count', 0)
                }
                
                results.append(prediction_info)
        
        # Sort by confidence score (highest first)
        results.sort(key=lambda x: x['confidence_score'], reverse=True)
        return results
    
    def query_model_comparison_summary(self) -> Dict:
        """
        Get a summary of model comparison data in the knowledge graph.
        
        Returns:
            Dictionary with model comparison statistics
        """
        summary = {
            'total_models': 0,
            'total_predictions': 0,
            'total_similarity_rankings': 0,
            'model_performance': {},
            'scenario_coverage': {},
            'go_term_coverage': 0
        }
        
        # Count models
        models = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'llm_model']
        summary['total_models'] = len(models)
        
        # Count predictions and analyze by model
        predictions = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'model_prediction']
        summary['total_predictions'] = len(predictions)
        
        # Count similarity rankings
        rankings = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'similarity_ranking']
        summary['total_similarity_rankings'] = len(rankings)
        
        # Analyze model performance
        for model_node in models:
            model_data = self.graph.nodes[model_node]
            model_name = model_data.get('model_name', '')
            
            summary['model_performance'][model_name] = {
                'mean_confidence': model_data.get('mean_confidence', 0.0),
                'mean_similarity': model_data.get('mean_similarity', 0.0),
                'mean_percentile': model_data.get('mean_percentile', 0.0),
                'contamination_robustness': model_data.get('contamination_robustness_score', 0.0),
                'stable_performance': model_data.get('stable_performance', 0),
                'severe_drops': model_data.get('severe_drops', 0)
            }
        
        # Count scenarios
        scenario_counts = {}
        go_terms_with_predictions = set()
        
        for prediction_node in predictions:
            prediction_data = self.graph.nodes[prediction_node]
            scenario = prediction_data.get('scenario', '')
            go_id = prediction_data.get('go_id', '')
            
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
            if go_id:
                go_terms_with_predictions.add(go_id)
        
        summary['scenario_coverage'] = scenario_counts
        summary['go_term_coverage'] = len(go_terms_with_predictions)
        
        return summary
    
    def query_cc_mf_terms(self, namespace: str = None) -> List[Dict]:
        """Query CC and MF GO terms.
        
        Args:
            namespace: 'CC', 'MF', or None for both
            
        Returns:
            List of GO term information
        """
        terms = []
        
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get('node_type') == 'go_term' and attrs.get('source') == 'CC_MF_branch':
                if namespace is None or attrs.get('namespace') == namespace:
                    term_info = {
                        'go_id': node_id,
                        'name': attrs.get('name', ''),
                        'namespace': attrs.get('namespace'),
                        'description': attrs.get('description', ''),
                        'gene_count': attrs.get('gene_count', 0)
                    }
                    terms.append(term_info)
        
        return sorted(terms, key=lambda x: x['gene_count'], reverse=True)
    
    def query_cc_mf_llm_interpretations(self, go_id: str = None, namespace: str = None) -> List[Dict]:
        """Query LLM interpretations for CC and MF terms.
        
        Args:
            go_id: Specific GO term to query
            namespace: 'CC', 'MF', or None for both
            
        Returns:
            List of LLM interpretation information
        """
        interpretations = []
        
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get('node_type') == 'llm_interpretation' and attrs.get('source') == 'CC_MF_branch':
                if go_id is None or attrs.get('go_term_id') == go_id:
                    if namespace is None or attrs.get('namespace') == namespace:
                        interp_info = {
                            'interpretation_id': node_id,
                            'go_term_id': attrs.get('go_term_id'),
                            'namespace': attrs.get('namespace'),
                            'llm_name': attrs.get('llm_name', ''),
                            'llm_analysis': attrs.get('llm_analysis', ''),
                            'llm_score': attrs.get('llm_score', 0.0),
                            'term_description': attrs.get('term_description', ''),
                            'gene_count': attrs.get('gene_count', 0)
                        }
                        interpretations.append(interp_info)
        
        return sorted(interpretations, key=lambda x: x['llm_score'], reverse=True)
    
    def query_cc_mf_similarity_rankings(self, go_id: str = None, namespace: str = None) -> List[Dict]:
        """Query similarity rankings for CC and MF terms.
        
        Args:
            go_id: Specific GO term to query
            namespace: 'CC', 'MF', or None for both
            
        Returns:
            List of similarity ranking information
        """
        rankings = []
        
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get('node_type') == 'similarity_ranking' and attrs.get('source') == 'CC_MF_branch':
                if go_id is None or attrs.get('go_term_id') == go_id:
                    if namespace is None or attrs.get('namespace') == namespace:
                        ranking_info = {
                            'ranking_id': node_id,
                            'go_term_id': attrs.get('go_term_id'),
                            'namespace': attrs.get('namespace'),
                            'llm_name_go_term_sim': attrs.get('llm_name_go_term_sim', 0.0),
                            'sim_rank': attrs.get('sim_rank', 0),
                            'true_go_term_sim_percentile': attrs.get('true_go_term_sim_percentile', 0.0),
                            'random_go_name': attrs.get('random_go_name', ''),
                            'top_3_hits': attrs.get('top_3_hits', []),
                            'top_3_similarities': attrs.get('top_3_similarities', [])
                        }
                        rankings.append(ranking_info)
        
        return sorted(rankings, key=lambda x: x['true_go_term_sim_percentile'], reverse=True)
    
    def query_gene_cc_mf_profile(self, gene_symbol: str) -> Dict[str, Any]:
        """Query comprehensive CC and MF profile for a gene.
        
        Args:
            gene_symbol: Gene symbol to query
            
        Returns:
            Dictionary containing CC and MF associations, interpretations, and rankings
        """
        gene_node_id = f"GENE:{gene_symbol}"
        
        if gene_node_id not in self.graph:
            return {'error': f'Gene {gene_symbol} not found in knowledge graph'}
        
        profile = {
            'gene_symbol': gene_symbol,
            'cc_associations': [],
            'mf_associations': [],
            'cc_interpretations': [],
            'mf_interpretations': [],
            'cc_similarity_rankings': [],
            'mf_similarity_rankings': []
        }
        
        # Find GO term associations
        for neighbor in self.graph.neighbors(gene_node_id):
            edge_data = self.graph.get_edge_data(gene_node_id, neighbor, 0)
            
            if edge_data and edge_data.get('edge_type') == 'gene_go_association':
                namespace = edge_data.get('namespace')
                go_node_attrs = self.graph.nodes[neighbor]
                
                association_info = {
                    'go_id': neighbor,
                    'name': go_node_attrs.get('name', ''),
                    'description': go_node_attrs.get('description', ''),
                    'gene_count': go_node_attrs.get('gene_count', 0)
                }
                
                if namespace == 'CC':
                    profile['cc_associations'].append(association_info)
                elif namespace == 'MF':
                    profile['mf_associations'].append(association_info)
                
                # Find related interpretations and rankings
                go_id = neighbor
                interpretations = self.query_cc_mf_llm_interpretations(go_id=go_id, namespace=namespace)
                rankings = self.query_cc_mf_similarity_rankings(go_id=go_id, namespace=namespace)
                
                if namespace == 'CC':
                    profile['cc_interpretations'].extend(interpretations)
                    profile['cc_similarity_rankings'].extend(rankings)
                elif namespace == 'MF':
                    profile['mf_interpretations'].extend(interpretations)
                    profile['mf_similarity_rankings'].extend(rankings)
        
        # Sort results
        profile['cc_associations'] = sorted(profile['cc_associations'], key=lambda x: x['gene_count'], reverse=True)
        profile['mf_associations'] = sorted(profile['mf_associations'], key=lambda x: x['gene_count'], reverse=True)
        profile['cc_interpretations'] = sorted(profile['cc_interpretations'], key=lambda x: x['llm_score'], reverse=True)
        profile['mf_interpretations'] = sorted(profile['mf_interpretations'], key=lambda x: x['llm_score'], reverse=True)
        
        return profile
    
    def get_cc_mf_branch_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for CC_MF_Branch data integration."""
        stats = {
            'cc_go_terms': 0,
            'mf_go_terms': 0,
            'cc_interpretations': 0,
            'mf_interpretations': 0,
            'cc_similarity_rankings': 0,
            'mf_similarity_rankings': 0,
            'cc_mf_gene_associations': 0,
            'unique_cc_mf_genes': 0
        }
        
        unique_genes = set()
        
        for node_id, attrs in self.graph.nodes(data=True):
            source = attrs.get('source')
            node_type = attrs.get('node_type')
            namespace = attrs.get('namespace')
            
            if source == 'CC_MF_branch':
                if node_type == 'go_term':
                    if namespace == 'CC':
                        stats['cc_go_terms'] += 1
                    elif namespace == 'MF':
                        stats['mf_go_terms'] += 1
                elif node_type == 'llm_interpretation':
                    if namespace == 'CC':
                        stats['cc_interpretations'] += 1
                    elif namespace == 'MF':
                        stats['mf_interpretations'] += 1
                elif node_type == 'similarity_ranking':
                    if namespace == 'CC':
                        stats['cc_similarity_rankings'] += 1
                    elif namespace == 'MF':
                        stats['mf_similarity_rankings'] += 1
        
        # Count gene associations and unique genes
        for u, v, edge_attrs in self.graph.edges(data=True):
            if (edge_attrs.get('source') == 'CC_MF_branch' and 
                edge_attrs.get('edge_type') == 'gene_go_association'):
                stats['cc_mf_gene_associations'] += 1
                
                # Extract gene symbol from node ID
                if u.startswith('GENE:'):
                    unique_genes.add(u.split(':', 1)[1])
        
        stats['unique_cc_mf_genes'] = len(unique_genes)
        stats['total_cc_mf_terms'] = stats['cc_go_terms'] + stats['mf_go_terms']
        stats['total_cc_mf_interpretations'] = stats['cc_interpretations'] + stats['mf_interpretations']
        stats['total_cc_mf_rankings'] = stats['cc_similarity_rankings'] + stats['mf_similarity_rankings']
        
        return stats
    
    # ============= LLM_PROCESSED DATA INTEGRATION METHODS =============
    
    def _add_llm_processed_data(self):
        """Add LLM_processed data (multi-model interpretations, contamination analysis, similarity rankings) to the knowledge graph."""
        logger.info("Adding LLM_processed data...")
        
        if 'llm_processed_data' not in self.parsed_data:
            logger.info("No LLM_processed data available")
            return
        
        llm_data = self.parsed_data['llm_processed_data']
        
        # Add main LLM interpretation nodes
        self._add_llm_interpretation_nodes(llm_data)
        
        # Add contamination analysis nodes for multiple models
        self._add_contamination_analysis_nodes(llm_data)
        
        # Add similarity ranking nodes
        self._add_llm_similarity_ranking_nodes(llm_data)
        
        # Add similarity p-value nodes
        self._add_similarity_pvalue_nodes(llm_data)
        
        # Add model comparison nodes
        self._add_model_comparison_nodes(llm_data)
        
        # Connect LLM data to existing GO terms and genes
        self._connect_llm_data_to_graph(llm_data)
        
        logger.info("LLM_processed data integration complete")
    
    def _add_llm_interpretation_nodes(self, llm_data):
        """Add main LLM interpretation nodes to the graph."""
        main_interpretations = llm_data.get('main_interpretations', {})
        
        for dataset_name, interpretations in main_interpretations.items():
            for go_id, interp_data in interpretations.items():
                # Create interpretation node
                interp_id = f"llm_interp_{dataset_name}_{go_id}"
                
                node_attrs = {
                    'node_type': 'llm_interpretation',
                    'source': 'LLM_processed',
                    'dataset': dataset_name,
                    'go_term_id': go_id,
                    'model': interp_data.get('model', 'gpt_4'),
                    'llm_name': interp_data.get('llm_name', ''),
                    'llm_analysis': interp_data.get('llm_analysis', ''),
                    'llm_score': interp_data.get('llm_score', 0.0),
                    'gene_count': interp_data.get('gene_count', 0),
                    'term_description': interp_data.get('term_description', '')
                }
                
                self.graph.add_node(interp_id, **node_attrs)
                
                # Connect to GO term if it exists
                if go_id in self.graph.nodes:
                    self.graph.add_edge(interp_id, go_id, 
                                      edge_type='interprets',
                                      source='LLM_processed',
                                      model=node_attrs['model'])
                
                # Connect to genes if they exist
                for gene_symbol in interp_data.get('genes', []):
                    if gene_symbol in self.graph.nodes:
                        self.graph.add_edge(interp_id, gene_symbol,
                                          edge_type='interprets_gene',
                                          source='LLM_processed',
                                          model=node_attrs['model'])
    
    def _add_contamination_analysis_nodes(self, llm_data):
        """Add contamination analysis nodes for multiple models to the graph."""
        contamination_analysis = llm_data.get('contamination_analysis', {})
        
        for model_name, model_analysis in contamination_analysis.items():
            for go_id, analysis_data in model_analysis.items():
                # Create contamination analysis node for each model
                contam_id = f"contam_analysis_{model_name}_{go_id}"
                
                node_attrs = {
                    'node_type': 'contamination_analysis',
                    'source': 'LLM_processed',
                    'model': model_name,
                    'go_term_id': go_id,
                    'gene_count': analysis_data.get('gene_count', 0),
                    'term_description': analysis_data.get('term_description', ''),
                    'scenarios': len(analysis_data.get('scenarios', {}))
                }
                
                # Add scenario-specific data
                scenarios = analysis_data.get('scenarios', {})
                for scenario, scenario_data in scenarios.items():
                    node_attrs[f'{scenario}_name'] = scenario_data.get('name', '')
                    node_attrs[f'{scenario}_score'] = scenario_data.get('score', 0.0)
                    node_attrs[f'{scenario}_analysis'] = scenario_data.get('analysis', '')
                
                self.graph.add_node(contam_id, **node_attrs)
                
                # Connect to GO term if it exists
                if go_id in self.graph.nodes:
                    self.graph.add_edge(contam_id, go_id,
                                      edge_type='contamination_analysis',
                                      source='LLM_processed',
                                      model=model_name)
                
                # Connect to genes
                for gene_symbol in analysis_data.get('genes', []):
                    if gene_symbol in self.graph.nodes:
                        self.graph.add_edge(contam_id, gene_symbol,
                                          edge_type='contamination_analysis_gene',
                                          source='LLM_processed',
                                          model=model_name)
    
    def _add_llm_similarity_ranking_nodes(self, llm_data):
        """Add LLM similarity ranking nodes to the graph."""
        similarity_rankings = llm_data.get('similarity_rankings', {})
        
        for dataset_name, rankings in similarity_rankings.items():
            for go_id, ranking_data in rankings.items():
                # Create similarity ranking node
                ranking_id = f"sim_rank_{dataset_name}_{go_id}"
                
                node_attrs = {
                    'node_type': 'similarity_ranking',
                    'source': 'LLM_processed',
                    'dataset': dataset_name,
                    'go_term_id': go_id,
                    'llm_name': ranking_data.get('llm_name', ''),
                    'llm_score': ranking_data.get('llm_score', 0.0),
                    'score_bin': ranking_data.get('score_bin', ''),
                    'llm_go_similarity': ranking_data.get('llm_go_similarity', 0.0),
                    'similarity_rank': ranking_data.get('similarity_rank', 0),
                    'similarity_percentile': ranking_data.get('similarity_percentile', 0.0),
                    'random_go_name': ranking_data.get('random_go_name', ''),
                    'random_go_similarity': ranking_data.get('random_go_similarity', 0.0),
                    'random_similarity_rank': ranking_data.get('random_similarity_rank', 0),
                    'random_similarity_percentile': ranking_data.get('random_similarity_percentile', 0.0),
                    'top_3_hits': str(ranking_data.get('top_3_hits', [])),
                    'top_3_similarities': str(ranking_data.get('top_3_similarities', []))
                }
                
                self.graph.add_node(ranking_id, **node_attrs)
                
                # Connect to GO term if it exists
                if go_id in self.graph.nodes:
                    self.graph.add_edge(ranking_id, go_id,
                                      edge_type='similarity_ranking',
                                      source='LLM_processed',
                                      similarity_percentile=node_attrs['similarity_percentile'])
    
    def _add_similarity_pvalue_nodes(self, llm_data):
        """Add similarity p-value nodes to the graph."""
        similarity_pvalues = llm_data.get('similarity_pvalues', {})
        
        for go_id, pval_data in similarity_pvalues.items():
            # Create p-value node
            pval_id = f"sim_pval_{go_id}"
            
            node_attrs = {
                'node_type': 'similarity_pvalues',
                'source': 'LLM_processed',
                'go_term_id': go_id,
                'gene_count': pval_data.get('gene_count', 0),
                'term_description': pval_data.get('term_description', '')
            }
            
            # Add p-value data
            pvalues = pval_data.get('pvalues', {})
            for pval_key, pval_value in pvalues.items():
                node_attrs[pval_key] = pval_value
            
            self.graph.add_node(pval_id, **node_attrs)
            
            # Connect to GO term if it exists
            if go_id in self.graph.nodes:
                self.graph.add_edge(pval_id, go_id,
                                  edge_type='similarity_pvalues',
                                  source='LLM_processed')
    
    def _add_model_comparison_nodes(self, llm_data):
        """Add model comparison nodes to the graph."""
        model_comparison = llm_data.get('model_comparison', {})
        
        for go_id, comp_data in model_comparison.items():
            # Create model comparison node
            comp_id = f"model_comp_{go_id}"
            
            node_attrs = {
                'node_type': 'model_comparison',
                'source': 'LLM_processed',
                'go_term_id': go_id,
                'gene_count': comp_data.get('gene_count', 0),
                'term_description': comp_data.get('term_description', ''),
                'contaminated_genes_50perc_count': len(comp_data.get('contaminated_genes_50perc', [])),
                'contaminated_genes_100perc_count': len(comp_data.get('contaminated_genes_100perc', []))
            }
            
            self.graph.add_node(comp_id, **node_attrs)
            
            # Connect to GO term if it exists
            if go_id in self.graph.nodes:
                self.graph.add_edge(comp_id, go_id,
                                  edge_type='model_comparison',
                                  source='LLM_processed')
    
    def _connect_llm_data_to_graph(self, llm_data):
        """Create additional connections between LLM data and existing graph elements."""
        # This method can be used for creating more complex relationships
        # between LLM interpretations, contamination analysis, and existing nodes
        logger.info("Creating additional LLM data connections...")
        
        # Count genes mentioned in LLM data
        genes_in_llm = set()
        for dataset in llm_data.get('main_interpretations', {}).values():
            for interp in dataset.values():
                genes_in_llm.update(interp.get('genes', []))
        
        logger.info(f"LLM data covers {len(genes_in_llm)} unique genes")
    
    # ============= GO ANALYSIS DATA INTEGRATION =============
    
    def _add_go_analysis_data(self):
        """Add GO Analysis Data (core terms, contamination, confidence evaluations, hierarchy) to the knowledge graph."""
        logger.info("Adding GO Analysis Data...")
        
        if 'go_analysis_data' not in self.parsed_data:
            logger.info("No GO Analysis Data available")
            return
        
        analysis_data = self.parsed_data['go_analysis_data']
        
        # Add core GO term analysis nodes
        self._add_core_go_analysis_nodes(analysis_data)
        
        # Add contamination dataset nodes
        self._add_contamination_dataset_nodes(analysis_data)
        
        # Add confidence evaluation nodes
        self._add_confidence_evaluation_nodes(analysis_data)
        
        # Add hierarchy relationship nodes
        self._add_hierarchy_relationship_nodes(analysis_data)
        
        # Add similarity score nodes
        self._add_similarity_score_nodes(analysis_data)
        
        # Connect GO analysis data to existing graph elements
        self._connect_go_analysis_to_graph(analysis_data)
        
        logger.info("GO Analysis Data integration complete")
    
    def _add_core_go_analysis_nodes(self, analysis_data):
        """Add core GO term analysis nodes to the graph."""
        core_terms = analysis_data.get('core_go_terms', {})
        
        for dataset_name, terms in core_terms.items():
            for go_id, term_data in terms.items():
                # Create core analysis node
                analysis_id = f"go_analysis_{dataset_name}_{go_id}"
                
                node_attrs = {
                    'node_type': 'go_core_analysis',
                    'source': 'GO_Analysis_Data',
                    'dataset': dataset_name,
                    'dataset_type': term_data.get('dataset_type', 'core_terms'),
                    'go_term_id': go_id,
                    'gene_count': term_data.get('gene_count', 0),
                    'term_description': term_data.get('term_description', ''),
                    'genes': term_data.get('genes', [])
                }
                
                # Add enrichment analysis data if available
                if 'enrichment_analysis' in term_data:
                    node_attrs['enrichment_analysis'] = term_data['enrichment_analysis']
                    node_attrs['has_enrichment_data'] = True
                else:
                    node_attrs['has_enrichment_data'] = False
                
                self.graph.add_node(analysis_id, **node_attrs)
                
                # Connect to GO term if it exists
                if go_id in self.graph:
                    self.graph.add_edge(analysis_id, go_id, 
                                      edge_type='analyzes_go_term',
                                      source='GO_Analysis_Data')
                
                # Connect to genes if they exist
                for gene in term_data.get('genes', []):
                    gene_nodes = [n for n, attrs in self.graph.nodes(data=True) 
                                if attrs.get('node_type') == 'gene' and 
                                   (attrs.get('symbol') == gene or attrs.get('gene_id') == gene)]
                    
                    for gene_node in gene_nodes:
                        self.graph.add_edge(analysis_id, gene_node,
                                          edge_type='analyzes_gene',
                                          source='GO_Analysis_Data',
                                          dataset=dataset_name)
        
        logger.info(f"Added {sum(len(terms) for terms in core_terms.values())} core GO analysis nodes")
    
    def _add_contamination_dataset_nodes(self, analysis_data):
        """Add contamination dataset nodes to the graph."""
        contamination_datasets = analysis_data.get('contamination_datasets', {})
        
        for dataset_name, terms in contamination_datasets.items():
            for go_id, term_data in terms.items():
                # Create contamination analysis node
                contam_id = f"go_contamination_{dataset_name}_{go_id}"
                
                node_attrs = {
                    'node_type': 'go_contamination_analysis',
                    'source': 'GO_Analysis_Data',
                    'dataset': dataset_name,
                    'dataset_type': term_data.get('dataset_type', 'contamination_analysis'),
                    'go_term_id': go_id,
                    'gene_count': term_data.get('gene_count', 0),
                    'term_description': term_data.get('term_description', ''),
                    'original_genes': term_data.get('genes', []),
                    'contaminated_50perc': term_data.get('contaminated_50perc', []),
                    'contaminated_100perc': term_data.get('contaminated_100perc', []),
                    'contamination_levels': len([x for x in [term_data.get('contaminated_50perc'), term_data.get('contaminated_100perc')] if x])
                }
                
                self.graph.add_node(contam_id, **node_attrs)
                
                # Connect to GO term if it exists
                if go_id in self.graph:
                    self.graph.add_edge(contam_id, go_id, 
                                      edge_type='contamination_study_of_go_term',
                                      source='GO_Analysis_Data')
                
                # Connect to genes (original and contaminated)
                all_genes = set(term_data.get('genes', []))
                all_genes.update(term_data.get('contaminated_50perc', []))
                all_genes.update(term_data.get('contaminated_100perc', []))
                
                for gene in all_genes:
                    gene_nodes = [n for n, attrs in self.graph.nodes(data=True) 
                                if attrs.get('node_type') == 'gene' and 
                                   (attrs.get('symbol') == gene or attrs.get('gene_id') == gene)]
                    
                    for gene_node in gene_nodes:
                        # Determine contamination status
                        if gene in term_data.get('genes', []):
                            contamination_status = 'original'
                        elif gene in term_data.get('contaminated_50perc', []):
                            contamination_status = '50perc_contaminated'
                        elif gene in term_data.get('contaminated_100perc', []):
                            contamination_status = '100perc_contaminated'
                        else:
                            contamination_status = 'unknown'
                        
                        self.graph.add_edge(contam_id, gene_node,
                                          edge_type='contamination_gene_association',
                                          source='GO_Analysis_Data',
                                          dataset=dataset_name,
                                          contamination_status=contamination_status)
        
        logger.info(f"Added {sum(len(terms) for terms in contamination_datasets.values())} contamination analysis nodes")
    
    def _add_confidence_evaluation_nodes(self, analysis_data):
        """Add confidence evaluation nodes to the graph."""
        confidence_evaluations = analysis_data.get('confidence_evaluations', {})
        
        for dataset_name, evaluations in confidence_evaluations.items():
            for go_id, eval_data in evaluations.items():
                # Create confidence evaluation node
                conf_id = f"go_confidence_{dataset_name}_{go_id}"
                
                node_attrs = {
                    'node_type': 'go_confidence_evaluation',
                    'source': 'GO_Analysis_Data',
                    'dataset': dataset_name,
                    'dataset_type': eval_data.get('dataset_type', 'human_evaluation'),
                    'go_term_id': go_id,
                    'gene_count': eval_data.get('gene_count', 0),
                    'llm_name': eval_data.get('llm_name', ''),
                    'llm_analysis': eval_data.get('llm_analysis', ''),
                    'reviewer_score_bin': eval_data.get('reviewer_score_bin', ''),
                    'raw_score': eval_data.get('raw_score', 0),
                    'notes': eval_data.get('notes', ''),
                    'reviewer_score_bin_final': eval_data.get('reviewer_score_bin_final', ''),
                    'has_human_review': True
                }
                
                self.graph.add_node(conf_id, **node_attrs)
                
                # Connect to GO term if it exists
                if go_id in self.graph:
                    self.graph.add_edge(conf_id, go_id, 
                                      edge_type='confidence_evaluation_of_go_term',
                                      source='GO_Analysis_Data')
                
                # Connect to genes
                for gene in eval_data.get('genes', []):
                    gene_nodes = [n for n, attrs in self.graph.nodes(data=True) 
                                if attrs.get('node_type') == 'gene' and 
                                   (attrs.get('symbol') == gene or attrs.get('gene_id') == gene)]
                    
                    for gene_node in gene_nodes:
                        self.graph.add_edge(conf_id, gene_node,
                                          edge_type='confidence_gene_association',
                                          source='GO_Analysis_Data',
                                          dataset=dataset_name)
        
        logger.info(f"Added {sum(len(evals) for evals in confidence_evaluations.values())} confidence evaluation nodes")
    
    def _add_hierarchy_relationship_nodes(self, analysis_data):
        """Add hierarchy relationship nodes to the graph."""
        hierarchy_data = analysis_data.get('hierarchy_data', {})
        relationships = hierarchy_data.get('relationships', [])
        
        for rel in relationships:
            # Create hierarchy relationship edge between GO terms
            child_id = rel.get('child')
            parent_id = rel.get('parent')
            
            # Add hierarchy edge if both nodes exist
            if child_id in self.graph and parent_id in self.graph:
                self.graph.add_edge(child_id, parent_id,
                                  edge_type='go_hierarchy_relationship',
                                  source='GO_Analysis_Data',
                                  relationship_type=rel.get('relationship_type', 'parent_child'))
        
        logger.info(f"Added {len(relationships)} GO hierarchy relationships")
    
    def _add_similarity_score_nodes(self, analysis_data):
        """Add similarity score data to the graph."""
        similarity_scores = analysis_data.get('similarity_scores', {})
        
        for score_type, score_data in similarity_scores.items():
            # Create a single node representing the similarity score dataset
            sim_id = f"similarity_scores_{score_type}"
            
            node_attrs = {
                'node_type': 'similarity_scores',
                'source': 'GO_Analysis_Data',
                'score_type': score_type,
                'sample_count': score_data.get('sample_count', 0),
                'total_lines': score_data.get('total_lines', 0),
                'file_size': score_data.get('file_size', 0),
                'score_range': score_data.get('score_range', {}),
                'sample_scores': score_data.get('sample_scores', [])
            }
            
            self.graph.add_node(sim_id, **node_attrs)
        
        logger.info(f"Added {len(similarity_scores)} similarity score datasets")
    
    def _connect_go_analysis_to_graph(self, analysis_data):
        """Create additional connections between GO analysis data and existing graph elements."""
        logger.info("Creating additional GO analysis data connections...")
        
        # Count genes mentioned in GO analysis data
        genes_in_analysis = set()
        
        # Count from core terms
        for dataset in analysis_data.get('core_go_terms', {}).values():
            for term in dataset.values():
                genes_in_analysis.update(term.get('genes', []))
        
        # Count from contamination datasets
        for dataset in analysis_data.get('contamination_datasets', {}).values():
            for term in dataset.values():
                genes_in_analysis.update(term.get('genes', []))
                genes_in_analysis.update(term.get('contaminated_50perc', []))
                genes_in_analysis.update(term.get('contaminated_100perc', []))
        
        # Count from confidence evaluations
        for dataset in analysis_data.get('confidence_evaluations', {}).values():
            for eval_data in dataset.values():
                genes_in_analysis.update(eval_data.get('genes', []))
        
        logger.info(f"GO analysis data covers {len(genes_in_analysis)} unique genes")
    
    def _add_remaining_data(self):
        """Add remaining data files integration (GMT, reference evaluation, L1000, embeddings, supplement table)."""
        
        if 'remaining_data' not in self.parsed_data:
            logger.info("No remaining data available for integration")
            return
        
        logger.info("Adding remaining data integration...")
        remaining_data = self.parsed_data['remaining_data']
        
        # Add GMT data (GO gene sets)
        self._add_gmt_data(remaining_data.get('gmt_data', {}))
        
        # Add reference evaluation data (literature support)
        self._add_reference_evaluation_data(remaining_data.get('reference_evaluation_data', {}))
        
        # Add L1000 data (perturbation experiments)  
        self._add_l1000_data(remaining_data.get('l1000_data', {}))
        
        # Add GO term embeddings data
        self._add_embeddings_data(remaining_data.get('embeddings_data', {}))
        
        # Add supplement table data (additional LLM evaluations)
        self._add_supplement_table_data(remaining_data.get('supplement_table_data', {}))
        
        logger.info("Remaining data integration complete")
    
    def _add_talisman_gene_sets(self):
        """Add talisman gene sets integration (HALLMARK, bicluster, pathways, GO custom, disease sets)."""
        
        if 'talisman_gene_sets' not in self.parsed_data:
            logger.info("No talisman gene sets available for integration")
            return
        
        logger.info("Adding talisman gene sets integration...")
        talisman_data = self.parsed_data['talisman_gene_sets']
        
        # Add HALLMARK gene sets
        self._add_hallmark_gene_sets(talisman_data.get('hallmark_sets', {}))
        
        # Add bicluster gene sets
        self._add_bicluster_gene_sets(talisman_data.get('bicluster_sets', {}))
        
        # Add pathway gene sets
        self._add_pathway_gene_sets(talisman_data.get('pathway_sets', {}))
        
        # Add GO custom gene sets
        self._add_go_custom_gene_sets(talisman_data.get('go_custom_sets', {}))
        
        # Add disease gene sets
        self._add_disease_gene_sets(talisman_data.get('disease_sets', {}))
        
        # Add other gene sets
        self._add_other_gene_sets(talisman_data.get('other_sets', {}))
        
        logger.info("Talisman gene sets integration complete")
    
    def _add_hallmark_gene_sets(self, hallmark_sets):
        """Add HALLMARK pathway gene sets to the knowledge graph."""
        if not hallmark_sets:
            logger.info("No HALLMARK gene sets to add")
            return
        
        logger.info(f"Adding {len(hallmark_sets)} HALLMARK gene sets...")
        
        for gene_set_id, gene_set_data in hallmark_sets.items():
            # Create HALLMARK gene set node
            node_id = f"hallmark_{gene_set_id}"
            self.graph.add_node(node_id,
                node_type='hallmark_gene_set',
                gene_set_id=gene_set_id,
                name=gene_set_data['name'],
                description=gene_set_data.get('description', ''),
                gene_count=gene_set_data['gene_count'],
                source='talisman_hallmark',
                systematic_name=gene_set_data['metadata'].get('systematic_name'),
                pmid=gene_set_data['metadata'].get('pmid'),
                collection=gene_set_data['metadata'].get('collection'),
                msigdb_url=gene_set_data['metadata'].get('msigdb_url')
            )
            
            # Add gene associations
            for gene_symbol in gene_set_data['genes']:
                gene_node_id = f"gene_{gene_symbol}"
                
                # Ensure gene node exists
                if gene_node_id not in self.graph:
                    self._add_gene_node_if_missing(gene_symbol, 'talisman_hallmark')
                
                # Add association edge
                self.graph.add_edge(gene_node_id, node_id,
                    relationship='member_of_hallmark_pathway',
                    source='talisman_hallmark',
                    pathway_type='hallmark',
                    evidence='literature_curated'
                )
        
        logger.info(f"Added {len(hallmark_sets)} HALLMARK gene sets")
    
    def _add_bicluster_gene_sets(self, bicluster_sets):
        """Add bicluster gene sets to the knowledge graph."""
        if not bicluster_sets:
            logger.info("No bicluster gene sets to add")
            return
        
        logger.info(f"Adding {len(bicluster_sets)} bicluster gene sets...")
        
        for gene_set_id, gene_set_data in bicluster_sets.items():
            # Create bicluster gene set node
            node_id = f"bicluster_{gene_set_id}"
            self.graph.add_node(node_id,
                node_type='bicluster_gene_set',
                gene_set_id=gene_set_id,
                name=gene_set_data['name'],
                description=gene_set_data.get('description', ''),
                gene_count=gene_set_data['gene_count'],
                source='talisman_bicluster'
            )
            
            # Add gene associations
            for gene_symbol in gene_set_data['genes']:
                gene_node_id = f"gene_{gene_symbol}"
                
                # Ensure gene node exists
                if gene_node_id not in self.graph:
                    self._add_gene_node_if_missing(gene_symbol, 'talisman_bicluster')
                
                # Add association edge
                self.graph.add_edge(gene_node_id, node_id,
                    relationship='member_of_bicluster',
                    source='talisman_bicluster',
                    cluster_type='expression_based',
                    evidence='computational'
                )
        
        logger.info(f"Added {len(bicluster_sets)} bicluster gene sets")
    
    def _add_pathway_gene_sets(self, pathway_sets):
        """Add custom pathway gene sets to the knowledge graph."""
        if not pathway_sets:
            logger.info("No pathway gene sets to add")
            return
        
        logger.info(f"Adding {len(pathway_sets)} pathway gene sets...")
        
        for gene_set_id, gene_set_data in pathway_sets.items():
            # Create pathway gene set node
            node_id = f"pathway_{gene_set_id}"
            self.graph.add_node(node_id,
                node_type='custom_pathway_gene_set',
                gene_set_id=gene_set_id,
                name=gene_set_data['name'],
                description=gene_set_data.get('description', ''),
                gene_count=gene_set_data['gene_count'],
                source='talisman_pathway'
            )
            
            # Add gene associations
            for gene_symbol in gene_set_data['genes']:
                gene_node_id = f"gene_{gene_symbol}"
                
                # Ensure gene node exists
                if gene_node_id not in self.graph:
                    self._add_gene_node_if_missing(gene_symbol, 'talisman_pathway')
                
                # Add association edge
                self.graph.add_edge(gene_node_id, node_id,
                    relationship='member_of_custom_pathway',
                    source='talisman_pathway',
                    pathway_type='custom',
                    evidence='literature_derived'
                )
        
        logger.info(f"Added {len(pathway_sets)} pathway gene sets")
    
    def _add_go_custom_gene_sets(self, go_custom_sets):
        """Add GO custom gene sets to the knowledge graph."""
        if not go_custom_sets:
            logger.info("No GO custom gene sets to add")
            return
        
        logger.info(f"Adding {len(go_custom_sets)} GO custom gene sets...")
        
        for gene_set_id, gene_set_data in go_custom_sets.items():
            # Create GO custom gene set node
            node_id = f"go_custom_{gene_set_id}"
            self.graph.add_node(node_id,
                node_type='go_custom_gene_set',
                gene_set_id=gene_set_id,
                name=gene_set_data['name'],
                description=gene_set_data.get('description', ''),
                gene_count=gene_set_data['gene_count'],
                source='talisman_go_custom'
            )
            
            # Add gene associations
            for gene_symbol in gene_set_data['genes']:
                gene_node_id = f"gene_{gene_symbol}"
                
                # Ensure gene node exists
                if gene_node_id not in self.graph:
                    self._add_gene_node_if_missing(gene_symbol, 'talisman_go_custom')
                
                # Add association edge
                self.graph.add_edge(gene_node_id, node_id,
                    relationship='member_of_go_custom_set',
                    source='talisman_go_custom',
                    set_type='go_derived',
                    evidence='ontology_based'
                )
        
        logger.info(f"Added {len(go_custom_sets)} GO custom gene sets")
    
    def _add_disease_gene_sets(self, disease_sets):
        """Add disease-specific gene sets to the knowledge graph."""
        if not disease_sets:
            logger.info("No disease gene sets to add")
            return
        
        logger.info(f"Adding {len(disease_sets)} disease gene sets...")
        
        for gene_set_id, gene_set_data in disease_sets.items():
            # Create disease gene set node
            node_id = f"disease_{gene_set_id}"
            self.graph.add_node(node_id,
                node_type='disease_gene_set',
                gene_set_id=gene_set_id,
                name=gene_set_data['name'],
                description=gene_set_data.get('description', ''),
                gene_count=gene_set_data['gene_count'],
                source='talisman_disease'
            )
            
            # Add gene associations
            for gene_symbol in gene_set_data['genes']:
                gene_node_id = f"gene_{gene_symbol}"
                
                # Ensure gene node exists
                if gene_node_id not in self.graph:
                    self._add_gene_node_if_missing(gene_symbol, 'talisman_disease')
                
                # Add association edge
                self.graph.add_edge(gene_node_id, node_id,
                    relationship='associated_with_disease_set',
                    source='talisman_disease',
                    association_type='disease_related',
                    evidence='literature_curated'
                )
        
        logger.info(f"Added {len(disease_sets)} disease gene sets")
    
    def _add_other_gene_sets(self, other_sets):
        """Add other specialized gene sets to the knowledge graph."""
        if not other_sets:
            logger.info("No other gene sets to add")
            return
        
        logger.info(f"Adding {len(other_sets)} other gene sets...")
        
        for gene_set_id, gene_set_data in other_sets.items():
            # Create other gene set node
            node_id = f"other_{gene_set_id}"
            self.graph.add_node(node_id,
                node_type='specialized_gene_set',
                gene_set_id=gene_set_id,
                name=gene_set_data['name'],
                description=gene_set_data.get('description', ''),
                gene_count=gene_set_data['gene_count'],
                source='talisman_other'
            )
            
            # Add gene associations
            for gene_symbol in gene_set_data['genes']:
                gene_node_id = f"gene_{gene_symbol}"
                
                # Ensure gene node exists
                if gene_node_id not in self.graph:
                    self._add_gene_node_if_missing(gene_symbol, 'talisman_other')
                
                # Add association edge
                self.graph.add_edge(gene_node_id, node_id,
                    relationship='member_of_specialized_set',
                    source='talisman_other',
                    set_type='specialized',
                    evidence='curated'
                )
        
        logger.info(f"Added {len(other_sets)} other gene sets")
    
    def _add_gene_node_if_missing(self, gene_symbol, source):
        """Add a gene node if it doesn't exist in the graph."""
        gene_node_id = f"gene_{gene_symbol}"
        
        if gene_node_id not in self.graph:
            self.graph.add_node(gene_node_id,
                node_type='gene',
                gene_symbol=gene_symbol,
                source=source,
                name=gene_symbol,
                added_from_talisman=True
            )
    
    def _add_gmt_data(self, gmt_data):
        """Add GMT file data (GO gene sets) to the knowledge graph."""
        if not gmt_data:
            logger.info("No GMT data to add")
            return
        
        logger.info("Adding GMT data (GO gene sets)...")
        
        # Add gene sets as nodes
        gene_sets = gmt_data.get('gene_sets', [])
        go_terms = gmt_data.get('go_terms', {})
        genes_to_go_terms = gmt_data.get('genes_to_go_terms', {})
        
        # Add GMT gene set nodes
        for gene_set in gene_sets:
            go_id = gene_set['go_id']
            node_id = f"gmt_{go_id}"
            
            node_attrs = {
                'go_id': go_id,
                'node_type': 'gmt_gene_set',
                'source': 'GMT_File',
                'description': gene_set['description'],
                'gene_count': gene_set['gene_count'],
                'genes': gene_set['genes'],
                'url_or_desc': gene_set.get('url_or_desc', '')
            }
            
            self.graph.add_node(node_id, **node_attrs)
            
            # Connect genes to this gene set
            for gene in gene_set['genes']:
                gene_node_id = f"gene_{gene}"
                
                # Add gene node if not exists
                if gene_node_id not in self.graph:
                    self.graph.add_node(gene_node_id, 
                                      gene_symbol=gene, 
                                      node_type='gene',
                                      source='GMT_File')
                
                # Add association edge
                self.graph.add_edge(gene_node_id, node_id, 
                                  relationship='associated_with_gmt_gene_set',
                                  source='GMT_File',
                                  go_id=go_id)
        
        logger.info(f"Added {len(gene_sets)} GMT gene sets with {len(genes_to_go_terms)} gene associations")
    
    def _add_reference_evaluation_data(self, ref_eval_data):
        """Add reference evaluation data (literature support) to the knowledge graph."""
        if not ref_eval_data:
            logger.info("No reference evaluation data to add")
            return
        
        logger.info("Adding reference evaluation data...")
        
        references = ref_eval_data.get('references', {})
        gene_sets = ref_eval_data.get('gene_sets', {})
        paragraphs = ref_eval_data.get('paragraphs', {})
        
        # Add reference nodes
        for ref_id, ref_data in references.items():
            node_attrs = {
                'node_type': 'literature_reference',
                'source': 'Reference_Evaluation',
                'reference': ref_data['reference'],
                'paragraph': ref_data['paragraph'],
                'comment': ref_data['comment'],
                'gene_set_name': ref_data['gene_set_name'],
                'dataset': ref_data['dataset'],
                'title_supports': ref_data.get('title_supports'),
                'abstract_supports': ref_data.get('abstract_supports')
            }
            
            self.graph.add_node(ref_id, **node_attrs)
        
        # Add gene set evaluation nodes
        for gene_set_name, gene_set_data in gene_sets.items():
            eval_node_id = f"gene_set_eval_{gene_set_name.replace(' ', '_')}"
            
            node_attrs = {
                'node_type': 'gene_set_evaluation',
                'source': 'Reference_Evaluation',
                'gene_set_name': gene_set_name,
                'total_references': len(gene_set_data['references']),
                'datasets': gene_set_data['datasets'],
                'evaluation_count': len(gene_set_data['evaluations'])
            }
            
            self.graph.add_node(eval_node_id, **node_attrs)
            
            # Connect references to gene set evaluations
            for reference in gene_set_data['references']:
                # Find matching reference node
                for ref_id, ref_data in references.items():
                    if ref_data['reference'] == reference:
                        self.graph.add_edge(ref_id, eval_node_id,
                                          relationship='supports_gene_set',
                                          source='Reference_Evaluation')
        
        logger.info(f"Added {len(references)} references and {len(gene_sets)} gene set evaluations")
    
    def _add_l1000_data(self, l1000_data):
        """Add L1000 perturbation data to the knowledge graph."""
        if not l1000_data:
            logger.info("No L1000 data to add")
            return
        
        logger.info("Adding L1000 perturbation data...")
        
        perturbations = l1000_data.get('perturbations', {})
        cell_lines = l1000_data.get('cell_lines', {})
        reagents = l1000_data.get('reagents', {})
        
        # Add perturbation nodes
        for pert_id, pert_data in perturbations.items():
            node_attrs = {
                'node_type': 'l1000_perturbation',
                'source': 'L1000',
                'reagent': pert_data['reagent'],
                'cell_line': pert_data['cell_line'],
                'duration': pert_data['duration'],
                'duration_unit': pert_data['duration_unit'],
                'dosage': pert_data['dosage'],
                'dosage_unit': pert_data['dosage_unit'],
                'n_genesets': pert_data['n_genesets']
            }
            
            self.graph.add_node(pert_id, **node_attrs)
        
        # Add cell line nodes
        for cell_line, cell_data in cell_lines.items():
            cell_node_id = f"cell_line_{cell_line}"
            
            node_attrs = {
                'node_type': 'cell_line',
                'source': 'L1000',
                'cell_line_name': cell_line,
                'perturbation_count': len(cell_data['perturbations']),
                'unique_reagents': len(cell_data['reagents']),
                'total_genesets': cell_data['total_genesets']
            }
            
            self.graph.add_node(cell_node_id, **node_attrs)
            
            # Connect perturbations to cell lines
            for pert_id in cell_data['perturbations']:
                if pert_id in perturbations:
                    self.graph.add_edge(pert_id, cell_node_id,
                                      relationship='performed_in_cell_line',
                                      source='L1000')
        
        # Add reagent nodes
        for reagent, reagent_data in reagents.items():
            reagent_node_id = f"reagent_{reagent.replace('-', '_')}"
            
            node_attrs = {
                'node_type': 'l1000_reagent',
                'source': 'L1000',
                'reagent_name': reagent,
                'perturbation_count': len(reagent_data['perturbations']),
                'unique_cell_lines': len(reagent_data['cell_lines']),
                'total_genesets': reagent_data['total_genesets']
            }
            
            self.graph.add_node(reagent_node_id, **node_attrs)
            
            # Connect perturbations to reagents
            for pert_id in reagent_data['perturbations']:
                if pert_id in perturbations:
                    self.graph.add_edge(pert_id, reagent_node_id,
                                      relationship='uses_reagent',
                                      source='L1000')
        
        logger.info(f"Added {len(perturbations)} perturbations, {len(cell_lines)} cell lines, {len(reagents)} reagents")
    
    def _add_embeddings_data(self, embeddings_data):
        """Add GO term embeddings data to the knowledge graph."""
        if not embeddings_data:
            logger.info("No embeddings data to add") 
            return
        
        logger.info("Adding GO term embeddings...")
        
        embeddings = embeddings_data.get('embeddings', {})
        
        # Add embedding nodes and connect to existing GO terms
        embedding_count = 0
        connected_count = 0
        
        for go_term, embedding_info in embeddings.items():
            embedding_node_id = f"embedding_{go_term}"
            
            node_attrs = {
                'node_type': 'go_term_embedding',
                'source': 'Embeddings',
                'go_term': go_term,
                'embedding_dimension': embedding_info['dimension'],
                'has_embedding_vector': True
                # Note: Not storing the actual vector in node attributes for performance
            }
            
            self.graph.add_node(embedding_node_id, **node_attrs)
            embedding_count += 1
            
            # Try to connect to existing GO term nodes
            for node_id, node_attrs in self.graph.nodes(data=True):
                if (node_attrs.get('node_type') in ['go_term', 'gmt_gene_set'] and 
                    node_attrs.get('go_id') == go_term):
                    
                    self.graph.add_edge(node_id, embedding_node_id,
                                      relationship='has_embedding',
                                      source='Embeddings')
                    connected_count += 1
                    break
        
        logger.info(f"Added {embedding_count} GO term embeddings, {connected_count} connected to existing GO terms")
    
    def _add_supplement_table_data(self, supp_data):
        """Add supplementary LLM evaluation data to the knowledge graph.""" 
        if not supp_data:
            logger.info("No supplement table data to add")
            return
        
        logger.info("Adding supplement table LLM evaluation data...")
        
        evaluations = supp_data.get('evaluations', {})
        gene_sets = supp_data.get('gene_sets', {})
        llm_analyses = supp_data.get('llm_analyses', {})
        
        # Add evaluation nodes
        for eval_id, eval_data in evaluations.items():
            node_attrs = {
                'node_type': 'supplement_llm_evaluation',
                'source': 'Supplement_Table',
                'data_source': eval_data['source'],
                'gene_set_name': eval_data['gene_set_name'],
                'gene_list': eval_data['gene_list'],
                'n_genes': eval_data['n_genes'],
                'llm_name': eval_data['llm_name'],
                'referenced_analysis': eval_data['referenced_analysis'],
                'score': eval_data.get('score')
            }
            
            # Add other metadata columns
            for key, value in eval_data.items():
                if key not in node_attrs:
                    node_attrs[key] = value
            
            self.graph.add_node(eval_id, **node_attrs)
        
        # Add LLM analysis summary nodes
        for llm_name, llm_data in llm_analyses.items():
            llm_node_id = f"supplement_llm_{llm_name.replace(' ', '_')[:50]}"
            
            node_attrs = {
                'node_type': 'supplement_llm_analysis',
                'source': 'Supplement_Table',
                'llm_name': llm_name,
                'total_evaluations': len(llm_data['evaluations']),
                'unique_analyses': len(llm_data['analyses']),
                'unique_gene_sets': len(llm_data['gene_sets'])
            }
            
            self.graph.add_node(llm_node_id, **node_attrs)
            
            # Connect evaluations to LLM analysis
            for eval_id in llm_data['evaluations']:
                if eval_id in evaluations:
                    self.graph.add_edge(eval_id, llm_node_id,
                                      relationship='analyzed_by_llm',
                                      source='Supplement_Table')
        
        logger.info(f"Added {len(evaluations)} supplement evaluations and {len(llm_analyses)} LLM analyses")
    
    # ============= LLM_PROCESSED QUERY METHODS =============
    
    def query_llm_interpretations(self, dataset: str = None, go_id: str = None, model: str = None) -> List[Dict]:
        """Query LLM interpretations with optional filtering."""
        results = []
        
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get('node_type') == 'llm_interpretation' and attrs.get('source') == 'LLM_processed':
                # Apply filters
                if dataset and attrs.get('dataset') != dataset:
                    continue
                if go_id and attrs.get('go_term_id') != go_id:
                    continue
                if model and attrs.get('model') != model:
                    continue
                
                # Format result
                result = {
                    'interpretation_id': node_id,
                    'dataset': attrs.get('dataset'),
                    'go_term_id': attrs.get('go_term_id'),
                    'model': attrs.get('model'),
                    'llm_name': attrs.get('llm_name'),
                    'llm_analysis': attrs.get('llm_analysis'),
                    'llm_score': attrs.get('llm_score'),
                    'gene_count': attrs.get('gene_count'),
                    'term_description': attrs.get('term_description')
                }
                results.append(result)
        
        return results
    
    def query_contamination_analysis(self, model: str = None, go_id: str = None) -> List[Dict]:
        """Query contamination analysis with optional filtering."""
        results = []
        
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get('node_type') == 'contamination_analysis' and attrs.get('source') == 'LLM_processed':
                # Apply filters
                if model and attrs.get('model') != model:
                    continue
                if go_id and attrs.get('go_term_id') != go_id:
                    continue
                
                # Format result
                result = {
                    'analysis_id': node_id,
                    'model': attrs.get('model'),
                    'go_term_id': attrs.get('go_term_id'),
                    'gene_count': attrs.get('gene_count'),
                    'term_description': attrs.get('term_description'),
                    'scenarios': attrs.get('scenarios'),
                    'default_score': attrs.get('default_score'),
                    '50perc_contaminated_score': attrs.get('50perc_contaminated_score'),
                    '100perc_contaminated_score': attrs.get('100perc_contaminated_score')
                }
                results.append(result)
        
        return results
    
    def query_llm_similarity_rankings(self, dataset: str = None, go_id: str = None) -> List[Dict]:
        """Query LLM similarity rankings with optional filtering."""
        results = []
        
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get('node_type') == 'similarity_ranking' and attrs.get('source') == 'LLM_processed':
                # Apply filters
                if dataset and attrs.get('dataset') != dataset:
                    continue
                if go_id and attrs.get('go_term_id') != go_id:
                    continue
                
                # Format result
                result = {
                    'ranking_id': node_id,
                    'dataset': attrs.get('dataset'),
                    'go_term_id': attrs.get('go_term_id'),
                    'llm_name': attrs.get('llm_name'),
                    'llm_score': attrs.get('llm_score'),
                    'similarity_rank': attrs.get('similarity_rank'),
                    'similarity_percentile': attrs.get('similarity_percentile'),
                    'random_similarity_percentile': attrs.get('random_similarity_percentile'),
                    'top_3_hits': attrs.get('top_3_hits'),
                    'top_3_similarities': attrs.get('top_3_similarities')
                }
                results.append(result)
        
        return results
    
    def query_gene_llm_profile(self, gene_symbol: str) -> Dict[str, Any]:
        """Get comprehensive LLM profile for a specific gene."""
        profile = {
            'gene_symbol': gene_symbol,
            'llm_interpretations': [],
            'contamination_analyses': [],
            'similarity_rankings': [],
            'model_comparisons': []
        }
        
        # Find all LLM data connected to this gene
        if gene_symbol in self.graph.nodes:
            for neighbor in self.graph.neighbors(gene_symbol):
                neighbor_attrs = self.graph.nodes[neighbor]
                neighbor_type = neighbor_attrs.get('node_type')
                source = neighbor_attrs.get('source')
                
                if source == 'LLM_processed':
                    if neighbor_type == 'llm_interpretation':
                        profile['llm_interpretations'].append({
                            'interpretation_id': neighbor,
                            'dataset': neighbor_attrs.get('dataset'),
                            'model': neighbor_attrs.get('model'),
                            'llm_name': neighbor_attrs.get('llm_name'),
                            'llm_score': neighbor_attrs.get('llm_score')
                        })
                    elif neighbor_type == 'contamination_analysis':
                        profile['contamination_analyses'].append({
                            'analysis_id': neighbor,
                            'model': neighbor_attrs.get('model'),
                            'default_score': neighbor_attrs.get('default_score'),
                            '50perc_contaminated_score': neighbor_attrs.get('50perc_contaminated_score'),
                            '100perc_contaminated_score': neighbor_attrs.get('100perc_contaminated_score')
                        })
        
        return profile
    
    def get_llm_processed_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for LLM_processed data integration."""
        stats = {
            'llm_interpretations': 0,
            'contamination_analyses': 0,
            'similarity_rankings': 0,
            'similarity_pvalues': 0,
            'model_comparisons': 0,
            'models_analyzed': set(),
            'datasets_analyzed': set(),
            'unique_go_terms': set(),
            'unique_genes': set()
        }
        
        for node_id, attrs in self.graph.nodes(data=True):
            source = attrs.get('source')
            node_type = attrs.get('node_type')
            
            if source == 'LLM_processed':
                if node_type == 'llm_interpretation':
                    stats['llm_interpretations'] += 1
                    stats['datasets_analyzed'].add(attrs.get('dataset'))
                    stats['models_analyzed'].add(attrs.get('model'))
                    stats['unique_go_terms'].add(attrs.get('go_term_id'))
                elif node_type == 'contamination_analysis':
                    stats['contamination_analyses'] += 1
                    stats['models_analyzed'].add(attrs.get('model'))
                    stats['unique_go_terms'].add(attrs.get('go_term_id'))
                elif node_type == 'similarity_ranking':
                    stats['similarity_rankings'] += 1
                    stats['datasets_analyzed'].add(attrs.get('dataset'))
                    stats['unique_go_terms'].add(attrs.get('go_term_id'))
                elif node_type == 'similarity_pvalues':
                    stats['similarity_pvalues'] += 1
                    stats['unique_go_terms'].add(attrs.get('go_term_id'))
                elif node_type == 'model_comparison':
                    stats['model_comparisons'] += 1
                    stats['unique_go_terms'].add(attrs.get('go_term_id'))
        
        # Count edges to genes
        for u, v, edge_attrs in self.graph.edges(data=True):
            if (edge_attrs.get('source') == 'LLM_processed' and 
                'gene' in edge_attrs.get('edge_type', '')):
                # v should be the gene
                if self.graph.nodes[v].get('node_type') == 'gene':
                    stats['unique_genes'].add(v)
        
        # Convert sets to counts
        stats['models_analyzed'] = len(stats['models_analyzed'])
        stats['datasets_analyzed'] = len(stats['datasets_analyzed'])
        stats['unique_go_terms'] = len(stats['unique_go_terms'])
        stats['unique_genes'] = len(stats['unique_genes'])
        
        return stats
    
    # ============= GO ANALYSIS DATA QUERY METHODS =============
    
    def query_go_core_analysis(self, dataset: str = None, go_id: str = None) -> List[Dict]:
        """Query GO core analysis data with optional filtering."""
        results = []
        
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get('node_type') == 'go_core_analysis' and attrs.get('source') == 'GO_Analysis_Data':
                # Apply filters
                if dataset and attrs.get('dataset') != dataset:
                    continue
                if go_id and attrs.get('go_term_id') != go_id:
                    continue
                
                results.append({
                    'node_id': node_id,
                    'dataset': attrs.get('dataset'),
                    'go_term_id': attrs.get('go_term_id'),
                    'gene_count': attrs.get('gene_count'),
                    'term_description': attrs.get('term_description'),
                    'genes': attrs.get('genes'),
                    'has_enrichment_data': attrs.get('has_enrichment_data'),
                    'enrichment_analysis': attrs.get('enrichment_analysis')
                })
        
        return results
    
    def query_go_contamination_analysis(self, dataset: str = None, go_id: str = None) -> List[Dict]:
        """Query GO contamination analysis data with optional filtering."""
        results = []
        
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get('node_type') == 'go_contamination_analysis' and attrs.get('source') == 'GO_Analysis_Data':
                # Apply filters
                if dataset and attrs.get('dataset') != dataset:
                    continue
                if go_id and attrs.get('go_term_id') != go_id:
                    continue
                
                results.append({
                    'node_id': node_id,
                    'dataset': attrs.get('dataset'),
                    'go_term_id': attrs.get('go_term_id'),
                    'gene_count': attrs.get('gene_count'),
                    'term_description': attrs.get('term_description'),
                    'original_genes': attrs.get('original_genes'),
                    'contaminated_50perc': attrs.get('contaminated_50perc'),
                    'contaminated_100perc': attrs.get('contaminated_100perc'),
                    'contamination_levels': attrs.get('contamination_levels')
                })
        
        return results
    
    def query_go_confidence_evaluations(self, dataset: str = None, go_id: str = None) -> List[Dict]:
        """Query GO confidence evaluation data with optional filtering."""
        results = []
        
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get('node_type') == 'go_confidence_evaluation' and attrs.get('source') == 'GO_Analysis_Data':
                # Apply filters
                if dataset and attrs.get('dataset') != dataset:
                    continue
                if go_id and attrs.get('go_term_id') != go_id:
                    continue
                
                results.append({
                    'node_id': node_id,
                    'dataset': attrs.get('dataset'),
                    'go_term_id': attrs.get('go_term_id'),
                    'gene_count': attrs.get('gene_count'),
                    'llm_name': attrs.get('llm_name'),
                    'llm_analysis': attrs.get('llm_analysis'),
                    'reviewer_score_bin': attrs.get('reviewer_score_bin'),
                    'raw_score': attrs.get('raw_score'),
                    'notes': attrs.get('notes'),
                    'reviewer_score_bin_final': attrs.get('reviewer_score_bin_final')
                })
        
        return results
    
    def query_gene_go_analysis_profile(self, gene_symbol: str) -> Dict[str, Any]:
        """Get comprehensive GO analysis profile for a gene."""
        profile = {
            'gene_symbol': gene_symbol,
            'core_analyses': [],
            'contamination_analyses': [],
            'confidence_evaluations': [],
            'total_analyses': 0
        }
        
        # Find gene node
        gene_nodes = [n for n, attrs in self.graph.nodes(data=True) 
                     if attrs.get('node_type') == 'gene' and 
                        (attrs.get('symbol') == gene_symbol or attrs.get('gene_id') == gene_symbol)]
        
        if not gene_nodes:
            return profile
        
        gene_node = gene_nodes[0]
        
        # Find connected GO analysis nodes
        for neighbor in self.graph.neighbors(gene_node):
            neighbor_attrs = self.graph.nodes[neighbor]
            
            if neighbor_attrs.get('source') == 'GO_Analysis_Data':
                edge_attrs = self.graph.edges[neighbor, gene_node]
                
                if neighbor_attrs.get('node_type') == 'go_core_analysis':
                    profile['core_analyses'].append({
                        'node_id': neighbor,
                        'dataset': neighbor_attrs.get('dataset'),
                        'go_term_id': neighbor_attrs.get('go_term_id'),
                        'term_description': neighbor_attrs.get('term_description'),
                        'gene_count': neighbor_attrs.get('gene_count')
                    })
                
                elif neighbor_attrs.get('node_type') == 'go_contamination_analysis':
                    contamination_status = edge_attrs.get('contamination_status', 'unknown')
                    profile['contamination_analyses'].append({
                        'node_id': neighbor,
                        'dataset': neighbor_attrs.get('dataset'),
                        'go_term_id': neighbor_attrs.get('go_term_id'),
                        'term_description': neighbor_attrs.get('term_description'),
                        'contamination_status': contamination_status,
                        'contamination_levels': neighbor_attrs.get('contamination_levels')
                    })
                
                elif neighbor_attrs.get('node_type') == 'go_confidence_evaluation':
                    profile['confidence_evaluations'].append({
                        'node_id': neighbor,
                        'dataset': neighbor_attrs.get('dataset'),
                        'go_term_id': neighbor_attrs.get('go_term_id'),
                        'llm_name': neighbor_attrs.get('llm_name'),
                        'reviewer_score_bin': neighbor_attrs.get('reviewer_score_bin'),
                        'raw_score': neighbor_attrs.get('raw_score')
                    })
        
        profile['total_analyses'] = len(profile['core_analyses']) + len(profile['contamination_analyses']) + len(profile['confidence_evaluations'])
        return profile
    
    def get_go_analysis_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for GO Analysis Data integration."""
        stats = {
            'core_analyses': 0,
            'contamination_analyses': 0,
            'confidence_evaluations': 0,
            'hierarchy_relationships': 0,
            'similarity_datasets': 0,
            'datasets_analyzed': set(),
            'unique_go_terms': set(),
            'unique_genes': set(),
            'enrichment_analyses': 0,
            'human_reviewed': 0
        }
        
        for node_id, attrs in self.graph.nodes(data=True):
            source = attrs.get('source')
            node_type = attrs.get('node_type')
            
            if source == 'GO_Analysis_Data':
                if node_type == 'go_core_analysis':
                    stats['core_analyses'] += 1
                    stats['datasets_analyzed'].add(attrs.get('dataset'))
                    stats['unique_go_terms'].add(attrs.get('go_term_id'))
                    if attrs.get('has_enrichment_data'):
                        stats['enrichment_analyses'] += 1
                
                elif node_type == 'go_contamination_analysis':
                    stats['contamination_analyses'] += 1
                    stats['datasets_analyzed'].add(attrs.get('dataset'))
                    stats['unique_go_terms'].add(attrs.get('go_term_id'))
                
                elif node_type == 'go_confidence_evaluation':
                    stats['confidence_evaluations'] += 1
                    stats['datasets_analyzed'].add(attrs.get('dataset'))
                    stats['unique_go_terms'].add(attrs.get('go_term_id'))
                    if attrs.get('has_human_review'):
                        stats['human_reviewed'] += 1
                
                elif node_type == 'similarity_scores':
                    stats['similarity_datasets'] += 1
        
        # Count hierarchy relationships
        for u, v, attrs in self.graph.edges(data=True):
            if attrs.get('edge_type') == 'go_hierarchy_relationship' and attrs.get('source') == 'GO_Analysis_Data':
                stats['hierarchy_relationships'] += 1
        
        # Count genes from edges
        for u, v, attrs in self.graph.edges(data=True):
            if attrs.get('source') == 'GO_Analysis_Data':
                # Check if one node is a gene
                u_attrs = self.graph.nodes[u]
                v_attrs = self.graph.nodes[v]
                
                if u_attrs.get('node_type') == 'gene':
                    stats['unique_genes'].add(u_attrs.get('symbol', u_attrs.get('gene_id', u)))
                elif v_attrs.get('node_type') == 'gene':
                    stats['unique_genes'].add(v_attrs.get('symbol', v_attrs.get('gene_id', v)))
        
        # Convert sets to counts
        stats['datasets_analyzed'] = len(stats['datasets_analyzed'])
        stats['unique_go_terms'] = len(stats['unique_go_terms'])
        stats['unique_genes'] = len(stats['unique_genes'])
        
        return stats
    
    def save_comprehensive_graph(self, filepath: str):
        """Save the comprehensive biomedical knowledge graph."""
        save_graph_to_file(self.graph, filepath)
