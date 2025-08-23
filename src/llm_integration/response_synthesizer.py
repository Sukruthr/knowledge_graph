"""
Response Synthesizer for Knowledge Graph Q&A System

This component combines knowledge graph query results with LLM reasoning
to generate comprehensive, evidence-based answers to biomedical questions.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import re
from collections import defaultdict

from .ollama_client import OllamaClient, OllamaResponse
from .query_planner import QueryPlan, QueryIntent

logger = logging.getLogger(__name__)

@dataclass
class EvidenceSource:
    """Structure for evidence sources in synthesized responses."""
    source_type: str  # "knowledge_graph", "pathway", "literature", etc.
    content: str
    confidence: float
    details: Dict[str, Any]

@dataclass
class SynthesizedResponse:
    """Complete synthesized response with evidence and metadata."""
    answer: str
    confidence: float
    evidence_sources: List[EvidenceSource]
    follow_up_questions: List[str]
    query_coverage: Dict[str, bool]
    synthesis_metadata: Dict[str, Any]

class EvidenceProcessor:
    """Process and structure evidence from knowledge graph queries."""
    
    def __init__(self):
        self.evidence_weights = {
            'genes': 1.0,
            'go_annotations': 0.9,
            'pathways': 0.8,
            'disease_associations': 0.9,
            'drug_interactions': 0.8,
            'expression_data': 0.7,
            'literature_refs': 0.6,
            'model_predictions': 0.5
        }
    
    def process_kg_results(self, results: Dict[str, Any], query_plan: QueryPlan) -> List[EvidenceSource]:
        """
        Process knowledge graph results into structured evidence.
        
        Args:
            results: Raw results from KG queries
            query_plan: Original query plan for context
            
        Returns:
            List of structured evidence sources
        """
        evidence_sources = []
        
        # Process different types of results
        for query_method, result_data in results.items():
            # Handle different result formats
            if isinstance(result_data, dict):
                # Check for drug interaction results (has target_genes)
                if 'target_genes' in result_data and result_data['target_genes']:
                    evidence_sources.extend(self._process_single_result(query_method, result_data))
                # Check for traditional found results
                elif result_data.get('found', False):
                    evidence_sources.extend(self._process_single_result(query_method, result_data))
                # Check for any meaningful data (not just empty dict)
                elif any(key != 'found' and value for key, value in result_data.items()):
                    evidence_sources.extend(self._process_single_result(query_method, result_data))
            elif isinstance(result_data, list):
                for item in result_data:
                    if isinstance(item, dict):
                        evidence_sources.extend(self._process_single_result(query_method, item))
        
        # Rank evidence by relevance and confidence
        evidence_sources = self._rank_evidence(evidence_sources, query_plan)
        
        return evidence_sources[:20]  # Limit to top 20 pieces of evidence
    
    def _process_single_result(self, query_method: str, data: Dict[str, Any]) -> List[EvidenceSource]:
        """Process a single query result into evidence sources."""
        sources = []
        
        # Gene information processing
        if 'gene' in data and 'go_annotations' in data:
            gene = data['gene']
            for annotation in data.get('go_annotations', []):
                source = EvidenceSource(
                    source_type='go_annotation',
                    content=f"Gene {gene} is annotated with GO term {annotation.get('go_id')}: {annotation.get('name')} (namespace: {annotation.get('namespace')})",
                    confidence=0.9,
                    details={
                        'gene': gene,
                        'go_term': annotation.get('go_id'),
                        'go_name': annotation.get('name'),
                        'namespace': annotation.get('namespace'),
                        'evidence': annotation.get('evidence')
                    }
                )
                sources.append(source)
        
        # Disease associations
        if 'disease' in data or 'disease_associations' in data:
            for assoc in data.get('disease_associations', data.get('associations', [])):
                if assoc.get('type') == 'disease':
                    source = EvidenceSource(
                        source_type='disease_association',
                        content=f"Associated with disease: {assoc.get('name', assoc.get('entity'))} (relationship: {assoc.get('relationship_type', 'unknown')})",
                        confidence=0.8,
                        details=assoc
                    )
                    sources.append(source)
        
        # Drug interactions
        if 'drug' in data or 'drug_interactions' in data or 'target_genes' in data:
            target_genes = data.get('target_genes', data.get('drug_interactions', []))
            drug_name = data.get('drug', data.get('drug_name', 'Unknown drug'))
            
            # Handle both list of strings (gene names) and list of dicts
            for target in target_genes:
                if isinstance(target, str):
                    # Simple gene name
                    gene_name = target
                    interaction_type = 'targets'
                elif isinstance(target, dict):
                    # Complex interaction data
                    gene_name = target.get('gene', target.get('target', target.get('name', 'Unknown gene')))
                    interaction_type = target.get('interaction_type', 'targets')
                else:
                    continue
                
                source = EvidenceSource(
                    source_type='drug_interaction',
                    content=f"{drug_name} {interaction_type} gene {gene_name}",
                    confidence=0.8,
                    details={'drug': drug_name, 'gene': gene_name, 'interaction_type': interaction_type, 'raw_data': target}
                )
                sources.append(source)
        
        # Pathway information
        if 'pathway' in data or 'pathways' in data:
            for pathway in data.get('pathways', data.get('associated_genes', [])):
                source = EvidenceSource(
                    source_type='pathway',
                    content=f"Involved in pathway: {pathway.get('go_data', {}).get('name', pathway.get('name', 'Unknown pathway'))}",
                    confidence=0.8,
                    details=pathway
                )
                sources.append(source)
        
        # Search results - handle all categories including 'other'
        if 'total_matches' in data:
            # Process standard categories
            for category in ['genes', 'go_terms', 'diseases', 'drugs']:
                for item in data.get(category, []):
                    source = EvidenceSource(
                        source_type=f'search_{category}',
                        content=f"Found {category[:-1]}: {item.get('name', item.get('node'))} (match score: {item.get('match_score', 0)})",
                        confidence=min(item.get('match_score', 0) / 3.0, 1.0),
                        details=item
                    )
                    sources.append(source)
            
            # Process 'other' category which often contains the actual results
            for item in data.get('other', []):
                if isinstance(item, dict):
                    node_id = item.get('node', 'Unknown')
                    node_name = item.get('name', node_id)
                    match_score = item.get('match_score', 1.0)
                    
                    # Determine source type based on node ID
                    if node_id.startswith('GO:'):
                        source_type = 'go_term'
                        content = f"GO term: {node_name} ({node_id})"
                    elif node_id.startswith('DISEASE:'):
                        source_type = 'disease'
                        content = f"Disease: {node_name or node_id.replace('DISEASE:', '')}"
                    elif node_id.startswith('DRUG:'):
                        source_type = 'drug'  
                        content = f"Drug: {node_name or node_id.replace('DRUG:', '')}"
                    elif node_id.startswith('GENE:') or len(node_id.split(':')) == 1:
                        source_type = 'gene'
                        content = f"Gene: {node_name or node_id}"
                    else:
                        source_type = 'entity'
                        content = f"Entity: {node_name} ({node_id})"
                    
                    source = EvidenceSource(
                        source_type=source_type,
                        content=content,
                        confidence=min(match_score, 1.0),
                        details=item
                    )
                    sources.append(source)
        
        # Viral expression results
        if 'top_genes' in data and data.get('found', False):
            for i, gene_info in enumerate(data.get('top_genes', [])[:10]):  # Top 10 genes
                gene_symbol = gene_info.get('gene_symbol', 'Unknown gene')
                max_expr = gene_info.get('max_expression', 0)
                avg_expr = gene_info.get('avg_expression', 0)
                conditions = gene_info.get('condition_count', 0)
                
                source = EvidenceSource(
                    source_type='viral_expression',
                    content=f"Gene {gene_symbol}: max viral expression {max_expr:.2f}, average {avg_expr:.2f} across {conditions} viral conditions",
                    confidence=0.9,  # High confidence for quantitative data
                    details={
                        'gene': gene_symbol,
                        'max_expression': max_expr,
                        'avg_expression': avg_expr,
                        'condition_count': conditions,
                        'rank': i + 1
                    }
                )
                sources.append(source)
        
        return sources
    
    def _rank_evidence(self, evidence_sources: List[EvidenceSource], query_plan: QueryPlan) -> List[EvidenceSource]:
        """Rank evidence sources by relevance and confidence."""
        
        def calculate_relevance_score(evidence: EvidenceSource) -> float:
            base_score = evidence.confidence
            
            # Boost based on evidence type relevance to query intent
            type_bonus = self.evidence_weights.get(evidence.source_type, 0.5)
            base_score *= type_bonus
            
            # Boost if evidence mentions entities from the query
            entity_mentions = 0
            content_lower = evidence.content.lower()
            for entity in query_plan.entities:
                if entity.normalized_form.lower() in content_lower:
                    entity_mentions += 1
            
            entity_bonus = min(entity_mentions * 0.1, 0.3)
            base_score += entity_bonus
            
            return base_score
        
        # Calculate relevance scores and sort
        scored_evidence = [(calculate_relevance_score(ev), ev) for ev in evidence_sources]
        scored_evidence.sort(key=lambda x: x[0], reverse=True)
        
        return [ev for score, ev in scored_evidence]

class PromptTemplateManager:
    """Manage prompt templates for different synthesis scenarios."""
    
    def __init__(self):
        self.templates = {
            'gene_function': """You are a biomedical expert. Based on the knowledge graph evidence below, provide a comprehensive answer about gene function.

Question: {question}

Evidence from Knowledge Graph:
{evidence}

Instructions:
1. Analyze the GO annotations, pathways, and associations
2. Explain the gene's molecular function, biological process, and cellular component
3. Provide specific mechanisms when available
4. Cite evidence using numbered references [1], [2], etc.
5. Indicate confidence levels where appropriate

Answer:""",

            'gene_disease': """You are a biomedical expert. Based on the knowledge graph evidence below, explain the relationship between genes and diseases.

Question: {question}

Evidence from Knowledge Graph:
{evidence}

Instructions:
1. Identify specific gene-disease associations
2. Explain potential mechanisms of pathogenesis
3. Discuss any known variants or mutations
4. Reference supporting pathways and biological processes
5. Use numbered citations [1], [2], etc. for evidence

Answer:""",

            'pathway_analysis': """You are a biomedical expert. Based on the knowledge graph evidence below, provide a detailed pathway analysis.

Question: {question}

Evidence from Knowledge Graph:
{evidence}

Instructions:
1. Map out the biological pathway or process
2. Identify key genes and their roles
3. Explain regulatory mechanisms
4. Discuss pathway interactions and crosstalk
5. Reference evidence with numbered citations [1], [2], etc.

Answer:""",

            'drug_targets': """You are a pharmaceutical expert. Based on the knowledge graph evidence below, analyze drug targets and interactions.

Question: {question}

Evidence from Knowledge Graph:
{evidence}

Instructions:
1. Identify drug targets and mechanisms of action
2. Explain molecular interactions
3. Discuss therapeutic implications
4. Consider off-target effects if mentioned
5. Use numbered citations [1], [2], etc. for evidence

Answer:""",

            'general': """You are a biomedical expert. Based on the knowledge graph evidence below, provide a comprehensive answer to the scientific question.

Question: {question}

Evidence from Knowledge Graph:
{evidence}

Instructions:
1. Synthesize information from multiple evidence sources
2. Provide a clear, scientific explanation
3. Highlight key findings and mechanisms
4. Use numbered citations [1], [2], etc. to reference evidence
5. Indicate any limitations in the available data

Answer:"""
        }
    
    def get_template(self, intent: QueryIntent) -> str:
        """Get appropriate template for query intent."""
        template_map = {
            QueryIntent.GENE_FUNCTION: 'gene_function',
            QueryIntent.GENE_DISEASE: 'gene_disease', 
            QueryIntent.PATHWAY_ANALYSIS: 'pathway_analysis',
            QueryIntent.DRUG_TARGETS: 'drug_targets',
            QueryIntent.GENE_DRUG: 'drug_targets',
            QueryIntent.DISEASE_GENES: 'gene_disease'
        }
        
        template_key = template_map.get(intent, 'general')
        return self.templates[template_key]

class ResponseSynthesizer:
    """
    Main response synthesizer that combines KG results with LLM reasoning
    to generate comprehensive, evidence-based answers.
    """
    
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        self.evidence_processor = EvidenceProcessor()
        self.template_manager = PromptTemplateManager()
        self.synthesis_cache = {}
    
    def synthesize_response(self, 
                          question: str,
                          kg_results: Dict[str, Any], 
                          query_plan: QueryPlan,
                          conversation_history: Optional[List[Dict[str, str]]] = None) -> SynthesizedResponse:
        """
        Synthesize a comprehensive response from knowledge graph results.
        
        Args:
            question: Original user question
            kg_results: Results from knowledge graph queries
            query_plan: Original query plan
            conversation_history: Optional conversation context
            
        Returns:
            SynthesizedResponse with answer, evidence, and metadata
        """
        logger.info(f"Synthesizing response for question: {question[:100]}...")
        
        # Check cache
        cache_key = hash(question + str(kg_results))
        if cache_key in self.synthesis_cache:
            logger.info("Using cached synthesis")
            return self.synthesis_cache[cache_key]
        
        try:
            # Debug: Log the KG results structure
            logger.info(f"DEBUG: KG results keys: {list(kg_results.keys())}")
            for key, value in kg_results.items():
                if isinstance(value, dict):
                    logger.info(f"DEBUG: {key} = dict with keys: {list(value.keys())}")
                    if 'target_genes' in value:
                        logger.info(f"DEBUG: {key} has target_genes with {len(value['target_genes'])} entries")
                        if len(value['target_genes']) > 0:
                            logger.info(f"DEBUG: First few target genes: {value['target_genes'][:5]}")
                    if 'found' in value:
                        logger.info(f"DEBUG: {key} has found={value['found']}")
                else:
                    logger.info(f"DEBUG: {key} = {type(value)} - {len(value) if hasattr(value, '__len__') else value}")
            
            # Process evidence from KG results
            evidence_sources = self.evidence_processor.process_kg_results(kg_results, query_plan)
            logger.info(f"Processed {len(evidence_sources)} evidence sources")
            
            # If no evidence found, try smart entity search as fallback
            if len(evidence_sources) == 0:
                logger.info("No evidence found, trying smart entity search...")
                from .kg_service import KnowledgeGraphService
                # Try to get KG service from somewhere or create temporary one
                # For now, we'll add a flag to indicate this needs better integration
                pass
            
            # Generate evidence summary for LLM
            evidence_text = self._format_evidence_for_llm(evidence_sources)
            
            # Select appropriate prompt template
            template = self.template_manager.get_template(query_plan.intent)
            
            # Create synthesis prompt
            synthesis_prompt = template.format(
                question=question,
                evidence=evidence_text
            )
            
            # Generate response using LLM
            logger.info("Generating LLM response...")
            ollama_response = self.ollama_client.generate_response(synthesis_prompt)
            
            # Extract citations and validate them
            answer, citation_map = self._process_citations(ollama_response.content, evidence_sources)
            
            # Generate follow-up questions
            follow_ups = self._generate_follow_ups(question, answer, evidence_sources, query_plan)
            
            # Assess query coverage
            coverage = self._assess_query_coverage(kg_results, query_plan)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_confidence(evidence_sources, coverage, query_plan)
            
            # Create synthesis metadata
            metadata = {
                'evidence_count': len(evidence_sources),
                'llm_model': self.ollama_client.model,
                'template_used': query_plan.intent.value,
                'query_steps_executed': len(query_plan.query_strategy),
                'citation_count': len(citation_map),
                'response_length': len(answer)
            }
            
            # Create final response
            synthesized_response = SynthesizedResponse(
                answer=answer,
                confidence=overall_confidence,
                evidence_sources=evidence_sources,
                follow_up_questions=follow_ups,
                query_coverage=coverage,
                synthesis_metadata=metadata
            )
            
            # Cache the response
            self.synthesis_cache[cache_key] = synthesized_response
            
            logger.info(f"Response synthesized successfully (confidence: {overall_confidence:.2f})")
            return synthesized_response
            
        except Exception as e:
            logger.error(f"Failed to synthesize response: {e}")
            # Return fallback response
            return self._create_fallback_response(question, kg_results, str(e))
    
    def _format_evidence_for_llm(self, evidence_sources: List[EvidenceSource]) -> str:
        """Format evidence sources for LLM consumption."""
        formatted_evidence = []
        
        # Group evidence by type for better organization
        evidence_by_type = defaultdict(list)
        for evidence in evidence_sources:
            evidence_by_type[evidence.source_type].append(evidence)
        
        # Format each evidence type
        citation_num = 1
        for evidence_type, evidences in evidence_by_type.items():
            if not evidences:
                continue
                
            formatted_evidence.append(f"\n## {evidence_type.replace('_', ' ').title()}")
            
            for evidence in evidences[:5]:  # Limit per type
                formatted_evidence.append(
                    f"[{citation_num}] {evidence.content} (Confidence: {evidence.confidence:.2f})"
                )
                citation_num += 1
        
        return "\n".join(formatted_evidence)
    
    def _process_citations(self, response_text: str, evidence_sources: List[EvidenceSource]) -> Tuple[str, Dict[int, EvidenceSource]]:
        """Process and validate citations in the response."""
        citation_pattern = r'\[(\d+)\]'
        citations_found = re.findall(citation_pattern, response_text)
        
        citation_map = {}
        for i, evidence in enumerate(evidence_sources, 1):
            if str(i) in citations_found:
                citation_map[i] = evidence
        
        return response_text, citation_map
    
    def _generate_follow_ups(self, 
                           question: str, 
                           answer: str, 
                           evidence_sources: List[EvidenceSource],
                           query_plan: QueryPlan) -> List[str]:
        """Generate relevant follow-up questions."""
        try:
            # Use Ollama to generate follow-ups
            follow_ups = self.ollama_client.generate_follow_up_questions(
                question, answer, {'evidence_types': [e.source_type for e in evidence_sources]}
            )
            return follow_ups[:5]  # Limit to 5
        except Exception as e:
            logger.warning(f"Failed to generate follow-up questions: {e}")
            return self._generate_default_follow_ups(query_plan)
    
    def _generate_default_follow_ups(self, query_plan: QueryPlan) -> List[str]:
        """Generate default follow-up questions based on query intent."""
        intent_follow_ups = {
            QueryIntent.GENE_FUNCTION: [
                "What pathways involve this gene?",
                "Are there any disease associations?",
                "What are the regulatory mechanisms?"
            ],
            QueryIntent.GENE_DISEASE: [
                "What mutations are associated with this condition?", 
                "Are there therapeutic targets?",
                "What is the molecular mechanism?"
            ],
            QueryIntent.PATHWAY_ANALYSIS: [
                "Which genes are key regulators?",
                "How is this pathway controlled?",
                "What diseases involve this pathway?"
            ]
        }
        
        return intent_follow_ups.get(query_plan.intent, [
            "What additional information is available?",
            "Are there related biological processes?",
            "What are the clinical implications?"
        ])
    
    def _assess_query_coverage(self, kg_results: Dict[str, Any], query_plan: QueryPlan) -> Dict[str, bool]:
        """Assess how well the KG results covered the query plan."""
        coverage = {}
        
        for step in query_plan.query_strategy:
            method = step.get('method')
            coverage[method] = method in kg_results and bool(kg_results.get(method))
        
        # Overall coverage metrics
        total_steps = len(query_plan.query_strategy)
        covered_steps = sum(coverage.values())
        coverage['overall_percentage'] = covered_steps / total_steps if total_steps > 0 else 0
        
        return coverage
    
    def _calculate_confidence(self, 
                            evidence_sources: List[EvidenceSource],
                            coverage: Dict[str, bool], 
                            query_plan: QueryPlan) -> float:
        """Calculate overall confidence in the synthesized response."""
        if not evidence_sources:
            return 0.1
        
        # Evidence quality score
        evidence_scores = [e.confidence for e in evidence_sources]
        avg_evidence_confidence = sum(evidence_scores) / len(evidence_scores)
        
        # Coverage score
        coverage_score = coverage.get('overall_percentage', 0)
        
        # Query plan confidence
        plan_confidence = query_plan.confidence
        
        # Weighted combination
        overall_confidence = (
            0.4 * avg_evidence_confidence +
            0.3 * coverage_score +
            0.3 * plan_confidence
        )
        
        return min(overall_confidence, 1.0)
    
    def _create_fallback_response(self, 
                                 question: str, 
                                 kg_results: Dict[str, Any], 
                                 error_msg: str) -> SynthesizedResponse:
        """Create a fallback response when synthesis fails."""
        fallback_answer = f"""I encountered an issue while synthesizing a response to your question: "{question}"

Available data from the knowledge graph:
{self._format_raw_results(kg_results)}

Error details: {error_msg}

Please try rephrasing your question or asking about specific aspects of this topic."""
        
        return SynthesizedResponse(
            answer=fallback_answer,
            confidence=0.2,
            evidence_sources=[],
            follow_up_questions=[
                "Could you rephrase the question?",
                "Are you looking for specific gene information?",
                "Would you like to explore related topics?"
            ],
            query_coverage={'fallback': True},
            synthesis_metadata={'error': error_msg, 'fallback_used': True}
        )
    
    def _format_raw_results(self, kg_results: Dict[str, Any]) -> str:
        """Format raw KG results for fallback response."""
        formatted = []
        for key, value in kg_results.items():
            if isinstance(value, dict):
                item_count = len([k for k, v in value.items() if isinstance(v, list) and v])
                formatted.append(f"- {key}: {item_count} items found")
            elif isinstance(value, list):
                formatted.append(f"- {key}: {len(value)} items")
            else:
                formatted.append(f"- {key}: {str(value)[:100]}")
        
        return "\n".join(formatted[:10])  # Limit to 10 items
    
    def enhance_response_with_context(self, 
                                    response: SynthesizedResponse,
                                    conversation_history: List[Dict[str, str]]) -> SynthesizedResponse:
        """Enhance response with conversation context."""
        if not conversation_history:
            return response
        
        try:
            context_prompt = f"""Given this conversation history and current response, provide an enhanced answer that builds on previous discussion:

Conversation History:
{self._format_conversation_history(conversation_history)}

Current Response:
{response.answer}

Enhanced Response:"""
            
            enhanced_response = self.ollama_client.generate_response(context_prompt)
            
            # Update the response
            response.answer = enhanced_response.content
            response.synthesis_metadata['context_enhanced'] = True
            
        except Exception as e:
            logger.warning(f"Failed to enhance response with context: {e}")
        
        return response
    
    def _format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history for context."""
        formatted = []
        for i, msg in enumerate(history[-5:]):  # Last 5 messages
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:200]  # Truncate long messages
            formatted.append(f"{role.title()}: {content}")
        
        return "\n".join(formatted)
    
    def get_synthesis_stats(self) -> Dict[str, Any]:
        """Get synthesis statistics."""
        return {
            'cache_size': len(self.synthesis_cache),
            'available_templates': list(self.template_manager.templates.keys()),
            'evidence_types_supported': list(self.evidence_processor.evidence_weights.keys()),
            'ollama_model': self.ollama_client.model
        }