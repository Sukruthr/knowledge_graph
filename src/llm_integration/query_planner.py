"""
Query Planning Agent for Knowledge Graph Q&A System

This agent analyzes natural language questions and converts them into structured
knowledge graph query strategies, handling complex biomedical reasoning workflows.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Types of query intents for biomedical questions."""
    GENE_FUNCTION = "gene_function"
    GENE_DISEASE = "gene_disease"
    GENE_DRUG = "gene_drug"
    PATHWAY_ANALYSIS = "pathway_analysis"
    DISEASE_GENES = "disease_genes"
    DRUG_TARGETS = "drug_targets"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    MECHANISM_EXPLORATION = "mechanism_exploration"
    GENERAL_SEARCH = "general_search"
    GO_TERM_QUERY = "go_term_query"
    VIRAL_RESPONSE = "viral_response"
    MODEL_COMPARISON = "model_comparison"
    EXPRESSION_ANALYSIS = "expression_analysis"

@dataclass
class EntityMention:
    """Represents a biological entity mentioned in the query."""
    text: str
    entity_type: str  # gene, disease, drug, pathway, go_term, etc.
    confidence: float
    start_pos: int
    end_pos: int
    normalized_form: Optional[str] = None

@dataclass 
class QueryPlan:
    """Structured plan for executing knowledge graph queries."""
    intent: QueryIntent
    entities: List[EntityMention]
    query_strategy: List[Dict[str, Any]]
    synthesis_instructions: str
    confidence: float
    expected_result_types: List[str]
    multi_step: bool = False

class BiomedicalEntityExtractor:
    """Extract biological entities from natural language text."""
    
    def __init__(self):
        # Define patterns for different entity types
        self.patterns = {
            'gene': [
                r'\b[A-Z][A-Z0-9]{1,6}\b',  # Standard gene symbols (TP53, BRCA1)
                r'\b[A-Z][a-z]{2,10}[0-9]{1,3}\b',  # Mixed case genes (Pten1, Myc)
                r'\bgene[s]?\s+([A-Z][A-Z0-9]{1,6})\b',  # "gene TP53"
                r'\b([A-Z][A-Z0-9]{1,6})\s+gene[s]?\b',  # "TP53 gene"
            ],
            'disease': [
                r'\bcancer\b', r'\btumor\b', r'\btumour\b', r'\bcarcinoma\b',
                r'\bdiabetes\b', r'\balzheimer[\'s]?\b', r'\bparkinson[\'s]?\b',
                r'\bheart\s+disease\b', r'\bstroke\b', r'\bhypertension\b',
                r'\basthma\b', r'\barbritis\b', r'\bobesity\b'
            ],
            'drug': [
                r'\bestradiol\b', r'\btestosterone\b', r'\bcortisol\b', r'\binsulin\b',
                r'\baspirin\b', r'\bmetformin\b', r'\bwarfarin\b', r'\btamoxifen\b',
                r'\bdoxorubicin\b', r'\bcisplatin\b', r'\bpaclitaxel\b', r'\bchemotherapy\b',
                r'\btreatment[s]?\b', r'\bmedication[s]?\b', r'\bdrug[s]?\b', 
                r'\btherapy\b', r'\bcompound[s]?\b', r'\bhormone[s]?\b'
            ],
            'pathway': [
                r'\bpathway[s]?\b', r'\bsignaling\b', r'\bmetabolism\b',
                r'\bbiosynthesis\b', r'\bregulation\b', r'\bprocess[es]?\b',
                r'\bmechanism[s]?\b', r'\bfunction[s]?\b'
            ],
            'go_term': [
                r'\bGO:\d{7}\b',  # GO term IDs
                r'\bbiology\b', r'\bbiological\s+process\b',
                r'\bcellular\s+component\b', r'\bmolecular\s+function\b'
            ],
            'organism': [
                r'\bhuman[s]?\b', r'\bmice\b', r'\bmouse\b', r'\brat[s]?\b',
                r'\bbird[s]?\b', r'\bavian\b', r'\bzebrafish\b', r'\bfly\b',
                r'\byeast\b', r'\bbacteria[l]?\b'
            ],
            'tissue': [
                r'\bbrain\b', r'\bheart\b', r'\bliver\b', r'\blung[s]?\b',
                r'\bkidney[s]?\b', r'\bmuscle\b', r'\bskin\b', r'\bbone[s]?\b',
                r'\bblood\b', r'\btissue[s]?\b'
            ],
            'phenotype': [
                r'\bcolor\b', r'\bcolour\b', r'\bpigment[ation]?\b',
                r'\bfeather[s]?\b', r'\bheight\b', r'\bweight\b', r'\bsize\b',
                r'\bdevelopment\b', r'\bgrowth\b', r'\bbehavior\b', r'\bbehaviour\b'
            ]
        }
        
        # Gene symbol validation patterns
        self.valid_gene_patterns = [
            r'^[A-Z]{2,}[0-9]*[A-Z]*$',  # Standard human gene symbols
            r'^[A-Z][a-z]+[0-9]*$',      # Mouse/other species conventions
        ]
        
        # Common biological terms that aren't specific entities
        self.stopwords = {
            'gene', 'genes', 'protein', 'proteins', 'expression', 'activity',
            'level', 'levels', 'function', 'functions', 'role', 'roles',
            'involved', 'associated', 'related', 'linked', 'connected',
            'what', 'which', 'where', 'when', 'how', 'why', 'who', 'whose',
            'treatment', 'treatments', 'therapy', 'therapies'
        }
    
    def extract_entities(self, text: str) -> List[EntityMention]:
        """
        Extract biological entities from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of extracted entity mentions
        """
        entities = []
        text_lower = text.lower()
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group(1) if match.groups() else match.group(0)
                    
                    # Skip if it's a stopword
                    if entity_text.lower() in self.stopwords:
                        continue
                    
                    # Special validation for genes
                    if entity_type == 'gene':
                        if not self._is_valid_gene_symbol(entity_text):
                            continue
                    
                    # Calculate confidence based on pattern specificity and context
                    confidence = self._calculate_confidence(entity_text, entity_type, text_lower)
                    
                    entity = EntityMention(
                        text=entity_text,
                        entity_type=entity_type,
                        confidence=confidence,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        normalized_form=self._normalize_entity(entity_text, entity_type)
                    )
                    entities.append(entity)
        
        # Remove duplicates and overlapping entities
        entities = self._deduplicate_entities(entities)
        
        # Sort by confidence
        entities.sort(key=lambda x: x.confidence, reverse=True)
        
        return entities
    
    def _is_valid_gene_symbol(self, symbol: str) -> bool:
        """Check if a symbol looks like a valid gene symbol."""
        # Too short or too long
        if len(symbol) < 2 or len(symbol) > 15:
            return False
        
        # Check against valid patterns
        for pattern in self.valid_gene_patterns:
            if re.match(pattern, symbol):
                return True
        
        return False
    
    def _calculate_confidence(self, entity_text: str, entity_type: str, full_text: str) -> float:
        """Calculate confidence score for an entity mention."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on context
        context_words = {
            'gene': ['gene', 'protein', 'encodes', 'expression', 'mutation'],
            'disease': ['disease', 'disorder', 'syndrome', 'cancer', 'patients'],
            'drug': ['drug', 'treatment', 'therapy', 'medication', 'compound'],
            'pathway': ['pathway', 'signaling', 'cascade', 'process', 'mechanism']
        }
        
        if entity_type in context_words:
            for word in context_words[entity_type]:
                if word in full_text:
                    confidence += 0.1
        
        # Boost for specific patterns
        if entity_type == 'gene' and re.match(r'^[A-Z]{3,6}[0-9]*$', entity_text):
            confidence += 0.2
        
        if entity_type == 'go_term' and entity_text.startswith('GO:'):
            confidence += 0.3
        
        return min(confidence, 1.0)
    
    def _normalize_entity(self, entity_text: str, entity_type: str) -> str:
        """Normalize entity text for better matching."""
        if entity_type == 'gene':
            # Standardize gene symbols to uppercase
            return entity_text.upper()
        elif entity_type == 'disease':
            # Standardize disease names
            return entity_text.lower()
        else:
            return entity_text.lower()
    
    def _deduplicate_entities(self, entities: List[EntityMention]) -> List[EntityMention]:
        """Remove duplicate and overlapping entity mentions."""
        # Sort by start position
        entities.sort(key=lambda x: x.start_pos)
        
        deduplicated = []
        for entity in entities:
            # Check for overlap with existing entities
            overlaps = False
            for existing in deduplicated:
                if (entity.start_pos < existing.end_pos and 
                    entity.end_pos > existing.start_pos):
                    # If overlap, keep the one with higher confidence
                    if entity.confidence > existing.confidence:
                        deduplicated.remove(existing)
                    else:
                        overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(entity)
        
        return deduplicated

class IntentClassifier:
    """Classify the intent of biomedical questions."""
    
    def __init__(self):
        # Intent patterns and keywords
        self.intent_patterns = {
            QueryIntent.GENE_FUNCTION: [
                r'what.*(do|does|is|are).*(gene|protein).*(do|function|role)',
                r'function.*(of|for).*(gene|protein)',
                r'(role|purpose).*(of|for)',
                r'what.*(gene|protein).*responsible.*for'
            ],
            QueryIntent.GENE_DISEASE: [
                r'gene.*associated.*with.*(disease|cancer|disorder)',
                r'(disease|cancer|disorder).*caused.*by.*gene',
                r'genetic.*basis.*(of|for)',
                r'mutation.*in.*(gene|protein).*cause'
            ],
            QueryIntent.GENE_DRUG: [
                r'drug.*target.*(gene|protein)',
                r'(gene|protein).*targeted.*by.*drug',
                r'therapeutic.*target',
                r'drug.*interact.*with'
            ],
            QueryIntent.PATHWAY_ANALYSIS: [
                r'pathway.*involv',
                r'signaling.*cascade',
                r'biological.*process',
                r'metabolic.*pathway',
                r'mechanism.*of.*action'
            ],
            QueryIntent.DISEASE_GENES: [
                r'gene.*cause.*(disease|cancer|disorder)',
                r'genetic.*factor.*(in|for|of)',
                r'which.*gene.*associated.*with',
                r'gene.*involved.*in.*(disease|cancer|disorder)'
            ],
            QueryIntent.DRUG_TARGETS: [
                r'target.*of.*(drug|compound|treatment)',
                r'(drug|compound|treatment).*target',
                r'molecular.*target',
                r'therapeutic.*target'
            ],
            QueryIntent.COMPARATIVE_ANALYSIS: [
                r'compar.*between',
                r'difference.*between',
                r'similar.*between',
                r'versus|vs\.|vs',
                r'which.*better'
            ],
            QueryIntent.MECHANISM_EXPLORATION: [
                r'how.*(do|does).*work',
                r'mechanism.*of',
                r'molecular.*basis',
                r'biochemical.*pathway',
                r'cellular.*mechanism'
            ],
            QueryIntent.GO_TERM_QUERY: [
                r'GO:\d{7}',
                r'gene.*ontology',
                r'biological.*process',
                r'molecular.*function',
                r'cellular.*component'
            ],
            QueryIntent.VIRAL_RESPONSE: [
                r'viral.*infection',
                r'immune.*response',
                r'virus.*interaction',
                r'viral.*replication',
                r'antiviral.*response'
            ],
            QueryIntent.EXPRESSION_ANALYSIS: [
                r'expression.*level',
                r'expression.*pattern',
                r'upregulated|downregulated',
                r'expression.*profile',
                r'highly.*expressed',
                r'high.*expression',
                r'expression.*changes',
                r'gene.*expression',
                r'transcript.*level'
            ]
        }
        
        self.intent_keywords = {
            QueryIntent.GENE_FUNCTION: ['function', 'role', 'purpose', 'activity'],
            QueryIntent.PATHWAY_ANALYSIS: ['pathway', 'signaling', 'cascade', 'process'],
            QueryIntent.DISEASE_GENES: ['disease', 'cancer', 'disorder', 'syndrome'],
            QueryIntent.DRUG_TARGETS: ['drug', 'compound', 'treatment', 'therapy'],
            QueryIntent.COMPARATIVE_ANALYSIS: ['compare', 'versus', 'difference', 'similar'],
            QueryIntent.MECHANISM_EXPLORATION: ['mechanism', 'how', 'molecular', 'biochemical'],
            QueryIntent.EXPRESSION_ANALYSIS: ['expression', 'level', 'upregulated', 'downregulated']
        }
    
    def classify_intent(self, text: str, entities: List[EntityMention]) -> Tuple[QueryIntent, float]:
        """
        Classify the intent of a question.
        
        Args:
            text: Question text
            entities: Extracted entities
            
        Returns:
            Tuple of (intent, confidence)
        """
        text_lower = text.lower()
        intent_scores = {}
        
        # Score based on pattern matching
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            intent_scores[intent] = score
        
        # Score based on keyword presence
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    intent_scores[intent] = intent_scores.get(intent, 0) + 0.5
        
        # Boost based on entity types
        entity_types = [e.entity_type for e in entities]
        
        if 'gene' in entity_types and 'disease' in entity_types:
            intent_scores[QueryIntent.GENE_DISEASE] = intent_scores.get(QueryIntent.GENE_DISEASE, 0) + 1
        
        if 'gene' in entity_types and 'drug' in entity_types:
            intent_scores[QueryIntent.GENE_DRUG] = intent_scores.get(QueryIntent.GENE_DRUG, 0) + 1
        
        if 'pathway' in entity_types or 'go_term' in entity_types:
            intent_scores[QueryIntent.PATHWAY_ANALYSIS] = intent_scores.get(QueryIntent.PATHWAY_ANALYSIS, 0) + 1
        
        # Default to general search if no strong signals
        if not intent_scores or max(intent_scores.values()) < 0.5:
            return QueryIntent.GENERAL_SEARCH, 0.3
        
        # Return highest scoring intent
        best_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[best_intent]
        confidence = min(max_score / 3.0, 1.0)  # Normalize to 0-1
        
        return best_intent, confidence

class QueryStrategist:
    """Generate query strategies for different intents."""
    
    def __init__(self):
        # Define query strategies for each intent type
        self.strategies = {
            QueryIntent.GENE_FUNCTION: self._gene_function_strategy,
            QueryIntent.GENE_DISEASE: self._gene_disease_strategy,
            QueryIntent.GENE_DRUG: self._gene_drug_strategy,
            QueryIntent.PATHWAY_ANALYSIS: self._pathway_analysis_strategy,
            QueryIntent.DISEASE_GENES: self._disease_genes_strategy,
            QueryIntent.DRUG_TARGETS: self._drug_targets_strategy,
            QueryIntent.COMPARATIVE_ANALYSIS: self._comparative_analysis_strategy,
            QueryIntent.MECHANISM_EXPLORATION: self._mechanism_exploration_strategy,
            QueryIntent.GENERAL_SEARCH: self._general_search_strategy,
            QueryIntent.GO_TERM_QUERY: self._go_term_query_strategy,
            QueryIntent.VIRAL_RESPONSE: self._viral_response_strategy,
            QueryIntent.EXPRESSION_ANALYSIS: self._expression_analysis_strategy
        }
    
    def generate_strategy(self, intent: QueryIntent, entities: List[EntityMention], text: str) -> Dict[str, Any]:
        """
        Generate query strategy for given intent and entities.
        
        Args:
            intent: Classified intent
            entities: Extracted entities
            text: Original question text
            
        Returns:
            Dictionary containing query strategy
        """
        strategy_func = self.strategies.get(intent, self._general_search_strategy)
        return strategy_func(entities, text)
    
    def _gene_function_strategy(self, entities: List[EntityMention], text: str) -> Dict[str, Any]:
        """Strategy for gene function queries."""
        genes = [e for e in entities if e.entity_type == 'gene']
        
        if not genes:
            return self._general_search_strategy(entities, text)
        
        strategy = []
        for gene in genes[:3]:  # Limit to top 3 genes
            strategy.extend([
                {
                    "method": "query_gene_information",
                    "params": {"gene": gene.normalized_form},
                    "description": f"Get comprehensive information for gene {gene.text}"
                },
                {
                    "method": "query_pathway_information", 
                    "params": {"pathway": f"involving {gene.text}"},
                    "description": f"Find pathways involving {gene.text}"
                }
            ])
        
        return {
            "query_steps": strategy,
            "synthesis_focus": "gene function and biological pathways",
            "expected_results": ["go_annotations", "pathways", "molecular_functions"],
            "multi_step": len(genes) > 1
        }
    
    def _gene_disease_strategy(self, entities: List[EntityMention], text: str) -> Dict[str, Any]:
        """Strategy for gene-disease association queries."""
        genes = [e for e in entities if e.entity_type == 'gene']
        diseases = [e for e in entities if e.entity_type == 'disease']
        
        strategy = []
        
        if genes:
            for gene in genes[:2]:
                strategy.append({
                    "method": "query_gene_information",
                    "params": {"gene": gene.normalized_form},
                    "description": f"Get disease associations for gene {gene.text}"
                })
        
        if diseases:
            for disease in diseases[:2]:
                strategy.append({
                    "method": "query_disease_associations",
                    "params": {"disease": disease.normalized_form},
                    "description": f"Find genes associated with {disease.text}"
                })
        
        # Add pathway analysis for mechanism exploration
        strategy.append({
            "method": "search_by_keywords",
            "params": {"keywords": [e.normalized_form for e in genes + diseases]},
            "description": "Cross-reference genes and diseases in pathways"
        })
        
        return {
            "query_steps": strategy,
            "synthesis_focus": "gene-disease associations and mechanisms",
            "expected_results": ["disease_associations", "genetic_variants", "pathways"],
            "multi_step": True
        }
    
    def _pathway_analysis_strategy(self, entities: List[EntityMention], text: str) -> Dict[str, Any]:
        """Strategy for pathway analysis queries."""
        pathways = [e for e in entities if e.entity_type == 'pathway']
        genes = [e for e in entities if e.entity_type == 'gene']
        
        strategy = []
        
        if pathways:
            for pathway in pathways[:2]:
                strategy.append({
                    "method": "query_pathway_information",
                    "params": {"pathway": pathway.normalized_form},
                    "description": f"Analyze {pathway.text} pathway"
                })
        
        if genes:
            for gene in genes[:3]:
                strategy.append({
                    "method": "query_gene_information", 
                    "params": {"gene": gene.normalized_form},
                    "description": f"Get pathway information for {gene.text}"
                })
        
        # Add keyword search for broader pathway context - break compound terms into searchable words
        search_terms = []
        for entity in entities:
            # Break compound terms into individual words for better matching
            words = entity.normalized_form.replace('_', ' ').split()
            search_terms.extend(words)
        
        # Add contextual terms
        search_terms.extend(["pathway", "signaling"])
        
        # Remove duplicates and filter out common stop words
        search_terms = list(set(search_terms))
        search_terms = [term for term in search_terms if len(term) > 2]
        
        if search_terms:
            strategy.append({
                "method": "search_by_keywords",
                "params": {"keywords": search_terms[:8]},  # Limit to avoid too many terms
                "description": "Search for related pathways and processes"
            })
        
        return {
            "query_steps": strategy,
            "synthesis_focus": "pathway mechanisms and gene interactions",
            "expected_results": ["pathways", "gene_interactions", "biological_processes"],
            "multi_step": True
        }
    
    def _disease_genes_strategy(self, entities: List[EntityMention], text: str) -> Dict[str, Any]:
        """Strategy for finding genes associated with diseases."""
        diseases = [e for e in entities if e.entity_type == 'disease']
        
        if not diseases:
            return self._general_search_strategy(entities, text)
        
        strategy = []
        for disease in diseases[:2]:
            strategy.extend([
                {
                    "method": "query_disease_associations",
                    "params": {"disease": disease.normalized_form},
                    "description": f"Find genes associated with {disease.text}"
                },
                {
                    "method": "search_by_keywords",
                    "params": {"keywords": [disease.normalized_form, "genetic", "mutation"]},
                    "description": f"Search for genetic factors in {disease.text}"
                }
            ])
        
        return {
            "query_steps": strategy,
            "synthesis_focus": "disease-associated genes and genetic mechanisms",
            "expected_results": ["associated_genes", "genetic_variants", "pathways"],
            "multi_step": len(diseases) > 1
        }
    
    def _drug_targets_strategy(self, entities: List[EntityMention], text: str) -> Dict[str, Any]:
        """Strategy for drug target queries."""
        drugs = [e for e in entities if e.entity_type == 'drug']
        genes = [e for e in entities if e.entity_type == 'gene']
        
        strategy = []
        
        if drugs:
            for drug in drugs[:2]:
                strategy.append({
                    "method": "query_drug_interactions",
                    "params": {"drug": drug.normalized_form},
                    "description": f"Find targets and interactions for {drug.text}"
                })
        
        if genes:
            for gene in genes[:2]:
                strategy.append({
                    "method": "query_gene_information",
                    "params": {"gene": gene.normalized_form},
                    "description": f"Get drug interaction information for {gene.text}"
                })
        
        return {
            "query_steps": strategy,
            "synthesis_focus": "drug-target interactions and mechanisms",
            "expected_results": ["drug_targets", "interactions", "mechanisms"],
            "multi_step": True
        }
    
    def _gene_drug_strategy(self, entities: List[EntityMention], text: str) -> Dict[str, Any]:
        """Strategy for gene-drug interaction queries."""
        genes = [e for e in entities if e.entity_type == 'gene']
        drugs = [e for e in entities if e.entity_type == 'drug']
        
        strategy = []
        
        if genes:
            for gene in genes[:2]:
                strategy.append({
                    "method": "query_gene_information",
                    "params": {"gene": gene.normalized_form},
                    "description": f"Get drug interaction information for gene {gene.text}"
                })
        
        if drugs:
            for drug in drugs[:2]:
                strategy.append({
                    "method": "query_drug_interactions",
                    "params": {"drug": drug.normalized_form},
                    "description": f"Find gene targets for drug {drug.text}"
                })
        
        return {
            "query_steps": strategy,
            "synthesis_focus": "gene-drug interactions and pharmacogenomics",
            "expected_results": ["drug_interactions", "gene_targets", "pharmacological_effects"],
            "multi_step": True
        }
    
    def _comparative_analysis_strategy(self, entities: List[EntityMention], text: str) -> Dict[str, Any]:
        """Strategy for comparative analysis queries."""
        genes = [e for e in entities if e.entity_type == 'gene']
        diseases = [e for e in entities if e.entity_type == 'disease']
        drugs = [e for e in entities if e.entity_type == 'drug']
        
        strategy = []
        
        # Get information for all entities to enable comparison
        for gene in genes:
            strategy.append({
                "method": "query_gene_information",
                "params": {"gene": gene.normalized_form},
                "description": f"Get comprehensive data for {gene.text}"
            })
        
        for disease in diseases:
            strategy.append({
                "method": "query_disease_associations",
                "params": {"disease": disease.normalized_form},
                "description": f"Get gene associations for {disease.text}"
            })
        
        for drug in drugs:
            strategy.append({
                "method": "query_drug_interactions",
                "params": {"drug": drug.normalized_form},
                "description": f"Get target information for {drug.text}"
            })
        
        return {
            "query_steps": strategy,
            "synthesis_focus": "comparative analysis between entities",
            "expected_results": ["comparative_data", "similarities", "differences"],
            "multi_step": True
        }
    
    def _mechanism_exploration_strategy(self, entities: List[EntityMention], text: str) -> Dict[str, Any]:
        """Strategy for mechanism exploration queries."""
        all_entities = entities[:5]  # Limit to avoid too many queries
        
        strategy = []
        
        for entity in all_entities:
            if entity.entity_type == 'gene':
                strategy.append({
                    "method": "query_gene_information",
                    "params": {"gene": entity.normalized_form},
                    "description": f"Get mechanism data for {entity.text}"
                })
            elif entity.entity_type == 'pathway':
                strategy.append({
                    "method": "query_pathway_information",
                    "params": {"pathway": entity.normalized_form},
                    "description": f"Explore {entity.text} mechanisms"
                })
        
        # Add related entity search for broader mechanism context
        strategy.append({
            "method": "get_related_entities",
            "params": {
                "entity": entities[0].normalized_form if entities else "",
                "relation_types": ["pathway", "interaction", "regulation"]
            },
            "description": "Find mechanistically related entities"
        })
        
        return {
            "query_steps": strategy,
            "synthesis_focus": "molecular mechanisms and interactions",
            "expected_results": ["mechanisms", "pathways", "interactions"],
            "multi_step": True
        }
    
    def _general_search_strategy(self, entities: List[EntityMention], text: str) -> Dict[str, Any]:
        """Default strategy for general searches."""
        keywords = [e.normalized_form for e in entities]
        
        # Extract additional keywords from text
        important_words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
        keywords.extend([w.lower() for w in important_words if w.lower() not in {
            'what', 'which', 'where', 'when', 'why', 'how', 'does', 'do', 'is', 'are'
        }])
        
        keywords = list(set(keywords))[:10]  # Deduplicate and limit
        
        strategy = [
            {
                "method": "search_by_keywords",
                "params": {"keywords": keywords},
                "description": "Search knowledge graph by keywords"
            }
        ]
        
        # If specific entities found, get detailed info
        for entity in entities[:3]:
            if entity.entity_type == 'gene':
                strategy.append({
                    "method": "query_gene_information",
                    "params": {"gene": entity.normalized_form},
                    "description": f"Get information for {entity.text}"
                })
        
        return {
            "query_steps": strategy,
            "synthesis_focus": "general biomedical information",
            "expected_results": ["search_results", "entity_information"],
            "multi_step": len(strategy) > 1
        }
    
    def _go_term_query_strategy(self, entities: List[EntityMention], text: str) -> Dict[str, Any]:
        """Strategy for GO term and functional analysis queries."""
        go_terms = [e for e in entities if e.entity_type == 'go_term']
        pathways = [e for e in entities if e.entity_type == 'pathway']
        genes = [e for e in entities if e.entity_type == 'gene']
        
        strategy = []
        
        # Query GO terms using search
        for go_term in go_terms:
            if "molecular function" in go_term.text.lower():
                strategy.append({
                    "method": "search_by_keywords",
                    "params": {"keywords": ["molecular_function", "enzyme", "binding", "catalytic"]},
                    "description": f"Search for molecular function concepts"
                })
            else:
                strategy.append({
                    "method": "search_by_keywords",
                    "params": {"keywords": [go_term.normalized_form]},
                    "description": f"Search for GO term {go_term.text}"
                })
        
        # Query pathways/gene categories using more specific search terms
        for pathway in pathways:
            if "disease" in pathway.text.lower():
                # For disease-related concepts, search for specific disease terms
                strategy.append({
                    "method": "search_by_keywords", 
                    "params": {"keywords": ["cancer", "disease", "tumor", "pathology"]},
                    "description": f"Search for disease-related concepts"
                })
            else:
                strategy.append({
                    "method": "search_by_keywords", 
                    "params": {"keywords": [pathway.normalized_form]},
                    "description": f"Search for {pathway.text}"
                })
            
        # Query specific genes if mentioned
        for gene in genes:
            strategy.append({
                "method": "query_gene_information",
                "params": {"gene": gene.normalized_form},
                "description": f"Query gene {gene.text}"
            })
        
        # If no specific entities, do general functional search
        if not (go_terms or pathways or genes):
            strategy.append({
                "method": "search_by_keywords",
                "params": {"keywords": ["molecular_function", "disease", "gene"]},
                "description": "General functional analysis search"
            })
        
        return {
            "query_steps": strategy,
            "synthesis_focus": "gene ontology and functional annotations",
            "expected_results": ["go_terms", "annotations", "gene_functions", "pathways"],
            "multi_step": len(strategy) > 1
        }
    
    def _viral_response_strategy(self, entities: List[EntityMention], text: str) -> Dict[str, Any]:
        """Strategy for viral response queries."""
        strategy = [
            {
                "method": "search_by_keywords",
                "params": {"keywords": ["viral", "immune", "response", "infection"]},
                "description": "Search for viral response mechanisms"
            }
        ]
        
        return {
            "query_steps": strategy,
            "synthesis_focus": "viral response and immune mechanisms",
            "expected_results": ["viral_associations", "immune_response", "pathways"],
            "multi_step": False
        }
    
    def _expression_analysis_strategy(self, entities: List[EntityMention], text: str) -> Dict[str, Any]:
        """Strategy for expression analysis queries."""
        genes = [e for e in entities if e.entity_type == 'gene']
        text_lower = text.lower()
        
        strategy = []
        
        # Check if this is specifically a viral expression question
        if ('viral' in text_lower or 'virus' in text_lower) and ('expression' in text_lower or 'levels' in text_lower):
            if 'highest' in text_lower or 'top' in text_lower or 'most' in text_lower:
                # This is asking for top genes with viral expression
                strategy.append({
                    "method": "query_viral_expression",
                    "params": {"limit": 20},
                    "description": "Get top genes with highest viral expression levels"
                })
                
                return {
                    "query_steps": strategy,
                    "synthesis_focus": "viral expression levels and gene rankings",
                    "expected_results": ["viral_expression", "gene_rankings", "expression_levels"],
                    "multi_step": False
                }
        
        # Regular expression analysis for specific genes
        for gene in genes[:3]:
            strategy.append({
                "method": "query_gene_information",
                "params": {"gene": gene.normalized_form},
                "description": f"Get expression data for {gene.text}"
            })
        
        # If no specific genes, add context-specific expression searches
        if not genes:
            if 'viral' in text_lower:
                strategy.append({
                    "method": "search_by_keywords",
                    "params": {"keywords": ["viral", "expression", "infection", "response"]},
                    "description": "Search for viral expression patterns"
                })
            elif any(term in text_lower for term in ['drug', 'medication', 'treatment', 'estradiol', 'insulin', 'aspirin', 'metformin']):
                # For drug-expression questions, use both drug queries and expression search
                strategy.append({
                    "method": "search_by_keywords",
                    "params": {"keywords": ["drug", "expression", "regulation", "transcription", "medication"]},
                    "description": "Search for drug-expression interactions"
                })
                # Add specific drug interaction queries for common drugs
                common_drugs = ['estradiol', 'insulin', 'aspirin', 'metformin']
                for drug in common_drugs:
                    if drug in text_lower:
                        strategy.append({
                            "method": "query_drug_interactions",
                            "params": {"drug": drug},
                            "description": f"Get expression targets for {drug}"
                        })
                        break
            else:
                strategy.append({
                    "method": "search_by_keywords",
                    "params": {"keywords": ["expression", "levels", "regulation"]},
                    "description": "Search for expression patterns and related genes"
                })
        
        return {
            "query_steps": strategy,
            "synthesis_focus": "gene expression patterns and regulation",
            "expected_results": ["expression_data", "regulation", "conditions"],
            "multi_step": len(strategy) > 1
        }

class QueryPlanner:
    """
    Main query planning agent that orchestrates entity extraction,
    intent classification, and strategy generation.
    """
    
    def __init__(self):
        self.entity_extractor = BiomedicalEntityExtractor()
        self.intent_classifier = IntentClassifier()
        self.query_strategist = QueryStrategist()
        self.planning_cache = {}
        # Import here to avoid circular imports
        from .ollama_client import OllamaClient
        self.ollama_client = OllamaClient(model="llama3.2:1b")
    
    def plan_query(self, question: str) -> QueryPlan:
        """
        Plan a query strategy for the given question.
        
        Args:
            question: Natural language question
            
        Returns:
            QueryPlan with structured execution strategy
        """
        logger.info(f"Planning query for: {question[:100]}...")
        
        # Check cache
        cache_key = hash(question)
        if cache_key in self.planning_cache:
            logger.info("Using cached query plan")
            return self.planning_cache[cache_key]
        
        # Use hybrid approach: regex first for reliability, LLM for complex cases
        regex_entities = self.entity_extractor.extract_entities(question)
        
        # Add keyword-based extraction for common concepts
        keyword_entities = self._extract_entities_with_keywords(question)
        
        # Combine and deduplicate
        all_entities = regex_entities + keyword_entities
        entities = self._deduplicate_entity_mentions(all_entities)
        
        # LLM fallback only if both approaches fail
        if len(entities) == 0:
            logger.info("Regex and keyword extraction returned 0 entities, trying LLM fallback")
            entities = self._extract_entities_with_llm(question)
        
        logger.info(f"Extracted {len(entities)} entities: {[e.text for e in entities]}")
        
        # Classify intent
        intent, intent_confidence = self.intent_classifier.classify_intent(question, entities)
        logger.info(f"Classified intent: {intent.value} (confidence: {intent_confidence:.2f})")
        
        # Generate strategy
        strategy_info = self.query_strategist.generate_strategy(intent, entities, question)
        
        # Create query plan
        query_plan = QueryPlan(
            intent=intent,
            entities=entities,
            query_strategy=strategy_info["query_steps"],
            synthesis_instructions=strategy_info["synthesis_focus"],
            confidence=intent_confidence,
            expected_result_types=strategy_info["expected_results"],
            multi_step=strategy_info["multi_step"]
        )
        
        # Cache the plan
        self.planning_cache[cache_key] = query_plan
        
        logger.info(f"Generated query plan with {len(query_plan.query_strategy)} steps")
        return query_plan
    
    def refine_plan(self, plan: QueryPlan, feedback: str) -> QueryPlan:
        """
        Refine a query plan based on feedback or results.
        
        Args:
            plan: Original query plan
            feedback: Feedback or additional context
            
        Returns:
            Refined query plan
        """
        # Simple refinement - could be made more sophisticated
        if "not found" in feedback.lower() or "no results" in feedback.lower():
            # Add broader search strategy
            broader_search = {
                "method": "search_by_keywords",
                "params": {"keywords": [e.text.lower() for e in plan.entities[:5]]},
                "description": "Broader keyword search"
            }
            plan.query_strategy.append(broader_search)
        
        return plan
    
    def get_planning_stats(self) -> Dict[str, Any]:
        """Get statistics about query planning."""
        return {
            "cache_size": len(self.planning_cache),
            "supported_intents": [intent.value for intent in QueryIntent],
            "entity_types": list(self.entity_extractor.patterns.keys())
        }
    
    def _extract_entities_with_llm(self, question: str) -> List[EntityMention]:
        """
        Extract entities using LLM-based natural language understanding.
        This provides more flexible entity extraction than rigid regex patterns.
        """
        prompt = f"""Extract ONLY the biomedical entities that appear in this specific question. Do NOT add examples or entities not mentioned.

QUESTION: "{question}"

RULES:
1. Extract ONLY words/phrases from the question above
2. Do NOT add TP53, cancer, or other examples  
3. Do NOT make up entities

FORMAT: entity_text|entity_type

TYPES: gene, disease, drug, go_term, pathway

For questions about molecular functions and disease genes:
- "molecular function" or "molecular functions" → go_term
- "disease-related genes" → pathway
- "biological process" → go_term
- "cellular component" → go_term

Output only entities FROM THE QUESTION:"""

        try:
            response = self.ollama_client.generate_response(prompt)
            response_text = response.content.strip()
            
            # Parse pipe-separated format: entity_text|entity_type
            lines = response_text.strip().split('\n')
            entities = []
            
            for line in lines:
                line = line.strip()
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        entity_text = parts[0].strip()
                        entity_type = parts[1].strip()
                        
                        # Skip invalid entity types
                        valid_types = {'gene', 'disease', 'drug', 'go_term', 'pathway', 'phenotype', 'organism'}
                        if entity_type not in valid_types:
                            continue
                        
                        # Create normalized form
                        normalized_form = entity_text.lower().replace(' ', '_').replace('-', '_')
                        
                        entity = EntityMention(
                            text=entity_text,
                            entity_type=entity_type,
                            confidence=0.8,  # Fixed confidence since LLM can't provide objective confidence
                            start_pos=0,  # LLM doesn't provide position info
                            end_pos=len(entity_text),
                            normalized_form=normalized_form
                        )
                        entities.append(entity)
            
            logger.info(f"LLM extracted {len(entities)} entities successfully")
            return entities
            
        except (KeyError, ValueError) as e:
            logger.warning(f"LLM entity extraction parsing failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error in LLM entity extraction: {e}")
            return []
    
    def _extract_entities_with_keywords(self, question: str) -> List[EntityMention]:
        """Extract entities using reliable keyword matching."""
        entities = []
        text_lower = question.lower()
        
        # Define comprehensive keyword patterns for broad biomedical coverage
        keyword_patterns = {
            'go_term': [
                ('molecular function', 'molecular functions'),
                ('biological process', 'biological processes'), 
                ('cellular component', 'cellular components'),
                ('gene function', 'gene functions'),
                ('protein function', 'protein functions'),
                ('expression changes', 'gene expression'),
                ('protein expression', 'enzyme activity')
            ],
            'pathway': [
                ('disease-related genes', 'disease related genes'),
                ('cancer genes', 'cancer-related genes'),
                ('tumor suppressor genes', 'tumor suppressors'),
                ('oncogenes', 'proto-oncogenes'),
                ('immune response', 'immune system'),
                ('inflammatory response', 'inflammation'),
                ('stress response', 'cellular stress')
            ],
            'disease': [
                ('cancer', 'tumor'),
                ('diabetes', 'diabetic'),
                ('alzheimer', "alzheimer's"),
                ('parkinson', "parkinson's"),
                ('viral infections', 'viral infection'),
                ('bacterial infections', 'bacterial infection'),
                ('autoimmune diseases', 'autoimmune')
            ],
            'phenotype': [
                ('expression levels', 'expression level'),
                ('gene regulation', 'transcriptional regulation'),
                ('cell proliferation', 'cell growth'),
                ('cell death', 'apoptosis')
            ]
        }
        
        for entity_type, patterns in keyword_patterns.items():
            for pattern_variants in patterns:
                for pattern in pattern_variants:
                    if pattern in text_lower:
                        # Find the actual position in the original text
                        start_pos = text_lower.find(pattern)
                        if start_pos != -1:
                            # Extract the actual case from original text
                            entity_text = question[start_pos:start_pos + len(pattern)]
                            normalized_form = pattern.replace(' ', '_').replace('-', '_')
                            
                            entity = EntityMention(
                                text=entity_text,
                                entity_type=entity_type,
                                confidence=0.9,  # High confidence for keyword matches
                                start_pos=start_pos,
                                end_pos=start_pos + len(pattern),
                                normalized_form=normalized_form
                            )
                            entities.append(entity)
                            break  # Only match first variant of each pattern
        
        return entities
    
    def _deduplicate_entity_mentions(self, entities: List[EntityMention]) -> List[EntityMention]:
        """Remove duplicate entity mentions."""
        seen = set()
        deduplicated = []
        
        for entity in entities:
            # Create unique key based on text and type
            key = (entity.text.lower(), entity.entity_type)
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)
        
        return deduplicated