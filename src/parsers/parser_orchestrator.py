"""
Parser Orchestrator for Comprehensive Biomedical Data Integration.

This module contains the main orchestrator class that coordinates all specialized
parsers to create a unified biomedical knowledge graph from multiple data sources.
"""

from pathlib import Path
from typing import Dict, Set
import logging

# Import core parsers
from .core_parsers import CombinedGOParser, OmicsDataParser

# Import specialized parsers with clean error handling
specialized_parsers = {}

try:
    from .model_compare_parser import ModelCompareParser
    specialized_parsers['ModelCompareParser'] = ModelCompareParser
except ImportError:
    specialized_parsers['ModelCompareParser'] = None

try:
    from .cc_mf_branch_parser import CCMFBranchParser
    specialized_parsers['CCMFBranchParser'] = CCMFBranchParser
except ImportError:
    specialized_parsers['CCMFBranchParser'] = None

try:
    from .llm_processed_parser import LLMProcessedParser
    specialized_parsers['LLMProcessedParser'] = LLMProcessedParser
except ImportError:
    specialized_parsers['LLMProcessedParser'] = None

try:
    from .go_analysis_data_parser import GOAnalysisDataParser
    specialized_parsers['GOAnalysisDataParser'] = GOAnalysisDataParser
except ImportError:
    specialized_parsers['GOAnalysisDataParser'] = None

try:
    from .remaining_data_parser import RemainingDataParser
    specialized_parsers['RemainingDataParser'] = RemainingDataParser
except ImportError:
    specialized_parsers['RemainingDataParser'] = None

try:
    from .talisman_gene_sets_parser import TalismanGeneSetsParser
    specialized_parsers['TalismanGeneSetsParser'] = TalismanGeneSetsParser
except ImportError:
    specialized_parsers['TalismanGeneSetsParser'] = None

# Configure logging
logger = logging.getLogger(__name__)


class CombinedBiomedicalParser:
    """Parser for comprehensive biomedical data (GO + Omics integration)."""
    
    def __init__(self, base_data_dir: str):
        """
        Initialize combined biomedical parser for GO and Omics data.
        
        Args:
            base_data_dir: Base directory containing GO_BP, GO_CC, GO_MF, Omics_data, Omics_data2, and GO_term_analysis subdirectories
        """
        self.base_data_dir = Path(base_data_dir)
        
        # Initialize GO parser
        self.go_parser = CombinedGOParser(str(base_data_dir))
        
        # Initialize Omics parser with both Omics_data and Omics_data2
        omics_dir = self.base_data_dir / "Omics_data"
        omics_data2_dir = self.base_data_dir / "Omics_data2"
        
        if omics_dir.exists():
            omics_data2_path = str(omics_data2_dir) if omics_data2_dir.exists() else None
            self.omics_parser = OmicsDataParser(str(omics_dir), omics_data2_path)
            logger.info("Initialized Omics data parser")
            if omics_data2_path:
                logger.info("Enhanced semantic data integration enabled")
        else:
            self.omics_parser = None
            logger.warning(f"Omics_data directory not found: {omics_dir}")
        
        # Initialize Model Comparison parser
        model_compare_dir = self.base_data_dir / "GO_term_analysis" / "model_compare"
        ModelCompareParser = specialized_parsers.get('ModelCompareParser')
        if model_compare_dir.exists() and ModelCompareParser is not None:
            self.model_compare_parser = ModelCompareParser(str(model_compare_dir))
            logger.info("Initialized Model Comparison parser")
        else:
            self.model_compare_parser = None
            if model_compare_dir.exists():
                logger.warning("Model comparison data found but parser not available")
            else:
                logger.info("No model comparison data directory found")
        
        # Initialize CC_MF_Branch parser
        cc_mf_branch_dir = self.base_data_dir / "GO_term_analysis" / "CC_MF_branch"
        CCMFBranchParser = specialized_parsers.get('CCMFBranchParser')
        if cc_mf_branch_dir.exists() and CCMFBranchParser is not None:
            self.cc_mf_branch_parser = CCMFBranchParser(str(self.base_data_dir))
            logger.info("Initialized CC_MF_Branch parser")
        else:
            self.cc_mf_branch_parser = None
            if cc_mf_branch_dir.exists():
                logger.warning("CC_MF_Branch data found but parser not available")
            else:
                logger.info("No CC_MF_Branch data directory found")
        
        # Initialize LLM_processed parser
        llm_processed_dir = self.base_data_dir / "GO_term_analysis" / "LLM_processed"
        LLMProcessedParser = specialized_parsers.get('LLMProcessedParser')
        if llm_processed_dir.exists() and LLMProcessedParser is not None:
            self.llm_processed_parser = LLMProcessedParser(str(llm_processed_dir))
            logger.info("Initialized LLM_processed parser")
        else:
            self.llm_processed_parser = None
            if llm_processed_dir.exists():
                logger.warning("LLM_processed data found but parser not available")
            else:
                logger.info("No LLM_processed data directory found")
        
        # Initialize GO Analysis Data parser
        go_analysis_data_dir = self.base_data_dir / "GO_term_analysis" / "data_files"
        GOAnalysisDataParser = specialized_parsers.get('GOAnalysisDataParser')
        if go_analysis_data_dir.exists() and GOAnalysisDataParser is not None:
            self.go_analysis_data_parser = GOAnalysisDataParser(str(go_analysis_data_dir))
            logger.info("Initialized GO Analysis Data parser")
        else:
            self.go_analysis_data_parser = None
            if go_analysis_data_dir.exists():
                logger.warning("GO Analysis Data found but parser not available")
            else:
                logger.info("No GO Analysis Data directory found")
        
        # Initialize Remaining Data parser
        remaining_data_dir = self.base_data_dir / "remaining_data_files"
        RemainingDataParser = specialized_parsers.get('RemainingDataParser')
        if remaining_data_dir.exists() and RemainingDataParser is not None:
            self.remaining_data_parser = RemainingDataParser(str(self.base_data_dir))
            logger.info("Initialized Remaining Data parser")
        else:
            self.remaining_data_parser = None
            if remaining_data_dir.exists():
                logger.warning("Remaining Data found but parser not available")
            else:
                logger.info("No Remaining Data directory found")
        
        # Initialize Talisman Gene Sets parser
        # Try multiple possible locations for talisman data
        possible_talisman_dirs = [
            self.base_data_dir / "talisman-paper" / "genesets" / "human",
            Path("talisman-paper") / "genesets" / "human"
        ]
        
        talisman_dir = None
        for candidate_dir in possible_talisman_dirs:
            if candidate_dir.exists():
                talisman_dir = candidate_dir
                break
        
        TalismanGeneSetsParser = specialized_parsers.get('TalismanGeneSetsParser')
        if talisman_dir and TalismanGeneSetsParser is not None:
            self.talisman_gene_sets_parser = TalismanGeneSetsParser(str(talisman_dir))
            self.talisman_parser = self.talisman_gene_sets_parser  # Alias for consistency
            logger.info(f"Initialized Talisman Gene Sets parser with directory: {talisman_dir}")
        else:
            self.talisman_gene_sets_parser = None
            self.talisman_parser = None
            if talisman_dir:
                logger.warning("Talisman Gene Sets found but parser not available")
            else:
                logger.info("No Talisman Gene Sets directory found in expected locations")
        
        self.parsed_data = {}
        
    def parse_all_biomedical_data(self) -> Dict[str, Dict]:
        """
        Parse all biomedical data (GO + Omics).
        
        Returns:
            Dictionary with parsed data from all sources
        """
        logger.info("Starting comprehensive biomedical data parsing...")
        
        # Parse GO data
        go_data = self.go_parser.parse_all_namespaces()
        self.parsed_data['go_data'] = go_data
        
        # Parse Omics data if available
        if self.omics_parser:
            omics_data = {
                'disease_associations': self.omics_parser.parse_disease_gene_associations(),
                'drug_associations': self.omics_parser.parse_drug_gene_associations(),
                'viral_associations': self.omics_parser.parse_viral_gene_associations(),
                'cluster_relationships': self.omics_parser.parse_cluster_relationships(),
                'disease_expression_matrix': self.omics_parser.parse_disease_expression_matrix(),
                'viral_expression_matrix': self.omics_parser.parse_viral_expression_matrix(),
                'unique_entities': self.omics_parser.get_unique_entities(),
                'validation': self.omics_parser.validate_omics_data(),
                'summary': self.omics_parser.get_omics_summary()
            }
            
            # Parse enhanced semantic data from Omics_data2 if available
            if self.omics_parser.omics_data2_dir:
                enhanced_data = self.omics_parser.parse_all_enhanced_data()
                omics_data['enhanced_data'] = enhanced_data
                logger.info("Enhanced semantic data integrated successfully")
            
            self.parsed_data['omics_data'] = omics_data
        
        # Parse Model Comparison data if available
        if self.model_compare_parser:
            model_compare_data = self.model_compare_parser.parse_all_model_data()
            self.parsed_data['model_compare_data'] = model_compare_data
            logger.info("Model comparison data integrated successfully")
        
        # Parse CC_MF_Branch data if available
        if self.cc_mf_branch_parser:
            cc_mf_branch_data = self.cc_mf_branch_parser.parse_all_cc_mf_data()
            self.parsed_data['cc_mf_branch_data'] = cc_mf_branch_data
            logger.info("CC_MF_Branch data integrated successfully")
        
        # Parse LLM_processed data if available
        if self.llm_processed_parser:
            llm_processed_data = self.llm_processed_parser.parse_all_llm_processed_data()
            self.parsed_data['llm_processed_data'] = llm_processed_data
            logger.info("LLM_processed data integrated successfully")
        
        # Parse GO Analysis Data if available
        if self.go_analysis_data_parser:
            go_analysis_data = self.go_analysis_data_parser.parse_all_go_analysis_data()
            self.parsed_data['go_analysis_data'] = go_analysis_data
            logger.info("GO Analysis Data integrated successfully")
        
        # Parse Remaining Data if available
        if self.remaining_data_parser:
            remaining_data = self.remaining_data_parser.parse_all_remaining_data()
            self.parsed_data['remaining_data'] = remaining_data
            logger.info("Remaining Data integrated successfully")
        
        # Parse Talisman Gene Sets if available
        if self.talisman_gene_sets_parser:
            talisman_data = self.talisman_gene_sets_parser.parse_all_gene_sets()
            self.parsed_data['talisman_gene_sets'] = talisman_data
            logger.info("Talisman Gene Sets integrated successfully")
        
        logger.info("Comprehensive biomedical data parsing complete")
        return self.parsed_data
    
    def get_comprehensive_summary(self) -> Dict:
        """
        Get comprehensive summary across all parsed data sources.
        
        Returns:
            Combined summary dictionary
        """
        summary = {
            'data_sources': [],
            'go_summary': {},
            'omics_summary': {},
            'integration_stats': {}
        }
        
        # GO summary
        if 'go_data' in self.parsed_data:
            summary['data_sources'].append('GO_ontology')
            summary['go_summary'] = self.go_parser.get_combined_summary()
        
        # Omics summary
        if 'omics_data' in self.parsed_data and self.omics_parser:
            summary['data_sources'].append('Omics_associations')
            summary['omics_summary'] = self.omics_parser.get_omics_summary()
        
        # Model comparison summary
        if 'model_compare_data' in self.parsed_data and self.model_compare_parser:
            summary['data_sources'].append('Model_comparison')
            summary['model_compare_summary'] = self.model_compare_parser.get_model_compare_summary()
        
        # CC_MF_Branch summary  
        if 'cc_mf_branch_data' in self.parsed_data and self.cc_mf_branch_parser:
            summary['data_sources'].append('CC_MF_Branch')
            summary['cc_mf_branch_summary'] = self.cc_mf_branch_parser.get_stats()
        
        # Integration statistics
        if 'go_data' in self.parsed_data and 'omics_data' in self.parsed_data:
            # Find gene overlaps between GO and Omics data
            go_genes = set()
            for namespace_data in self.parsed_data['go_data'].values():
                for assoc in namespace_data.get('gene_associations', []):
                    go_genes.add(assoc['gene_symbol'])
            
            omics_genes = set()
            for assoc in self.parsed_data['omics_data']['disease_associations']:
                omics_genes.add(assoc['gene_symbol'])
            for assoc in self.parsed_data['omics_data']['drug_associations']:
                omics_genes.add(assoc['gene_symbol'])
            for assoc in self.parsed_data['omics_data']['viral_associations']:
                omics_genes.add(assoc['gene_symbol'])
            
            overlap_genes = go_genes & omics_genes
            
            summary['integration_stats'] = {
                'go_genes': len(go_genes),
                'omics_genes': len(omics_genes),
                'overlapping_genes': len(overlap_genes),
                'integration_coverage': len(overlap_genes) / len(go_genes) if go_genes else 0,
                'can_integrate': len(overlap_genes) > 0
            }
        
        return summary
    
    def validate_comprehensive_data(self) -> Dict[str, bool]:
        """
        Validate all parsed biomedical data.
        
        Returns:
            Comprehensive validation results
        """
        validation = {
            'go_data_valid': False,
            'omics_data_valid': False,
            'integration_possible': False,
            'overall_valid': False
        }
        
        # Validate GO data
        if 'go_data' in self.parsed_data:
            go_valid = True
            for namespace_data in self.parsed_data['go_data'].values():
                if not namespace_data.get('go_terms') or not namespace_data.get('gene_associations'):
                    go_valid = False
                    break
            validation['go_data_valid'] = go_valid
        
        # Validate Omics data
        if 'omics_data' in self.parsed_data and self.omics_parser:
            omics_validation = self.parsed_data['omics_data']['validation']
            validation['omics_data_valid'] = omics_validation.get('overall_valid', False)
        
        # Check integration possibility
        summary = self.get_comprehensive_summary()
        integration_stats = summary.get('integration_stats', {})
        validation['integration_possible'] = integration_stats.get('can_integrate', False)
        
        # Overall validation
        validation['overall_valid'] = (validation['go_data_valid'] and 
                                     validation['omics_data_valid'] and 
                                     validation['integration_possible'])
        
        logger.info(f"Comprehensive biomedical data validation: {validation}")
        return validation
    
    def get_available_parsers(self) -> Dict[str, bool]:
        """
        Get status of all available parsers.
        
        Returns:
            Dictionary with parser availability status
        """
        return {
            'go_parser': self.go_parser is not None,
            'omics_parser': self.omics_parser is not None,
            'model_compare_parser': self.model_compare_parser is not None,
            'cc_mf_branch_parser': self.cc_mf_branch_parser is not None,
            'llm_processed_parser': self.llm_processed_parser is not None,
            'go_analysis_data_parser': self.go_analysis_data_parser is not None,
            'remaining_data_parser': self.remaining_data_parser is not None,
            'talisman_gene_sets_parser': self.talisman_gene_sets_parser is not None
        }