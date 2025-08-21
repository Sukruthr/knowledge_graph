#!/usr/bin/env python3
"""
Talisman Paper Gene Sets Analysis

Comprehensive analysis of gene set files from talisman-paper/genesets/human/ 
to understand data types, formats, and integration potential.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TalismanGeneSetsAnalyzer:
    """Analyze talisman paper gene set files for integration potential"""
    
    def __init__(self, data_dir: str = "talisman-paper/genesets/human"):
        self.data_dir = Path(data_dir)
        self.analysis_results = {
            'file_summary': {},
            'data_types': defaultdict(list),
            'format_summary': {},
            'content_analysis': {},
            'integration_assessment': {},
            'duplication_risk': {}
        }
        
    def analyze_all_files(self) -> Dict[str, Any]:
        """Perform comprehensive analysis of all gene set files"""
        logger.info("Starting comprehensive analysis of talisman gene sets...")
        
        # Get all files
        all_files = list(self.data_dir.glob("*.yaml")) + list(self.data_dir.glob("*.json"))
        logger.info(f"Found {len(all_files)} gene set files")
        
        # Analyze each file
        for file_path in all_files:
            try:
                self._analyze_single_file(file_path)
            except Exception as e:
                logger.warning(f"Failed to analyze {file_path.name}: {str(e)}")
        
        # Perform summary analysis
        self._generate_summary_analysis()
        self._assess_integration_potential()
        self._assess_duplication_risks()
        
        return self.analysis_results
    
    def _analyze_single_file(self, file_path: Path) -> None:
        """Analyze a single gene set file"""
        file_name = file_path.name
        file_ext = file_path.suffix
        
        # Load file content
        content = self._load_file_content(file_path)
        if content is None:
            return
        
        # Extract metadata
        file_analysis = {
            'name': file_name,
            'format': file_ext,
            'size_kb': file_path.stat().st_size / 1024,
            'content_structure': self._analyze_content_structure(content),
            'gene_info': self._extract_gene_information(content),
            'data_type': self._classify_data_type(file_name, content),
            'quality_metrics': self._assess_data_quality(content)
        }
        
        self.analysis_results['file_summary'][file_name] = file_analysis
        self.analysis_results['data_types'][file_analysis['data_type']].append(file_name)
        
    def _load_file_content(self, file_path: Path) -> Any:
        """Load YAML or JSON file content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.yaml':
                    return yaml.safe_load(f)
                elif file_path.suffix == '.json':
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {file_path.name}: {str(e)}")
            return None
    
    def _analyze_content_structure(self, content: Any) -> Dict[str, Any]:
        """Analyze the structure of file content"""
        if isinstance(content, dict):
            # Check for different structures
            structure = {
                'type': 'dict',
                'keys': list(content.keys()),
                'has_gene_symbols': 'gene_symbols' in content,
                'has_gene_ids': 'gene_ids' in content,
                'has_name': 'name' in content,
                'has_description': 'description' in content,
                'nested_structure': any(isinstance(v, dict) for v in content.values())
            }
            
            # Special case for MSigDB-style JSON
            if len(content) == 1 and all(isinstance(v, dict) for v in content.values()):
                structure['msigdb_format'] = True
                inner_dict = list(content.values())[0]
                structure['inner_keys'] = list(inner_dict.keys())
                structure['has_gene_symbols'] = 'geneSymbols' in inner_dict
            
            return structure
        else:
            return {'type': 'other', 'content_type': type(content).__name__}
    
    def _extract_gene_information(self, content: Any) -> Dict[str, Any]:
        """Extract gene-related information from content"""
        gene_info = {
            'gene_count': 0,
            'gene_symbols': [],
            'gene_ids': [],
            'id_types': set()
        }
        
        if isinstance(content, dict):
            # Direct gene_symbols
            if 'gene_symbols' in content and isinstance(content['gene_symbols'], list):
                gene_info['gene_symbols'] = content['gene_symbols']
                gene_info['gene_count'] = len(content['gene_symbols'])
            
            # Direct gene_ids  
            if 'gene_ids' in content and isinstance(content['gene_ids'], list):
                gene_info['gene_ids'] = content['gene_ids']
                if gene_info['gene_count'] == 0:
                    gene_info['gene_count'] = len(content['gene_ids'])
                
                # Extract ID types
                for gene_id in content['gene_ids'][:10]:  # Sample first 10
                    if ':' in str(gene_id):
                        id_type = str(gene_id).split(':')[0]
                        gene_info['id_types'].add(id_type)
            
            # MSigDB format
            if len(content) == 1:
                inner_dict = list(content.values())[0]
                if isinstance(inner_dict, dict) and 'geneSymbols' in inner_dict:
                    gene_info['gene_symbols'] = inner_dict['geneSymbols']
                    gene_info['gene_count'] = len(inner_dict['geneSymbols'])
        
        gene_info['id_types'] = list(gene_info['id_types'])
        return gene_info
    
    def _classify_data_type(self, file_name: str, content: Any) -> str:
        """Classify the type of gene set based on filename and content"""
        file_lower = file_name.lower()
        
        # HALLMARK gene sets
        if file_lower.startswith('hallmark_'):
            return 'hallmark'
        
        # GO-related sets
        if file_lower.startswith('go-') or 'go:' in file_lower:
            return 'go_custom'
        
        # Bicluster sets
        if file_lower.startswith('bicluster_'):
            return 'bicluster'
        
        # Disease-specific
        disease_terms = ['eds', 'fa', 'progeria', 'ataxia']
        if any(term in file_lower for term in disease_terms):
            return 'disease'
        
        # Pathway-specific
        pathway_terms = ['dopamine', 'glycolysis', 'mtorc1', 'tf-', 'hydrolase']
        if any(term in file_lower for term in pathway_terms):
            return 'pathway'
        
        # Cellular component
        if any(term in file_lower for term in ['endocytosis', 'peroxisome', 'membrane']):
            return 'cellular_component'
        
        # Gene function
        function_terms = ['yamanaka', 'transcription', 'proliferation', 'meiosis']
        if any(term in file_lower for term in function_terms):
            return 'gene_function'
        
        # Other specific types
        if 'canonical-' in file_lower:
            return 'canonical_pathway'
        if 'ig-receptor' in file_lower:
            return 'immunology'
        
        return 'other'
    
    def _assess_data_quality(self, content: Any) -> Dict[str, Any]:
        """Assess data quality metrics"""
        quality = {
            'has_metadata': False,
            'has_description': False,
            'gene_count_reasonable': False,
            'consistent_format': True,
            'complete_data': True
        }
        
        if isinstance(content, dict):
            # Check for metadata
            quality['has_metadata'] = any(key in content for key in ['name', 'description', 'taxon'])
            quality['has_description'] = 'description' in content or (
                len(content) == 1 and 'exactSource' in list(content.values())[0]
            )
            
            # Check gene count
            gene_count = 0
            if 'gene_symbols' in content:
                gene_count = len(content.get('gene_symbols', []))
            elif len(content) == 1:
                inner_dict = list(content.values())[0]
                if 'geneSymbols' in inner_dict:
                    gene_count = len(inner_dict['geneSymbols'])
            
            quality['gene_count_reasonable'] = 5 <= gene_count <= 1000
            quality['gene_count'] = gene_count
            
            # Check completeness
            if 'gene_symbols' in content:
                quality['complete_data'] = all(symbol for symbol in content['gene_symbols'])
            
        return quality
    
    def _generate_summary_analysis(self) -> None:
        """Generate summary statistics across all files"""
        summary = {
            'total_files': len(self.analysis_results['file_summary']),
            'format_distribution': defaultdict(int),
            'data_type_distribution': {k: len(v) for k, v in self.analysis_results['data_types'].items()},
            'gene_count_stats': {
                'total_unique_files': 0,
                'min_genes': float('inf'),
                'max_genes': 0,
                'avg_genes': 0,
                'files_with_genes': 0
            },
            'quality_assessment': {
                'high_quality': 0,
                'medium_quality': 0,
                'low_quality': 0
            }
        }
        
        gene_counts = []
        
        for file_name, analysis in self.analysis_results['file_summary'].items():
            # Format distribution
            summary['format_distribution'][analysis['format']] += 1
            
            # Gene count statistics
            gene_count = analysis['gene_info']['gene_count']
            if gene_count > 0:
                gene_counts.append(gene_count)
                summary['gene_count_stats']['files_with_genes'] += 1
                summary['gene_count_stats']['min_genes'] = min(summary['gene_count_stats']['min_genes'], gene_count)
                summary['gene_count_stats']['max_genes'] = max(summary['gene_count_stats']['max_genes'], gene_count)
            
            # Quality assessment
            quality = analysis['quality_metrics']
            if (quality['has_metadata'] and quality['gene_count_reasonable'] and 
                quality['complete_data'] and quality['gene_count'] >= 20):
                summary['quality_assessment']['high_quality'] += 1
            elif quality['gene_count'] >= 10 and quality['complete_data']:
                summary['quality_assessment']['medium_quality'] += 1
            else:
                summary['quality_assessment']['low_quality'] += 1
        
        if gene_counts:
            summary['gene_count_stats']['avg_genes'] = sum(gene_counts) / len(gene_counts)
            summary['gene_count_stats']['total_unique_genes'] = len(set(gene_counts))
        
        self.analysis_results['content_analysis'] = summary
    
    def _assess_integration_potential(self) -> None:
        """Assess integration potential for each data type"""
        integration = {}
        
        for data_type, files in self.analysis_results['data_types'].items():
            file_count = len(files)
            
            # Calculate average quality for this data type
            total_genes = 0
            high_quality_files = 0
            
            for file_name in files:
                file_analysis = self.analysis_results['file_summary'][file_name]
                total_genes += file_analysis['gene_info']['gene_count']
                if file_analysis['quality_metrics']['gene_count'] >= 20:
                    high_quality_files += 1
            
            avg_genes = total_genes / file_count if file_count > 0 else 0
            
            # Integration value scoring
            if data_type == 'hallmark':
                value_score = 95  # Very high value - well-curated pathways
            elif data_type == 'go_custom':
                value_score = 85  # High value - GO-based custom sets
            elif data_type in ['pathway', 'disease']:
                value_score = 80  # High value - specific biological contexts
            elif data_type == 'bicluster':
                value_score = 75  # Good value - expression-based clustering
            elif data_type in ['gene_function', 'immunology']:
                value_score = 70  # Good value - functional categories
            else:
                value_score = 60  # Medium value
            
            # Adjust for quality and size
            if high_quality_files / file_count > 0.8:
                value_score += 10
            if avg_genes > 50:
                value_score += 5
            
            integration[data_type] = {
                'file_count': file_count,
                'avg_genes_per_set': avg_genes,
                'high_quality_ratio': high_quality_files / file_count,
                'integration_value_score': min(100, value_score),
                'recommended_for_integration': value_score >= 70
            }
        
        self.analysis_results['integration_assessment'] = integration
    
    def _assess_duplication_risks(self) -> None:
        """Assess potential duplication with existing KG data"""
        risks = {}
        
        # Known existing data in KG
        existing_data_types = {
            'go_terms': 'GO terms already integrated from GO_BP, GO_CC, GO_MF',
            'hallmark_pathways': 'Some pathway data may overlap with existing omics data',
            'gene_associations': 'Gene associations present in multiple existing datasets'
        }
        
        for data_type, files in self.analysis_results['data_types'].items():
            risk_level = 'LOW'
            risk_details = []
            
            if data_type == 'hallmark':
                risk_level = 'MEDIUM'
                risk_details.append('May overlap with existing pathway/disease associations')
                risk_details.append('However, HALLMARK sets are highly curated and specific')
            
            elif data_type == 'go_custom':
                risk_level = 'MEDIUM' 
                risk_details.append('Custom GO queries may overlap with existing GO terms')
                risk_details.append('But likely represents specific gene combinations not in standard GO')
            
            elif data_type in ['bicluster', 'pathway']:
                risk_level = 'LOW'
                risk_details.append('Expression-based and specific pathway data likely unique')
            
            elif data_type == 'disease':
                risk_level = 'MEDIUM'
                risk_details.append('May have some overlap with existing disease association data')
            
            else:
                risk_level = 'LOW'
                risk_details.append('Likely unique specialized gene sets')
            
            risks[data_type] = {
                'risk_level': risk_level,
                'risk_details': risk_details,
                'recommended_action': 'INTEGRATE' if risk_level in ['LOW', 'MEDIUM'] else 'REVIEW'
            }
        
        self.analysis_results['duplication_risk'] = risks
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report"""
        report = []
        report.append("# TALISMAN GENE SETS ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Summary
        summary = self.analysis_results['content_analysis']
        report.append(f"\n## SUMMARY")
        report.append(f"- **Total Files**: {summary['total_files']}")
        report.append(f"- **YAML Files**: {summary['format_distribution']['.yaml']}")
        report.append(f"- **JSON Files**: {summary['format_distribution']['.json']}")
        report.append(f"- **Files with Genes**: {summary['gene_count_stats']['files_with_genes']}")
        report.append(f"- **Average Genes per Set**: {summary['gene_count_stats']['avg_genes']:.1f}")
        
        # Data Types
        report.append(f"\n## DATA TYPE DISTRIBUTION")
        for data_type, count in summary['data_type_distribution'].items():
            report.append(f"- **{data_type}**: {count} files")
        
        # Quality Assessment
        quality = summary['quality_assessment']
        report.append(f"\n## QUALITY ASSESSMENT")
        report.append(f"- **High Quality**: {quality['high_quality']} files")
        report.append(f"- **Medium Quality**: {quality['medium_quality']} files")
        report.append(f"- **Low Quality**: {quality['low_quality']} files")
        
        # Integration Assessment
        report.append(f"\n## INTEGRATION ASSESSMENT")
        for data_type, assessment in self.analysis_results['integration_assessment'].items():
            status = "✅ RECOMMENDED" if assessment['recommended_for_integration'] else "⚠️ REVIEW"
            report.append(f"\n### {data_type.upper()} ({assessment['file_count']} files)")
            report.append(f"- **Status**: {status}")
            report.append(f"- **Integration Value**: {assessment['integration_value_score']}/100")
            report.append(f"- **Avg Genes per Set**: {assessment['avg_genes_per_set']:.1f}")
            report.append(f"- **High Quality Ratio**: {assessment['high_quality_ratio']:.2%}")
        
        # Duplication Risk
        report.append(f"\n## DUPLICATION RISK ASSESSMENT")
        for data_type, risk in self.analysis_results['duplication_risk'].items():
            report.append(f"\n### {data_type.upper()}")
            report.append(f"- **Risk Level**: {risk['risk_level']}")
            report.append(f"- **Action**: {risk['recommended_action']}")
            for detail in risk['risk_details']:
                report.append(f"  - {detail}")
        
        return "\n".join(report)

def main():
    """Main analysis function"""
    logger.info("🔬 STARTING TALISMAN GENE SETS ANALYSIS")
    logger.info("=" * 60)
    
    try:
        analyzer = TalismanGeneSetsAnalyzer()
        results = analyzer.analyze_all_files()
        
        # Generate and save report
        report = analyzer.generate_report()
        
        with open('talisman_gene_sets_analysis_report.txt', 'w') as f:
            f.write(report)
        
        print(report)
        
        logger.info("✅ ANALYSIS COMPLETE")
        logger.info("Report saved to: talisman_gene_sets_analysis_report.txt")
        
        return results
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()