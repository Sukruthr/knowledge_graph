#!/usr/bin/env python3
"""
Comprehensive Analysis of Remaining Data Files

This script analyzes all files in the remaining_data_files folder to:
1. Understand their structure and content
2. Assess their value for knowledge graph integration
3. Check for potential duplications with existing data
4. Provide recommendations for integration
"""

import os
import pandas as pd
import json
import pickle
import csv
from pathlib import Path
import logging
from typing import Dict, List, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RemainingDataFilesAnalyzer:
    """Comprehensive analyzer for remaining data files"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.file_analysis = {}
        self.integration_assessment = {}
        
    def analyze_all_files(self) -> Dict[str, Any]:
        """Analyze all files in the remaining_data_files directory"""
        
        files_to_analyze = [
            "GO_BP_20231115.gmt",
            "L1000_sep_count_DF.txt", 
            "MarkedParagraphs.pickle",
            "NeST_table_All.csv",
            "SupplementTable3_0715.tsv",
            "all_go_terms_embeddings_dict.pkl",
            "go_terms.csv",
            "num_citations_per_paragraph.json",
            "reference_evaluation.tsv",
            "supporting_gene_log.json"
        ]
        
        for filename in files_to_analyze:
            file_path = self.data_dir / filename
            if file_path.exists():
                logger.info(f"Analyzing {filename}...")
                self.file_analysis[filename] = self._analyze_single_file(file_path)
            else:
                logger.warning(f"File not found: {filename}")
        
        return self.file_analysis
    
    def _analyze_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single file and return its characteristics"""
        
        analysis = {
            'filename': file_path.name,
            'size_mb': file_path.stat().st_size / (1024 * 1024),
            'file_type': file_path.suffix,
            'content_type': None,
            'structure': {},
            'sample_data': {},
            'data_quality': {},
            'integration_potential': 'unknown'
        }
        
        try:
            if file_path.suffix == '.gmt':
                analysis.update(self._analyze_gmt_file(file_path))
            elif file_path.suffix == '.txt':
                analysis.update(self._analyze_txt_file(file_path))
            elif file_path.suffix == '.csv':
                analysis.update(self._analyze_csv_file(file_path))
            elif file_path.suffix == '.tsv':
                analysis.update(self._analyze_tsv_file(file_path))
            elif file_path.suffix == '.json':
                analysis.update(self._analyze_json_file(file_path))
            elif file_path.suffix == '.pkl' or file_path.suffix == '.pickle':
                analysis.update(self._analyze_pickle_file(file_path))
                
        except Exception as e:
            logger.error(f"Error analyzing {file_path.name}: {str(e)}")
            analysis['error'] = str(e)
            
        return analysis
    
    def _analyze_gmt_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze GMT (Gene Matrix Transposed) file format"""
        
        total_lines = 0
        total_genes = set()
        go_terms = set()
        sample_entries = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < 5:  # Store first 5 entries as samples
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        go_term = parts[0]
                        description = parts[1] if parts[1] else "No description"
                        genes = parts[2:] if len(parts) > 2 else []
                        
                        sample_entries.append({
                            'go_term': go_term,
                            'description': description,
                            'gene_count': len(genes),
                            'genes_sample': genes[:10] if genes else []  # First 10 genes
                        })
                        
                        go_terms.add(go_term)
                        total_genes.update(genes)
                
                total_lines += 1
                if total_lines > 20000:  # Limit processing for very large files
                    break
        
        return {
            'content_type': 'GO Gene Sets (GMT format)',
            'structure': {
                'total_gene_sets': total_lines,
                'unique_go_terms': len(go_terms),
                'unique_genes': len(total_genes),
                'format': 'GO_TERM<tab>DESCRIPTION<tab>GENE1<tab>GENE2...'
            },
            'sample_data': sample_entries[:3],
            'data_quality': {
                'complete_entries': len([e for e in sample_entries if e['gene_count'] > 0]),
                'avg_genes_per_set': sum(e['gene_count'] for e in sample_entries) / len(sample_entries) if sample_entries else 0
            },
            'integration_potential': 'high'  # GO gene sets are highly valuable
        }
    
    def _analyze_csv_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze CSV files"""
        
        try:
            # Read a sample of the file
            df_sample = pd.read_csv(file_path, nrows=100)
            df_info = pd.read_csv(file_path, nrows=0)  # Just to get column info
            
            # Get basic statistics
            total_rows = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
            
            return {
                'content_type': 'Tabular Data (CSV)',
                'structure': {
                    'total_rows': total_rows,
                    'total_columns': len(df_info.columns),
                    'columns': list(df_info.columns),
                    'column_types': df_sample.dtypes.to_dict()
                },
                'sample_data': df_sample.head(3).to_dict('records') if not df_sample.empty else [],
                'data_quality': {
                    'null_counts': df_sample.isnull().sum().to_dict(),
                    'unique_counts': {col: df_sample[col].nunique() for col in df_sample.columns}
                },
                'integration_potential': 'medium'
            }
            
        except Exception as e:
            return {'error': f"Failed to read CSV: {str(e)}", 'integration_potential': 'low'}
    
    def _analyze_tsv_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze TSV files"""
        
        try:
            # Read a sample of the file
            df_sample = pd.read_csv(file_path, sep='\t', nrows=100)
            df_info = pd.read_csv(file_path, sep='\t', nrows=0)
            
            total_rows = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
            
            return {
                'content_type': 'Tabular Data (TSV)',
                'structure': {
                    'total_rows': total_rows,
                    'total_columns': len(df_info.columns),
                    'columns': list(df_info.columns),
                    'column_types': df_sample.dtypes.to_dict()
                },
                'sample_data': df_sample.head(3).to_dict('records') if not df_sample.empty else [],
                'data_quality': {
                    'null_counts': df_sample.isnull().sum().to_dict(),
                    'unique_counts': {col: df_sample[col].nunique() for col in df_sample.columns}
                },
                'integration_potential': 'medium'
            }
            
        except Exception as e:
            return {'error': f"Failed to read TSV: {str(e)}", 'integration_potential': 'low'}
    
    def _analyze_txt_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze TXT files"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [f.readline().strip() for _ in range(20)]  # Read first 20 lines
                
            # Try to parse as tab-separated
            parsed_lines = []
            for line in lines:
                if line:
                    parts = line.split('\t')
                    parsed_lines.append(parts)
            
            total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
            
            # Detect if it's structured data
            if len(parsed_lines) > 0 and all(len(parts) == len(parsed_lines[0]) for parts in parsed_lines[:5]):
                # Looks like structured tab-separated data
                columns = parsed_lines[0] if len(parsed_lines) > 0 else []
                sample_data = parsed_lines[1:4] if len(parsed_lines) > 1 else []
                
                return {
                    'content_type': 'Structured Text Data (Tab-separated)',
                    'structure': {
                        'total_rows': total_lines,
                        'total_columns': len(columns),
                        'columns': columns,
                        'format': 'tab-separated'
                    },
                    'sample_data': sample_data,
                    'data_quality': {
                        'consistent_structure': True,
                        'sample_size': len(sample_data)
                    },
                    'integration_potential': 'medium'
                }
            else:
                return {
                    'content_type': 'Unstructured Text Data',
                    'structure': {
                        'total_lines': total_lines,
                        'format': 'plain text'
                    },
                    'sample_data': lines[:5],
                    'integration_potential': 'low'
                }
                
        except Exception as e:
            return {'error': f"Failed to read TXT: {str(e)}", 'integration_potential': 'low'}
    
    def _analyze_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze JSON files"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                return {
                    'content_type': 'JSON Object',
                    'structure': {
                        'keys': list(data.keys())[:20],  # First 20 keys
                        'total_keys': len(data.keys()),
                        'data_type': 'dictionary'
                    },
                    'sample_data': {k: v for k, v in list(data.items())[:3]},
                    'data_quality': {
                        'has_nested_structure': any(isinstance(v, (dict, list)) for v in data.values()),
                        'value_types': list(set(type(v).__name__ for v in data.values()))
                    },
                    'integration_potential': 'medium'
                }
            elif isinstance(data, list):
                return {
                    'content_type': 'JSON Array',
                    'structure': {
                        'total_items': len(data),
                        'data_type': 'array'
                    },
                    'sample_data': data[:3] if data else [],
                    'data_quality': {
                        'item_types': list(set(type(item).__name__ for item in data[:100]))
                    },
                    'integration_potential': 'medium'
                }
            else:
                return {
                    'content_type': 'JSON Primitive',
                    'structure': {'data_type': type(data).__name__},
                    'sample_data': data,
                    'integration_potential': 'low'
                }
                
        except Exception as e:
            return {'error': f"Failed to read JSON: {str(e)}", 'integration_potential': 'low'}
    
    def _analyze_pickle_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze pickle files (with caution)"""
        
        try:
            # Note: Loading pickle files can be dangerous, but these are from a trusted source
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            data_type = type(data).__name__
            
            if isinstance(data, dict):
                return {
                    'content_type': 'Pickled Dictionary',
                    'structure': {
                        'keys': list(data.keys())[:20] if hasattr(data, 'keys') else [],
                        'total_keys': len(data) if hasattr(data, '__len__') else 'unknown',
                        'data_type': data_type
                    },
                    'sample_data': {k: str(v)[:100] for k, v in list(data.items())[:3]} if hasattr(data, 'items') else {},
                    'data_quality': {
                        'serialized_size_mb': file_path.stat().st_size / (1024 * 1024),
                        'python_type': data_type
                    },
                    'integration_potential': 'medium'
                }
            elif isinstance(data, list):
                return {
                    'content_type': 'Pickled List',
                    'structure': {
                        'total_items': len(data),
                        'data_type': data_type
                    },
                    'sample_data': [str(item)[:100] for item in data[:3]],
                    'data_quality': {
                        'serialized_size_mb': file_path.stat().st_size / (1024 * 1024),
                        'python_type': data_type
                    },
                    'integration_potential': 'medium'
                }
            else:
                return {
                    'content_type': f'Pickled {data_type}',
                    'structure': {'data_type': data_type},
                    'sample_data': str(data)[:200],
                    'data_quality': {
                        'serialized_size_mb': file_path.stat().st_size / (1024 * 1024),
                        'python_type': data_type
                    },
                    'integration_potential': 'low'
                }
                
        except Exception as e:
            return {'error': f"Failed to read pickle: {str(e)}", 'integration_potential': 'low'}
    
    def assess_integration_value(self) -> Dict[str, Any]:
        """Assess the value of each file for knowledge graph integration"""
        
        assessment = {
            'high_value_files': [],
            'medium_value_files': [],
            'low_value_files': [],
            'duplicate_risk_files': [],
            'recommended_integrations': [],
            'overall_score': 0
        }
        
        # Scoring criteria
        for filename, analysis in self.file_analysis.items():
            score = 0
            reasons = []
            duplicate_risk = False
            
            # Content type scoring
            content_type = analysis.get('content_type', '')
            
            if 'GO' in content_type or 'Gene Set' in content_type:
                score += 25
                reasons.append("Contains GO/gene set data")
                
                # Check for potential duplication with existing GO data
                if 'GO_BP' in filename:
                    duplicate_risk = True
                    reasons.append("May duplicate existing GO_BP data")
            
            if 'LLM' in content_type or 'evaluation' in filename.lower():
                score += 20
                reasons.append("Contains LLM evaluation data")
            
            if 'expression' in filename.lower() or 'L1000' in filename:
                score += 20
                reasons.append("Contains expression/perturbation data")
            
            if 'reference' in filename.lower() or 'citation' in filename.lower():
                score += 15
                reasons.append("Contains reference/citation data")
            
            if 'embedding' in filename.lower():
                score += 15
                reasons.append("Contains embedding/vector data")
            
            # Data quality scoring
            if analysis.get('integration_potential') == 'high':
                score += 15
            elif analysis.get('integration_potential') == 'medium':
                score += 10
            elif analysis.get('integration_potential') == 'low':
                score += 0
            
            # Structure scoring
            structure = analysis.get('structure', {})
            if structure.get('total_rows', 0) > 1000 or structure.get('total_keys', 0) > 1000:
                score += 10
                reasons.append("Substantial data volume")
            
            # File size scoring
            if analysis.get('size_mb', 0) > 1:
                score += 5
                reasons.append("Significant data size")
            
            # Data completeness scoring
            data_quality = analysis.get('data_quality', {})
            if data_quality and not analysis.get('error'):
                score += 5
                reasons.append("Good data quality")
            
            # Categorize files
            file_info = {
                'filename': filename,
                'score': score,
                'reasons': reasons,
                'duplicate_risk': duplicate_risk,
                'analysis': analysis
            }
            
            if score >= 50:
                assessment['high_value_files'].append(file_info)
            elif score >= 25:
                assessment['medium_value_files'].append(file_info)
            else:
                assessment['low_value_files'].append(file_info)
            
            if duplicate_risk:
                assessment['duplicate_risk_files'].append(file_info)
        
        # Calculate overall score
        total_files = len(self.file_analysis)
        total_score = sum(f['score'] for f in assessment['high_value_files'] + 
                         assessment['medium_value_files'] + assessment['low_value_files'])
        assessment['overall_score'] = total_score / total_files if total_files > 0 else 0
        
        # Generate recommendations
        assessment['recommended_integrations'] = self._generate_integration_recommendations(assessment)
        
        return assessment
    
    def _generate_integration_recommendations(self, assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific integration recommendations"""
        
        recommendations = []
        
        # High-value files should definitely be integrated
        for file_info in assessment['high_value_files']:
            filename = file_info['filename']
            
            if filename == 'GO_BP_20231115.gmt':
                if not file_info['duplicate_risk']:
                    recommendations.append({
                        'filename': filename,
                        'action': 'integrate',
                        'priority': 'high',
                        'integration_type': 'go_gene_sets',
                        'description': 'Integrate GO Biological Process gene sets - provides comprehensive GO-gene associations',
                        'expected_nodes': 'GO terms, genes',
                        'expected_edges': 'gene-GO associations'
                    })
                else:
                    recommendations.append({
                        'filename': filename,
                        'action': 'compare_then_integrate',
                        'priority': 'medium',
                        'integration_type': 'go_gene_sets_comparison',
                        'description': 'Compare with existing GO_BP data, integrate if newer/more comprehensive',
                        'expected_nodes': 'Additional GO terms, genes',
                        'expected_edges': 'Additional gene-GO associations'
                    })
        
        # Medium-value files need evaluation
        for file_info in assessment['medium_value_files']:
            filename = file_info['filename']
            
            if 'L1000' in filename:
                recommendations.append({
                    'filename': filename,
                    'action': 'integrate',
                    'priority': 'medium',
                    'integration_type': 'expression_data',
                    'description': 'Integrate L1000 expression data - provides perturbation-gene expression relationships',
                    'expected_nodes': 'compounds, cell lines, experimental conditions',
                    'expected_edges': 'perturbation-expression relationships'
                })
            
            if 'reference_evaluation' in filename:
                recommendations.append({
                    'filename': filename,
                    'action': 'integrate',
                    'priority': 'medium',
                    'integration_type': 'literature_evaluation',
                    'description': 'Integrate literature evaluation data - provides citation and reference context',
                    'expected_nodes': 'papers, gene sets, evaluations',
                    'expected_edges': 'citation-geneset relationships'
                })
            
            if 'NeST' in filename:
                recommendations.append({
                    'filename': filename,
                    'action': 'integrate',
                    'priority': 'medium',
                    'integration_type': 'pathway_networks',
                    'description': 'Integrate NeST pathway networks - provides curated pathway information',
                    'expected_nodes': 'pathways, genes, network modules',
                    'expected_edges': 'gene-pathway, pathway-pathway relationships'
                })
        
        return recommendations
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report"""
        
        assessment = self.assess_integration_value()
        
        report = []
        report.append("# REMAINING DATA FILES ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        report.append("## SUMMARY STATISTICS")
        report.append(f"Total files analyzed: {len(self.file_analysis)}")
        report.append(f"High-value files: {len(assessment['high_value_files'])}")
        report.append(f"Medium-value files: {len(assessment['medium_value_files'])}")
        report.append(f"Low-value files: {len(assessment['low_value_files'])}")
        report.append(f"Files with duplication risk: {len(assessment['duplicate_risk_files'])}")
        report.append(f"Overall integration score: {assessment['overall_score']:.2f}/100")
        report.append("")
        
        # High-value files
        if assessment['high_value_files']:
            report.append("## HIGH-VALUE FILES (Score ≥ 50)")
            for file_info in assessment['high_value_files']:
                report.append(f"### {file_info['filename']} (Score: {file_info['score']})")
                report.append(f"Reasons: {', '.join(file_info['reasons'])}")
                if file_info['duplicate_risk']:
                    report.append("⚠️  DUPLICATE RISK: May overlap with existing data")
                
                analysis = file_info['analysis']
                report.append(f"Content: {analysis.get('content_type', 'Unknown')}")
                report.append(f"Size: {analysis.get('size_mb', 0):.2f} MB")
                
                structure = analysis.get('structure', {})
                if structure:
                    report.append(f"Structure: {structure}")
                report.append("")
        
        # Medium-value files
        if assessment['medium_value_files']:
            report.append("## MEDIUM-VALUE FILES (Score 25-49)")
            for file_info in assessment['medium_value_files']:
                report.append(f"### {file_info['filename']} (Score: {file_info['score']})")
                report.append(f"Reasons: {', '.join(file_info['reasons'])}")
                
                analysis = file_info['analysis']
                report.append(f"Content: {analysis.get('content_type', 'Unknown')}")
                report.append(f"Size: {analysis.get('size_mb', 0):.2f} MB")
                report.append("")
        
        # Integration recommendations
        if assessment['recommended_integrations']:
            report.append("## INTEGRATION RECOMMENDATIONS")
            for rec in assessment['recommended_integrations']:
                report.append(f"### {rec['filename']}")
                report.append(f"**Action**: {rec['action']}")
                report.append(f"**Priority**: {rec['priority']}")
                report.append(f"**Type**: {rec['integration_type']}")
                report.append(f"**Description**: {rec['description']}")
                report.append(f"**Expected nodes**: {rec['expected_nodes']}")
                report.append(f"**Expected edges**: {rec['expected_edges']}")
                report.append("")
        
        # Low-value files
        if assessment['low_value_files']:
            report.append("## LOW-VALUE FILES (Score < 25)")
            report.append("These files have limited integration value:")
            for file_info in assessment['low_value_files']:
                report.append(f"- {file_info['filename']} (Score: {file_info['score']})")
            report.append("")
        
        return "\n".join(report)

def main():
    """Main execution function"""
    
    # Configure paths
    data_dir = "llm_evaluation_for_gene_set_interpretation/data/remaining_data_files"
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Initialize analyzer
    analyzer = RemainingDataFilesAnalyzer(data_dir)
    
    # Run comprehensive analysis
    logger.info("Starting comprehensive analysis of remaining data files...")
    file_analysis = analyzer.analyze_all_files()
    
    # Generate report
    report = analyzer.generate_report()
    
    # Save results
    output_file = "remaining_data_files_analysis_report.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Analysis complete! Report saved to: {output_file}")
    print(report)
    
    # Save detailed analysis as JSON for further processing
    detailed_output = "remaining_data_files_detailed_analysis.json"
    with open(detailed_output, 'w', encoding='utf-8') as f:
        json.dump({
            'file_analysis': analyzer.file_analysis,
            'integration_assessment': analyzer.assess_integration_value()
        }, f, indent=2, default=str)
    
    logger.info(f"Detailed analysis saved to: {detailed_output}")

if __name__ == "__main__":
    main()