#!/usr/bin/env python3
"""
Comprehensive analysis of GO_term_analysis/data_files folder
to understand data structure and assess integration value
"""

import os
import csv
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Any, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataFilesAnalyzer:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.analysis_results = {}
        
    def analyze_file_structure(self, file_path: Path) -> Dict[str, Any]:
        """Analyze the structure and content of a single file"""
        result = {
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size,
            'file_type': file_path.suffix,
            'line_count': 0,
            'columns': [],
            'sample_rows': [],
            'unique_values': {},
            'data_quality': {},
            'errors': []
        }
        
        try:
            if file_path.suffix in ['.csv', '.tsv']:
                separator = '\t' if file_path.suffix == '.tsv' else ','
                
                # Get line count
                with open(file_path, 'r', encoding='utf-8') as f:
                    result['line_count'] = sum(1 for _ in f)
                
                # Read first few rows to understand structure
                try:
                    df = pd.read_csv(file_path, sep=separator, nrows=10)
                    result['columns'] = list(df.columns)
                    result['sample_rows'] = df.head(5).to_dict('records')
                    
                    # Analyze column data types and unique values for first 1000 rows
                    df_sample = pd.read_csv(file_path, sep=separator, nrows=1000)
                    for col in df_sample.columns:
                        result['unique_values'][col] = {
                            'count': df_sample[col].nunique(),
                            'sample_values': df_sample[col].dropna().head(10).tolist()
                        }
                        
                        # Data quality checks
                        result['data_quality'][col] = {
                            'null_count': df_sample[col].isnull().sum(),
                            'null_percentage': (df_sample[col].isnull().sum() / len(df_sample)) * 100,
                            'data_type': str(df_sample[col].dtype)
                        }
                        
                except Exception as e:
                    result['errors'].append(f"CSV/TSV parsing error: {str(e)}")
                    
            elif file_path.suffix == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    result['line_count'] = len(lines)
                    result['sample_rows'] = [line.strip() for line in lines[:10]]
                    
                    # Check if it's structured data
                    if lines and '\t' in lines[0]:
                        try:
                            first_line = lines[0].strip().split('\t')
                            result['columns'] = first_line
                            result['potential_tsv'] = True
                        except:
                            pass
                            
            elif file_path.suffix == '.xlsx':
                try:
                    df = pd.read_excel(file_path, nrows=10)
                    result['line_count'] = len(df)
                    result['columns'] = list(df.columns)
                    result['sample_rows'] = df.head(5).to_dict('records')
                except Exception as e:
                    result['errors'].append(f"Excel parsing error: {str(e)}")
                    
        except Exception as e:
            result['errors'].append(f"File reading error: {str(e)}")
            
        return result
    
    def assess_integration_value(self, file_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the integration value of a file for the knowledge graph"""
        assessment = {
            'integration_score': 0,
            'value_factors': {},
            'integration_potential': 'low',
            'recommended_action': 'skip',
            'reasoning': []
        }
        
        file_name = file_analysis['file_name'].lower()
        columns = file_analysis.get('columns', [])
        
        # Factor 1: Data completeness and quality (0-25 points)
        if file_analysis['line_count'] > 100:
            assessment['value_factors']['data_completeness'] = 20
            assessment['reasoning'].append(f"Good data volume: {file_analysis['line_count']} records")
        elif file_analysis['line_count'] > 10:
            assessment['value_factors']['data_completeness'] = 10
            assessment['reasoning'].append(f"Moderate data volume: {file_analysis['line_count']} records")
        else:
            assessment['value_factors']['data_completeness'] = 2
            assessment['reasoning'].append(f"Limited data volume: {file_analysis['line_count']} records")
        
        # Factor 2: GO term relevance (0-25 points)
        go_relevance = 0
        if any('go' in col.lower() for col in columns):
            go_relevance += 15
            assessment['reasoning'].append("Contains GO term columns")
        if 'go:' in str(file_analysis.get('sample_rows', [])).lower():
            go_relevance += 10
            assessment['reasoning'].append("Contains GO term identifiers")
        assessment['value_factors']['go_relevance'] = go_relevance
        
        # Factor 3: Gene information (0-20 points)
        gene_relevance = 0
        if any(keyword in str(columns).lower() for keyword in ['gene', 'protein', 'symbol']):
            gene_relevance += 15
            assessment['reasoning'].append("Contains gene/protein information")
        assessment['value_factors']['gene_relevance'] = gene_relevance
        
        # Factor 4: LLM/Analysis data (0-15 points)
        llm_relevance = 0
        if any(keyword in str(columns).lower() for keyword in ['llm', 'analysis', 'score', 'confidence', 'similarity']):
            llm_relevance += 10
            assessment['reasoning'].append("Contains analysis/scoring data")
        if any(keyword in file_name for keyword in ['confidence', 'eval', 'review', 'sim', 'contaminated']):
            llm_relevance += 5
            assessment['reasoning'].append("File name suggests analysis data")
        assessment['value_factors']['llm_relevance'] = llm_relevance
        
        # Factor 5: Data structure and parsability (0-15 points)
        structure_score = 0
        if file_analysis['file_type'] in ['.csv', '.tsv']:
            structure_score += 10
            assessment['reasoning'].append("Well-structured CSV/TSV format")
        elif file_analysis['file_type'] == '.xlsx':
            structure_score += 8
            assessment['reasoning'].append("Excel format (parsable)")
        elif file_analysis['file_type'] == '.txt' and file_analysis.get('potential_tsv'):
            structure_score += 7
            assessment['reasoning'].append("Structured text file")
        
        if len(file_analysis.get('errors', [])) == 0:
            structure_score += 5
            assessment['reasoning'].append("No parsing errors")
        assessment['value_factors']['structure_quality'] = structure_score
        
        # Calculate total score
        assessment['integration_score'] = sum(assessment['value_factors'].values())
        
        # Determine integration potential and recommendation
        if assessment['integration_score'] >= 60:
            assessment['integration_potential'] = 'high'
            assessment['recommended_action'] = 'integrate_priority'
        elif assessment['integration_score'] >= 40:
            assessment['integration_potential'] = 'medium'
            assessment['recommended_action'] = 'integrate_secondary'
        elif assessment['integration_score'] >= 20:
            assessment['integration_potential'] = 'low'
            assessment['recommended_action'] = 'consider_integration'
        else:
            assessment['integration_potential'] = 'minimal'
            assessment['recommended_action'] = 'skip'
            
        return assessment
    
    def analyze_all_files(self) -> Dict[str, Any]:
        """Analyze all files in the data directory"""
        results = {
            'files_analyzed': 0,
            'total_data_volume': 0,
            'file_analyses': {},
            'integration_assessments': {},
            'summary': {
                'high_value_files': [],
                'medium_value_files': [],
                'low_value_files': [],
                'skip_files': []
            }
        }
        
        # Get all data files
        data_files = []
        for ext in ['*.csv', '*.tsv', '*.txt', '*.xlsx']:
            data_files.extend(self.data_path.glob(ext))
            
        logger.info(f"Found {len(data_files)} data files to analyze")
        
        for file_path in data_files:
            logger.info(f"Analyzing: {file_path.name}")
            
            # Analyze file structure
            file_analysis = self.analyze_file_structure(file_path)
            results['file_analyses'][file_path.name] = file_analysis
            
            # Assess integration value
            integration_assessment = self.assess_integration_value(file_analysis)
            results['integration_assessments'][file_path.name] = integration_assessment
            
            # Update summary
            action = integration_assessment['recommended_action']
            if action == 'integrate_priority':
                results['summary']['high_value_files'].append(file_path.name)
            elif action == 'integrate_secondary':
                results['summary']['medium_value_files'].append(file_path.name)
            elif action == 'consider_integration':
                results['summary']['low_value_files'].append(file_path.name)
            else:
                results['summary']['skip_files'].append(file_path.name)
                
            results['files_analyzed'] += 1
            results['total_data_volume'] += file_analysis['line_count']
            
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive analysis report"""
        report = []
        report.append("=" * 80)
        report.append("GO_TERM_ANALYSIS DATA_FILES COMPREHENSIVE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 40)
        report.append(f"Total files analyzed: {results['files_analyzed']}")
        report.append(f"Total data volume: {results['total_data_volume']:,} records")
        report.append(f"High-value files: {len(results['summary']['high_value_files'])}")
        report.append(f"Medium-value files: {len(results['summary']['medium_value_files'])}")
        report.append(f"Low-value files: {len(results['summary']['low_value_files'])}")
        report.append(f"Skip files: {len(results['summary']['skip_files'])}")
        report.append("")
        
        # High-value files detailed analysis
        if results['summary']['high_value_files']:
            report.append("HIGH-VALUE FILES FOR INTEGRATION")
            report.append("-" * 40)
            for file_name in results['summary']['high_value_files']:
                file_analysis = results['file_analyses'][file_name]
                integration_assessment = results['integration_assessments'][file_name]
                
                report.append(f"\n📊 {file_name}")
                report.append(f"   Integration Score: {integration_assessment['integration_score']}/100")
                report.append(f"   Records: {file_analysis['line_count']:,}")
                report.append(f"   Columns: {len(file_analysis.get('columns', []))}")
                if file_analysis.get('columns'):
                    report.append(f"   Key Columns: {', '.join(file_analysis['columns'][:5])}")
                report.append(f"   Value Factors:")
                for factor, score in integration_assessment['value_factors'].items():
                    report.append(f"     - {factor}: {score} points")
                report.append(f"   Reasoning: {'; '.join(integration_assessment['reasoning'])}")
                
        # Medium-value files
        if results['summary']['medium_value_files']:
            report.append("\n\nMEDIUM-VALUE FILES FOR INTEGRATION")
            report.append("-" * 40)
            for file_name in results['summary']['medium_value_files']:
                file_analysis = results['file_analyses'][file_name]
                integration_assessment = results['integration_assessments'][file_name]
                
                report.append(f"\n📈 {file_name}")
                report.append(f"   Integration Score: {integration_assessment['integration_score']}/100")
                report.append(f"   Records: {file_analysis['line_count']:,}")
                report.append(f"   Recommendation: {integration_assessment['recommended_action']}")
                
        # File-by-file breakdown
        report.append("\n\nDETAILED FILE-BY-FILE ANALYSIS")
        report.append("-" * 40)
        
        for file_name, file_analysis in results['file_analyses'].items():
            integration_assessment = results['integration_assessments'][file_name]
            
            report.append(f"\n🔍 {file_name}")
            report.append(f"   Size: {file_analysis['file_size']:,} bytes")
            report.append(f"   Type: {file_analysis['file_type']}")
            report.append(f"   Records: {file_analysis['line_count']:,}")
            report.append(f"   Integration Score: {integration_assessment['integration_score']}/100")
            report.append(f"   Potential: {integration_assessment['integration_potential']}")
            report.append(f"   Action: {integration_assessment['recommended_action']}")
            
            if file_analysis.get('columns'):
                report.append(f"   Columns ({len(file_analysis['columns'])}): {', '.join(file_analysis['columns'])}")
                
            if file_analysis.get('errors'):
                report.append(f"   ⚠️  Errors: {'; '.join(file_analysis['errors'])}")
                
        # Integration recommendations
        report.append("\n\nINTEGRATION RECOMMENDATIONS")
        report.append("-" * 40)
        
        total_high_medium = len(results['summary']['high_value_files']) + len(results['summary']['medium_value_files'])
        if total_high_medium > 0:
            report.append("✅ RECOMMENDED FOR INTEGRATION:")
            for file_name in results['summary']['high_value_files'] + results['summary']['medium_value_files']:
                integration_assessment = results['integration_assessments'][file_name]
                report.append(f"   • {file_name} (Score: {integration_assessment['integration_score']}/100)")
                
            report.append(f"\n📈 INTEGRATION VALUE ASSESSMENT:")
            total_records = sum(results['file_analyses'][f]['line_count'] for f in results['summary']['high_value_files'] + results['summary']['medium_value_files'])
            report.append(f"   • Total valuable records: {total_records:,}")
            report.append(f"   • Files to integrate: {total_high_medium}")
            
            # Calculate overall integration score
            avg_score = np.mean([results['integration_assessments'][f]['integration_score'] 
                               for f in results['summary']['high_value_files'] + results['summary']['medium_value_files']])
            report.append(f"   • Average integration score: {avg_score:.1f}/100")
            
            if avg_score >= 70:
                report.append("   • Overall Assessment: EXCEPTIONAL VALUE - Highly recommended for integration")
            elif avg_score >= 50:
                report.append("   • Overall Assessment: HIGH VALUE - Recommended for integration")
            elif avg_score >= 30:
                report.append("   • Overall Assessment: MODERATE VALUE - Consider integration")
            else:
                report.append("   • Overall Assessment: LIMITED VALUE - Low priority integration")
        else:
            report.append("❌ No files meet the integration criteria threshold")
            
        return "\n".join(report)

def main():
    """Main execution function"""
    data_path = "/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data/GO_term_analysis/data_files"
    
    if not os.path.exists(data_path):
        logger.error(f"Data path does not exist: {data_path}")
        return
        
    analyzer = DataFilesAnalyzer(data_path)
    
    logger.info("Starting comprehensive data files analysis...")
    results = analyzer.analyze_all_files()
    
    logger.info("Generating analysis report...")
    report = analyzer.generate_report(results)
    
    # Save report
    report_path = "/home/mreddy1/knowledge_graph/data_files_analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
        
    logger.info(f"Analysis complete. Report saved to: {report_path}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("DATA FILES ANALYSIS SUMMARY")
    print("="*60)
    print(f"Files analyzed: {results['files_analyzed']}")
    print(f"Total records: {results['total_data_volume']:,}")
    print(f"High-value files: {len(results['summary']['high_value_files'])}")
    print(f"Medium-value files: {len(results['summary']['medium_value_files'])}")
    
    if results['summary']['high_value_files']:
        print(f"\nTop integration candidates:")
        for file_name in results['summary']['high_value_files']:
            score = results['integration_assessments'][file_name]['integration_score']
            records = results['file_analyses'][file_name]['line_count']
            print(f"  • {file_name} (Score: {score}/100, Records: {records:,})")
            
    print(f"\nFull report: {report_path}")
    print("="*60)

if __name__ == "__main__":
    main()