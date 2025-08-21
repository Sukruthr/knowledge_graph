#!/usr/bin/env python3
"""
Comprehensive analysis of LLM_processed data for integration value assessment.

Analyzes the LLM_processed data structure, content, and potential value 
for biomedical knowledge graph integration.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMProcessedAnalyzer:
    """Comprehensive analyzer for LLM_processed data files."""
    
    def __init__(self, data_dir: str = "llm_evaluation_for_gene_set_interpretation/data/GO_term_analysis/LLM_processed"):
        self.data_dir = Path(data_dir)
        self.files = {}
        self.analysis_results = {}
        
    def analyze_all_llm_processed_data(self) -> Dict[str, Any]:
        """Perform comprehensive analysis of all LLM_processed data."""
        logger.info("🔍 Starting comprehensive LLM_processed data analysis...")
        
        # Discover and categorize files
        self._discover_files()
        
        # Analyze each category
        self._analyze_main_datasets()
        self._analyze_model_comparison_files()
        self._analyze_similarity_rankings()
        self._analyze_contamination_datasets()
        
        # Generate integration value assessment
        integration_value = self._assess_integration_value()
        
        # Compile final results
        results = {
            'file_inventory': self.files,
            'main_datasets': self.analysis_results.get('main_datasets', {}),
            'model_comparison': self.analysis_results.get('model_comparison', {}),
            'similarity_rankings': self.analysis_results.get('similarity_rankings', {}),
            'contamination_analysis': self.analysis_results.get('contamination_analysis', {}),
            'integration_assessment': integration_value,
            'recommendations': self._generate_recommendations()
        }
        
        return results
    
    def _discover_files(self):
        """Discover and categorize all files in the LLM_processed directory."""
        logger.info("📁 Discovering LLM_processed files...")
        
        categories = {
            'main_datasets': [],
            'model_comparison': [],
            'similarity_rankings': [],
            'contamination_files': [],
            'other': []
        }
        
        for file_path in self.data_dir.glob("*"):
            if file_path.is_file():
                filename = file_path.name
                
                # Categorize files
                if 'simrank' in filename and 'pval' not in filename:
                    categories['similarity_rankings'].append(filename)
                elif 'simrank_pval' in filename:
                    categories['similarity_rankings'].append(filename)
                elif 'model_comparison' in filename:
                    categories['model_comparison'].append(filename)
                elif 'contamination' in filename:
                    categories['contamination_files'].append(filename)
                elif 'selected_1000_go_terms' in filename and 'bad_coverage' not in filename and 'simrank' not in filename:
                    categories['main_datasets'].append(filename)
                elif 'GO_representative' in filename:
                    categories['main_datasets'].append(filename)
                else:
                    categories['other'].append(filename)
        
        self.files = categories
        
        # Print file inventory
        print("\n📊 FILE INVENTORY:")
        for category, files in categories.items():
            print(f"   {category}: {len(files)} files")
            for file in files:
                print(f"     - {file}")
    
    def _analyze_main_datasets(self):
        """Analyze main LLM datasets."""
        logger.info("🔬 Analyzing main LLM datasets...")
        
        main_analysis = {}
        
        for filename in self.files['main_datasets']:
            file_path = self.data_dir / filename
            
            try:
                if filename.endswith('.tsv'):
                    df = pd.read_csv(file_path, sep='\t', dtype=str)
                else:
                    df = pd.read_csv(file_path, dtype=str)
                
                analysis = {
                    'file_size': os.path.getsize(file_path),
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'columns': list(df.columns),
                    'go_terms': df['GO'].nunique() if 'GO' in df.columns else 0,
                    'sample_data': df.head(2).to_dict('records') if len(df) > 0 else []
                }
                
                # Analyze specific data types
                if 'gpt_4_default Score' in df.columns:
                    scores = pd.to_numeric(df['gpt_4_default Score'], errors='coerce')
                    analysis['llm_scores'] = {
                        'mean': float(scores.mean()) if not scores.isna().all() else None,
                        'std': float(scores.std()) if not scores.isna().all() else None,
                        'min': float(scores.min()) if not scores.isna().all() else None,
                        'max': float(scores.max()) if not scores.isna().all() else None,
                        'count': int(scores.count())
                    }
                
                # Analyze gene coverage
                if 'Genes' in df.columns:
                    total_genes = set()
                    for gene_list in df['Genes'].dropna():
                        genes = str(gene_list).split()
                        total_genes.update(genes)
                    analysis['unique_genes'] = len(total_genes)
                    analysis['sample_genes'] = list(total_genes)[:10]  # First 10 for inspection
                
                main_analysis[filename] = analysis
                
            except Exception as e:
                logger.error(f"Error analyzing {filename}: {e}")
                main_analysis[filename] = {'error': str(e)}
        
        self.analysis_results['main_datasets'] = main_analysis
    
    def _analyze_model_comparison_files(self):
        """Analyze multi-model comparison files."""
        logger.info("🤖 Analyzing model comparison files...")
        
        model_analysis = {}
        model_names = ['gpt_4', 'gpt_35', 'gemini_pro', 'llama2_70b', 'llama2_7b', 'mistral_7b', 'mixtral_instruct', 'mixtral_latest']
        
        for filename in self.files['contamination_files']:
            file_path = self.data_dir / filename
            
            try:
                df = pd.read_csv(file_path, sep='\t', dtype=str)
                
                # Extract model name from filename
                model_name = None
                for model in model_names:
                    if model in filename:
                        model_name = model
                        break
                
                if not model_name:
                    continue
                
                analysis = {
                    'model_name': model_name,
                    'file_size': os.path.getsize(file_path),
                    'row_count': len(df),
                    'columns': list(df.columns),
                    'go_terms': df['GO'].nunique() if 'GO' in df.columns else 0,
                }
                
                # Analyze contamination scenarios
                contamination_scenarios = []
                for col in df.columns:
                    if 'Score' in col and model_name in col:
                        scenario = col.replace(f'{model_name}_', '').replace(' Score', '')
                        contamination_scenarios.append(scenario)
                        
                        scores = pd.to_numeric(df[col], errors='coerce')
                        analysis[f'{scenario}_scores'] = {
                            'mean': float(scores.mean()) if not scores.isna().all() else None,
                            'std': float(scores.std()) if not scores.isna().all() else None,
                            'count': int(scores.count())
                        }
                
                analysis['contamination_scenarios'] = contamination_scenarios
                model_analysis[filename] = analysis
                
            except Exception as e:
                logger.error(f"Error analyzing {filename}: {e}")
                model_analysis[filename] = {'error': str(e)}
        
        self.analysis_results['model_comparison'] = model_analysis
    
    def _analyze_similarity_rankings(self):
        """Analyze similarity ranking files."""
        logger.info("📈 Analyzing similarity ranking files...")
        
        similarity_analysis = {}
        
        for filename in self.files['similarity_rankings']:
            file_path = self.data_dir / filename
            
            try:
                df = pd.read_csv(file_path, sep='\t', dtype=str)
                
                analysis = {
                    'file_size': os.path.getsize(file_path),
                    'row_count': len(df),
                    'columns': list(df.columns),
                    'go_terms': df['GO'].nunique() if 'GO' in df.columns else 0,
                }
                
                # Analyze similarity metrics
                if 'sim_rank' in df.columns:
                    sim_ranks = pd.to_numeric(df['sim_rank'], errors='coerce')
                    analysis['similarity_ranks'] = {
                        'mean': float(sim_ranks.mean()) if not sim_ranks.isna().all() else None,
                        'max': float(sim_ranks.max()) if not sim_ranks.isna().all() else None,
                        'count': int(sim_ranks.count())
                    }
                
                if 'true_GO_term_sim_percentile' in df.columns:
                    percentiles = pd.to_numeric(df['true_GO_term_sim_percentile'], errors='coerce')
                    analysis['similarity_percentiles'] = {
                        'mean': float(percentiles.mean()) if not percentiles.isna().all() else None,
                        'std': float(percentiles.std()) if not percentiles.isna().all() else None,
                        'count': int(percentiles.count())
                    }
                
                # Check for p-value data
                if 'pval' in filename:
                    analysis['contains_pvalues'] = True
                    p_val_columns = [col for col in df.columns if 'pval' in col.lower()]
                    analysis['pvalue_columns'] = p_val_columns
                
                similarity_analysis[filename] = analysis
                
            except Exception as e:
                logger.error(f"Error analyzing {filename}: {e}")
                similarity_analysis[filename] = {'error': str(e)}
        
        self.analysis_results['similarity_rankings'] = similarity_analysis
    
    def _analyze_contamination_datasets(self):
        """Analyze contamination analysis datasets."""
        logger.info("🧪 Analyzing contamination datasets...")
        
        contamination_analysis = {}
        
        # Get unique models from contamination files
        models_analyzed = set()
        for filename in self.files['contamination_files']:
            for model in ['gpt_4', 'gpt_35', 'gemini_pro', 'llama2_70b', 'llama2_7b', 'mistral_7b', 'mixtral_instruct', 'mixtral_latest']:
                if model in filename:
                    models_analyzed.add(model)
                    break
        
        contamination_analysis['models_with_contamination_data'] = list(models_analyzed)
        contamination_analysis['contamination_file_count'] = len(self.files['contamination_files'])
        
        # Analyze toy example files
        toy_files = [f for f in self.files['contamination_files'] if 'toy_example' in f]
        contamination_analysis['toy_example_files'] = toy_files
        
        self.analysis_results['contamination_analysis'] = contamination_analysis
    
    def _assess_integration_value(self) -> Dict[str, Any]:
        """Assess the integration value of LLM_processed data."""
        logger.info("💎 Assessing integration value...")
        
        # Calculate value factors
        factors = {}
        
        # 1. Data Completeness (25 points)
        main_files = len(self.files['main_datasets'])
        similarity_files = len(self.files['similarity_rankings'])
        model_files = len(self.files['contamination_files'])
        total_files = main_files + similarity_files + model_files
        
        completeness_score = min(25, (total_files / 15) * 25)  # Expect ~15 files
        factors['data_completeness'] = {
            'score': completeness_score,
            'max': 25,
            'details': f"{total_files} files ({main_files} main, {similarity_files} similarity, {model_files} model comparison)"
        }
        
        # 2. Multi-Model Analysis Depth (25 points)
        models_available = len(self.analysis_results.get('contamination_analysis', {}).get('models_with_contamination_data', []))
        multi_model_score = min(25, (models_available / 8) * 25)  # Expect 8 models
        factors['multi_model_analysis'] = {
            'score': multi_model_score,
            'max': 25,
            'details': f"{models_available} LLM models with contamination analysis"
        }
        
        # 3. Advanced Analytics (20 points)
        has_similarity = bool(self.files['similarity_rankings'])
        has_contamination = bool(self.files['contamination_files'])
        has_main_llm = bool(self.files['main_datasets'])
        
        analytics_score = 0
        if has_similarity: analytics_score += 7
        if has_contamination: analytics_score += 8
        if has_main_llm: analytics_score += 5
        
        factors['advanced_analytics'] = {
            'score': analytics_score,
            'max': 20,
            'details': f"Similarity rankings: {has_similarity}, Contamination analysis: {has_contamination}, LLM interpretations: {has_main_llm}"
        }
        
        # 4. Gene Coverage (15 points)
        main_analysis = self.analysis_results.get('main_datasets', {})
        max_genes = 0
        for file_analysis in main_analysis.values():
            if 'unique_genes' in file_analysis:
                max_genes = max(max_genes, file_analysis['unique_genes'])
        
        gene_coverage_score = min(15, (max_genes / 20000) * 15)  # Expect ~20K genes
        factors['gene_coverage'] = {
            'score': gene_coverage_score,
            'max': 15,
            'details': f"Maximum {max_genes} unique genes across datasets"
        }
        
        # 5. Research Enhancement Value (15 points)
        # This is high value due to LLM interpretations and model comparison
        research_score = 15  # Full score for comprehensive LLM analysis
        factors['research_enhancement'] = {
            'score': research_score,
            'max': 15,
            'details': "LLM interpretations, model comparison, contamination robustness analysis"
        }
        
        # Calculate total score
        total_score = sum(factor['score'] for factor in factors.values())
        max_possible = sum(factor['max'] for factor in factors.values())
        
        return {
            'total_score': total_score,
            'max_possible': max_possible,
            'percentage': (total_score / max_possible) * 100,
            'factors': factors,
            'recommendation': 'HIGHLY RECOMMENDED' if total_score >= 80 else 'RECOMMENDED' if total_score >= 60 else 'CONDITIONAL'
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate integration recommendations."""
        recommendations = []
        
        integration_score = self.analysis_results.get('integration_assessment', {}).get('total_score', 0)
        
        if integration_score >= 80:
            recommendations.extend([
                "PROCEED with full LLM_processed data integration",
                "Implement multi-model LLM comparison parser",
                "Add contamination robustness analysis to KG",
                "Include similarity ranking capabilities",
                "Create advanced LLM interpretation query methods"
            ])
        elif integration_score >= 60:
            recommendations.extend([
                "CONDITIONAL integration recommended",
                "Focus on highest-value datasets first",
                "Implement core LLM interpretation functionality",
                "Consider phased integration approach"
            ])
        else:
            recommendations.extend([
                "Review data quality and completeness",
                "Consider selective integration of specific components",
                "Evaluate alternative data sources"
            ])
        
        # Add specific technical recommendations
        recommendations.extend([
            "Design LLMProcessedParser following established patterns",
            "Integrate with existing CombinedBiomedicalParser architecture",
            "Add comprehensive testing for multi-model scenarios",
            "Document contamination analysis capabilities"
        ])
        
        return recommendations

def main():
    """Main analysis function."""
    print("🚀 LLM_PROCESSED DATA ANALYSIS")
    print("=" * 60)
    
    analyzer = LLMProcessedAnalyzer()
    results = analyzer.analyze_all_llm_processed_data()
    
    # Print summary
    print(f"\n📊 INTEGRATION VALUE ASSESSMENT:")
    assessment = results['integration_assessment']
    print(f"   Score: {assessment['total_score']:.1f}/{assessment['max_possible']} ({assessment['percentage']:.1f}%)")
    print(f"   Recommendation: {assessment['recommendation']}")
    
    print(f"\n🎯 FACTOR BREAKDOWN:")
    for factor_name, factor_data in assessment['factors'].items():
        print(f"   {factor_name}: {factor_data['score']:.1f}/{factor_data['max']} - {factor_data['details']}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    # Save detailed results
    import json
    with open('llm_processed_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to llm_processed_analysis_results.json")
    
    return results

if __name__ == "__main__":
    results = main()