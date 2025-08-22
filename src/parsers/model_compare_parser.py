"""
Model Comparison Data Parser for LLM evaluation results.
Integrates model performance and prediction confidence data.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Set
import json

logger = logging.getLogger(__name__)

class ModelCompareParser:
    """Parser for model comparison evaluation data."""
    
    def __init__(self, model_compare_dir: str):
        """
        Initialize model comparison parser.
        
        Args:
            model_compare_dir: Path to model_compare directory
        """
        self.model_compare_dir = Path(model_compare_dir)
        
        # Core data structures
        self.model_predictions = {}
        self.similarity_rankings = {}
        self.evaluation_metrics = {}
        self.contamination_results = {}
        
        # Model types covered
        self.available_models = set()
        
        logger.info(f"Initialized ModelCompareParser for {model_compare_dir}")
    
    def parse_all_model_data(self) -> Dict[str, Dict]:
        """
        Parse all model comparison data.
        
        Returns:
            Dictionary containing all parsed model evaluation data
        """
        logger.info("Parsing all model comparison data...")
        
        # Parse LLM processed files
        self.model_predictions = self.parse_llm_processed_files()
        
        # Parse similarity ranking files
        self.similarity_rankings = self.parse_similarity_ranking_files()
        
        # Extract evaluation metrics
        self.evaluation_metrics = self.extract_evaluation_metrics()
        
        # Analyze contamination effects
        self.contamination_results = self.analyze_contamination_effects()
        
        # Compute summary statistics
        summary_stats = self.compute_summary_statistics()
        
        return {
            'model_predictions': self.model_predictions,
            'similarity_rankings': self.similarity_rankings,
            'evaluation_metrics': self.evaluation_metrics,
            'contamination_results': self.contamination_results,
            'summary_stats': summary_stats,
            'available_models': list(self.available_models)
        }
    
    def parse_llm_processed_files(self) -> Dict[str, Dict]:
        """Parse LLM processed model comparison files."""
        logger.info("Parsing LLM processed files...")
        
        llm_files = list(self.model_compare_dir.glob("LLM_processed_model_compare_*.tsv"))
        predictions = {}
        
        for file_path in llm_files:
            model_name = self.extract_model_name(file_path.name)
            if model_name == "unknown":
                continue
            
            self.available_models.add(model_name)
            
            try:
                df = pd.read_csv(file_path, sep='\t')
                
                model_data = {
                    'file_path': str(file_path),
                    'total_evaluations': len(df),
                    'go_predictions': {},
                    'score_distributions': {},
                    'confidence_bins': {}
                }
                
                # Parse each GO term evaluation
                for _, row in df.iterrows():
                    go_id = row.get('GO', '')
                    if not go_id or not go_id.startswith('GO:'):
                        continue
                    
                    # Extract predictions for all contamination levels
                    go_prediction = {
                        'go_id': go_id,
                        'true_description': row.get('Term_Description', ''),
                        'genes': self._parse_gene_list(row.get('Genes', '')),
                        'gene_count': int(row.get('Gene_Count', 0)) if pd.notna(row.get('Gene_Count')) else 0,
                        'scenarios': {}
                    }
                    
                    # Parse default, 50% contaminated, and 100% contaminated scenarios
                    scenarios = ['default', '50perc_contaminated', '100perc_contaminated']
                    
                    for scenario in scenarios:
                        name_col = f"{model_name}_{scenario} Name"
                        analysis_col = f"{model_name}_{scenario} Analysis"
                        score_col = f"{model_name}_{scenario} Score"
                        bin_col = f"{model_name}_{scenario} Score Bin"
                        
                        if scenario == '50perc_contaminated':
                            genes_col = '50perc_contaminated_Genes'
                        elif scenario == '100perc_contaminated':
                            genes_col = '100perc_contaminated_Genes'
                        else:
                            genes_col = 'Genes'
                        
                        scenario_data = {
                            'predicted_name': row.get(name_col, ''),
                            'analysis': row.get(analysis_col, ''),
                            'confidence_score': float(row.get(score_col, 0)) if pd.notna(row.get(score_col)) else 0.0,
                            'confidence_bin': row.get(bin_col, ''),
                            'genes_used': self._parse_gene_list(row.get(genes_col, ''))
                        }
                        
                        go_prediction['scenarios'][scenario] = scenario_data
                    
                    model_data['go_predictions'][go_id] = go_prediction
                
                # Compute score distributions
                model_data['score_distributions'] = self._compute_score_distributions(df, model_name)
                model_data['confidence_bins'] = self._compute_confidence_bins(df, model_name)
                
                predictions[model_name] = model_data
                logger.info(f"Parsed {len(model_data['go_predictions'])} GO predictions for {model_name}")
                
            except Exception as e:
                logger.error(f"Error parsing {file_path}: {e}")
                
        return predictions
    
    def parse_similarity_ranking_files(self) -> Dict[str, Dict]:
        """Parse similarity ranking files."""
        logger.info("Parsing similarity ranking files...")
        
        sim_files = list(self.model_compare_dir.glob("sim_rank_LLM_processed_*.tsv"))
        rankings = {}
        
        for file_path in sim_files:
            model_name = self.extract_model_name(file_path.name)
            if model_name == "unknown":
                continue
            
            try:
                df = pd.read_csv(file_path, sep='\t')
                
                ranking_data = {
                    'file_path': str(file_path),
                    'total_rankings': len(df),
                    'similarity_metrics': {},
                    'ranking_performance': {}
                }
                
                # Parse similarity metrics for each GO term
                for _, row in df.iterrows():
                    go_id = row.get('GO', '')
                    if not go_id or not go_id.startswith('GO:'):
                        continue
                    
                    similarity_info = {
                        'llm_go_similarity': float(row.get('LLM_name_GO_term_sim', 0)) if pd.notna(row.get('LLM_name_GO_term_sim')) else 0.0,
                        'similarity_rank': int(row.get('sim_rank', 0)) if pd.notna(row.get('sim_rank')) else 0,
                        'true_percentile': float(row.get('true_GO_term_sim_percentile', 0)) if pd.notna(row.get('true_GO_term_sim_percentile')) else 0.0,
                        'random_comparison': {
                            'random_go_name': row.get('random_GO_name', ''),
                            'random_similarity': float(row.get('random_go_llm_sim', 0)) if pd.notna(row.get('random_go_llm_sim')) else 0.0,
                            'random_rank': int(row.get('random_sim_rank', 0)) if pd.notna(row.get('random_sim_rank')) else 0,
                            'random_percentile': float(row.get('random_sim_percentile', 0)) if pd.notna(row.get('random_sim_percentile')) else 0.0
                        },
                        'top_matches': {
                            'top_3_hits': row.get('top_3_hits', '').split('|') if pd.notna(row.get('top_3_hits')) else [],
                            'top_3_similarities': [float(x) for x in str(row.get('top_3_sim', '')).split('|') if x] if pd.notna(row.get('top_3_sim')) else []
                        }
                    }
                    
                    ranking_data['similarity_metrics'][go_id] = similarity_info
                
                # Compute ranking performance metrics
                ranking_data['ranking_performance'] = self._compute_ranking_performance(df)
                
                rankings[model_name] = ranking_data
                logger.info(f"Parsed similarity rankings for {len(ranking_data['similarity_metrics'])} GO terms for {model_name}")
                
            except Exception as e:
                logger.error(f"Error parsing ranking file {file_path}: {e}")
        
        return rankings
    
    def extract_evaluation_metrics(self) -> Dict[str, Dict]:
        """Extract evaluation metrics across all models."""
        logger.info("Extracting evaluation metrics...")
        
        metrics = {}
        
        for model_name in self.available_models:
            model_metrics = {
                'confidence_stats': {},
                'similarity_stats': {},
                'contamination_impact': {}
            }
            
            # Confidence score statistics
            if model_name in self.model_predictions:
                predictions = self.model_predictions[model_name]['go_predictions']
                
                all_scores = []
                scenario_scores = {'default': [], '50perc_contaminated': [], '100perc_contaminated': []}
                
                for go_data in predictions.values():
                    for scenario, scenario_data in go_data['scenarios'].items():
                        score = scenario_data['confidence_score']
                        if score > 0:  # Valid score
                            all_scores.append(score)
                            scenario_scores[scenario].append(score)
                
                if all_scores:
                    model_metrics['confidence_stats'] = {
                        'mean_confidence': sum(all_scores) / len(all_scores),
                        'median_confidence': sorted(all_scores)[len(all_scores)//2],
                        'high_confidence_count': len([s for s in all_scores if s >= 0.8]),
                        'low_confidence_count': len([s for s in all_scores if s < 0.5]),
                        'scenario_means': {scenario: sum(scores)/len(scores) if scores else 0 
                                         for scenario, scores in scenario_scores.items()}
                    }
            
            # Similarity ranking statistics
            if model_name in self.similarity_rankings:
                rankings = self.similarity_rankings[model_name]['similarity_metrics']
                
                similarities = [data['llm_go_similarity'] for data in rankings.values()]
                percentiles = [data['true_percentile'] for data in rankings.values()]
                ranks = [data['similarity_rank'] for data in rankings.values() if data['similarity_rank'] > 0]
                
                if similarities:
                    model_metrics['similarity_stats'] = {
                        'mean_similarity': sum(similarities) / len(similarities),
                        'mean_percentile': sum(percentiles) / len(percentiles),
                        'mean_rank': sum(ranks) / len(ranks) if ranks else 0,
                        'top_10_percent_count': len([p for p in percentiles if p >= 0.9]),
                        'bottom_50_percent_count': len([p for p in percentiles if p < 0.5])
                    }
            
            metrics[model_name] = model_metrics
        
        return metrics
    
    def analyze_contamination_effects(self) -> Dict[str, Dict]:
        """Analyze the effects of gene contamination on model performance."""
        logger.info("Analyzing contamination effects...")
        
        contamination_analysis = {}
        
        for model_name in self.available_models:
            if model_name not in self.model_predictions:
                continue
            
            predictions = self.model_predictions[model_name]['go_predictions']
            
            # Track score changes across contamination levels
            score_changes = {
                'default_to_50perc': [],
                'default_to_100perc': [],
                '50perc_to_100perc': []
            }
            
            performance_degradation = {
                'severe_drop': 0,  # >0.3 drop
                'moderate_drop': 0,  # 0.1-0.3 drop
                'stable': 0,  # <0.1 change
                'improvement': 0  # increase
            }
            
            for go_data in predictions.values():
                scenarios = go_data['scenarios']
                
                if all(scenario in scenarios for scenario in ['default', '50perc_contaminated', '100perc_contaminated']):
                    default_score = scenarios['default']['confidence_score']
                    cont_50_score = scenarios['50perc_contaminated']['confidence_score']
                    cont_100_score = scenarios['100perc_contaminated']['confidence_score']
                    
                    # Calculate score changes
                    if default_score > 0 and cont_50_score >= 0:
                        change_50 = cont_50_score - default_score
                        score_changes['default_to_50perc'].append(change_50)
                        
                        # Categorize performance change
                        if change_50 < -0.3:
                            performance_degradation['severe_drop'] += 1
                        elif change_50 < -0.1:
                            performance_degradation['moderate_drop'] += 1
                        elif abs(change_50) <= 0.1:
                            performance_degradation['stable'] += 1
                        else:
                            performance_degradation['improvement'] += 1
                    
                    if default_score > 0 and cont_100_score >= 0:
                        change_100 = cont_100_score - default_score
                        score_changes['default_to_100perc'].append(change_100)
                    
                    if cont_50_score > 0 and cont_100_score >= 0:
                        change_50_to_100 = cont_100_score - cont_50_score
                        score_changes['50perc_to_100perc'].append(change_50_to_100)
            
            # Compute statistics
            contamination_stats = {}
            for change_type, changes in score_changes.items():
                if changes:
                    contamination_stats[change_type] = {
                        'mean_change': sum(changes) / len(changes),
                        'median_change': sorted(changes)[len(changes)//2],
                        'negative_changes': len([c for c in changes if c < 0]),
                        'positive_changes': len([c for c in changes if c > 0])
                    }
            
            contamination_analysis[model_name] = {
                'score_changes': contamination_stats,
                'performance_degradation': performance_degradation,
                'robustness_score': performance_degradation['stable'] / max(sum(performance_degradation.values()), 1)
            }
        
        return contamination_analysis
    
    def compute_summary_statistics(self) -> Dict:
        """Compute summary statistics across all models."""
        logger.info("Computing summary statistics...")
        
        summary = {
            'total_models': len(self.available_models),
            'models_covered': list(self.available_models),
            'total_go_evaluations': 0,
            'total_similarity_rankings': 0,
            'cross_model_comparison': {}
        }
        
        # Count total evaluations
        for model_data in self.model_predictions.values():
            summary['total_go_evaluations'] += model_data['total_evaluations']
        
        for ranking_data in self.similarity_rankings.values():
            summary['total_similarity_rankings'] += ranking_data['total_rankings']
        
        # Cross-model performance comparison
        if len(self.available_models) >= 2:
            # Compare confidence scores
            model_confidence_means = {}
            for model_name, metrics in self.evaluation_metrics.items():
                if 'confidence_stats' in metrics and 'mean_confidence' in metrics['confidence_stats']:
                    model_confidence_means[model_name] = metrics['confidence_stats']['mean_confidence']
            
            if model_confidence_means:
                best_model = max(model_confidence_means.items(), key=lambda x: x[1])
                worst_model = min(model_confidence_means.items(), key=lambda x: x[1])
                
                summary['cross_model_comparison'] = {
                    'best_confidence_model': best_model[0],
                    'best_confidence_score': best_model[1],
                    'worst_confidence_model': worst_model[0],
                    'worst_confidence_score': worst_model[1],
                    'confidence_range': best_model[1] - worst_model[1]
                }
        
        return summary
    
    def _parse_gene_list(self, gene_string: str) -> List[str]:
        """Parse space-separated gene list."""
        if pd.isna(gene_string) or not gene_string:
            return []
        return str(gene_string).strip().split()
    
    def _compute_score_distributions(self, df: pd.DataFrame, model_name: str) -> Dict:
        """Compute score distributions for a model."""
        distributions = {}
        
        scenarios = ['default', '50perc_contaminated', '100perc_contaminated']
        
        for scenario in scenarios:
            score_col = f"{model_name}_{scenario} Score"
            if score_col in df.columns:
                scores = df[score_col].dropna()
                scores = scores[scores >= 0]  # Valid scores only
                
                if len(scores) > 0:
                    distributions[scenario] = {
                        'mean': float(scores.mean()),
                        'std': float(scores.std()),
                        'min': float(scores.min()),
                        'max': float(scores.max()),
                        'count': len(scores)
                    }
        
        return distributions
    
    def _compute_confidence_bins(self, df: pd.DataFrame, model_name: str) -> Dict:
        """Compute confidence bin distributions."""
        bins = {}
        
        scenarios = ['default', '50perc_contaminated', '100perc_contaminated']
        
        for scenario in scenarios:
            bin_col = f"{model_name}_{scenario} Score Bin"
            if bin_col in df.columns:
                bin_counts = df[bin_col].value_counts().to_dict()
                bins[scenario] = bin_counts
        
        return bins
    
    def _compute_ranking_performance(self, df: pd.DataFrame) -> Dict:
        """Compute ranking performance metrics."""
        if 'true_GO_term_sim_percentile' not in df.columns:
            return {}
        
        percentiles = df['true_GO_term_sim_percentile'].dropna()
        
        if len(percentiles) == 0:
            return {}
        
        return {
            'mean_percentile': float(percentiles.mean()),
            'median_percentile': float(percentiles.median()),
            'top_10_percent': len(percentiles[percentiles >= 0.9]),
            'top_25_percent': len(percentiles[percentiles >= 0.75]),
            'bottom_25_percent': len(percentiles[percentiles < 0.25]),
            'total_rankings': len(percentiles)
        }
    
    def extract_model_name(self, filename: str) -> str:
        """Extract model name from filename."""
        if "gemini_pro" in filename:
            return "gemini_pro"
        elif "gpt_4" in filename:
            return "gpt_4"
        elif "gpt_35" in filename or "gpt_3.5" in filename:
            return "gpt_35"
        elif "llama2_70b" in filename:
            return "llama2_70b"
        elif "mixtral_instruct" in filename:
            return "mixtral_instruct"
        else:
            return "unknown"
    
    def get_model_compare_summary(self) -> Dict:
        """Get summary of model comparison data."""
        return {
            'available_models': list(self.available_models),
            'total_model_predictions': len(self.model_predictions),
            'total_similarity_rankings': len(self.similarity_rankings),
            'evaluation_metrics_available': len(self.evaluation_metrics),
            'contamination_results_available': len(self.contamination_results)
        }