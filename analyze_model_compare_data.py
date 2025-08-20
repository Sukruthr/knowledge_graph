#!/usr/bin/env python3
"""
Analysis of model comparison data to determine integration value for the knowledge graph.
"""

import sys
import pandas as pd
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_model_compare_data():
    """Analyze the model comparison data structure and content."""
    logger.info("="*70)
    logger.info("ANALYZING MODEL COMPARISON DATA")
    logger.info("="*70)
    
    data_dir = Path("llm_evaluation_for_gene_set_interpretation/data/GO_term_analysis/model_compare")
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return False
    
    # Get all files
    files = list(data_dir.glob("*.tsv"))
    logger.info(f"Found {len(files)} TSV files")
    
    analysis_results = {}
    
    # Categorize files
    llm_files = [f for f in files if f.name.startswith("LLM_processed_model_compare")]
    sim_rank_files = [f for f in files if f.name.startswith("sim_rank_LLM_processed")]
    void_files = [f for f in files if f.name.startswith("void_")]
    
    logger.info(f"LLM processed files: {len(llm_files)}")
    logger.info(f"Similarity ranked files: {len(sim_rank_files)}")
    logger.info(f"Void files: {len(void_files)}")
    
    # Analyze each category
    analysis_results['llm_files'] = analyze_llm_files(llm_files)
    analysis_results['sim_rank_files'] = analyze_sim_rank_files(sim_rank_files)
    analysis_results['models_covered'] = extract_models_from_filenames(llm_files)
    
    # Determine integration value
    integration_value = assess_integration_value(analysis_results)
    
    return analysis_results, integration_value

def analyze_llm_files(llm_files):
    """Analyze LLM processed model comparison files."""
    logger.info("\n📊 ANALYZING LLM PROCESSED FILES")
    logger.info("-" * 50)
    
    results = {}
    
    for file_path in llm_files[:3]:  # Sample first 3 files
        model_name = extract_model_name(file_path.name)
        logger.info(f"\n🔍 Analyzing {model_name} file...")
        
        try:
            df = pd.read_csv(file_path, sep='\t')
            
            file_analysis = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'sample_data': {}
            }
            
            logger.info(f"  Rows: {len(df):,}")
            logger.info(f"  Columns: {len(df.columns)}")
            logger.info(f"  Column names: {list(df.columns)}")
            
            # Sample data from each column
            for col in df.columns:
                if len(df) > 0:
                    sample_value = str(df[col].iloc[0])
                    file_analysis['sample_data'][col] = sample_value[:100] + "..." if len(sample_value) > 100 else sample_value
                    logger.info(f"  Sample {col}: {file_analysis['sample_data'][col]}")
            
            # Check for specific valuable columns
            valuable_columns = check_valuable_columns(df)
            file_analysis['valuable_columns'] = valuable_columns
            
            results[model_name] = file_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results

def analyze_sim_rank_files(sim_rank_files):
    """Analyze similarity ranked files."""
    logger.info("\n📈 ANALYZING SIMILARITY RANKED FILES")
    logger.info("-" * 50)
    
    results = {}
    
    for file_path in sim_rank_files[:2]:  # Sample first 2 files
        model_name = extract_model_name(file_path.name)
        logger.info(f"\n🔍 Analyzing {model_name} similarity rankings...")
        
        try:
            df = pd.read_csv(file_path, sep='\t')
            
            file_analysis = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'sample_data': {}
            }
            
            logger.info(f"  Rows: {len(df):,}")
            logger.info(f"  Columns: {len(df.columns)}")
            logger.info(f"  Column names: {list(df.columns)}")
            
            # Sample data from each column
            for col in df.columns:
                if len(df) > 0:
                    sample_value = str(df[col].iloc[0])
                    file_analysis['sample_data'][col] = sample_value[:100] + "..." if len(sample_value) > 100 else sample_value
                    logger.info(f"  Sample {col}: {file_analysis['sample_data'][col]}")
            
            results[model_name] = file_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results

def extract_model_name(filename):
    """Extract model name from filename."""
    if "gemini_pro" in filename:
        return "gemini_pro"
    elif "gpt_4" in filename:
        return "gpt_4"
    elif "gpt_35" in filename:
        return "gpt_35"
    elif "llama2_70b" in filename:
        return "llama2_70b"
    elif "mixtral_instruct" in filename:
        return "mixtral_instruct"
    elif "mixtral_latest" in filename:
        return "mixtral_latest"
    else:
        return "unknown"

def extract_models_from_filenames(files):
    """Extract all model names from filenames."""
    models = set()
    for file_path in files:
        model = extract_model_name(file_path.name)
        if model != "unknown":
            models.add(model)
    return list(models)

def check_valuable_columns(df):
    """Check for columns that would add value to our knowledge graph."""
    valuable_indicators = {
        'model_predictions': ['prediction', 'llm_response', 'generated', 'output'],
        'confidence_scores': ['confidence', 'score', 'probability', 'certainty'],
        'go_terms': ['go_id', 'go_term', 'ontology', 'term'],
        'gene_sets': ['gene_set', 'cluster', 'genes', 'gene_list'],
        'evaluation_metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
        'similarities': ['similarity', 'distance', 'cosine', 'jaccard'],
        'rankings': ['rank', 'ranking', 'position', 'order']
    }
    
    found_categories = {}
    
    for category, indicators in valuable_indicators.items():
        matching_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in indicators):
                matching_columns.append(col)
        
        if matching_columns:
            found_categories[category] = matching_columns
    
    return found_categories

def assess_integration_value(analysis_results):
    """Assess whether this data adds significant value to our knowledge graph."""
    logger.info("\n💡 ASSESSING INTEGRATION VALUE")
    logger.info("-" * 50)
    
    value_score = 0
    reasons = []
    
    # Check if we have multiple model comparisons
    models = analysis_results.get('models_covered', [])
    if len(models) >= 3:
        value_score += 20
        reasons.append(f"Multiple model comparisons ({len(models)} models)")
    
    # Check for valuable data types in LLM files
    llm_data = analysis_results.get('llm_files', {})
    for model, data in llm_data.items():
        if 'valuable_columns' in data:
            valuable_cols = data['valuable_columns']
            
            if 'model_predictions' in valuable_cols:
                value_score += 15
                reasons.append("Model predictions available")
            
            if 'confidence_scores' in valuable_cols:
                value_score += 15
                reasons.append("Confidence scores available")
            
            if 'evaluation_metrics' in valuable_cols:
                value_score += 10
                reasons.append("Evaluation metrics available")
            
            if 'go_terms' in valuable_cols:
                value_score += 10
                reasons.append("GO term data available")
    
    # Check for similarity rankings
    sim_data = analysis_results.get('sim_rank_files', {})
    if sim_data:
        value_score += 10
        reasons.append("Similarity rankings available")
    
    # Determine recommendation
    if value_score >= 40:
        recommendation = "HIGH VALUE - Recommend integration"
    elif value_score >= 25:
        recommendation = "MODERATE VALUE - Consider integration"
    elif value_score >= 10:
        recommendation = "LOW VALUE - Optional integration"
    else:
        recommendation = "MINIMAL VALUE - Skip integration"
    
    logger.info(f"Value Score: {value_score}/100")
    logger.info(f"Recommendation: {recommendation}")
    logger.info("Reasons:")
    for reason in reasons:
        logger.info(f"  + {reason}")
    
    return {
        'score': value_score,
        'recommendation': recommendation,
        'reasons': reasons,
        'should_integrate': value_score >= 25
    }

def generate_detailed_analysis_report(analysis_results, integration_value):
    """Generate a detailed analysis report."""
    logger.info("\n📋 DETAILED ANALYSIS REPORT")
    logger.info("=" * 50)
    
    models = analysis_results.get('models_covered', [])
    logger.info(f"\n🤖 MODELS COVERED: {', '.join(models)}")
    
    logger.info(f"\n📊 DATA STRUCTURE ANALYSIS:")
    llm_data = analysis_results.get('llm_files', {})
    for model, data in llm_data.items():
        if 'error' not in data:
            logger.info(f"  {model}: {data['rows']:,} rows × {data['columns']} columns")
    
    logger.info(f"\n🔍 POTENTIAL VALUE-ADD FEATURES:")
    found_features = set()
    for model, data in llm_data.items():
        if 'valuable_columns' in data:
            for category, columns in data['valuable_columns'].items():
                found_features.add(category)
                logger.info(f"  {category}: {', '.join(columns)}")
    
    logger.info(f"\n💰 INTEGRATION ASSESSMENT:")
    logger.info(f"  Score: {integration_value['score']}/100")
    logger.info(f"  Decision: {integration_value['recommendation']}")
    logger.info(f"  Should integrate: {integration_value['should_integrate']}")
    
    return integration_value['should_integrate']

def main():
    """Main analysis function."""
    logger.info("🔍 STARTING MODEL COMPARISON DATA ANALYSIS")
    
    try:
        analysis_results, integration_value = analyze_model_compare_data()
        should_integrate = generate_detailed_analysis_report(analysis_results, integration_value)
        
        # Save analysis results
        output_file = "model_compare_analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                'analysis_results': analysis_results,
                'integration_value': integration_value,
                'should_integrate': should_integrate
            }, f, indent=2, default=str)
        
        logger.info(f"\n💾 Analysis results saved to: {output_file}")
        
        if should_integrate:
            logger.info("\n✅ RECOMMENDATION: PROCEED WITH INTEGRATION")
            logger.info("This data appears to add significant value to the knowledge graph")
        else:
            logger.info("\n❌ RECOMMENDATION: SKIP INTEGRATION")
            logger.info("This data does not appear to add sufficient value")
        
        return should_integrate, analysis_results, integration_value
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

if __name__ == "__main__":
    should_integrate, analysis_results, integration_value = main()