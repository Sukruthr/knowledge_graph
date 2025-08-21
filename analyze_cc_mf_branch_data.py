#!/usr/bin/env python3
"""
Analysis script for CC_MF_branch data to understand structure, content, and integration value
for the existing biomedical knowledge graph.
"""

import os
import pandas as pd
import json
from typing import Dict, Any, List, Set
import sys

def analyze_cc_mf_branch_data():
    """Comprehensive analysis of CC_MF_branch data."""
    
    # Data paths
    data_dir = "llm_evaluation_for_gene_set_interpretation/data/GO_term_analysis/CC_MF_branch/"
    
    # File definitions
    files_info = {
        'cc_go_terms': 'CC_go_terms.csv',
        'mf_go_terms': 'MF_go_terms.csv', 
        'cc_1000_selected': 'CC_1000_selected_go_terms.csv',
        'mf_1000_selected': 'MF_1000_selected_go_terms.csv',
        'llm_processed_cc': 'LLM_processed_selected_1000_go_CCterms.tsv',
        'llm_processed_mf': 'LLM_processed_selected_1000_go_MFterms.tsv',
        'sim_rank_cc': 'sim_rank_LLM_processed_selected_1000_go_CCterms.tsv',
        'sim_rank_mf': 'sim_rank_LLM_processed_selected_1000_go_MFterms.tsv'
    }
    
    analysis_results = {}
    
    print("🔍 CC_MF_BRANCH DATA ANALYSIS")
    print("=" * 60)
    
    # 1. Basic file structure analysis
    print("\n📊 FILE STRUCTURE ANALYSIS:")
    for key, filename in files_info.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            try:
                # Determine delimiter
                delimiter = '\t' if filename.endswith('.tsv') else ','
                df = pd.read_csv(filepath, delimiter=delimiter)
                
                print(f"  {key}: {df.shape[0]} rows, {df.shape[1]} columns")
                print(f"    Columns: {list(df.columns)}")
                
                analysis_results[key] = {
                    'rows': df.shape[0],
                    'columns': df.shape[1],
                    'column_names': list(df.columns),
                    'file_path': filepath
                }
                
            except Exception as e:
                print(f"  {key}: ERROR reading file - {e}")
                analysis_results[key] = {'error': str(e)}
        else:
            print(f"  {key}: FILE NOT FOUND")
    
    # 2. Detailed content analysis for key files
    print("\n🧬 CONTENT ANALYSIS:")
    
    # Analyze GO terms structure
    for namespace, file_key in [('CC', 'cc_go_terms'), ('MF', 'mf_go_terms')]:
        if file_key in analysis_results and 'error' not in analysis_results[file_key]:
            filepath = analysis_results[file_key]['file_path']
            df = pd.read_csv(filepath)
            
            print(f"\n  {namespace} GO Terms:")
            print(f"    Total terms: {len(df)}")
            print(f"    Gene count range: {df['Gene_Count'].min()}-{df['Gene_Count'].max()}")
            print(f"    Median gene count: {df['Gene_Count'].median()}")
            
            # Extract unique genes
            all_genes = set()
            for genes_str in df['Genes'].dropna():
                genes = [g.strip() for g in str(genes_str).split()]
                all_genes.update(genes)
            
            print(f"    Unique genes: {len(all_genes)}")
            
            # Sample terms
            print(f"    Sample terms:")
            for i, row in df.head(3).iterrows():
                print(f"      {row['GO']}: {row['Term_Description']} ({row['Gene_Count']} genes)")
            
            analysis_results[f"{namespace.lower()}_genes"] = all_genes
    
    # 3. Analyze LLM processed data structure 
    print("\n🤖 LLM PROCESSED DATA ANALYSIS:")
    
    for namespace, file_key in [('CC', 'llm_processed_cc'), ('MF', 'llm_processed_mf')]:
        if file_key in analysis_results and 'error' not in analysis_results[file_key]:
            filepath = analysis_results[file_key]['file_path']
            df = pd.read_csv(filepath, delimiter='\t')
            
            print(f"\n  {namespace} LLM Processed:")
            print(f"    Shape: {df.shape}")
            
            # Key columns analysis
            llm_columns = [col for col in df.columns if 'gpt_4' in col.lower()]
            print(f"    LLM columns: {llm_columns}")
            
            if 'gpt_4_default Score' in df.columns:
                scores = df['gpt_4_default Score'].dropna()
                print(f"    GPT-4 scores: {len(scores)} entries")
                print(f"      Range: {scores.min():.3f} - {scores.max():.3f}")
                print(f"      Mean: {scores.mean():.3f}")
            
            # Sample analysis content
            if 'gpt_4_default Analysis' in df.columns:
                analysis_lengths = df['gpt_4_default Analysis'].dropna().str.len()
                print(f"    Analysis text lengths: {analysis_lengths.min()}-{analysis_lengths.max()} chars")
                print(f"      Mean length: {analysis_lengths.mean():.0f} chars")
    
    # 4. Analyze similarity ranking data
    print("\n📈 SIMILARITY RANKING ANALYSIS:")
    
    for namespace, file_key in [('CC', 'sim_rank_cc'), ('MF', 'sim_rank_mf')]:
        if file_key in analysis_results and 'error' not in analysis_results[file_key]:
            filepath = analysis_results[file_key]['file_path']
            df = pd.read_csv(filepath, delimiter='\t')
            
            print(f"\n  {namespace} Similarity Rankings:")
            
            # Key similarity metrics
            sim_columns = [col for col in df.columns if 'sim' in col.lower()]
            print(f"    Similarity columns: {sim_columns}")
            
            if 'sim_rank' in df.columns:
                ranks = df['sim_rank'].dropna()
                print(f"    Similarity ranks: {len(ranks)} entries")
                print(f"      Range: {ranks.min()}-{ranks.max()}")
                print(f"      Mean rank: {ranks.mean():.1f}")
            
            if 'true_GO_term_sim_percentile' in df.columns:
                percentiles = df['true_GO_term_sim_percentile'].dropna()
                print(f"    GO term sim percentiles: {len(percentiles)} entries")
                print(f"      Range: {percentiles.min():.3f}-{percentiles.max():.3f}")
                print(f"      Mean: {percentiles.mean():.3f}")
    
    # 5. Cross-data integration analysis
    print("\n🔗 CROSS-DATA INTEGRATION ANALYSIS:")
    
    # Gene overlap analysis
    if 'cc_genes' in analysis_results and 'mf_genes' in analysis_results:
        cc_genes = analysis_results['cc_genes']
        mf_genes = analysis_results['mf_genes']
        
        overlap = cc_genes.intersection(mf_genes)
        union = cc_genes.union(mf_genes)
        
        print(f"  Gene overlap between CC and MF:")
        print(f"    CC unique genes: {len(cc_genes)}")
        print(f"    MF unique genes: {len(mf_genes)}")
        print(f"    Overlapping genes: {len(overlap)}")
        print(f"    Total unique genes: {len(union)}")
        print(f"    Overlap ratio: {len(overlap)/len(union):.3f}")
        
        analysis_results['gene_overlap'] = {
            'cc_genes': len(cc_genes),
            'mf_genes': len(mf_genes),
            'overlap': len(overlap),
            'total_unique': len(union),
            'overlap_ratio': len(overlap)/len(union)
        }
    
    return analysis_results

def assess_integration_value(analysis_results: Dict) -> Dict[str, Any]:
    """Assess the integration value of CC_MF_branch data for the existing KG."""
    
    print("\n🎯 INTEGRATION VALUE ASSESSMENT:")
    print("=" * 60)
    
    value_assessment = {
        'integration_score': 0,
        'max_score': 100,
        'factors': {}
    }
    
    # Factor 1: Data completeness and quality (25 points)
    completeness_score = 0
    expected_files = 8
    actual_files = len([k for k in analysis_results if 'error' not in analysis_results.get(k, {})])
    completeness_score = (actual_files / expected_files) * 25
    
    value_assessment['factors']['data_completeness'] = {
        'score': completeness_score,
        'max': 25,
        'details': f'{actual_files}/{expected_files} files successfully loaded'
    }
    
    # Factor 2: LLM analysis depth (25 points)
    llm_score = 0
    has_analysis = any('llm_processed' in k for k in analysis_results)
    has_scoring = True  # Assuming from structure
    has_similarity = any('sim_rank' in k for k in analysis_results)
    
    if has_analysis: llm_score += 10
    if has_scoring: llm_score += 10
    if has_similarity: llm_score += 5
    
    value_assessment['factors']['llm_analysis_depth'] = {
        'score': llm_score,
        'max': 25,
        'details': f'Analysis:{has_analysis}, Scoring:{has_scoring}, Similarity:{has_similarity}'
    }
    
    # Factor 3: Gene coverage and diversity (20 points)
    gene_coverage_score = 0
    if 'gene_overlap' in analysis_results:
        total_genes = analysis_results['gene_overlap']['total_unique']
        # Assume >1000 genes is excellent coverage
        gene_coverage_score = min(20, (total_genes / 1000) * 20)
    
    value_assessment['factors']['gene_coverage'] = {
        'score': gene_coverage_score,
        'max': 20,
        'details': f"Total unique genes: {analysis_results.get('gene_overlap', {}).get('total_unique', 'unknown')}"
    }
    
    # Factor 4: Complementarity to existing data (15 points)
    # CC and MF are different GO namespaces from BP, adding significant value
    complementarity_score = 15  # Full points for CC/MF complementing BP
    
    value_assessment['factors']['complementarity'] = {
        'score': complementarity_score,
        'max': 15,
        'details': 'CC/MF namespaces complement existing GO_BP data'
    }
    
    # Factor 5: Advanced analytics features (15 points)
    analytics_score = 0
    has_confidence_scores = True  # GPT-4 scores visible
    has_ranking_system = any('sim_rank' in k for k in analysis_results)
    has_comparative_analysis = True  # Multiple models/scenarios
    
    if has_confidence_scores: analytics_score += 5
    if has_ranking_system: analytics_score += 5
    if has_comparative_analysis: analytics_score += 5
    
    value_assessment['factors']['advanced_analytics'] = {
        'score': analytics_score,
        'max': 15,
        'details': f'Confidence:{has_confidence_scores}, Ranking:{has_ranking_system}, Comparative:{has_comparative_analysis}'
    }
    
    # Calculate total score
    total_score = sum(factor['score'] for factor in value_assessment['factors'].values())
    value_assessment['integration_score'] = total_score
    
    # Integration recommendation
    if total_score >= 80:
        recommendation = "HIGHLY RECOMMENDED - Exceptional value for KG integration"
    elif total_score >= 60:
        recommendation = "RECOMMENDED - Significant value for KG integration"  
    elif total_score >= 40:
        recommendation = "CONDITIONALLY RECOMMENDED - Moderate value, consider selective integration"
    else:
        recommendation = "NOT RECOMMENDED - Limited value for KG integration"
    
    value_assessment['recommendation'] = recommendation
    
    # Print assessment
    print(f"\nINTEGRATION SCORE: {total_score:.1f}/100")
    print(f"RECOMMENDATION: {recommendation}")
    print("\nFACTOR BREAKDOWN:")
    for factor, details in value_assessment['factors'].items():
        print(f"  {factor}: {details['score']:.1f}/{details['max']} - {details['details']}")
    
    return value_assessment

def generate_integration_plan(analysis_results: Dict, value_assessment: Dict) -> Dict:
    """Generate detailed integration plan if data adds value."""
    
    if value_assessment['integration_score'] < 40:
        print("\n❌ INTEGRATION NOT RECOMMENDED - Score too low")
        return {'integrate': False, 'reason': 'Insufficient value score'}
    
    print(f"\n✅ INTEGRATION RECOMMENDED - Score: {value_assessment['integration_score']:.1f}/100")
    print("\n📋 INTEGRATION PLAN:")
    print("=" * 60)
    
    integration_plan = {
        'integrate': True,
        'new_parser_class': 'CCMFBranchParser',
        'integration_points': [],
        'new_query_methods': [],
        'testing_requirements': []
    }
    
    # 1. Parser integration
    print("\n1. PARSER INTEGRATION:")
    print("   - Create CCMFBranchParser class")
    print("   - Parse CC and MF GO terms with gene associations")
    print("   - Parse LLM analysis data (GPT-4 interpretations and scores)")
    print("   - Parse similarity ranking data")
    print("   - Integrate with CombinedBiomedicalParser")
    
    integration_plan['integration_points'] = [
        'cc_mf_go_terms',
        'llm_interpretations', 
        'similarity_rankings',
        'confidence_scores'
    ]
    
    # 2. KG builder integration
    print("\n2. KNOWLEDGE GRAPH INTEGRATION:")
    print("   - Add CC and MF GO term nodes")
    print("   - Add LLM interpretation nodes")
    print("   - Add gene-GO term associations (CC/MF)")
    print("   - Add interpretation-confidence edges")
    print("   - Add similarity ranking relationships")
    
    # 3. New query capabilities
    print("\n3. NEW QUERY METHODS:")
    query_methods = [
        'query_cc_mf_terms()',
        'query_llm_interpretations()',
        'query_go_similarity_rankings()',
        'query_gene_cc_mf_profile()',
        'query_interpretation_confidence()'
    ]
    
    for method in query_methods:
        print(f"   - {method}")
    
    integration_plan['new_query_methods'] = query_methods
    
    # 4. Testing requirements
    print("\n4. TESTING REQUIREMENTS:")
    testing_reqs = [
        'Data parsing validation',
        'KG construction integrity',
        'New query method functionality',
        'Cross-namespace integration',
        'Performance impact assessment',
        'Regression testing for existing functionality'
    ]
    
    for req in testing_reqs:
        print(f"   - {req}")
    
    integration_plan['testing_requirements'] = testing_reqs
    
    return integration_plan

def main():
    """Main analysis function."""
    print("🚀 Starting CC_MF_Branch data analysis...")
    
    # Run comprehensive analysis
    analysis_results = analyze_cc_mf_branch_data()
    
    # Assess integration value
    value_assessment = assess_integration_value(analysis_results)
    
    # Generate integration plan
    integration_plan = generate_integration_plan(analysis_results, value_assessment)
    
    # Save results
    final_results = {
        'analysis_results': analysis_results,
        'value_assessment': value_assessment,
        'integration_plan': integration_plan,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Write results to file
    with open('cc_mf_branch_analysis_results.json', 'w') as f:
        # Convert sets to lists for JSON serialization
        def convert_sets(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_sets(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets(item) for item in obj]
            return obj
        
        json.dump(convert_sets(final_results), f, indent=2, default=str)
    
    print(f"\n📄 Results saved to cc_mf_branch_analysis_results.json")
    
    return integration_plan['integrate']

if __name__ == "__main__":
    should_integrate = main()
    sys.exit(0 if should_integrate else 1)