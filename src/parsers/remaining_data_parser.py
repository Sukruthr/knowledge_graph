#!/usr/bin/env python3
"""
Remaining Data Files Parser

This parser handles the integration of valuable files from remaining_data_files:
1. GO_BP_20231115.gmt - GO Biological Process gene sets
2. reference_evaluation.tsv - Literature evaluation data  
3. L1000_sep_count_DF.txt - Expression perturbation data
4. all_go_terms_embeddings_dict.pkl - GO term embeddings
5. SupplementTable3_0715.tsv - Supplementary LLM evaluation data

Based on duplication analysis:
- GMT file: 100% new gene-term associations (HIGH VALUE)
- Reference evaluation: Literature support context (HIGH VALUE)  
- L1000 data: Perturbation-expression relationships (HIGH VALUE)
- Embeddings: Computational representations (HIGH VALUE)
- Supplement table: Additional LLM evaluation (MEDIUM VALUE)
"""

import os
import pandas as pd
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RemainingDataParser:
    """Parser for remaining data files with high integration value"""
    
    def __init__(self, data_directory: str = None):
        """
        Initialize the Remaining Data Parser
        
        Args:
            data_directory (str): Path to the data directory containing remaining_data_files
        """
        self.data_directory = data_directory if data_directory else "llm_evaluation_for_gene_set_interpretation/data"
        self.remaining_files_dir = os.path.join(self.data_directory, "remaining_data_files")
        
        # Parsed data storage
        self.gmt_data = {}
        self.reference_evaluation_data = {}
        self.l1000_data = {}
        self.embeddings_data = {}
        self.supplement_table_data = {}
        
        logger.info(f"Initialized RemainingDataParser with directory: {self.data_directory}")
    
    def parse_all_remaining_data(self) -> Dict[str, Dict]:
        """Parse all high-value remaining data files"""
        
        logger.info("Starting comprehensive parsing of remaining data files...")
        
        # Parse each data type
        self._parse_gmt_file()
        self._parse_reference_evaluation()  
        self._parse_l1000_data()
        self._parse_embeddings()
        self._parse_supplement_table()
        
        # Return combined parsed data
        return {
            'gmt_data': self.gmt_data,
            'reference_evaluation_data': self.reference_evaluation_data,
            'l1000_data': self.l1000_data,
            'embeddings_data': self.embeddings_data,
            'supplement_table_data': self.supplement_table_data
        }
    
    def _parse_gmt_file(self):
        """Parse GO_BP_20231115.gmt file - Gene Matrix Transposed format"""
        
        gmt_file = os.path.join(self.remaining_files_dir, "GO_BP_20231115.gmt")
        
        if not os.path.exists(gmt_file):
            logger.warning(f"GMT file not found: {gmt_file}")
            return
        
        logger.info("Parsing GMT file (GO gene sets)...")
        
        gene_sets = []
        go_terms = {}
        genes_to_go_terms = {}
        statistics = {
            'total_gene_sets': 0,
            'total_genes': set(),
            'total_go_terms': set(),
            'gene_term_associations': 0
        }
        
        try:
            with open(gmt_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split('\t')
                    
                    if len(parts) >= 3:
                        go_info = parts[0]  # Description with GO ID
                        url_or_desc = parts[1] if parts[1] else ""  # Usually empty or URL
                        genes = [gene.strip() for gene in parts[2:] if gene.strip()]
                        
                        # Extract GO ID and description
                        go_id = None
                        description = go_info
                        
                        if '(GO:' in go_info:
                            start_idx = go_info.find('(GO:')
                            end_idx = go_info.find(')', start_idx)
                            if end_idx > start_idx:
                                go_id = go_info[start_idx+1:end_idx]
                                description = go_info[:start_idx].strip()
                        
                        if go_id and genes:
                            # Store gene set
                            gene_set = {
                                'go_id': go_id,
                                'description': description,
                                'genes': genes,
                                'gene_count': len(genes),
                                'url_or_desc': url_or_desc
                            }
                            gene_sets.append(gene_set)
                            
                            # Store GO term info
                            go_terms[go_id] = {
                                'description': description,
                                'gene_count': len(genes),
                                'genes': genes
                            }
                            
                            # Store gene to GO term mappings
                            for gene in genes:
                                if gene not in genes_to_go_terms:
                                    genes_to_go_terms[gene] = []
                                genes_to_go_terms[gene].append(go_id)
                                
                                statistics['total_genes'].add(gene)
                            
                            statistics['total_go_terms'].add(go_id)
                            statistics['gene_term_associations'] += len(genes)
                            statistics['total_gene_sets'] += 1
                    
                    # Progress logging for large files
                    if line_num % 1000 == 0:
                        logger.info(f"Processed {line_num} lines of GMT file...")
            
            # Convert sets to counts for statistics
            statistics['total_genes'] = len(statistics['total_genes'])
            statistics['total_go_terms'] = len(statistics['total_go_terms'])
            
            self.gmt_data = {
                'gene_sets': gene_sets,
                'go_terms': go_terms,
                'genes_to_go_terms': genes_to_go_terms,
                'statistics': statistics
            }
            
            logger.info(f"GMT file parsed successfully: {statistics['total_gene_sets']} gene sets, "
                       f"{statistics['total_go_terms']} GO terms, {statistics['total_genes']} genes")
                       
        except Exception as e:
            logger.error(f"Error parsing GMT file: {str(e)}")
            self.gmt_data = {}
    
    def _parse_reference_evaluation(self):
        """Parse reference_evaluation.tsv - Literature evaluation data"""
        
        ref_file = os.path.join(self.remaining_files_dir, "reference_evaluation.tsv")
        
        if not os.path.exists(ref_file):
            logger.warning(f"Reference evaluation file not found: {ref_file}")
            return
        
        logger.info("Parsing reference evaluation data...")
        
        try:
            df = pd.read_csv(ref_file, sep='\t')
            
            references = {}
            gene_sets = {}
            paragraphs = {}
            statistics = {
                'total_references': 0,
                'total_gene_sets': 0,
                'total_paragraphs': 0,
                'datasets': set()
            }
            
            for idx, row in df.iterrows():
                reference = row.get('Reference', '')
                paragraph = row.get('Paragraph', '')
                comment = row.get('Comment', '')
                gene_set_name = row.get('Gene Set Name', '')
                dataset = row.get('Data Set', '')
                
                # Store reference information
                if reference:
                    ref_key = f"ref_{idx}"
                    references[ref_key] = {
                        'reference': reference,
                        'paragraph': paragraph,
                        'comment': comment,
                        'gene_set_name': gene_set_name,
                        'dataset': dataset,
                        'title_supports': row.get('Title Supports', None),
                        'abstract_supports': row.get('Abstract Supports', None)
                    }
                    statistics['total_references'] += 1
                
                # Store gene set evaluation information
                if gene_set_name:
                    if gene_set_name not in gene_sets:
                        gene_sets[gene_set_name] = {
                            'references': [],
                            'datasets': set(),
                            'evaluations': []
                        }
                    
                    gene_sets[gene_set_name]['references'].append(reference)
                    gene_sets[gene_set_name]['datasets'].add(dataset)
                    gene_sets[gene_set_name]['evaluations'].append({
                        'reference': reference,
                        'comment': comment,
                        'title_supports': row.get('Title Supports', None),
                        'abstract_supports': row.get('Abstract Supports', None)
                    })
                
                # Store paragraph information
                if paragraph:
                    para_key = f"para_{idx}"
                    paragraphs[para_key] = {
                        'content': paragraph,
                        'reference': reference,
                        'comment': comment,
                        'gene_set_name': gene_set_name
                    }
                    statistics['total_paragraphs'] += 1
                
                if dataset:
                    statistics['datasets'].add(dataset)
            
            # Convert gene sets datasets from sets to lists
            for gene_set_name in gene_sets:
                gene_sets[gene_set_name]['datasets'] = list(gene_sets[gene_set_name]['datasets'])
            
            statistics['total_gene_sets'] = len(gene_sets)
            statistics['datasets'] = list(statistics['datasets'])
            statistics['total_datasets'] = len(statistics['datasets'])
            
            self.reference_evaluation_data = {
                'references': references,
                'gene_sets': gene_sets,
                'paragraphs': paragraphs,
                'statistics': statistics
            }
            
            logger.info(f"Reference evaluation data parsed: {statistics['total_references']} references, "
                       f"{statistics['total_gene_sets']} gene sets, {statistics['total_datasets']} datasets")
                       
        except Exception as e:
            logger.error(f"Error parsing reference evaluation data: {str(e)}")
            self.reference_evaluation_data = {}
    
    def _parse_l1000_data(self):
        """Parse L1000_sep_count_DF.txt - Expression perturbation data"""
        
        l1000_file = os.path.join(self.remaining_files_dir, "L1000_sep_count_DF.txt")
        
        if not os.path.exists(l1000_file):
            logger.warning(f"L1000 file not found: {l1000_file}")
            return
        
        logger.info("Parsing L1000 perturbation data...")
        
        try:
            df = pd.read_csv(l1000_file, sep='\t')
            
            perturbations = {}
            cell_lines = {}
            reagents = {}
            statistics = {
                'total_perturbations': 0,
                'unique_reagents': set(),
                'unique_cell_lines': set(),
                'unique_durations': set(),
                'unique_dosages': set(),
                'total_gene_sets': 0
            }
            
            for idx, row in df.iterrows():
                reagent = str(row.get('Reagent', ''))
                cell_line = str(row.get('Cellline', ''))
                duration = row.get('duration', None)
                duration_unit = row.get('duration_unit', '')
                dosage = row.get('dosage', None)
                dosage_unit = row.get('dosage_unit', '')
                n_genesets = row.get('n_genesets', 0)
                
                # Create perturbation entry
                pert_key = f"pert_{idx}"
                perturbation = {
                    'reagent': reagent,
                    'cell_line': cell_line,
                    'duration': duration,
                    'duration_unit': duration_unit,
                    'dosage': dosage,
                    'dosage_unit': dosage_unit,
                    'n_genesets': n_genesets,
                    'perturbation_id': pert_key
                }
                perturbations[pert_key] = perturbation
                
                # Track cell line information
                if cell_line not in cell_lines:
                    cell_lines[cell_line] = {
                        'perturbations': [],
                        'reagents': set(),
                        'total_genesets': 0
                    }
                cell_lines[cell_line]['perturbations'].append(pert_key)
                cell_lines[cell_line]['reagents'].add(reagent)
                cell_lines[cell_line]['total_genesets'] += n_genesets if n_genesets else 0
                
                # Track reagent information
                if reagent not in reagents:
                    reagents[reagent] = {
                        'perturbations': [],
                        'cell_lines': set(),
                        'total_genesets': 0
                    }
                reagents[reagent]['perturbations'].append(pert_key)
                reagents[reagent]['cell_lines'].add(cell_line)
                reagents[reagent]['total_genesets'] += n_genesets if n_genesets else 0
                
                # Update statistics
                statistics['unique_reagents'].add(reagent)
                statistics['unique_cell_lines'].add(cell_line)
                if duration:
                    statistics['unique_durations'].add(f"{duration} {duration_unit}")
                if dosage:
                    statistics['unique_dosages'].add(f"{dosage} {dosage_unit}")
                statistics['total_gene_sets'] += n_genesets if n_genesets else 0
                statistics['total_perturbations'] += 1
            
            # Convert sets to lists for serialization
            for cell_line in cell_lines:
                cell_lines[cell_line]['reagents'] = list(cell_lines[cell_line]['reagents'])
            
            for reagent in reagents:
                reagents[reagent]['cell_lines'] = list(reagents[reagent]['cell_lines'])
            
            statistics['unique_reagents'] = len(statistics['unique_reagents'])
            statistics['unique_cell_lines'] = len(statistics['unique_cell_lines'])
            statistics['unique_durations'] = len(statistics['unique_durations'])
            statistics['unique_dosages'] = len(statistics['unique_dosages'])
            
            self.l1000_data = {
                'perturbations': perturbations,
                'cell_lines': cell_lines,
                'reagents': reagents,
                'statistics': statistics
            }
            
            logger.info(f"L1000 data parsed: {statistics['total_perturbations']} perturbations, "
                       f"{statistics['unique_reagents']} reagents, {statistics['unique_cell_lines']} cell lines")
                       
        except Exception as e:
            logger.error(f"Error parsing L1000 data: {str(e)}")
            self.l1000_data = {}
    
    def _parse_embeddings(self):
        """Parse all_go_terms_embeddings_dict.pkl - GO term embeddings"""
        
        embeddings_file = os.path.join(self.remaining_files_dir, "all_go_terms_embeddings_dict.pkl")
        
        if not os.path.exists(embeddings_file):
            logger.warning(f"Embeddings file not found: {embeddings_file}")
            return
        
        logger.info("Parsing GO term embeddings...")
        
        try:
            with open(embeddings_file, 'rb') as f:
                embeddings_dict = pickle.load(f)
            
            embeddings = {}
            statistics = {
                'total_embeddings': 0,
                'embedding_dimension': None,
                'go_terms_with_embeddings': set()
            }
            
            for go_term, embedding in embeddings_dict.items():
                embeddings[go_term] = {
                    'embedding_vector': embedding,
                    'dimension': len(embedding) if hasattr(embedding, '__len__') else None
                }
                
                statistics['go_terms_with_embeddings'].add(go_term)
                statistics['total_embeddings'] += 1
                
                # Set embedding dimension from first entry
                if statistics['embedding_dimension'] is None and hasattr(embedding, '__len__'):
                    statistics['embedding_dimension'] = len(embedding)
            
            statistics['go_terms_with_embeddings'] = len(statistics['go_terms_with_embeddings'])
            
            self.embeddings_data = {
                'embeddings': embeddings,
                'statistics': statistics
            }
            
            logger.info(f"Embeddings parsed: {statistics['total_embeddings']} GO term embeddings, "
                       f"dimension: {statistics['embedding_dimension']}")
                       
        except Exception as e:
            logger.error(f"Error parsing embeddings: {str(e)}")
            self.embeddings_data = {}
    
    def _parse_supplement_table(self):
        """Parse SupplementTable3_0715.tsv - Supplementary LLM evaluation data"""
        
        supp_file = os.path.join(self.remaining_files_dir, "SupplementTable3_0715.tsv")
        
        if not os.path.exists(supp_file):
            logger.warning(f"Supplement table file not found: {supp_file}")
            return
        
        logger.info("Parsing supplementary LLM evaluation data...")
        
        try:
            # Read with error handling for potential encoding issues
            df = pd.read_csv(supp_file, sep='\t', encoding='utf-8', low_memory=False)
            
            evaluations = {}
            gene_sets = {}
            llm_analyses = {}
            statistics = {
                'total_evaluations': 0,
                'total_gene_sets': 0,
                'data_sources': set(),
                'llm_names': set()
            }
            
            for idx, row in df.iterrows():
                source = row.get('Source', '')
                gene_set_name = row.get('GeneSetName', '')
                gene_list = row.get('updated GeneList', '')
                n_genes = row.get('n_Genes', 0)
                llm_name = row.get('LLM Name', '')
                referenced_analysis = row.get('referenced_analysis', '')
                score = row.get('Score', None)
                
                # Create evaluation entry
                eval_key = f"eval_{idx}"
                evaluation = {
                    'source': source,
                    'gene_set_name': gene_set_name,
                    'gene_list': gene_list,
                    'n_genes': n_genes,
                    'llm_name': llm_name,
                    'referenced_analysis': referenced_analysis,
                    'score': score,
                    'evaluation_id': eval_key
                }
                
                # Add all other columns as additional metadata
                for col in df.columns:
                    if col not in ['Source', 'GeneSetName', 'updated GeneList', 'n_Genes', 
                                   'LLM Name', 'referenced_analysis', 'Score']:
                        evaluation[col.lower().replace(' ', '_')] = row.get(col, None)
                
                evaluations[eval_key] = evaluation
                
                # Track gene set information
                if gene_set_name:
                    if gene_set_name not in gene_sets:
                        gene_sets[gene_set_name] = {
                            'evaluations': [],
                            'sources': set(),
                            'llm_names': set(),
                            'total_genes': 0
                        }
                    
                    gene_sets[gene_set_name]['evaluations'].append(eval_key)
                    gene_sets[gene_set_name]['sources'].add(source)
                    gene_sets[gene_set_name]['llm_names'].add(llm_name)
                    if n_genes:
                        gene_sets[gene_set_name]['total_genes'] = max(
                            gene_sets[gene_set_name]['total_genes'], n_genes
                        )
                
                # Track LLM analysis information
                if llm_name and referenced_analysis:
                    if llm_name not in llm_analyses:
                        llm_analyses[llm_name] = {
                            'evaluations': [],
                            'analyses': set(),
                            'gene_sets': set()
                        }
                    
                    llm_analyses[llm_name]['evaluations'].append(eval_key)
                    llm_analyses[llm_name]['analyses'].add(referenced_analysis)
                    llm_analyses[llm_name]['gene_sets'].add(gene_set_name)
                
                # Update statistics
                statistics['data_sources'].add(source)
                statistics['llm_names'].add(llm_name)
                statistics['total_evaluations'] += 1
            
            # Convert sets to lists for serialization
            for gene_set_name in gene_sets:
                gene_sets[gene_set_name]['sources'] = list(gene_sets[gene_set_name]['sources'])
                gene_sets[gene_set_name]['llm_names'] = list(gene_sets[gene_set_name]['llm_names'])
            
            for llm_name in llm_analyses:
                llm_analyses[llm_name]['analyses'] = list(llm_analyses[llm_name]['analyses'])
                llm_analyses[llm_name]['gene_sets'] = list(llm_analyses[llm_name]['gene_sets'])
            
            statistics['total_gene_sets'] = len(gene_sets)
            statistics['data_sources'] = list(statistics['data_sources'])
            statistics['llm_names'] = list(statistics['llm_names'])
            statistics['total_data_sources'] = len(statistics['data_sources'])
            statistics['total_llm_names'] = len(statistics['llm_names'])
            
            self.supplement_table_data = {
                'evaluations': evaluations,
                'gene_sets': gene_sets,
                'llm_analyses': llm_analyses,
                'statistics': statistics
            }
            
            logger.info(f"Supplement table parsed: {statistics['total_evaluations']} evaluations, "
                       f"{statistics['total_gene_sets']} gene sets, {statistics['total_llm_names']} LLMs")
                       
        except Exception as e:
            logger.error(f"Error parsing supplement table: {str(e)}")
            self.supplement_table_data = {}
    
    def get_parsing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive parsing statistics for all data types"""
        
        stats = {
            'gmt_data': self.gmt_data.get('statistics', {}),
            'reference_evaluation_data': self.reference_evaluation_data.get('statistics', {}),
            'l1000_data': self.l1000_data.get('statistics', {}),
            'embeddings_data': self.embeddings_data.get('statistics', {}),
            'supplement_table_data': self.supplement_table_data.get('statistics', {}),
            'overall_summary': {
                'total_data_types_parsed': 0,
                'successfully_parsed': []
            }
        }
        
        # Count successfully parsed data types
        data_types = ['gmt_data', 'reference_evaluation_data', 'l1000_data', 
                     'embeddings_data', 'supplement_table_data']
        
        for data_type in data_types:
            if getattr(self, data_type) and stats[data_type]:
                stats['overall_summary']['total_data_types_parsed'] += 1
                stats['overall_summary']['successfully_parsed'].append(data_type)
        
        return stats

def main():
    """Main function for testing the parser"""
    
    # Initialize parser
    parser = RemainingDataParser()
    
    # Parse all data
    parsed_data = parser.parse_all_remaining_data()
    
    # Get statistics
    stats = parser.get_parsing_statistics()
    
    print("REMAINING DATA FILES PARSING COMPLETE")
    print("=" * 50)
    
    for data_type, data_stats in stats.items():
        if data_type != 'overall_summary' and data_stats:
            print(f"\n{data_type.upper()}:")
            for key, value in data_stats.items():
                print(f"  {key}: {value}")
    
    print(f"\nOVERALL SUMMARY:")
    print(f"  Data types parsed: {stats['overall_summary']['total_data_types_parsed']}")
    print(f"  Successfully parsed: {', '.join(stats['overall_summary']['successfully_parsed'])}")

if __name__ == "__main__":
    main()