#!/usr/bin/env python3
"""
Phase 1: Build and Persist Complete Knowledge Graph

This script builds the complete biomedical knowledge graph once and saves it
in multiple optimized formats for reuse in all subsequent QC tests.

Key Benefits:
- One-time construction cost (~37 seconds)
- All QC tests use pre-built graph (no rebuild needed) 
- Multiple persistence formats for different use cases
- Integrity validation of save/load operations
"""

import sys
import os
import time
import pickle
import logging
import json
import traceback
from datetime import datetime
from pathlib import Path
import psutil
import networkx as nx

# Add project root to path
sys.path.append('/home/mreddy1/knowledge_graph/src')

# Import the comprehensive biomedical knowledge graph
from kg_builders import ComprehensiveBiomedicalKnowledgeGraph

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/mreddy1/knowledge_graph/quality_control/1_build_and_save_kg/build_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    """Builds and persists the complete biomedical knowledge graph."""
    
    def __init__(self, data_path='/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data'):
        self.data_path = data_path
        self.output_dir = '/home/mreddy1/knowledge_graph/quality_control/1_build_and_save_kg/saved_graphs'
        self.kg = None
        self.build_metrics = {}
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def monitor_memory_usage(self):
        """Monitor current memory usage."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            'percent': process.memory_percent()
        }
    
    def build_complete_knowledge_graph(self):
        """Build the complete biomedical knowledge graph with comprehensive monitoring."""
        logger.info("🚀 STARTING COMPLETE KNOWLEDGE GRAPH CONSTRUCTION")
        logger.info("=" * 80)
        
        start_time = time.time()
        initial_memory = self.monitor_memory_usage()
        
        try:
            # Initialize the knowledge graph
            logger.info("Initializing ComprehensiveBiomedicalKnowledgeGraph...")
            self.kg = ComprehensiveBiomedicalKnowledgeGraph()
            
            init_memory = self.monitor_memory_usage()
            logger.info(f"Memory after initialization: {init_memory['rss_mb']:.1f} MB")
            
            # Load data
            logger.info(f"Loading data from: {self.data_path}")
            data_load_start = time.time()
            self.kg.load_data(self.data_path)
            data_load_time = time.time() - data_load_start
            
            load_memory = self.monitor_memory_usage()
            logger.info(f"Data loading completed in {data_load_time:.2f} seconds")
            logger.info(f"Memory after data loading: {load_memory['rss_mb']:.1f} MB")
            
            # Build comprehensive graph
            logger.info("Building comprehensive biomedical knowledge graph...")
            graph_build_start = time.time()
            self.kg.build_comprehensive_graph()
            graph_build_time = time.time() - graph_build_start
            
            build_memory = self.monitor_memory_usage()
            logger.info(f"Graph construction completed in {graph_build_time:.2f} seconds")
            logger.info(f"Memory after graph building: {build_memory['rss_mb']:.1f} MB")
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Collect build metrics
            self.build_metrics = {
                'build_timestamp': datetime.now().isoformat(),
                'total_build_time_seconds': total_time,
                'data_loading_time_seconds': data_load_time,
                'graph_construction_time_seconds': graph_build_time,
                'memory_usage': {
                    'initial_mb': initial_memory['rss_mb'],
                    'after_init_mb': init_memory['rss_mb'],
                    'after_loading_mb': load_memory['rss_mb'],
                    'after_building_mb': build_memory['rss_mb'],
                    'peak_mb': build_memory['rss_mb']
                },
                'graph_statistics': self.get_graph_statistics()
            }
            
            logger.info("✅ KNOWLEDGE GRAPH CONSTRUCTION COMPLETED")
            logger.info(f"Total construction time: {total_time:.2f} seconds")
            logger.info(f"Peak memory usage: {build_memory['rss_mb']:.1f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Knowledge graph construction failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def get_graph_statistics(self):
        """Get comprehensive statistics about the built knowledge graph."""
        if not self.kg or not hasattr(self.kg, 'graph'):
            return {}
        
        try:
            graph = self.kg.graph
            stats = self.kg.get_comprehensive_stats()
            
            return {
                'total_nodes': graph.number_of_nodes(),
                'total_edges': graph.number_of_edges(),
                'node_types': stats.get('node_types', {}),
                'edge_types': stats.get('edge_types', {}),
                'connected_components': nx.number_connected_components(graph.to_undirected()) if graph.is_directed() else nx.number_connected_components(graph),
                'density': nx.density(graph),
                'is_directed': graph.is_directed()
            }
        except Exception as e:
            logger.warning(f"Could not collect graph statistics: {e}")
            return {}
    
    def save_knowledge_graph(self):
        """Save the knowledge graph in multiple formats for different use cases."""
        if not self.kg:
            logger.error("No knowledge graph to save!")
            return False
        
        logger.info("💾 SAVING KNOWLEDGE GRAPH IN MULTIPLE FORMATS")
        logger.info("=" * 60)
        
        save_results = {
            'timestamp': datetime.now().isoformat(),
            'formats_saved': [],
            'save_times': {},
            'file_sizes': {}
        }
        
        try:
            # 1. Save complete KG object as pickle (fastest loading)
            logger.info("Saving complete KG object as pickle...")
            pickle_start = time.time()
            pickle_path = os.path.join(self.output_dir, 'complete_biomedical_kg.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(self.kg, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle_time = time.time() - pickle_start
            
            save_results['formats_saved'].append('pickle')
            save_results['save_times']['pickle'] = pickle_time
            save_results['file_sizes']['pickle_mb'] = os.path.getsize(pickle_path) / 1024 / 1024
            logger.info(f"✅ Pickle saved in {pickle_time:.2f}s ({save_results['file_sizes']['pickle_mb']:.1f} MB)")
            
            # 2. Save NetworkX graph separately (for graph analysis)
            logger.info("Saving NetworkX graph...")
            networkx_start = time.time()
            networkx_path = os.path.join(self.output_dir, 'biomedical_graph.gpickle')
            # Use standard pickle instead of deprecated nx.write_gpickle
            with open(networkx_path, 'wb') as f:
                pickle.dump(self.kg.graph, f)
            networkx_time = time.time() - networkx_start
            
            save_results['formats_saved'].append('networkx')
            save_results['save_times']['networkx'] = networkx_time
            save_results['file_sizes']['networkx_mb'] = os.path.getsize(networkx_path) / 1024 / 1024
            logger.info(f"✅ NetworkX graph saved in {networkx_time:.2f}s ({save_results['file_sizes']['networkx_mb']:.1f} MB)")
            
            # 3. Save graph statistics and metadata
            logger.info("Saving graph statistics and metadata...")
            stats_path = os.path.join(self.output_dir, 'graph_statistics.json')
            with open(stats_path, 'w') as f:
                json.dump(self.build_metrics, f, indent=2)
            
            save_results['formats_saved'].append('statistics')
            logger.info("✅ Statistics and metadata saved")
            
            # 4. Export for Neo4j (optional - generates Cypher statements)
            logger.info("Generating Neo4j export commands...")
            neo4j_start = time.time()
            neo4j_path = os.path.join(self.output_dir, 'neo4j_import_commands.cypher')
            self.generate_neo4j_export(neo4j_path)
            neo4j_time = time.time() - neo4j_start
            
            save_results['formats_saved'].append('neo4j_cypher')
            save_results['save_times']['neo4j_export'] = neo4j_time
            logger.info(f"✅ Neo4j export commands generated in {neo4j_time:.2f}s")
            
            # Save the save results
            results_path = os.path.join(self.output_dir, 'save_results.json')
            with open(results_path, 'w') as f:
                json.dump(save_results, f, indent=2)
            
            logger.info("✅ ALL FORMATS SAVED SUCCESSFULLY")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving knowledge graph: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def generate_neo4j_export(self, output_path, sample_size=1000):
        """Generate Neo4j Cypher commands for import (sample for demonstration)."""
        try:
            graph = self.kg.graph
            
            with open(output_path, 'w') as f:
                f.write("// Neo4j Import Commands for Biomedical Knowledge Graph\n")
                f.write("// Generated automatically - sample data for demonstration\n")
                f.write("// Full import would require custom processing of all node/edge types\n\n")
                
                # Sample nodes
                f.write("// Sample Node Creation (first 100 nodes)\n")
                node_count = 0
                for node, data in graph.nodes(data=True):
                    if node_count >= 100:
                        break
                    node_type = data.get('type', 'Unknown')
                    node_id = str(node).replace("'", "\\'")
                    f.write(f"CREATE (n{node_count}:{node_type} {{id: '{node_id}'}});\n")
                    node_count += 1
                
                f.write("\n// Sample Edge Creation (first 100 edges)\n")
                edge_count = 0
                for source, target, data in graph.edges(data=True):
                    if edge_count >= 100:
                        break
                    edge_type = data.get('type', 'RELATED_TO')
                    source_id = str(source).replace("'", "\\'")
                    target_id = str(target).replace("'", "\\'")
                    f.write(f"MATCH (s {{id: '{source_id}'}}), (t {{id: '{target_id}'}}) CREATE (s)-[:{edge_type}]->(t);\n")
                    edge_count += 1
                
                f.write(f"\n// Total nodes in full graph: {graph.number_of_nodes()}\n")
                f.write(f"// Total edges in full graph: {graph.number_of_edges()}\n")
                f.write("// This is a sample export. Full production export requires custom processing.\n")
                
        except Exception as e:
            logger.warning(f"Neo4j export generation failed: {e}")
    
    def validate_saved_graphs(self):
        """Validate the integrity of saved graph files."""
        logger.info("🔍 VALIDATING SAVED GRAPH INTEGRITY")
        logger.info("=" * 50)
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'validations': {}
        }
        
        try:
            # Validate pickle file
            logger.info("Validating pickle file integrity...")
            pickle_path = os.path.join(self.output_dir, 'complete_biomedical_kg.pkl')
            if os.path.exists(pickle_path):
                load_start = time.time()
                with open(pickle_path, 'rb') as f:
                    loaded_kg = pickle.load(f)
                load_time = time.time() - load_start
                
                # Basic integrity checks
                original_nodes = self.kg.graph.number_of_nodes()
                loaded_nodes = loaded_kg.graph.number_of_nodes()
                original_edges = self.kg.graph.number_of_edges()
                loaded_edges = loaded_kg.graph.number_of_edges()
                
                validation_results['validations']['pickle'] = {
                    'exists': True,
                    'load_time_seconds': load_time,
                    'nodes_match': original_nodes == loaded_nodes,
                    'edges_match': original_edges == loaded_edges,
                    'original_nodes': original_nodes,
                    'loaded_nodes': loaded_nodes,
                    'original_edges': original_edges,
                    'loaded_edges': loaded_edges
                }
                
                logger.info(f"✅ Pickle validation: {load_time:.2f}s load time")
                logger.info(f"   Nodes match: {original_nodes == loaded_nodes} ({original_nodes} vs {loaded_nodes})")
                logger.info(f"   Edges match: {original_edges == loaded_edges} ({original_edges} vs {loaded_edges})")
            else:
                validation_results['validations']['pickle'] = {'exists': False}
                logger.warning("❌ Pickle file not found")
            
            # Validate NetworkX graph
            logger.info("Validating NetworkX graph...")
            networkx_path = os.path.join(self.output_dir, 'biomedical_graph.gpickle')
            if os.path.exists(networkx_path):
                load_start = time.time()
                # Use standard pickle instead of deprecated nx.read_gpickle
                with open(networkx_path, 'rb') as f:
                    loaded_graph = pickle.load(f)
                load_time = time.time() - load_start
                
                validation_results['validations']['networkx'] = {
                    'exists': True,
                    'load_time_seconds': load_time,
                    'nodes_match': self.kg.graph.number_of_nodes() == loaded_graph.number_of_nodes(),
                    'edges_match': self.kg.graph.number_of_edges() == loaded_graph.number_of_edges()
                }
                
                logger.info(f"✅ NetworkX validation: {load_time:.2f}s load time")
                logger.info(f"   Structure matches: {validation_results['validations']['networkx']['nodes_match'] and validation_results['validations']['networkx']['edges_match']}")
            else:
                validation_results['validations']['networkx'] = {'exists': False}
                logger.warning("❌ NetworkX file not found")
            
            # Save validation results
            validation_path = os.path.join(self.output_dir, 'validation_results.json')
            with open(validation_path, 'w') as f:
                json.dump(validation_results, f, indent=2)
            
            logger.info("✅ VALIDATION COMPLETED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

def main():
    """Main execution function."""
    logger.info("🎯 COMPREHENSIVE BIOMEDICAL KNOWLEDGE GRAPH - BUILD & PERSIST")
    logger.info("=" * 80)
    
    try:
        # Initialize builder
        builder = KnowledgeGraphBuilder()
        
        # Build complete knowledge graph
        if not builder.build_complete_knowledge_graph():
            logger.error("❌ Knowledge graph construction failed")
            return 1
        
        # Save in multiple formats
        if not builder.save_knowledge_graph():
            logger.error("❌ Knowledge graph saving failed") 
            return 1
        
        # Validate saved graphs
        if not builder.validate_saved_graphs():
            logger.error("❌ Graph validation failed")
            return 1
        
        # Final summary
        logger.info("=" * 80)
        logger.info("🎉 PHASE 1 COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        stats = builder.build_metrics['graph_statistics']
        memory = builder.build_metrics['memory_usage']
        
        logger.info("📊 BUILD SUMMARY:")
        logger.info(f"   Construction Time: {builder.build_metrics['total_build_time_seconds']:.2f} seconds")
        logger.info(f"   Peak Memory Usage: {memory['peak_mb']:.1f} MB")
        logger.info(f"   Total Nodes: {stats.get('total_nodes', 'Unknown')}")
        logger.info(f"   Total Edges: {stats.get('total_edges', 'Unknown')}")
        
        logger.info("💾 SAVED FORMATS:")
        logger.info("   ✅ Complete KG Object (pickle) - Fastest loading")
        logger.info("   ✅ NetworkX Graph (gpickle) - Graph analysis")
        logger.info("   ✅ Statistics & Metadata (JSON) - QC reference")
        logger.info("   ✅ Neo4j Export Commands (Cypher) - Database import")
        
        logger.info(f"📁 All files saved to: {builder.output_dir}")
        logger.info("🚀 Ready for Phase 2: Structural Integrity Validation")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in main: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main())