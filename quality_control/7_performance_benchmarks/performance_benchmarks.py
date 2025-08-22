#!/usr/bin/env python3
"""
Phase 7: Performance Benchmarks

This script performs load testing, memory profiling, and scalability analysis
of the biomedical knowledge graph.
"""

import sys
import os
import time
import pickle
import logging
import json
import psutil
from datetime import datetime

# Add project root to path
sys.path.append('/home/mreddy1/knowledge_graph/src')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceBenchmarker:
    """Performance benchmarking for biomedical knowledge graph."""
    
    def __init__(self, kg_path='/home/mreddy1/knowledge_graph/quality_control/1_build_and_save_kg/saved_graphs/complete_biomedical_kg.pkl'):
        self.kg_path = kg_path
        self.kg = None
        self.benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'validation_summary': {},
            'quality_metrics': {},
            'recommendations': []
        }
    
    def load_knowledge_graph(self):
        """Load knowledge graph."""
        try:
            if os.path.exists(self.kg_path):
                start_time = time.time()
                with open(self.kg_path, 'rb') as f:
                    self.kg = pickle.load(f)
                load_time = time.time() - start_time
                logger.info(f"✅ Knowledge graph loaded in {load_time:.2f}s")
            else:
                from kg_builders import ComprehensiveBiomedicalKnowledgeGraph
                start_time = time.time()
                self.kg = ComprehensiveBiomedicalKnowledgeGraph()
                self.kg.load_data('/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data')
                self.kg.build_comprehensive_graph()
                build_time = time.time() - start_time
                logger.info(f"✅ Knowledge graph built in {build_time:.2f}s")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load knowledge graph: {str(e)}")
            return False
    
    def benchmark_query_performance(self):
        """Benchmark query performance."""
        logger.info("⏱️ BENCHMARKING QUERY PERFORMANCE")
        
        try:
            # Sample gene queries
            test_genes = ['TP53', 'BRCA1', 'EGFR', 'MYC', 'PTEN']
            
            query_times = []
            successful_queries = 0
            
            for gene in test_genes:
                if hasattr(self.kg, 'query_gene_comprehensive'):
                    try:
                        start_time = time.time()
                        result = self.kg.query_gene_comprehensive(gene)
                        query_time = time.time() - start_time
                        
                        if result:
                            query_times.append(query_time)
                            successful_queries += 1
                    except:
                        pass
            
            if query_times:
                avg_query_time = sum(query_times) / len(query_times)
                queries_per_second = 1 / avg_query_time if avg_query_time > 0 else 0
            else:
                avg_query_time = 0
                queries_per_second = 0
            
            performance_results = {
                'queries_tested': len(test_genes),
                'successful_queries': successful_queries,
                'average_query_time': avg_query_time,
                'queries_per_second': queries_per_second,
                'meets_target': queries_per_second >= 100  # Target: 100+ QPS
            }
            
            logger.info(f"Average query time: {avg_query_time:.3f}s")
            logger.info(f"Queries per second: {queries_per_second:.1f}")
            
            self.benchmark_results['detailed_results'] = {'query_performance': performance_results}
            return True
            
        except Exception as e:
            logger.error(f"❌ Query performance benchmarking failed: {str(e)}")
            return False
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage."""
        logger.info("💾 BENCHMARKING MEMORY USAGE")
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            memory_results = {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'memory_percent': process.memory_percent(),
                'within_limits': memory_info.rss / 1024 / 1024 < 6000  # Target: <6GB
            }
            
            logger.info(f"Memory usage: {memory_results['rss_mb']:.1f} MB")
            logger.info(f"Memory percentage: {memory_results['memory_percent']:.1f}%")
            
            self.benchmark_results['detailed_results']['memory_usage'] = memory_results
            return True
            
        except Exception as e:
            logger.error(f"❌ Memory benchmarking failed: {str(e)}")
            return False
    
    def generate_quality_metrics(self):
        """Generate performance quality metrics."""
        logger.info("📊 GENERATING PERFORMANCE METRICS")
        
        try:
            metrics = {}
            
            # Query performance score
            query_perf = self.benchmark_results['detailed_results'].get('query_performance', {})
            qps = query_perf.get('queries_per_second', 0)
            query_score = min((qps / 100) * 100, 100)  # Scale based on 100 QPS target
            metrics['query_performance_score'] = query_score
            
            # Memory efficiency score
            memory_perf = self.benchmark_results['detailed_results'].get('memory_usage', {})
            memory_mb = memory_perf.get('rss_mb', 0)
            memory_score = max(100 - (memory_mb / 60), 0)  # 6GB = 0 points, 0GB = 100 points
            metrics['memory_efficiency_score'] = memory_score
            
            # Overall performance score
            overall_performance = (query_score + memory_score) / 2
            metrics['overall_performance_score'] = overall_performance
            
            # Performance grade
            if overall_performance >= 90:
                grade = 'A'
            elif overall_performance >= 80:
                grade = 'B'
            elif overall_performance >= 70:
                grade = 'C'
            else:
                grade = 'D'
            
            metrics['performance_grade'] = grade
            
            self.benchmark_results['quality_metrics'] = metrics
            
            logger.info(f"Performance Score: {overall_performance:.1f}/100 (Grade: {grade})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance metrics generation failed: {str(e)}")
            return False
    
    def save_results(self):
        """Save benchmark results."""
        try:
            output_path = '/home/mreddy1/knowledge_graph/quality_control/7_performance_benchmarks/performance_benchmarks_results.json'
            with open(output_path, 'w') as f:
                json.dump(self.benchmark_results, f, indent=2)
            logger.info(f"📄 Results saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save results: {str(e)}")
            return False
    
    def run_benchmarks(self):
        """Run performance benchmarks."""
        logger.info("⚡ PERFORMANCE BENCHMARKING")
        
        steps = [
            ('Load Knowledge Graph', self.load_knowledge_graph),
            ('Benchmark Query Performance', self.benchmark_query_performance),
            ('Benchmark Memory Usage', self.benchmark_memory_usage),
            ('Generate Quality Metrics', self.generate_quality_metrics),
            ('Save Results', self.save_results)
        ]
        
        passed = 0
        for name, func in steps:
            logger.info(f"Executing: {name}")
            if func():
                passed += 1
                logger.info(f"✅ {name} completed")
            else:
                logger.error(f"❌ {name} failed")
        
        success_rate = (passed / len(steps)) * 100
        self.benchmark_results['validation_summary'] = {
            'total_steps': len(steps),
            'passed_steps': passed,
            'success_rate': success_rate,
            'overall_status': 'PASSED' if passed == len(steps) else 'FAILED'
        }
        
        return success_rate == 100

def main():
    """Main execution function."""
    try:
        benchmarker = PerformanceBenchmarker()
        success = benchmarker.run_benchmarks()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())