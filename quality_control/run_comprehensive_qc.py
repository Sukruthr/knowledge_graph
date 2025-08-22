#!/usr/bin/env python3
"""
Master Quality Control Orchestrator

This script orchestrates the complete quality control framework for the biomedical knowledge graph.
It runs all QC phases in sequence and generates comprehensive quality assessment reports.

QC Framework Phases:
1. Build & Persist Complete KG (one-time setup)
2. Structural Integrity Validation
3. Data Quality Validation  
4. Functional Testing
5. Integration Quality Validation
6. Semantic Validation
7. Performance Benchmarks
8. Regression Testing
9. Production Readiness Assessment
"""

import sys
import os
import time
import subprocess
import logging
import json
import traceback
from datetime import datetime
from pathlib import Path
import psutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/mreddy1/knowledge_graph/quality_control/comprehensive_qc_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveQCOrchestrator:
    """Orchestrates the complete quality control framework."""
    
    def __init__(self, base_dir='/home/mreddy1/knowledge_graph/quality_control'):
        self.base_dir = base_dir
        self.results_dir = os.path.join(base_dir, 'comprehensive_results')
        self.comprehensive_results = {
            'timestamp': datetime.now().isoformat(),
            'qc_framework_version': '1.0',
            'overall_summary': {},
            'phase_results': {},
            'quality_metrics': {},
            'final_assessment': {},
            'recommendations': []
        }
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # QC phase definitions
        self.qc_phases = [
            {
                'phase': 1,
                'name': 'Build & Persist KG',
                'description': 'Build complete knowledge graph and save in multiple formats',
                'script': '1_build_and_save_kg/build_complete_kg.py',
                'critical': True,
                'estimated_time': '60 seconds'
            },
            {
                'phase': 2,
                'name': 'Structural Integrity',
                'description': 'Validate graph topology, schema adherence, and structural consistency',
                'script': '2_structural_integrity/structural_validation.py',
                'critical': True,
                'estimated_time': '120 seconds'
            },
            {
                'phase': 3,
                'name': 'Data Quality',
                'description': 'Cross-reference against authoritative sources and validate data accuracy',
                'script': '3_data_quality_validation/data_quality_validation.py',
                'critical': True,
                'estimated_time': '180 seconds'
            },
            {
                'phase': 4,
                'name': 'Functional Testing',
                'description': 'Test all methods and validate query responses',
                'script': '4_functional_testing/functional_testing.py',
                'critical': True,
                'estimated_time': '300 seconds'
            },
            {
                'phase': 5,
                'name': 'Integration Quality',
                'description': 'Validate 9-phase data source integration',
                'script': '5_integration_quality/integration_quality_validation.py',
                'critical': False,
                'estimated_time': '120 seconds'
            },
            {
                'phase': 6,
                'name': 'Semantic Validation',
                'description': 'Validate biological logic and consistency',
                'script': '6_semantic_validation/semantic_validation.py',
                'critical': False,
                'estimated_time': '180 seconds'
            },
            {
                'phase': 7,
                'name': 'Performance Benchmarks',
                'description': 'Load testing and memory profiling',
                'script': '7_performance_benchmarks/performance_benchmarks.py',
                'critical': False,
                'estimated_time': '240 seconds'
            },
            {
                'phase': 8,
                'name': 'Regression Testing',
                'description': 'Backward compatibility validation',
                'script': '8_regression_testing/regression_testing.py',
                'critical': False,
                'estimated_time': '120 seconds'
            }
        ]
        
    def check_prerequisites(self):
        """Check system prerequisites for QC framework."""
        logger.info("🔍 CHECKING QC FRAMEWORK PREREQUISITES")
        logger.info("=" * 60)
        
        try:
            # Check Python version
            python_version = sys.version_info
            logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
            
            # Check available memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            logger.info(f"Available memory: {available_gb:.1f} GB")
            
            if available_gb < 4:
                logger.warning("⚠️ Less than 4GB available memory - performance may be impacted")
            
            # Check disk space
            disk = psutil.disk_usage(self.base_dir)
            free_gb = disk.free / (1024**3)
            logger.info(f"Available disk space: {free_gb:.1f} GB")
            
            if free_gb < 2:
                logger.warning("⚠️ Less than 2GB free disk space - may impact results storage")
            
            # Check data directory
            data_dir = '/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data'
            if not os.path.exists(data_dir):
                logger.error(f"❌ Data directory not found: {data_dir}")
                return False
            
            logger.info(f"✅ Data directory exists: {data_dir}")
            
            # Check src directory
            src_dir = '/home/mreddy1/knowledge_graph/src'
            if not os.path.exists(src_dir):
                logger.error(f"❌ Source directory not found: {src_dir}")
                return False
            
            logger.info(f"✅ Source directory exists: {src_dir}")
            
            logger.info("✅ All prerequisites satisfied")
            return True
            
        except Exception as e:
            logger.error(f"❌ Prerequisites check failed: {str(e)}")
            return False
    
    def run_qc_phase(self, phase_info):
        """Run a single QC phase."""
        phase_num = phase_info['phase']
        phase_name = phase_info['name']
        script_path = os.path.join(self.base_dir, phase_info['script'])
        
        logger.info(f"🚀 PHASE {phase_num}: {phase_name}")
        logger.info("=" * 60)
        logger.info(f"Description: {phase_info['description']}")
        logger.info(f"Estimated time: {phase_info['estimated_time']}")
        logger.info(f"Critical: {phase_info['critical']}")
        
        # Check if script exists
        if not os.path.exists(script_path):
            logger.error(f"❌ QC script not found: {script_path}")
            if phase_info['critical']:
                return False, {'error': f'Critical script missing: {script_path}'}
            else:
                logger.warning("⚠️ Non-critical script missing - skipping")
                return True, {'status': 'skipped', 'reason': 'script_missing'}
        
        # Run the QC phase
        start_time = time.time()
        
        try:
            logger.info(f"Executing: python {script_path}")
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(script_path),
                timeout=1200  # 20 minute timeout per phase
            )
            
            execution_time = time.time() - start_time
            
            phase_result = {
                'phase': phase_num,
                'name': phase_name,
                'execution_time_seconds': execution_time,
                'exit_code': result.returncode,
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            if result.returncode == 0:
                logger.info(f"✅ Phase {phase_num} completed successfully in {execution_time:.1f}s")
                
                # Try to load phase-specific results
                try:
                    results_file = script_path.replace('.py', '_results.json')
                    if os.path.exists(results_file):
                        with open(results_file, 'r') as f:
                            detailed_results = json.load(f)
                        phase_result['detailed_results'] = detailed_results
                        logger.info(f"✅ Loaded detailed results from {results_file}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load detailed results: {e}")
                
                return True, phase_result
            else:
                logger.error(f"❌ Phase {phase_num} failed with exit code {result.returncode}")
                logger.error(f"STDERR: {result.stderr[:500]}...")  # Truncate for readability
                
                if phase_info['critical']:
                    return False, phase_result
                else:
                    logger.warning("⚠️ Non-critical phase failed - continuing")
                    return True, phase_result
                    
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            logger.error(f"❌ Phase {phase_num} timed out after {execution_time:.1f}s")
            
            phase_result = {
                'phase': phase_num,
                'name': phase_name,
                'execution_time_seconds': execution_time,
                'error': 'timeout',
                'success': False
            }
            
            if phase_info['critical']:
                return False, phase_result
            else:
                return True, phase_result
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Phase {phase_num} failed with exception: {str(e)}")
            
            phase_result = {
                'phase': phase_num,
                'name': phase_name,
                'execution_time_seconds': execution_time,
                'error': str(e),
                'success': False
            }
            
            if phase_info['critical']:
                return False, phase_result
            else:
                return True, phase_result
    
    def generate_comprehensive_report(self):
        """Generate comprehensive quality assessment report."""
        logger.info("📊 GENERATING COMPREHENSIVE QC REPORT")
        logger.info("=" * 60)
        
        try:
            # Calculate overall metrics
            total_phases = len(self.qc_phases)
            executed_phases = len(self.comprehensive_results['phase_results'])
            successful_phases = sum(1 for phase in self.comprehensive_results['phase_results'].values() if phase.get('success', False))
            
            # Calculate execution time
            total_execution_time = sum(phase.get('execution_time_seconds', 0) for phase in self.comprehensive_results['phase_results'].values())
            
            # Overall success rate
            overall_success_rate = (successful_phases / executed_phases * 100) if executed_phases > 0 else 0
            
            # Collect quality metrics from all phases
            quality_metrics = {}
            
            for phase_name, phase_result in self.comprehensive_results['phase_results'].items():
                detailed_results = phase_result.get('detailed_results', {})
                
                if 'quality_metrics' in detailed_results:
                    quality_metrics[phase_name] = detailed_results['quality_metrics']
            
            # Generate final assessment
            final_assessment = self._generate_final_assessment(overall_success_rate, quality_metrics)
            
            # Update comprehensive results
            self.comprehensive_results.update({
                'overall_summary': {
                    'total_phases': total_phases,
                    'executed_phases': executed_phases,
                    'successful_phases': successful_phases,
                    'overall_success_rate': overall_success_rate,
                    'total_execution_time_seconds': total_execution_time,
                    'total_execution_time_minutes': total_execution_time / 60
                },
                'quality_metrics': quality_metrics,
                'final_assessment': final_assessment
            })
            
            # Generate recommendations
            self._generate_comprehensive_recommendations()
            
            logger.info("Comprehensive QC Report Summary:")
            logger.info(f"   Total phases: {total_phases}")
            logger.info(f"   Executed phases: {executed_phases}")
            logger.info(f"   Successful phases: {successful_phases}")
            logger.info(f"   Overall success rate: {overall_success_rate:.1f}%")
            logger.info(f"   Total execution time: {total_execution_time/60:.1f} minutes")
            logger.info(f"   Final assessment: {final_assessment.get('grade', 'N/A')}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Comprehensive report generation failed: {str(e)}")
            return False
    
    def _generate_final_assessment(self, success_rate, quality_metrics):
        """Generate final production readiness assessment."""
        assessment = {
            'overall_score': 0.0,
            'grade': 'F',
            'production_ready': False,
            'certification_level': 'NOT_CERTIFIED',
            'critical_issues': [],
            'recommendations': []
        }
        
        # Base score from success rate
        base_score = success_rate
        
        # Quality metric bonuses/penalties
        quality_bonus = 0
        quality_penalties = 0
        
        for phase_name, metrics in quality_metrics.items():
            if isinstance(metrics, dict):
                # Look for overall quality scores
                for key, value in metrics.items():
                    if 'quality_score' in key.lower() and isinstance(value, (int, float)):
                        if value >= 90:
                            quality_bonus += 5
                        elif value < 70:
                            quality_penalties += 10
        
        # Calculate final score
        final_score = base_score + quality_bonus - quality_penalties
        final_score = max(0, min(100, final_score))  # Clamp to 0-100
        
        # Assign grade and certification
        if final_score >= 95:
            grade = 'A+'
            production_ready = True
            certification = 'PRODUCTION_CERTIFIED'
        elif final_score >= 90:
            grade = 'A'
            production_ready = True
            certification = 'PRODUCTION_READY'
        elif final_score >= 85:
            grade = 'B+'
            production_ready = True
            certification = 'CONDITIONAL_READY'
        elif final_score >= 80:
            grade = 'B'
            production_ready = False
            certification = 'DEVELOPMENT_READY'
        elif final_score >= 70:
            grade = 'C'
            production_ready = False
            certification = 'TESTING_REQUIRED'
        else:
            grade = 'D'
            production_ready = False
            certification = 'NOT_CERTIFIED'
        
        assessment.update({
            'overall_score': final_score,
            'grade': grade,
            'production_ready': production_ready,
            'certification_level': certification
        })
        
        return assessment
    
    def _generate_comprehensive_recommendations(self):
        """Generate comprehensive recommendations."""
        recommendations = []
        
        # Analyze phase results for recommendations
        failed_phases = [phase for phase in self.comprehensive_results['phase_results'].values() if not phase.get('success', False)]
        
        if failed_phases:
            critical_failures = [phase for phase in failed_phases if any(qc_phase['critical'] for qc_phase in self.qc_phases if qc_phase['phase'] == phase.get('phase'))]
            
            if critical_failures:
                recommendations.append({
                    'category': 'Critical Issues',
                    'priority': 'CRITICAL',
                    'issue': f'{len(critical_failures)} critical QC phases failed',
                    'recommendation': 'Address critical failures before any deployment',
                    'action_required': True
                })
        
        # Quality-based recommendations
        overall_score = self.comprehensive_results['final_assessment'].get('overall_score', 0)
        
        if overall_score < 80:
            recommendations.append({
                'category': 'Quality Score',
                'priority': 'HIGH',
                'issue': f'Overall quality score is {overall_score:.1f}%',
                'recommendation': 'Comprehensive quality improvements required',
                'action_required': True
            })
        elif overall_score < 90:
            recommendations.append({
                'category': 'Quality Score',
                'priority': 'MEDIUM',
                'issue': f'Overall quality score is {overall_score:.1f}%',
                'recommendation': 'Additional quality improvements recommended',
                'action_required': False
            })
        
        if not recommendations:
            recommendations.append({
                'category': 'Overall',
                'priority': 'INFO',
                'issue': 'All QC phases passed successfully',
                'recommendation': 'System meets production quality standards',
                'action_required': False
            })
        
        self.comprehensive_results['recommendations'] = recommendations
    
    def save_comprehensive_results(self):
        """Save comprehensive QC results."""
        try:
            # Save main results file
            results_file = os.path.join(self.results_dir, 'comprehensive_qc_results.json')
            with open(results_file, 'w') as f:
                json.dump(self.comprehensive_results, f, indent=2)
            
            logger.info(f"📄 Comprehensive results saved to: {results_file}")
            
            # Generate executive summary
            self._generate_executive_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save comprehensive results: {str(e)}")
            return False
    
    def _generate_executive_summary(self):
        """Generate executive summary report."""
        try:
            summary_file = os.path.join(self.results_dir, 'EXECUTIVE_SUMMARY.md')
            
            overall = self.comprehensive_results['overall_summary']
            assessment = self.comprehensive_results['final_assessment']
            
            with open(summary_file, 'w') as f:
                f.write("# Biomedical Knowledge Graph - Quality Control Executive Summary\n\n")
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
                f.write(f"**QC Framework Version:** {self.comprehensive_results.get('qc_framework_version', '1.0')}  \n")
                f.write(f"**Overall Grade:** {assessment.get('grade', 'N/A')}  \n")
                f.write(f"**Production Ready:** {'✅ YES' if assessment.get('production_ready', False) else '❌ NO'}  \n\n")
                
                f.write("## Quality Assessment Summary\n\n")
                f.write(f"- **Overall Success Rate:** {overall.get('overall_success_rate', 0):.1f}%\n")
                f.write(f"- **Quality Score:** {assessment.get('overall_score', 0):.1f}/100\n")
                f.write(f"- **Certification Level:** {assessment.get('certification_level', 'N/A')}\n")
                f.write(f"- **Total Execution Time:** {overall.get('total_execution_time_minutes', 0):.1f} minutes\n\n")
                
                f.write("## QC Phase Results\n\n")
                f.write("| Phase | Name | Status | Duration |\n")
                f.write("|-------|------|--------|----------|\n")
                
                for phase_name, result in self.comprehensive_results['phase_results'].items():
                    status = "✅ PASSED" if result.get('success', False) else "❌ FAILED"
                    duration = f"{result.get('execution_time_seconds', 0):.1f}s"
                    phase_num = result.get('phase', 'N/A')
                    f.write(f"| {phase_num} | {phase_name} | {status} | {duration} |\n")
                
                f.write("\n## Key Recommendations\n\n")
                for rec in self.comprehensive_results.get('recommendations', [])[:5]:
                    priority = rec.get('priority', 'INFO')
                    category = rec.get('category', 'General')
                    recommendation = rec.get('recommendation', 'N/A')
                    f.write(f"- **[{priority}] {category}:** {recommendation}\n")
                
                f.write(f"\n## Production Readiness\n\n")
                if assessment.get('production_ready', False):
                    f.write("🎉 **PRODUCTION CERTIFIED**\n\n")
                    f.write("The biomedical knowledge graph has passed comprehensive quality control ")
                    f.write("and is certified for production deployment.\n")
                else:
                    f.write("⚠️ **PRODUCTION NOT READY**\n\n")
                    f.write("The biomedical knowledge graph requires additional improvements ")
                    f.write("before production deployment.\n")
                
                f.write(f"\n---\n")
                f.write(f"*Generated by Comprehensive QC Framework v{self.comprehensive_results.get('qc_framework_version', '1.0')}*\n")
            
            logger.info(f"📄 Executive summary saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate executive summary: {str(e)}")
    
    def run_comprehensive_qc(self):
        """Run the complete QC framework."""
        logger.info("🎯 COMPREHENSIVE BIOMEDICAL KNOWLEDGE GRAPH QC FRAMEWORK")
        logger.info("=" * 80)
        logger.info("This framework will perform comprehensive quality control across all dimensions:")
        logger.info("- Structural Integrity")
        logger.info("- Data Quality") 
        logger.info("- Functional Testing")
        logger.info("- Integration Quality")
        logger.info("- Semantic Validation")
        logger.info("- Performance Benchmarks")
        logger.info("- Regression Testing")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("❌ Prerequisites not satisfied - aborting QC framework")
            return False
        
        # Run QC phases
        for phase_info in self.qc_phases:
            success, phase_result = self.run_qc_phase(phase_info)
            
            # Store phase result
            self.comprehensive_results['phase_results'][phase_info['name']] = phase_result
            
            # Check if critical phase failed
            if not success and phase_info['critical']:
                logger.error(f"❌ Critical phase {phase_info['phase']} failed - aborting QC framework")
                break
            
            # Brief pause between phases
            time.sleep(2)
        
        # Generate comprehensive report
        if not self.generate_comprehensive_report():
            logger.error("❌ Failed to generate comprehensive report")
        
        # Save results
        if not self.save_comprehensive_results():
            logger.error("❌ Failed to save comprehensive results")
        
        total_time = time.time() - start_time
        
        # Final summary
        logger.info("=" * 80)
        logger.info("🏁 COMPREHENSIVE QC FRAMEWORK COMPLETED")
        logger.info("=" * 80)
        
        overall = self.comprehensive_results['overall_summary']
        assessment = self.comprehensive_results['final_assessment']
        
        logger.info(f"Total execution time: {total_time/60:.1f} minutes")
        logger.info(f"Phases executed: {overall.get('executed_phases', 0)}/{overall.get('total_phases', 0)}")
        logger.info(f"Overall success rate: {overall.get('overall_success_rate', 0):.1f}%")
        logger.info(f"Final quality score: {assessment.get('overall_score', 0):.1f}/100")
        logger.info(f"Quality grade: {assessment.get('grade', 'N/A')}")
        logger.info(f"Production ready: {'YES ✅' if assessment.get('production_ready', False) else 'NO ❌'}")
        
        logger.info(f"\nResults saved to: {self.results_dir}")
        logger.info("📄 Check EXECUTIVE_SUMMARY.md for detailed assessment")
        
        return assessment.get('production_ready', False)

def main():
    """Main execution function."""
    try:
        orchestrator = ComprehensiveQCOrchestrator()
        production_ready = orchestrator.run_comprehensive_qc()
        
        if production_ready:
            logger.info("🎉 KNOWLEDGE GRAPH IS PRODUCTION READY!")
            return 0
        else:
            logger.info("⚠️ Knowledge graph requires improvements before production")
            return 1
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in QC orchestrator: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main())