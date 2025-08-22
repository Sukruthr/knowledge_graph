#!/usr/bin/env python3
"""
Quick QC Framework Demonstration

This script provides a quick demonstration of the comprehensive QC framework
by running a subset of the most critical validation phases without the full
knowledge graph build (which can take 30+ minutes).

This is perfect for validating the framework functionality and getting rapid feedback.
"""

import sys
import os
import time
import logging
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_quick_demo():
    """Run a quick demonstration of the QC framework."""
    logger.info("🚀 COMPREHENSIVE QC FRAMEWORK - QUICK DEMONSTRATION")
    logger.info("=" * 80)
    logger.info("This demonstration shows the QC framework capabilities without")
    logger.info("the full 30+ minute knowledge graph build process.")
    logger.info("=" * 80)
    
    demo_results = {
        'timestamp': datetime.now().isoformat(),
        'demo_phases': [],
        'framework_capabilities': {},
        'recommendations': []
    }
    
    # Phase 1: Framework Structure Validation
    logger.info("📋 PHASE 1: VALIDATING QC FRAMEWORK STRUCTURE")
    logger.info("-" * 50)
    
    base_dir = '/home/mreddy1/knowledge_graph/quality_control'
    expected_phases = [
        '1_build_and_save_kg',
        '2_structural_integrity',
        '3_data_quality_validation',
        '4_functional_testing',
        '5_integration_quality',
        '6_semantic_validation',
        '7_performance_benchmarks',
        '8_regression_testing'
    ]
    
    framework_status = {}
    for phase in expected_phases:
        phase_dir = os.path.join(base_dir, phase)
        script_path = os.path.join(phase_dir, f"{phase.split('_', 1)[1]}.py")
        
        exists = os.path.exists(script_path)
        framework_status[phase] = {
            'directory_exists': os.path.exists(phase_dir),
            'script_exists': exists,
            'status': '✅ READY' if exists else '❌ MISSING'
        }
        
        logger.info(f"   {phase}: {framework_status[phase]['status']}")
    
    ready_phases = sum(1 for status in framework_status.values() if status['script_exists'])
    framework_completeness = (ready_phases / len(expected_phases)) * 100
    
    logger.info(f"Framework completeness: {framework_completeness:.1f}% ({ready_phases}/{len(expected_phases)} phases ready)")
    
    demo_results['framework_capabilities']['structure_validation'] = {
        'expected_phases': len(expected_phases),
        'ready_phases': ready_phases,
        'completeness_percentage': framework_completeness,
        'phase_status': framework_status
    }
    
    # Phase 2: Master Orchestrator Validation
    logger.info("\n📊 PHASE 2: VALIDATING MASTER ORCHESTRATOR")
    logger.info("-" * 50)
    
    orchestrator_path = os.path.join(base_dir, 'run_comprehensive_qc.py')
    orchestrator_exists = os.path.exists(orchestrator_path)
    
    if orchestrator_exists:
        try:
            with open(orchestrator_path, 'r') as f:
                orchestrator_content = f.read()
            
            # Check for key components
            has_phase_definitions = 'qc_phases' in orchestrator_content
            has_orchestration_logic = 'run_qc_phase' in orchestrator_content
            has_reporting = 'generate_comprehensive_report' in orchestrator_content
            has_assessment = 'generate_final_assessment' in orchestrator_content
            
            orchestrator_capabilities = {
                'file_exists': True,
                'has_phase_definitions': has_phase_definitions,
                'has_orchestration_logic': has_orchestration_logic,
                'has_reporting': has_reporting,
                'has_assessment': has_assessment,
                'content_size_kb': len(orchestrator_content) / 1024
            }
            
            logger.info(f"   Master orchestrator: ✅ READY ({orchestrator_capabilities['content_size_kb']:.1f} KB)")
            logger.info(f"   Phase definitions: {'✅' if has_phase_definitions else '❌'}")
            logger.info(f"   Orchestration logic: {'✅' if has_orchestration_logic else '❌'}")
            logger.info(f"   Comprehensive reporting: {'✅' if has_reporting else '❌'}")
            logger.info(f"   Final assessment: {'✅' if has_assessment else '❌'}")
            
        except Exception as e:
            orchestrator_capabilities = {
                'file_exists': True,
                'error': str(e)
            }
            logger.warning(f"   ⚠️ Could not analyze orchestrator content: {str(e)}")
    else:
        orchestrator_capabilities = {'file_exists': False}
        logger.error("   ❌ Master orchestrator not found")
    
    demo_results['framework_capabilities']['orchestrator_validation'] = orchestrator_capabilities
    
    # Phase 3: Quick Import and Compatibility Test
    logger.info("\n🔧 PHASE 3: QUICK COMPATIBILITY TEST")
    logger.info("-" * 50)
    
    compatibility_results = {
        'import_tests': {},
        'method_availability': {},
        'system_readiness': {}
    }
    
    # Test imports
    try:
        sys.path.append('/home/mreddy1/knowledge_graph/src')
        
        # Test new-style import
        from kg_builders import ComprehensiveBiomedicalKnowledgeGraph
        compatibility_results['import_tests']['new_style'] = '✅ SUCCESS'
        logger.info("   New-style import (kg_builders): ✅ SUCCESS")
        
        # Test old-style import (should work with deprecation warning)
        try:
            from kg_builder import ComprehensiveBiomedicalKnowledgeGraph as OldKG
            compatibility_results['import_tests']['old_style'] = '✅ SUCCESS (with deprecation)'
            logger.info("   Old-style import (kg_builder): ✅ SUCCESS (with deprecation)")
        except:
            compatibility_results['import_tests']['old_style'] = '⚠️ DEPRECATED'
            logger.info("   Old-style import (kg_builder): ⚠️ DEPRECATED")
        
        # Test method availability
        kg_instance = ComprehensiveBiomedicalKnowledgeGraph()
        key_methods = [
            'load_data', 'build_comprehensive_graph', 'get_comprehensive_stats',
            'query_gene_comprehensive', 'query_go_term'
        ]
        
        available_methods = 0
        for method in key_methods:
            if hasattr(kg_instance, method):
                available_methods += 1
                compatibility_results['method_availability'][method] = '✅ AVAILABLE'
            else:
                compatibility_results['method_availability'][method] = '❌ MISSING'
        
        method_coverage = (available_methods / len(key_methods)) * 100
        logger.info(f"   Key method availability: {method_coverage:.1f}% ({available_methods}/{len(key_methods)})")
        
        # System readiness check
        data_dir = '/home/mreddy1/knowledge_graph/llm_evaluation_for_gene_set_interpretation/data'
        src_dir = '/home/mreddy1/knowledge_graph/src'
        
        compatibility_results['system_readiness'] = {
            'data_directory': '✅ READY' if os.path.exists(data_dir) else '❌ MISSING',
            'source_directory': '✅ READY' if os.path.exists(src_dir) else '❌ MISSING',
            'imports_working': '✅ READY' if 'new_style' in compatibility_results['import_tests'] else '❌ FAILED'
        }
        
        for component, status in compatibility_results['system_readiness'].items():
            logger.info(f"   {component.replace('_', ' ').title()}: {status}")
        
    except Exception as e:
        compatibility_results['import_tests']['error'] = str(e)
        logger.error(f"   ❌ Import test failed: {str(e)}")
    
    demo_results['framework_capabilities']['compatibility_test'] = compatibility_results
    
    # Phase 4: Generate Demo Summary
    logger.info("\n📊 PHASE 4: DEMO SUMMARY & RECOMMENDATIONS")
    logger.info("-" * 50)
    
    # Calculate overall readiness score
    structure_score = framework_completeness
    orchestrator_score = 100 if orchestrator_capabilities.get('file_exists') and orchestrator_capabilities.get('has_orchestration_logic') else 0
    compatibility_score = method_coverage if 'method_coverage' in locals() else 0
    
    overall_readiness = (structure_score + orchestrator_score + compatibility_score) / 3
    
    # Generate assessment
    if overall_readiness >= 90:
        readiness_status = "🎉 FRAMEWORK FULLY READY"
        readiness_grade = "A+"
        recommendation = "The QC framework is production-ready and can be executed immediately."
    elif overall_readiness >= 80:
        readiness_status = "✅ FRAMEWORK READY"
        readiness_grade = "A"
        recommendation = "The QC framework is ready for execution with minor enhancements possible."
    elif overall_readiness >= 70:
        readiness_status = "⚠️ FRAMEWORK MOSTLY READY"
        readiness_grade = "B"
        recommendation = "The QC framework is functional but may need some components addressed."
    else:
        readiness_status = "❌ FRAMEWORK NEEDS WORK"
        readiness_grade = "C"
        recommendation = "The QC framework requires additional setup before execution."
    
    demo_summary = {
        'overall_readiness_score': overall_readiness,
        'readiness_grade': readiness_grade,
        'readiness_status': readiness_status,
        'component_scores': {
            'structure_completeness': structure_score,
            'orchestrator_readiness': orchestrator_score,
            'compatibility_score': compatibility_score
        },
        'recommendation': recommendation
    }
    
    logger.info(f"Overall Framework Readiness: {overall_readiness:.1f}% (Grade: {readiness_grade})")
    logger.info(f"Status: {readiness_status}")
    logger.info(f"Recommendation: {recommendation}")
    
    demo_results['demo_summary'] = demo_summary
    
    # Save demo results
    logger.info("\n💾 SAVING DEMO RESULTS")
    logger.info("-" * 50)
    
    try:
        results_dir = '/home/mreddy1/knowledge_graph/quality_control/demo_results'
        os.makedirs(results_dir, exist_ok=True)
        
        demo_file = os.path.join(results_dir, 'qc_framework_demo_results.json')
        with open(demo_file, 'w') as f:
            json.dump(demo_results, f, indent=2)
        
        logger.info(f"✅ Demo results saved to: {demo_file}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save demo results: {str(e)}")
    
    # Final execution guidance
    logger.info("\n🚀 EXECUTION GUIDANCE")
    logger.info("-" * 50)
    
    if overall_readiness >= 80:
        logger.info("✅ To run the complete QC framework:")
        logger.info("   cd /home/mreddy1/knowledge_graph/quality_control")
        logger.info("   python run_comprehensive_qc.py")
        logger.info("")
        logger.info("⏱️ Expected execution time: 2-3 hours")
        logger.info("💾 Expected results: Comprehensive quality assessment with production certification")
        logger.info("")
        logger.info("📊 The framework will validate:")
        logger.info("   • Graph structure & integrity")
        logger.info("   • Data quality & accuracy") 
        logger.info("   • All 97 methods & queries")
        logger.info("   • Cross-modal integration")
        logger.info("   • Biological semantics")
        logger.info("   • Performance benchmarks")
        logger.info("   • Backward compatibility")
        logger.info("   • Production readiness")
    else:
        logger.warning("⚠️ Framework needs additional setup before execution")
        logger.warning("   Please address the missing components identified above")
    
    logger.info("\n" + "=" * 80)
    logger.info("🎯 QC FRAMEWORK DEMONSTRATION COMPLETED")
    logger.info("=" * 80)
    
    return overall_readiness >= 80

def main():
    """Main execution function."""
    try:
        success = run_quick_demo()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"❌ Demo failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())