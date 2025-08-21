#!/usr/bin/env python3
"""
Test KG builder methods for GO Analysis Data
"""

import sys
sys.path.append('src')

def test_kg_methods():
    """Test that KG builder has the required GO Analysis Data methods."""
    
    print("=" * 60)
    print("KG BUILDER METHODS TEST")
    print("=" * 60)
    
    try:
        from kg_builder import ComprehensiveBiomedicalKnowledgeGraph
        
        # Check if methods exist
        kg = ComprehensiveBiomedicalKnowledgeGraph()
        
        methods_to_check = [
            '_add_go_analysis_data',
            '_add_core_go_analysis_nodes', 
            '_add_contamination_dataset_nodes',
            '_add_confidence_evaluation_nodes',
            '_add_hierarchy_relationship_nodes',
            '_add_similarity_score_nodes',
            '_connect_go_analysis_to_graph',
            'query_go_core_analysis',
            'query_go_contamination_analysis', 
            'query_go_confidence_evaluations',
            'query_gene_go_analysis_profile',
            'get_go_analysis_stats'
        ]
        
        print("✅ Method Availability Check:")
        missing_methods = []
        for method in methods_to_check:
            has_method = hasattr(kg, method)
            print(f"   {method}: {'✅' if has_method else '❌'}")
            if not has_method:
                missing_methods.append(method)
        
        if missing_methods:
            print(f"\n❌ Missing methods: {missing_methods}")
            return False
        else:
            print(f"\n🎉 All required methods are available!")
            
            # Test method signatures
            print(f"\n✅ Method Signature Test:")
            
            # Test calling methods with default parameters (should not crash)
            try:
                # These should work even without data loaded
                stats = kg.get_go_analysis_stats()
                print(f"   get_go_analysis_stats(): returns dict with {len(stats)} keys")
                
                core_analysis = kg.query_go_core_analysis()
                print(f"   query_go_core_analysis(): returns list with {len(core_analysis)} items")
                
                contamination = kg.query_go_contamination_analysis()
                print(f"   query_go_contamination_analysis(): returns list with {len(contamination)} items")
                
                confidence = kg.query_go_confidence_evaluations()
                print(f"   query_go_confidence_evaluations(): returns list with {len(confidence)} items")
                
                gene_profile = kg.query_gene_go_analysis_profile('TEST_GENE')
                print(f"   query_gene_go_analysis_profile(): returns dict with {len(gene_profile)} keys")
                
                print(f"\n🎉 All method signatures work correctly!")
                return True
                
            except Exception as e:
                print(f"\n❌ Method signature test failed: {e}")
                return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_kg_methods()
    sys.exit(0 if success else 1)