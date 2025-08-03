#!/usr/bin/env python3
"""
NMI 계산 방식 수정 및 검증
"""

from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
import numpy as np

def test_nmi_methods():
    """다양한 NMI 계산 방식 테스트"""
    print("🧮 NMI 계산 방식 비교")
    print("=" * 50)
    
    # 테스트 데이터
    true_labels = [0, 0, 1, 1, 2, 2]  # 3개 클래스, 각 2개씩
    pred_labels = [0, 1, 2, 3, 4, 5]  # 완전 세분화 (모든 노드 다른 커뮤니티)
    
    print(f"Ground truth: {true_labels}")
    print(f"Prediction:   {pred_labels}")
    print()
    
    # 다양한 평균 방식으로 NMI 계산
    methods = ['arithmetic', 'geometric', 'min', 'max']
    
    for method in methods:
        nmi = normalized_mutual_info_score(true_labels, pred_labels, average_method=method)
        print(f"NMI ({method:>10}): {nmi:.4f}")
    
    # Adjusted MI도 테스트
    ami = adjusted_mutual_info_score(true_labels, pred_labels)
    print(f"AMI (adjusted):   {ami:.4f}")
    
    print()
    print("🎯 올바른 결과:")
    print("   완전 세분화시 NMI는 0에 가까워야 함")
    print("   geometric나 min 방식이 더 적절할 수 있음")

def corrected_nmi(true_labels, pred_labels, method='min'):
    """수정된 NMI 계산"""
    return normalized_mutual_info_score(true_labels, pred_labels, average_method=method)

def test_corrected_evaluation():
    """수정된 평가로 Cora 재테스트"""
    print(f"\n🔧 수정된 평가 방식으로 재테스트")
    print("=" * 50)
    
    import sys
    sys.path.append('../../src')
    
    import networkx as nx
    from improved_leiden_lpa import LeidenLPAHybrid
    
    # Cora 데이터
    G = nx.read_edgelist("../../data/data/processed/cora/graph.edgelist")
    
    labels = {}
    with open("../../data/data/processed/cora/labels.txt", 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                node, label = parts[0], int(parts[1])
                labels[node] = label
    
    methods = ['pagerank', 'degree']
    
    for method in methods:
        print(f"\n🔧 {method.upper()}:")
        
        alg = LeidenLPAHybrid(core_ratio=0.4, centrality_method=method)
        pred_labels = alg.fit_predict(G)
        
        print(f"   커뮤니티 수: {len(set(pred_labels.values()))}")
        
        # 공통 노드로 NMI 계산
        common_nodes = set(G.nodes()) & set(labels.keys())
        true_list = [labels[node] for node in common_nodes]
        pred_list = [pred_labels[node] for node in common_nodes]
        
        # 다양한 방식으로 NMI 계산
        for avg_method in ['arithmetic', 'geometric', 'min']:
            nmi = normalized_mutual_info_score(true_list, pred_list, average_method=avg_method)
            print(f"   NMI ({avg_method}): {nmi:.4f}")

if __name__ == "__main__":
    test_nmi_methods()
    test_corrected_evaluation()