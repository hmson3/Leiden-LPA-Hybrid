#!/usr/bin/env python3
"""
데이터 전처리 검증 스크립트
실제 파일 내용을 확인해서 문제를 찾아보자
"""

import os
import sys
from pathlib import Path
import networkx as nx
project_root = Path(__file__).parent.parent.parent  # experiments/exp2_centrality에서 3단계 위로
print(f"프로젝트 루트: {project_root}")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))


from improved_leiden_lpa import LeidenLPAHybrid

def check_dataset(dataset_name):
    """단일 데이터셋 상세 확인"""
    print(f"\n🔍 {dataset_name.upper()} 데이터셋 검증")
    print("=" * 50)
    
    base_path = f"../../data/data/processed/{dataset_name}"
    edge_file = f"{base_path}/graph.edgelist"
    label_file = f"{base_path}/labels.txt"
    
    # 파일 존재 확인
    if not os.path.exists(edge_file):
        print(f"❌ 엣지 파일 없음: {edge_file}")
        return
    
    if not os.path.exists(label_file):
        print(f"❌ 레이블 파일 없음: {label_file}")
        return
    
    # 엣지 파일 샘플
    print(f"📄 엣지 파일 샘플 ({edge_file}):")
    with open(edge_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(f"  {i+1}: {line.strip()}")
    
    # 레이블 파일 샘플
    print(f"\n🏷️  레이블 파일 샘플 ({label_file}):")
    with open(label_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(f"  {i+1}: {line.strip()}")
    
    # 그래프 로드 및 분석
    try:
        G = nx.read_edgelist(edge_file)
        print(f"\n📊 그래프 정보:")
        print(f"  노드 수: {G.number_of_nodes()}")
        print(f"  엣지 수: {G.number_of_edges()}")
        print(f"  노드 타입: {type(list(G.nodes())[0])}")
        print(f"  노드 샘플: {list(G.nodes())[:5]}")
        
    except Exception as e:
        print(f"❌ 그래프 로드 실패: {e}")
        return
    
    # 레이블 로드 및 분석  
    try:
        labels = {}
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    node, label = parts[0], int(parts[1])
                    labels[node] = label
        
        print(f"\n🏷️  레이블 정보:")
        print(f"  레이블 노드 수: {len(labels)}")
        print(f"  클래스 수: {len(set(labels.values()))}")
        print(f"  노드 타입: {type(list(labels.keys())[0])}")
        print(f"  노드 샘플: {list(labels.keys())[:5]}")
        print(f"  클래스 분포: {dict(sorted({v: list(labels.values()).count(v) for v in set(labels.values())}.items()))}")
        
    except Exception as e:
        print(f"❌ 레이블 로드 실패: {e}")
        return
    
    # 매칭 확인
    graph_nodes = set(G.nodes())
    label_nodes = set(labels.keys())
    common_nodes = graph_nodes & label_nodes
    
    print(f"\n🔗 노드 매칭 분석:")
    print(f"  그래프 노드 수: {len(graph_nodes)}")
    print(f"  레이블 노드 수: {len(label_nodes)}")
    print(f"  공통 노드 수: {len(common_nodes)}")
    print(f"  매칭률: {len(common_nodes)/len(graph_nodes)*100:.1f}%")
    
    if len(common_nodes) == 0:
        print(f"⚠️ 노드 매칭 실패!")
        print(f"  그래프 노드 샘플: {list(graph_nodes)[:5]}")
        print(f"  레이블 노드 샘플: {list(label_nodes)[:5]}")
        
        # 타입 변환 시도
        try:
            # 문자열 → 정수 변환
            int_graph_nodes = {int(node) for node in graph_nodes if node.isdigit()}
            int_label_nodes = {int(node) for node in label_nodes if node.isdigit()}
            int_common = int_graph_nodes & int_label_nodes
            print(f"  정수 변환 후 공통: {len(int_common)}")
            
        except:
            print(f"  정수 변환 실패")
    else:
        print(f"✅ 노드 매칭 성공!")
        
        # 간단한 알고리즘 테스트
        try:
            
            
            # 공통 노드만 사용
            G_common = G.subgraph(common_nodes).copy()
            labels_common = {node: labels[node] for node in common_nodes}
            
            # PageRank 테스트
            alg = LeidenLPAHybrid(core_ratio=0.4, centrality_method='pagerank')
            result = alg.fit_predict(G_common)
            
            print(f"\n🧪 알고리즘 테스트 (PageRank):")
            print(f"  결과 노드 수: {len(result)}")
            print(f"  커뮤니티 수: {len(set(result.values()))}")
            print(f"  커뮤니티 분포: {dict(sorted({v: list(result.values()).count(v) for v in set(result.values())}.items()))}")
            
        except Exception as e:
            print(f"❌ 알고리즘 테스트 실패: {e}")

def main():
    """전체 데이터셋 검증"""
    datasets = ['karate', 'cora', 'citeseer', 'pubmed']
    
    print("🔍 데이터 전처리 검증 시작")
    print("=" * 60)
    
    for dataset in datasets:
        check_dataset(dataset)
    
    print(f"\n✅ 검증 완료!")

if __name__ == "__main__":
    main()