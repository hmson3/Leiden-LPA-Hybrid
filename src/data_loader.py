#!/usr/bin/env python3
"""
데이터 로더 모듈
다양한 네트워크 데이터셋을 표준 형식으로 로드
"""

import os
import networkx as nx
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict

def load_dataset(dataset_name: str) -> Tuple[nx.Graph, Dict[str, int]]:
    """
    데이터셋 로드 (표준 형식: graph.edgelist + labels.txt)
    
    Parameters:
    -----------
    dataset_name : str
        데이터셋 이름 ('karate', 'cora', 'citeseer', 'pubmed')
    
    Returns:
    --------
    G_nx : networkx.Graph
        네트워크 그래프
    true_labels : dict
        {node_id: label} 형태의 정답 라벨
    """
    
    # 프로젝트 루트 디렉토리 찾기
    current_dir = Path(__file__).parent
    while not (current_dir / 'data').exists() and current_dir != current_dir.parent:
        current_dir = current_dir.parent
    
    data_dir = current_dir / 'data' / 'processed' / dataset_name
    
    if dataset_name == 'karate':
        return load_karate_club()
    elif dataset_name in ['cora', 'citeseer', 'pubmed']:
        return load_citation_dataset(data_dir, dataset_name)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_karate_club() -> Tuple[nx.Graph, Dict[str, int]]:
    """
    Karate Club 데이터셋 로드 (NetworkX 내장)
    """
    G = nx.karate_club_graph()
    
    # 노드 ID를 문자열로 변환 (일관성을 위해)
    G = nx.relabel_nodes(G, {i: str(i) for i in G.nodes()})
    
    # Ground truth 라벨 생성 (Zachary's original split)
    true_labels = {}
    for node in G.nodes():
        club = G.nodes[node]['club']
        true_labels[node] = 0 if club == 'Mr. Hi' else 1
    
    print(f"✅ Karate Club: {G.number_of_nodes()} 노드, {G.number_of_edges()} 엣지")
    return G, true_labels

def load_citation_dataset(data_dir: Path, dataset_name: str) -> Tuple[nx.Graph, Dict[str, int]]:
    """
    Citation 데이터셋 로드 (Cora, CiteSeer, PubMed)
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    # 파일 경로
    edgelist_path = data_dir / 'graph.edgelist'
    labels_path = data_dir / 'labels.txt'
    
    if not edgelist_path.exists() or not labels_path.exists():
        # 원본 데이터에서 변환 시도
        print(f"⚠️  표준 형식 파일이 없습니다. 원본 데이터에서 변환 시도: {dataset_name}")
        convert_raw_dataset(dataset_name)
        
        # 다시 확인
        if not edgelist_path.exists() or not labels_path.exists():
            raise FileNotFoundError(f"Required files not found: {edgelist_path}, {labels_path}")
    
    try:
        # 그래프 로드
        print(f"📊 Loading {dataset_name} from {data_dir}")
        G = nx.read_edgelist(str(edgelist_path), nodetype=str)
        
        # 라벨 로드
        labels_df = pd.read_csv(labels_path, sep='\t', dtype={'node': str, 'label': int})
        true_labels = dict(zip(labels_df['node'], labels_df['label']))
        
        # 노드 매칭 확인
        graph_nodes = set(G.nodes())
        label_nodes = set(true_labels.keys())
        common_nodes = graph_nodes & label_nodes
        
        print(f"  📈 그래프: {len(graph_nodes)} 노드, {G.number_of_edges()} 엣지")
        print(f"  🏷️  라벨: {len(label_nodes)} 노드")
        print(f"  🎯 매칭: {len(common_nodes)} 노드 ({len(common_nodes)/len(graph_nodes)*100:.1f}%)")
        
        if len(common_nodes) == 0:
            raise ValueError("그래프와 라벨 간 노드 매칭이 없습니다!")
        
        # 매칭되는 노드만 유지
        if len(common_nodes) < len(graph_nodes):
            print(f"  ⚠️  {len(graph_nodes) - len(common_nodes)}개 노드를 제거합니다.")
            G = G.subgraph(common_nodes).copy()
            true_labels = {node: label for node, label in true_labels.items() if node in common_nodes}
        
        print(f"✅ {dataset_name}: {G.number_of_nodes()} 노드, {G.number_of_edges()} 엣지, {len(set(true_labels.values()))} 클래스")
        return G, true_labels
        
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        raise

def convert_raw_dataset(dataset_name: str):
    """
    원본 데이터를 표준 형식으로 변환
    """
    print(f"🔄 {dataset_name} 원본 데이터 변환 중...")
    
    # 프로젝트 루트 디렉토리 찾기
    current_dir = Path(__file__).parent
    while not (current_dir / 'data').exists() and current_dir != current_dir.parent:
        current_dir = current_dir.parent
    
    raw_dir = current_dir / 'data' / 'raw' / dataset_name
    processed_dir = current_dir / 'data' / 'processed' / dataset_name
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    
    if dataset_name in ['cora', 'citeseer']:
        convert_cora_citeseer(raw_dir, processed_dir, dataset_name)
    elif dataset_name == 'pubmed':
        convert_pubmed(raw_dir, processed_dir)
    else:
        raise ValueError(f"Conversion not supported for: {dataset_name}")

def convert_cora_citeseer(raw_dir: Path, processed_dir: Path, dataset_name: str):
    """
    Cora/CiteSeer 원본 데이터 변환
    """
    # 파일 찾기
    content_file = None
    cites_file = None
    
    for file_path in raw_dir.rglob('*'):
        if file_path.name.endswith('.content'):
            content_file = file_path
        elif file_path.name.endswith('.cites'):
            cites_file = file_path
    
    if not content_file or not cites_file:
        raise FileNotFoundError(f"Required files (.content, .cites) not found in {raw_dir}")
    
    print(f"  📄 Content file: {content_file}")
    print(f"  📄 Cites file: {cites_file}")
    
    # 1. 노드 라벨 읽기
    print("  🏷️  Reading node labels...")
    node_labels = {}
    class_to_id = {}
    class_counter = 0
    
    with open(content_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                paper_id = parts[0]
                class_label = parts[-1]
                
                if class_label not in class_to_id:
                    class_to_id[class_label] = class_counter
                    class_counter += 1
                
                node_labels[paper_id] = class_to_id[class_label]
    
    print(f"    - {len(node_labels)} 노드, {len(class_to_id)} 클래스")
    
    # 2. 엣지 읽기
    print("  🔗 Reading edges...")
    edges = []
    with open(cites_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                cited = parts[0]
                citing = parts[1]
                # 두 노드가 모두 라벨이 있는 경우만 추가
                if cited in node_labels and citing in node_labels:
                    edges.append((cited, citing))
    
    print(f"    - {len(edges)} 엣지")
    
    # 3. 그래프 생성 및 연결성 확인
    print("  🌐 Creating graph...")
    G = nx.Graph()
    G.add_edges_from(edges)
    
    # 가장 큰 연결 컴포넌트만 유지
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        node_labels = {node: label for node, label in node_labels.items() if node in G.nodes()}
        print(f"    - 가장 큰 연결 컴포넌트 선택: {G.number_of_nodes()} 노드")
    
    # 4. 파일 저장
    print("  💾 Saving processed data...")
    
    # 엣지리스트 저장
    edgelist_path = processed_dir / 'graph.edgelist'
    nx.write_edgelist(G, edgelist_path, data=False)
    
    # 라벨 저장
    labels_path = processed_dir / 'labels.txt'
    with open(labels_path, 'w') as f:
        f.write("node\tlabel\n")
        for node in G.nodes():
            if node in node_labels:
                f.write(f"{node}\t{node_labels[node]}\n")
    
    print(f"✅ {dataset_name} 변환 완료: {G.number_of_nodes()} 노드, {G.number_of_edges()} 엣지")

def convert_pubmed(raw_dir: Path, processed_dir: Path):
    """
    PubMed 원본 데이터 변환
    """
    # PubMed 파일 찾기
    cites_file = None
    content_file = None
    
    for file_path in raw_dir.rglob('*'):
        if 'cites' in file_path.name.lower():
            cites_file = file_path
        elif 'paper' in file_path.name.lower():
            content_file = file_path
    
    if not cites_file or not content_file:
        raise FileNotFoundError(f"Required PubMed files not found in {raw_dir}")
    
    print(f"  📄 Cites file: {cites_file}")
    print(f"  📄 Content file: {content_file}")
    
    # 1. 노드 라벨 읽기
    print("  🏷️  Reading node labels...")
    node_labels = {}
    class_to_id = {}
    class_counter = 0
    
    with open(content_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f):
            if line_num == 0:  # 헤더 스킵
                continue
            
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                # PubMed 형식: paper:ID \t label \t ...
                paper_id = parts[0]
                if paper_id.startswith('paper:'):
                    paper_id = paper_id[6:]  # 'paper:' 접두사 제거
                
                class_label = parts[1]
                
                if class_label not in class_to_id:
                    class_to_id[class_label] = class_counter
                    class_counter += 1
                
                node_labels[paper_id] = class_to_id[class_label]
    
    print(f"    - {len(node_labels)} 노드, {len(class_to_id)} 클래스")
    
    # 2. 엣지 읽기
    print("  🔗 Reading edges...")
    edges = []
    with open(cites_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f):
            if line_num == 0:  # 헤더 스킵
                continue
                
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                # PubMed 엣지 형식 처리
                cited = parts[0]
                citing = parts[1]
                
                # 'paper:' 접두사 제거
                if cited.startswith('paper:'):
                    cited = cited[6:]
                if citing.startswith('paper:'):
                    citing = citing[6:]
                
                # 두 노드가 모두 라벨이 있는 경우만 추가
                if cited in node_labels and citing in node_labels:
                    edges.append((cited, citing))
    
    print(f"    - {len(edges)} 엣지")
    
    # 3. 그래프 생성
    print("  🌐 Creating graph...")
    G = nx.Graph()
    G.add_edges_from(edges)
    
    # 가장 큰 연결 컴포넌트만 유지
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        node_labels = {node: label for node, label in node_labels.items() if node in G.nodes()}
        print(f"    - 가장 큰 연결 컴포넌트 선택: {G.number_of_nodes()} 노드")
    
    # 4. 파일 저장
    print("  💾 Saving processed data...")
    
    # 엣지리스트 저장
    edgelist_path = processed_dir / 'graph.edgelist'
    nx.write_edgelist(G, edgelist_path, data=False)
    
    # 라벨 저장
    labels_path = processed_dir / 'labels.txt'
    with open(labels_path, 'w') as f:
        f.write("node\tlabel\n")
        for node in G.nodes():
            if node in node_labels:
                f.write(f"{node}\t{node_labels[node]}\n")
    
    print(f"✅ PubMed 변환 완료: {G.number_of_nodes()} 노드, {G.number_of_edges()} 엣지")

def check_dataset_status():
    """
    모든 데이터셋의 상태 확인
    """
    datasets = ['karate', 'cora', 'citeseer', 'pubmed']
    
    print("📊 데이터셋 상태 확인")
    print("=" * 50)
    
    for dataset in datasets:
        try:
            G, true_labels = load_dataset(dataset)
            classes = len(set(true_labels.values()))
            print(f"✅ {dataset:8}: {G.number_of_nodes():5} 노드, {G.number_of_edges():5} 엣지, {classes} 클래스")
        except Exception as e:
            print(f"❌ {dataset:8}: {str(e)[:50]}...")

if __name__ == "__main__":
    # 데이터셋 상태 확인
    check_dataset_status()