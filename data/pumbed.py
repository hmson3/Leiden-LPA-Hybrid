import os
import networkx as nx

def fix_pubmed_id_matching(raw_dir="data/raw/pubmed", output_dir="data/processed/pubmed"):
    """PubMed ID 매칭 문제 해결된 변환기"""
    print("🔧 PubMed ID 매칭 문제 수정 중...")
    print("=" * 50)
    
    # 파일 찾기
    node_files = []
    cite_files = []
    
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if 'paper.tab' in file:
                node_files.append(os.path.join(root, file))
            elif 'cites.tab' in file:
                cite_files.append(os.path.join(root, file))
    
    node_file = node_files[0]
    cite_file = cite_files[0]
    
    print(f"📊 처리할 파일:")
    print(f"   노드: {os.path.basename(node_file)}")
    print(f"   인용: {os.path.basename(cite_file)}")
    
    # 1. 노드 파일에서 ID와 레이블 추출
    print(f"\n📋 노드 데이터 로딩...")
    node_labels = {}
    class_to_id = {}
    
    with open(node_file, 'r', encoding='utf-8') as f:
        f.readline()  # 헤더 스킵
        
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) < 2:
                continue
                
            try:
                # 노드 ID 추출 (숫자)
                node_id = parts[0].strip()
                
                # 레이블 추출 (label=숫자 형식)
                label_found = False
                for part in parts[1:]:
                    if part.startswith('label='):
                        class_label = part.split('=')[1]
                        
                        # 숫자 레이블만 처리 (1, 2, 3)
                        if class_label.isdigit():
                            if class_label not in class_to_id:
                                class_to_id[class_label] = len(class_to_id)
                            
                            node_labels[node_id] = class_to_id[class_label]
                            label_found = True
                            break
                
                if not label_found and line_num <= 5:
                    print(f"   ⚠️  라인 {line_num}: 레이블 없음")
                    
            except Exception as e:
                if line_num <= 5:
                    print(f"   ⚠️  라인 {line_num} 파싱 오류: {e}")
                continue
    
    print(f"   로딩된 노드: {len(node_labels):,}")
    print(f"   클래스: {list(class_to_id.keys())}")
    
    # 2. 인용 파일에서 엣지 추출 (ID 형식 변환)
    print(f"\n🔗 인용 데이터 로딩...")
    G = nx.Graph()
    edge_count = 0
    skip_count = 0
    format_issues = 0
    
    with open(cite_file, 'r', encoding='utf-8') as f:
        f.readline()  # 헤더 스킵
        
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line == "NO_FEATURES":
                continue
                
            parts = line.split('\t')
            if len(parts) < 4:  # 최소 4개 컬럼 필요
                skip_count += 1
                continue
                
            try:
                # 인용 파일 형식: [edge_id] [paper:citing_id] [|] [paper:cited_id]
                citing_raw = parts[1].strip()  # paper:citing_id
                cited_raw = parts[3].strip()   # paper:cited_id
                
                # "paper:" 제거해서 숫자 ID만 추출
                if citing_raw.startswith('paper:'):
                    citing_id = citing_raw[6:]  # "paper:" 제거
                else:
                    citing_id = citing_raw
                    
                if cited_raw.startswith('paper:'):
                    cited_id = cited_raw[6:]   # "paper:" 제거  
                else:
                    cited_id = cited_raw
                
                # 두 노드가 모두 레이블 데이터에 있는지 확인
                if citing_id in node_labels and cited_id in node_labels:
                    G.add_edge(citing_id, cited_id)
                    edge_count += 1
                else:
                    skip_count += 1
                    
            except Exception as e:
                format_issues += 1
                if format_issues <= 10:
                    print(f"   ⚠️  라인 {line_num} 형식 오류: {e}")
                continue
    
    print(f"   처리된 엣지: {edge_count:,}")
    print(f"   스킵된 라인: {skip_count:,}")
    print(f"   형식 오류: {format_issues}")
    
    # 3. 연결 컴포넌트 확인
    print(f"\n🌐 그래프 분석...")
    print(f"   전체 노드: {G.number_of_nodes():,}")
    print(f"   전체 엣지: {G.number_of_edges():,}")
    
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        largest_cc = max(components, key=len)
        print(f"   연결 컴포넌트: {len(components)}개")
        print(f"   최대 컴포넌트: {len(largest_cc):,} 노드")
        
        # 최대 컴포넌트만 사용
        G = G.subgraph(largest_cc).copy()
        
        # 레이블도 최대 컴포넌트에 있는 노드만 유지
        filtered_labels = {node: node_labels[node] for node in G.nodes() if node in node_labels}
        node_labels = filtered_labels
    
    # 4. 표준 형식으로 저장
    print(f"\n💾 표준 형식으로 저장...")
    os.makedirs(output_dir, exist_ok=True)
    
    # graph.edgelist
    edge_file = os.path.join(output_dir, "graph.edgelist")
    with open(edge_file, 'w') as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
    
    # labels.txt
    label_file = os.path.join(output_dir, "labels.txt")
    with open(label_file, 'w') as f:
        for node in G.nodes():
            if node in node_labels:
                f.write(f"{node} {node_labels[node]}\n")
    
    # metadata.txt
    meta_file = os.path.join(output_dir, "metadata.txt")
    with open(meta_file, 'w') as f:
        f.write(f"nodes: {G.number_of_nodes()}\n")
        f.write(f"edges: {G.number_of_edges()}\n")
        f.write(f"has_labels: True\n")
        f.write(f"num_classes: {len(class_to_id)}\n")
        f.write(f"density: {nx.density(G):.6f}\n")
        f.write(f"avg_degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}\n")
    
    print(f"   ✅ 저장 완료:")
    print(f"   📊 최종 통계:")
    print(f"      노드: {G.number_of_nodes():,}")
    print(f"      엣지: {G.number_of_edges():,}")  
    print(f"      클래스: {len(class_to_id)}")
    print(f"      밀도: {nx.density(G):.6f}")
    print(f"      평균 차수: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    
    return G, node_labels

def verify_pubmed_fix():
    """수정된 PubMed 데이터 검증"""
    print("\n🧪 PubMed 수정 결과 검증")
    print("=" * 50)
    
    try:
        G, labels = fix_pubmed_id_matching()
        
        # 예상 크기와 비교
        expected_nodes = 19717
        expected_edges_min = 40000  # 최소 예상 엣지 수
        
        actual_nodes = G.number_of_nodes()
        actual_edges = G.number_of_edges()
        
        print(f"\n📈 예상 vs 실제:")
        print(f"   노드: {expected_nodes:,} (예상) → {actual_nodes:,} (실제)")
        print(f"   엣지: {expected_edges_min:,}+ (예상) → {actual_edges:,} (실제)")
        
        # 성공 기준
        node_ratio = actual_nodes / expected_nodes
        edge_success = actual_edges >= expected_edges_min
        
        print(f"\n🎯 검증 결과:")
        print(f"   노드 비율: {node_ratio:.2%} ({'✅' if node_ratio > 0.8 else '❌'})")
        print(f"   엣지 충분: {'✅' if edge_success else '❌'}")
        
        if node_ratio > 0.8 and edge_success:
            print(f"\n🎉 PubMed 수정 성공! Large 카테고리 데이터 확보완료!")
            return True
        else:
            print(f"\n⚠️  PubMed 여전히 문제 있음. 추가 디버깅 필요.")
            return False
            
    except Exception as e:
        print(f"\n❌ PubMed 수정 실패: {e}")
        return False

def create_experiment_ready_summary():
    """실험 준비 완료 요약"""
    print("\n🎯 실험 준비 상태 요약")
    print("=" * 50)
    
    # 사용 가능한 데이터셋 확인
    datasets = {
        'karate': {'size': 'tiny', 'expected_nodes': 34, 'has_labels': True},
        'lesmis': {'size': 'small', 'expected_nodes': 77, 'has_labels': False}, 
        'cora': {'size': 'medium', 'expected_nodes': 2485, 'has_labels': True},
        'citeseer': {'size': 'medium', 'expected_nodes': 2110, 'has_labels': True},
        'pubmed': {'size': 'large', 'expected_nodes': 19717, 'has_labels': True}
    }
    
    available_datasets = []
    
    for name, info in datasets.items():
        processed_dir = f"data/processed/{name}"
        edge_file = os.path.join(processed_dir, "graph.edgelist")
        
        if os.path.exists(edge_file):
            # 메타데이터 읽기
            meta_file = os.path.join(processed_dir, "metadata.txt")
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as f:
                    meta = dict(line.strip().split(': ') for line in f if ': ' in line)
                actual_nodes = int(meta.get('nodes', '0'))
                actual_edges = int(meta.get('edges', '0'))
                has_labels = meta.get('has_labels', 'False') == 'True'
            else:
                actual_nodes = 0
                actual_edges = 0
                has_labels = False
            
            status = '✅' if actual_nodes > info['expected_nodes'] * 0.8 else '⚠️'
            available_datasets.append({
                'name': name,
                'size': info['size'],
                'nodes': actual_nodes,
                'edges': actual_edges,
                'labels': has_labels,
                'status': status
            })
    
    print(f"📊 사용 가능한 데이터셋: {len(available_datasets)}개")
    print()
    
    for dataset in available_datasets:
        labels_str = "✅" if dataset['labels'] else "❌"
        print(f"  {dataset['status']} {dataset['name']:10} ({dataset['size']:6}) | "
              f"{dataset['nodes']:5,} nodes, {dataset['edges']:5,} edges, labels: {labels_str}")
    
    # 크기별 분포
    size_distribution = {}
    for dataset in available_datasets:
        size = dataset['size']
        if size not in size_distribution:
            size_distribution[size] = []
        size_distribution[size].append(dataset['name'])
    
    print(f"\n📈 크기별 분포:")
    for size in ['tiny', 'small', 'medium', 'large']:
        if size in size_distribution:
            datasets_str = ', '.join(size_distribution[size])
            print(f"   {size:6}: {datasets_str}")
    
    # 실험 가능성 체크
    total_datasets = len(available_datasets)
    labeled_datasets = sum(1 for d in available_datasets if d['labels'])
    
    print(f"\n🎯 실험 준비도:")
    print(f"   전체 데이터셋: {total_datasets}/5 (60% 이상이면 충분)")
    print(f"   레이블 데이터: {labeled_datasets}/4 (정확도 측정용)")
    
    if total_datasets >= 3 and labeled_datasets >= 2:
        print(f"\n🚀 실험 진행 준비 완료! 첫 번째 실험 시작 가능!")
        return True
    else:
        print(f"\n⚠️  데이터가 부족합니다. 추가 처리 필요.")
        return False

if __name__ == "__main__":
    # 1. PubMed 수정
    success = verify_pubmed_fix()
    
    # 2. 전체 상황 요약
    ready = create_experiment_ready_summary()
    
    if ready:
        print(f"\n🎉 다음 단계: 실험 폴더 구조 생성 및 첫 번째 실험 시작!")