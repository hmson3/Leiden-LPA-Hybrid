import os
import wget
import tarfile
import networkx as nx

def download_citation_datasets(base_dir="citation_datasets"):
    """
    Cora, CiteSeer, PubMed 데이터셋 다운로드 및 변환
    """
    os.makedirs(base_dir, exist_ok=True)
    
    datasets = {
        "cora": {
            "url": "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
            "files": ["cora.cites", "cora.content"],
            "nodes": 2708,
            "edges": 5429,
            "classes": 7
        },
        "citeseer": {
            "url": "https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz", 
            "files": ["citeseer.cites", "citeseer.content"],
            "nodes": 3312,
            "edges": 4732,
            "classes": 6
        },
        "pubmed": {
            "url": "https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz",
            "files": ["data/Pubmed-Diabetes.DIRECTED.cites.tab", 
                     "data/Pubmed-Diabetes.NODE.paper.tab"],
            "nodes": 19717,
            "edges": 44338,
            "classes": 3
        }
    }
    
    for name, info in datasets.items():
        print(f"\n[INFO] Processing {name.upper()} dataset...")
        dataset_dir = os.path.join(base_dir, name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # 다운로드
        archive_path = os.path.join(dataset_dir, f"{name}.tgz")
        if not os.path.exists(archive_path):
            print(f"  Downloading from {info['url']}")
            wget.download(info['url'], archive_path)
        
        # 압축 해제
        extract_dir = os.path.join(dataset_dir, "raw")
        if not os.path.exists(extract_dir):
            print(f"  Extracting {archive_path}")
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
        
        # 변환
        convert_to_standard_format(name, extract_dir, dataset_dir)
        print(f"  ✅ {name} conversion complete!")

def convert_to_standard_format(dataset_name, raw_dir, output_dir):
    """
    원본 형식을 표준 형식으로 변환
    """
    if dataset_name in ["cora", "citeseer"]:
        convert_cora_citeseer(dataset_name, raw_dir, output_dir)
    elif dataset_name == "pubmed":
        convert_pubmed(raw_dir, output_dir)

def convert_cora_citeseer(dataset_name, raw_dir, output_dir):
    """
    Cora/CiteSeer 형식 변환
    """
    # 파일 찾기
    content_file = None
    cites_file = None
    
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.endswith('.content'):
                content_file = os.path.join(root, file)
            elif file.endswith('.cites'):
                cites_file = os.path.join(root, file)
    
    if not content_file or not cites_file:
        raise FileNotFoundError(f"Required files not found in {raw_dir}")
    
    # 노드 레이블 읽기
    node_labels = {}
    class_to_id = {}
    class_counter = 0
    
    with open(content_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            paper_id = parts[0]
            class_label = parts[-1]
            
            if class_label not in class_to_id:
                class_to_id[class_label] = class_counter
                class_counter += 1
            
            node_labels[paper_id] = class_to_id[class_label]
    
    # 엣지 읽기 및 그래프 생성
    G = nx.Graph()
    with open(cites_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                cited = parts[0]
                citing = parts[1]
                # 두 노드가 모두 레이블이 있는 경우만 추가
                if cited in node_labels and citing in node_labels:
                    G.add_edge(cited, citing)
    
    # 연결된 컴포넌트만 유지
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    # 표준 형식으로 저장
    # 1. graph.edgelist
    edge_file = os.path.join(output_dir, "graph.edgelist")
    with open(edge_file, 'w') as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
    
    # 2. labels.txt
    label_file = os.path.join(output_dir, "labels.txt")
    with open(label_file, 'w') as f:
        for node in G.nodes():
            if node in node_labels:
                f.write(f"{node} {node_labels[node]}\n")
    
    print(f"    Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"    Classes: {len(class_to_id)}")

def convert_pubmed(raw_dir, output_dir):
    """
    PubMed 형식 변환 - 실제 파일 구조에 맞게 수정
    """
    # 파일 찾기
    cites_file = None
    node_file = None
    
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if 'cites.tab' in file:
                cites_file = os.path.join(root, file)
            elif 'paper.tab' in file:
                node_file = os.path.join(root, file)
    
    if not cites_file or not node_file:
        raise FileNotFoundError(f"Required files not found in {raw_dir}")
    
    print(f"    Found files: {cites_file}, {node_file}")
    
    # 노드 파일 구조 파악
    print(f"    Inspecting node file structure...")
    with open(node_file, 'r') as f:
        header = f.readline().strip()
        print(f"    Header: {header}")
        
        # 첫 몇 줄 샘플 확인
        for i in range(3):
            line = f.readline().strip()
            if line:
                print(f"    Sample line {i+1}: {line}")
    
    # 노드 레이블 읽기
    node_labels = {}
    class_to_id = {}
    class_counter = 0
    
    with open(node_file, 'r') as f:
        f.readline()  # 헤더 스킵
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) < 2:
                continue
                
            try:
                paper_id = parts[0]
                
                # 다양한 레이블 형식 시도
                class_label = None
                if '=' in parts[1]:
                    # label=class_name 형식
                    class_label = parts[1].split('=')[1]
                else:
                    # 직접 레이블인 경우
                    class_label = parts[1]
                
                if class_label:
                    if class_label not in class_to_id:
                        class_to_id[class_label] = class_counter
                        class_counter += 1
                    
                    node_labels[paper_id] = class_to_id[class_label]
                    
            except Exception as e:
                print(f"    Warning: Error parsing line {line_num}: {e}")
                print(f"    Line content: {line}")
                continue
    
    print(f"    Loaded {len(node_labels)} nodes with {len(class_to_id)} classes")
    print(f"    Classes: {list(class_to_id.keys())}")
    
    # 엣지 파일 구조 파악
    print(f"    Inspecting citation file structure...")
    with open(cites_file, 'r') as f:
        header = f.readline().strip()
        print(f"    Header: {header}")
        
        # 첫 몇 줄 샘플 확인
        for i in range(3):
            line = f.readline().strip()
            if line:
                print(f"    Sample line {i+1}: {line}")
    
    # 엣지 읽기
    G = nx.Graph()
    edge_count = 0
    
    with open(cites_file, 'r') as f:
        f.readline()  # 헤더 스킵
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) < 2:
                continue
                
            try:
                # 다양한 ID 형식 시도
                if ':' in parts[0]:
                    citing = parts[0].split(':')[1]
                else:
                    citing = parts[0]
                    
                if ':' in parts[1]:
                    cited = parts[1].split(':')[1]
                else:
                    cited = parts[1]
                
                if citing in node_labels and cited in node_labels:
                    G.add_edge(citing, cited)
                    edge_count += 1
                    
            except Exception as e:
                if line_num <= 10:  # 처음 10개 에러만 출력
                    print(f"    Warning: Error parsing citation line {line_num}: {e}")
                continue
    
    print(f"    Loaded {edge_count} edges")
    
    # 연결된 컴포넌트만 유지
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        largest_cc = max(components, key=len)
        print(f"    Graph has {len(components)} components, using largest with {len(largest_cc)} nodes")
        G = G.subgraph(largest_cc).copy()
    
    # 표준 형식으로 저장
    edge_file = os.path.join(output_dir, "graph.edgelist")
    with open(edge_file, 'w') as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
    
    label_file = os.path.join(output_dir, "labels.txt")
    with open(label_file, 'w') as f:
        for node in G.nodes():
            if node in node_labels:
                f.write(f"{node} {node_labels[node]}\n")
    
    print(f"    Final: Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"    Classes: {len(class_to_id)}")

if __name__ == "__main__":
    download_citation_datasets()
    print("\n🎉 All citation datasets downloaded and converted!")