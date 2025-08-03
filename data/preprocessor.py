import os
import wget
import tarfile
import zipfile
import networkx as nx
from typing import Dict, List, Tuple, Optional
import pandas as pd

class DatasetPreprocessor:
    """
    다양한 네트워크 데이터셋을 표준 형식으로 변환하는 전처리기
    표준 형식: graph.edgelist + labels.txt
    """
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, "raw")
        self.processed_dir = os.path.join(base_dir, "processed")
        
        # 데이터셋 정보 정의
        self.datasets = {
            'tiny': {
                'karate': {'source': 'networkx', 'has_labels': True}
            },
            'medium': {
                'cora': {'source': 'citation', 'has_labels': True},
                'citeseer': {'source': 'citation', 'has_labels': True}
            },
            'large': {
                'pubmed': {'source': 'citation', 'has_labels': True}
            }
        }
        
        # 인용 네트워크 다운로드 URL
        self.citation_urls = {
            "cora": "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
            "citeseer": "https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz", 
            "pubmed": "https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz"
        }
        
        self._create_directories()
    
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        for size_category in self.datasets:
            for dataset_name in self.datasets[size_category]:
                dataset_path = os.path.join(self.processed_dir, dataset_name)
                os.makedirs(dataset_path, exist_ok=True)
        
        os.makedirs(self.raw_dir, exist_ok=True)
    
    def process_all_datasets(self):
        """모든 데이터셋 처리"""
        print("🚀 데이터셋 전처리 시작...")
        
        for size_category, datasets in self.datasets.items():
            print(f"\n📊 {size_category.upper()} 데이터셋 처리 중...")
            
            for dataset_name, info in datasets.items():
                try:
                    print(f"\n  [{dataset_name}] 처리 시작...")
                    self.process_single_dataset(dataset_name, info)
                    print(f"  ✅ [{dataset_name}] 완료!")
                
                except Exception as e:
                    print(f"  ❌ [{dataset_name}] 실패: {e}")
                    self._create_placeholder(dataset_name)
        
        print("\n🎉 전처리 완료!")
        self._print_summary()
    
    def process_single_dataset(self, dataset_name: str, info: Dict):
        """단일 데이터셋 처리"""
        output_dir = os.path.join(self.processed_dir, dataset_name)
        
        # 이미 처리된 경우 스킵
        if self._is_already_processed(output_dir):
            print(f"    이미 처리됨 - 스킵")
            return
        
        source_type = info['source']
        
        if source_type == 'networkx':
            self._process_networkx_dataset(dataset_name, output_dir)
        elif source_type == 'citation':
            self._process_citation_dataset(dataset_name, output_dir)
        elif source_type == 'manual':
            self._create_manual_placeholder(dataset_name, output_dir)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
    
    def _is_already_processed(self, output_dir: str) -> bool:
        """처리 완료 여부 확인"""
        edge_file = os.path.join(output_dir, "graph.edgelist")
        return os.path.exists(edge_file)
    
    def _process_networkx_dataset(self, dataset_name: str, output_dir: str):
        """NetworkX 내장 데이터셋 처리"""
        if dataset_name == 'karate':
            G = nx.karate_club_graph()
            # Ground truth 레이블 생성
            true_labels = {}
            for node in G.nodes():
                true_labels[node] = 0 if G.nodes[node]['club'] == 'Mr. Hi' else 1
        
        elif dataset_name == 'football':
            G = nx.read_gml("data/raw/football.gml")  # 수동 다운로드 필요
            true_labels = {node: G.nodes[node]['value'] for node in G.nodes()}
        
        elif dataset_name == 'lesmis':
            G = nx.les_miserables_graph()
            true_labels = None  # Ground truth 없음
        
        elif dataset_name == 'dolphins':
            G = nx.read_gml("data/raw/dolphins.gml")  # 수동 다운로드 필요
            true_labels = None  # Ground truth 없음
        
        else:
            raise ValueError(f"Unknown NetworkX dataset: {dataset_name}")
        
        self._save_standard_format(G, true_labels, output_dir)
    
    def _process_citation_dataset(self, dataset_name: str, output_dir: str):
        """인용 네트워크 데이터셋 처리"""
        # 다운로드
        raw_dataset_dir = os.path.join(self.raw_dir, dataset_name)
        self._download_citation_dataset(dataset_name, raw_dataset_dir)
        
        # 변환
        if dataset_name in ["cora", "citeseer"]:
            self._convert_cora_citeseer(dataset_name, raw_dataset_dir, output_dir)
        elif dataset_name == "pubmed":
            self._convert_pubmed(raw_dataset_dir, output_dir)
    
    def _download_citation_dataset(self, dataset_name: str, raw_dir: str):
        """인용 네트워크 다운로드"""
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir, exist_ok=True)
            
            url = self.citation_urls[dataset_name]
            archive_path = os.path.join(raw_dir, f"{dataset_name}.tgz")
            
            print(f"    다운로드 중: {url}")
            wget.download(url, archive_path)
            
            print(f"\n    압축 해제 중...")
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(raw_dir)
    
    def _convert_cora_citeseer(self, dataset_name: str, raw_dir: str, output_dir: str):
        """Cora/CiteSeer 형식 변환"""
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
        
        print(f"    Found: {os.path.basename(content_file)}, {os.path.basename(cites_file)}")
        
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
        
        # 엣지 읽기
        G = nx.Graph()
        with open(cites_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    cited = parts[0]
                    citing = parts[1]
                    if cited in node_labels and citing in node_labels:
                        G.add_edge(cited, citing)
        
        # 연결된 컴포넌트만 유지
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        
        print(f"    Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        print(f"    Classes: {len(class_to_id)}")
        
        self._save_standard_format(G, node_labels, output_dir)
    
    def _convert_pubmed(self, raw_dir: str, output_dir: str):
        """PubMed 형식 변환"""
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
        
        print(f"    Found: {os.path.basename(node_file)}, {os.path.basename(cites_file)}")
        
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
                    
                    # 레이블 추출 (다양한 형식 지원)
                    class_label = None
                    if '=' in parts[1]:
                        class_label = parts[1].split('=')[1]
                    else:
                        class_label = parts[1]
                    
                    if class_label:
                        if class_label not in class_to_id:
                            class_to_id[class_label] = class_counter
                            class_counter += 1
                        
                        node_labels[paper_id] = class_to_id[class_label]
                        
                except Exception as e:
                    if line_num <= 10:  # 처음 10개 에러만 출력
                        print(f"    Warning: Line {line_num} parse error")
                    continue
        
        # 엣지 읽기
        G = nx.Graph()
        edge_count = 0
        
        with open(cites_file, 'r') as f:
            f.readline()  # 헤더 스킵
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                    
                try:
                    citing = parts[0].split(':')[-1] if ':' in parts[0] else parts[0]
                    cited = parts[1].split(':')[-1] if ':' in parts[1] else parts[1]
                    
                    if citing in node_labels and cited in node_labels:
                        G.add_edge(citing, cited)
                        edge_count += 1
                        
                except Exception:
                    continue
        
        # 연결된 컴포넌트만 유지
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        
        print(f"    Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        print(f"    Classes: {len(class_to_id)}")
        
        self._save_standard_format(G, node_labels, output_dir)
    
    def _save_standard_format(self, G: nx.Graph, labels: Optional[Dict], output_dir: str):
        """표준 형식으로 저장"""
        # graph.edgelist
        edge_file = os.path.join(output_dir, "graph.edgelist")
        with open(edge_file, 'w') as f:
            for u, v in G.edges():
                f.write(f"{u} {v}\n")
        
        # labels.txt (라벨이 있는 경우만)
        if labels:
            label_file = os.path.join(output_dir, "labels.txt")
            with open(label_file, 'w') as f:
                for node in G.nodes():
                    if node in labels:
                        f.write(f"{node} {labels[node]}\n")
        
        # metadata.txt
        meta_file = os.path.join(output_dir, "metadata.txt")
        with open(meta_file, 'w') as f:
            f.write(f"nodes: {G.number_of_nodes()}\n")
            f.write(f"edges: {G.number_of_edges()}\n")
            f.write(f"has_labels: {labels is not None}\n")
            if labels:
                f.write(f"num_classes: {len(set(labels.values()))}\n")
    
    def _create_manual_placeholder(self, dataset_name: str, output_dir: str):
        """수동 다운로드 플레이스홀더 생성"""
        placeholder_file = os.path.join(output_dir, "MANUAL_DOWNLOAD_REQUIRED.txt")
        
        manual_instructions = {
            'polbooks': """
Political Books Dataset
======================
1. Download from: http://www-personal.umich.edu/~mejn/netdata/polbooks.zip
2. Extract to: data/raw/polbooks/
3. Run this preprocessor again

Expected files:
- polbooks.gml
            """,
            'email-eu': """
Email-Eu-core Dataset
====================
1. Download from: https://snap.stanford.edu/data/email-Eu-core.html
2. Extract to: data/raw/email-eu/
3. Run this preprocessor again

Expected files:
- email-Eu-core.txt
- email-Eu-core-department-labels.txt
            """,
            'dolphins': """
Dolphins Dataset
===============
1. Download from: http://www-personal.umich.edu/~mejn/netdata/dolphins.zip
2. Extract to: data/raw/dolphins/
3. Run this preprocessor again

Expected files:
- dolphins.gml
            """,
            'football': """
Football Dataset
===============
1. Download from: http://www-personal.umich.edu/~mejn/netdata/football.zip
2. Extract to: data/raw/football/
3. Run this preprocessor again

Expected files:
- football.gml
            """
        }
        
        with open(placeholder_file, 'w') as f:
            f.write(manual_instructions.get(dataset_name, f"Manual download required for {dataset_name}"))
        
        print(f"    ⚠️  수동 다운로드 필요 - 지침서 생성됨")
    
    def _create_placeholder(self, dataset_name: str):
        """실패한 데이터셋용 플레이스홀더"""
        output_dir = os.path.join(self.processed_dir, dataset_name)
        placeholder_file = os.path.join(output_dir, "ERROR.txt")
        
        with open(placeholder_file, 'w') as f:
            f.write(f"데이터셋 {dataset_name} 처리 실패\n")
            f.write("수동으로 처리하거나 오류를 확인하세요.\n")
    
    def _print_summary(self):
        """처리 결과 요약"""
        print("\n📋 처리 결과 요약:")
        print("=" * 50)
        
        for size_category, datasets in self.datasets.items():
            print(f"\n{size_category.upper()} 데이터셋:")
            for dataset_name in datasets:
                output_dir = os.path.join(self.processed_dir, dataset_name)
                edge_file = os.path.join(output_dir, "graph.edgelist")
                
                if os.path.exists(edge_file):
                    # 메타데이터 읽기
                    meta_file = os.path.join(output_dir, "metadata.txt")
                    if os.path.exists(meta_file):
                        with open(meta_file, 'r') as f:
                            meta = dict(line.strip().split(': ') for line in f if ': ' in line)
                        print(f"  ✅ {dataset_name:12} | {meta.get('nodes', '?')} nodes, {meta.get('edges', '?')} edges")
                    else:
                        print(f"  ✅ {dataset_name:12} | 처리됨")
                else:
                    print(f"  ❌ {dataset_name:12} | 실패/수동 다운로드 필요")
    
    def load_dataset(self, dataset_name: str) -> Tuple[nx.Graph, Optional[Dict]]:
        """처리된 데이터셋 로드"""
        dataset_dir = os.path.join(self.processed_dir, dataset_name)
        
        # 그래프 로드
        edge_file = os.path.join(dataset_dir, "graph.edgelist")
        if not os.path.exists(edge_file):
            raise FileNotFoundError(f"Dataset {dataset_name} not found or not processed")
        
        G = nx.read_edgelist(edge_file)
        
        # 라벨 로드 (있는 경우)
        label_file = os.path.join(dataset_dir, "labels.txt")
        labels = None
        
        if os.path.exists(label_file):
            labels = {}
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        node, label = parts[0], int(parts[1])
                        labels[node] = label
        
        return G, labels
    
    def get_available_datasets(self) -> Dict[str, List[str]]:
        """사용 가능한 데이터셋 목록 반환"""
        available = {}
        
        for size_category, datasets in self.datasets.items():
            available[size_category] = []
            for dataset_name in datasets:
                output_dir = os.path.join(self.processed_dir, dataset_name)
                edge_file = os.path.join(output_dir, "graph.edgelist")
                
                if os.path.exists(edge_file):
                    available[size_category].append(dataset_name)
        
        return available

def main():
    """메인 실행 함수"""
    preprocessor = DatasetPreprocessor()
    preprocessor.process_all_datasets()
    
    # 사용 가능한 데이터셋 확인
    available = preprocessor.get_available_datasets()
    print(f"\n🎯 사용 가능한 데이터셋: {sum(len(v) for v in available.values())}개")

if __name__ == "__main__":
    main()