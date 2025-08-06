"""
개선된 Leiden-LPA 하이브리드 커뮤니티 탐지 알고리즘 
- 3가지 앵커 전략 지원: Fixed_Single, Fixed_Iterative, Dynamic_Iterative
- Counter 제거로 성능 최적화
"""
import networkx as nx
import igraph as ig
from leidenalg import find_partition, ModularityVertexPartition
import time
import warnings
from typing import Dict, Optional, Union, List
from centrality import compute_centrality, get_top_nodes

class LeidenLPAHybrid:
    """
    Leiden-LPA 하이브리드 알고리즘 v2 - 3가지 앵커 전략 지원
    """
    
    def __init__(self, 
                 core_ratio: float = 0.4,
                 centrality_method: str = 'pagerank',
                 anchor_strategy: str = 'fixed_iterative',  # 새로운 파라미터!
                 max_lpa_iterations: int = 10,
                 seed: Optional[int] = None):
        """
        Parameters:
        -----------
        core_ratio : float
            핵심 노드 비율 (0.0 ~ 1.0)
        centrality_method : str
            중심성 지표 ('pagerank', 'degree', 'eigenvector', etc.)
        anchor_strategy : str
            앵커 전략 ('fixed_single', 'fixed_iterative', 'dynamic_iterative')
        max_lpa_iterations : int
            LPA 최대 반복 횟수
        seed : int, optional
            랜덤 시드
        """
        self.core_ratio = core_ratio
        self.centrality_method = centrality_method
        self.anchor_strategy = anchor_strategy
        self.max_lpa_iterations = max_lpa_iterations
        self.seed = seed
        
        # 앵커 전략 검증
        valid_strategies = ['fixed_single', 'fixed_iterative', 'dynamic_iterative']
        if anchor_strategy not in valid_strategies:
            raise ValueError(f"anchor_strategy must be one of {valid_strategies}")
        
        # 실행 통계
        self.stats = {
            'centrality_time': 0.0,
            'leiden_time': 0.0,
            'lpa_time': 0.0,
            'total_time': 0.0,
            'core_nodes_count': 0,
            'periphery_nodes_count': 0,
            'lpa_iterations': 0,
            'anchor_strategy': anchor_strategy
        }
    
    def fit_predict(self, G_nx: nx.Graph) -> Dict[str, int]:
        """
        메인 실행 함수
        
        Parameters:
        -----------
        G_nx : networkx.Graph
            입력 그래프
            
        Returns:
        --------
        labels : dict
            {node_id: community_label}
        """
        start_time = time.time()
        
        # 입력 검증
        if not isinstance(G_nx, nx.Graph):
            raise ValueError("Input must be a NetworkX Graph")
        
        if len(G_nx.nodes()) == 0:
            return {}
        
        if not nx.is_connected(G_nx):
            warnings.warn("Graph is not connected, using largest component")
            largest_cc = max(nx.connected_components(G_nx), key=len)
            G_nx = G_nx.subgraph(largest_cc).copy()
        
        # 극단적인 경우 처리
        if self.core_ratio <= 0.0:
            labels = self._run_pure_lpa(G_nx)
        elif self.core_ratio >= 1.0:
            labels = self._run_pure_leiden(G_nx)
        else:
            labels = self._run_hybrid(G_nx)
        
        self.stats['total_time'] = time.time() - start_time
        return labels
    
    def _get_most_common_label(self, neighbor_labels: List[int]) -> Optional[int]:
        """Counter 대신 수동으로 최빈값 계산 - 최적화!"""
        if not neighbor_labels:
            return None
        
        label_counts = {}
        max_count = 0
        most_common_label = None
        
        for label in neighbor_labels:
            count = label_counts.get(label, 0) + 1
            label_counts[label] = count
            if count > max_count:
                max_count = count
                most_common_label = label
        
        return most_common_label
    
    def _run_pure_lpa(self, G_nx: nx.Graph) -> Dict[str, int]:
        """순수 LPA 실행 - Counter 제거"""
        start_time = time.time()
        
        # 초기 라벨 (각 노드 고유 라벨)
        labels = {v: i for i, v in enumerate(G_nx.nodes())}
        
        # LPA 반복
        for iteration in range(self.max_lpa_iterations):
            updated = False
            nodes_list = list(G_nx.nodes())
            
            if self.seed is not None:
                import random
                random.seed(self.seed + iteration)
                random.shuffle(nodes_list)
            
            for v in nodes_list:
                neighbor_labels = [labels[n] for n in G_nx.neighbors(v)]
                if neighbor_labels:
                    most_common = self._get_most_common_label(neighbor_labels)
                    if labels[v] != most_common:
                        labels[v] = most_common
                        updated = True
            
            if not updated:
                break
        
        self.stats['lpa_time'] = time.time() - start_time
        self.stats['lpa_iterations'] = iteration + 1
        self.stats['core_nodes_count'] = 0
        self.stats['periphery_nodes_count'] = len(G_nx.nodes())
        
        return labels
    
    def _run_pure_leiden(self, G_nx: nx.Graph) -> Dict[str, int]:
        """순수 Leiden 실행 - 최적화된 igraph 변환"""
        start_time = time.time()
        
        # 🔥 최적화: 직접 igraph 생성 (TupleList 제거)
        nodes_list = list(G_nx.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes_list)}
        edges_idx = [(node_to_idx[u], node_to_idx[v]) for u, v in G_nx.edges()]
        
        G_ig = ig.Graph(edges_idx, directed=False)
        
        # Leiden 실행
        if self.seed is not None:
            partition = find_partition(G_ig, ModularityVertexPartition, seed=self.seed)
        else:
            partition = find_partition(G_ig, ModularityVertexPartition)
        
        # 🔥 최적화: 직접 매핑 (enumerate 제거)
        labels = {}
        for i in range(len(nodes_list)):
            labels[nodes_list[i]] = partition.membership[i]
        
        self.stats['leiden_time'] = time.time() - start_time
        self.stats['core_nodes_count'] = len(G_nx.nodes())
        self.stats['periphery_nodes_count'] = 0
        
        return labels
    
    def _run_hybrid(self, G_nx: nx.Graph) -> Dict[str, int]:
        """하이브리드 알고리즘 실행"""
        # 1. 중심성 계산
        centrality_start = time.time()
        centrality_scores = compute_centrality(G_nx, self.centrality_method)
        core_nodes = get_top_nodes(centrality_scores, self.core_ratio)
        
        # 🔥 핵심 최적화: 집합 사용!
        core_nodes_set = set(core_nodes)
        periphery_nodes = [v for v in G_nx.nodes() if v not in core_nodes_set]
        
        self.stats['centrality_time'] = time.time() - centrality_start
        self.stats['core_nodes_count'] = len(core_nodes)
        self.stats['periphery_nodes_count'] = len(periphery_nodes)
        
        # 2. 핵심 노드에 Leiden 적용
        leiden_start = time.time()
        if len(core_nodes) == 0:
            core_labels = {}
        else:
            G_core = G_nx.subgraph(core_nodes).copy()
            
            # 🔥 최적화: 직접 igraph 생성 및 빈 그래프 체크
            core_nodes_list = list(core_nodes)
            node_to_idx = {node: i for i, node in enumerate(core_nodes_list)}
            edges_idx = [(node_to_idx[u], node_to_idx[v]) for u, v in G_core.edges()]
            
            if not edges_idx:  # 엣지가 없으면 각 노드를 별도 커뮤니티로
                core_labels = {node: i for i, node in enumerate(core_nodes)}
            else:
                # igraph 생성 및 Leiden 실행
                G_core_ig = ig.Graph(edges_idx, directed=False)
                
                if self.seed is not None:
                    partition = find_partition(G_core_ig, ModularityVertexPartition, seed=self.seed)
                else:
                    partition = find_partition(G_core_ig, ModularityVertexPartition)
                
                # 🔥 수정: igraph 실제 노드 수 기준으로 매핑
                core_labels = {}
                # 연결된 노드들만 매핑 (igraph는 사용된 노드만 포함)
                used_nodes = set()
                for u, v in edges_idx:
                    used_nodes.add(u)
                    used_nodes.add(v)
                
                # 실제 사용된 노드 인덱스를 정렬하여 매핑
                used_indices = sorted(used_nodes)
                for igraph_idx, orig_idx in enumerate(used_indices):
                    original_node = core_nodes_list[orig_idx]
                    core_labels[original_node] = partition.membership[igraph_idx]
                
                # 연결되지 않은 핵심 노드들은 별도 커뮤니티로
                max_community = max(core_labels.values()) if core_labels else -1
                for node in core_nodes:
                    if node not in core_labels:
                        max_community += 1
                        core_labels[node] = max_community
        
        self.stats['leiden_time'] = time.time() - leiden_start
        
        # 3. 전체 라벨 초기화 - 최적화
        labels = dict(core_labels)  # 코어 라벨 복사
        for node in periphery_nodes:
            labels[node] = None  # 아직 할당되지 않음
        
        # 4. 앵커 전략에 따른 라벨 전파
        lpa_start = time.time()
        
        if self.anchor_strategy == 'fixed_single':
            self._propagate_fixed_single(G_nx, labels, periphery_nodes)
        elif self.anchor_strategy == 'fixed_iterative':
            self._propagate_fixed_iterative(G_nx, labels, periphery_nodes)
        elif self.anchor_strategy == 'dynamic_iterative':
            self._propagate_dynamic_iterative(G_nx, labels, core_nodes, periphery_nodes)
        
        self.stats['lpa_time'] = time.time() - lpa_start
        labels = self._clean_labels(labels)
        
        return labels
    
    def _propagate_fixed_single(self, G_nx: nx.Graph, labels: Dict[str, int], 
                               periphery_nodes: List[str]):
        """전략 A: 앵커 고정 + 1회 전파 - Counter 제거"""
        for v in periphery_nodes:
            labeled_neighbors = [labels[n] for n in G_nx.neighbors(v) 
                            if labels[n] is not None]
            if labeled_neighbors:
                most_common = self._get_most_common_label(labeled_neighbors)
                labels[v] = most_common
            else:
                # 라벨된 이웃이 없으면 새로운 커뮤니티 생성
                existing_labels = [l for l in labels.values() if l is not None]
                if existing_labels:
                    max_label = max(existing_labels)
                else:
                    max_label = -1
                labels[v] = max_label + 1
        
        self.stats['lpa_iterations'] = 1
    
    def _propagate_fixed_iterative(self, G_nx: nx.Graph, labels: Dict[str, int],
                                  periphery_nodes: List[str]):
        """전략 B: 앵커 고정 + 반복 전파 - Counter 제거"""
        for iteration in range(self.max_lpa_iterations):
            updated = False
            
            # 주변 노드만 업데이트 (핵심 노드는 절대 변경 안함!)
            for v in periphery_nodes:
                neighbor_labels = [labels[n] for n in G_nx.neighbors(v) 
                                if labels[n] is not None]
                if neighbor_labels:
                    most_common = self._get_most_common_label(neighbor_labels)
                    if labels[v] != most_common:
                        labels[v] = most_common
                        updated = True
                elif labels[v] is None:
                    # 새로운 커뮤니티 생성
                    existing_labels = [l for l in labels.values() if l is not None]
                    if existing_labels:
                        max_label = max(existing_labels)
                    else:
                        max_label = -1
                    labels[v] = max_label + 1
                    updated = True
            
            if not updated:  # 수렴하면 조기 종료
                break
        
        self.stats['lpa_iterations'] = iteration + 1
    
    def _propagate_dynamic_iterative(self, G_nx: nx.Graph, labels: Dict[str, int],
                                    core_nodes: List[str], periphery_nodes: List[str]):
        """전략 C: 앵커 비고정 + 반복 전파 - Counter 제거"""
        for iteration in range(self.max_lpa_iterations):
            updated = False
            
            # 1. 주변 노드 업데이트
            for v in periphery_nodes:
                neighbor_labels = [labels[n] for n in G_nx.neighbors(v) 
                                if labels[n] is not None]
                if neighbor_labels:
                    most_common = self._get_most_common_label(neighbor_labels)
                    if labels[v] != most_common:
                        labels[v] = most_common
                        updated = True
                elif labels[v] is None:
                    existing_labels = [l for l in labels.values() if l is not None]
                    if existing_labels:
                        max_label = max(existing_labels)
                    else:
                        max_label = -1
                    labels[v] = max_label + 1
                    updated = True
            
            # 2. 핵심 노드도 업데이트! (기존과 동일)
            for v in core_nodes:
                neighbor_labels = [labels[n] for n in G_nx.neighbors(v) 
                                if labels[n] is not None]
                if neighbor_labels:
                    most_common = self._get_most_common_label(neighbor_labels)
                    if labels[v] != most_common:
                        labels[v] = most_common
                        updated = True
            
            if not updated:
                break
        
        self.stats['lpa_iterations'] = iteration + 1
    
    def _clean_labels(self, labels: Dict[str, int]) -> Dict[str, int]:
        """None 값 제거 및 라벨 정리"""
        # None 값이 남아있으면 0으로 대체
        cleaned = {}
        for node, label in labels.items():
            if label is None:
                cleaned[node] = 0  # 기본값
            else:
                cleaned[node] = int(label)
        
        # 라벨 연속화 (0, 1, 2, ...)
        unique_labels = sorted(set(cleaned.values()))
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        
        final_labels = {node: label_mapping[label] for node, label in cleaned.items()}
        
        return final_labels
    
    def get_stats(self) -> Dict[str, Union[float, int]]:
        """실행 통계 반환"""
        return self.stats.copy()
    
    def get_config(self) -> Dict[str, Union[float, str, bool, int]]:
        """현재 설정 반환"""
        return {
            'core_ratio': self.core_ratio,
            'centrality_method': self.centrality_method,
            'anchor_strategy': self.anchor_strategy,
            'max_lpa_iterations': self.max_lpa_iterations,
            'seed': self.seed
        }

# 편의 함수들
def leiden_lpa_hybrid(G_nx: nx.Graph, 
                         core_ratio: float = 0.4,
                         centrality_method: str = 'pagerank',
                         anchor_strategy: str = 'fixed_single',
                         seed: Optional[int] = None) -> Dict[str, int]:
    """
    3가지 앵커 전략 지원 인터페이스 함수
    """
    alg = LeidenLPAHybrid(
        core_ratio=core_ratio,
        centrality_method=centrality_method,
        anchor_strategy=anchor_strategy,
        seed=seed
    )
    return alg.fit_predict(G_nx)

def run_3way_comparison(G_nx: nx.Graph, 
                       core_ratio: float = 0.4,
                       seed: Optional[int] = None) -> Dict[str, Dict]:
    """
    3가지 앵커 전략 비교 실행
    
    Returns:
    --------
    results : dict
        각 전략별 결과와 실행 시간
    """
    strategies = ['fixed_single', 'fixed_iterative', 'dynamic_iterative']
    results = {}
    
    for strategy in strategies:
        alg = LeidenLPAHybrid(
            core_ratio=core_ratio, 
            anchor_strategy=strategy,
            seed=seed
        )
        
        start_time = time.time()
        labels = alg.fit_predict(G_nx)
        runtime = time.time() - start_time
        
        results[strategy] = {
            'labels': labels,
            'runtime': runtime,
            'stats': alg.get_stats()
        }
    
    return results