"""
개선된 Leiden-LPA 하이브리드 커뮤니티 탐지 알고리즘
- 다양한 중심성 지표 지원
- 앵커 고정/비고정 옵션
- 더 나은 에러 핸들링
"""
import networkx as nx
import igraph as ig
from collections import Counter
from leidenalg import find_partition, ModularityVertexPartition
import time
import warnings
from typing import Dict, Optional, Union, List
from centrality import compute_centrality, get_top_nodes

class LeidenLPAHybrid:
    """
    Leiden-LPA 하이브리드 알고리즘 클래스
    """
    
    def __init__(self, 
                 core_ratio: float = 0.4,
                 centrality_method: str = 'pagerank',
                 anchor_fixed: bool = True,
                 max_lpa_iterations: int = 10,
                 seed: Optional[int] = None):
        """
        Parameters:
        -----------
        core_ratio : float
            핵심 노드 비율 (0.0 ~ 1.0)
        centrality_method : str
            중심성 지표 ('pagerank', 'degree', 'eigenvector', etc.)
        anchor_fixed : bool
            앵커 노드 고정 여부 (True: 고정, False: 업데이트 허용)
        max_lpa_iterations : int
            LPA 최대 반복 횟수
        seed : int, optional
            랜덤 시드
        """
        self.core_ratio = core_ratio
        self.centrality_method = centrality_method
        self.anchor_fixed = anchor_fixed
        self.max_lpa_iterations = max_lpa_iterations
        self.seed = seed
        
        # 실행 통계
        self.stats = {
            'centrality_time': 0.0,
            'leiden_time': 0.0,
            'lpa_time': 0.0,
            'total_time': 0.0,
            'core_nodes_count': 0,
            'periphery_nodes_count': 0,
            'lpa_iterations': 0
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
    
    def _run_pure_lpa(self, G_nx: nx.Graph) -> Dict[str, int]:
        """순수 LPA 실행"""
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
                    most_common = Counter(neighbor_labels).most_common(1)[0][0]
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
        """순수 Leiden 실행"""
        start_time = time.time()
        
        # NetworkX를 igraph로 변환
        G_ig = ig.Graph.TupleList(G_nx.edges(), directed=False)
        
        # Leiden 실행
        if self.seed is not None:
            partition = find_partition(G_ig, ModularityVertexPartition, seed=self.seed)
        else:
            partition = find_partition(G_ig, ModularityVertexPartition)
        
        # 결과 변환
        labels = {v["name"]: partition.membership[i] for i, v in enumerate(G_ig.vs)}
        
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
        periphery_nodes = [v for v in G_nx.nodes() if v not in core_nodes]
        
        self.stats['centrality_time'] = time.time() - centrality_start
        self.stats['core_nodes_count'] = len(core_nodes)
        self.stats['periphery_nodes_count'] = len(periphery_nodes)
        
        # 2. 핵심 노드에 Leiden 적용
        leiden_start = time.time()
        if len(core_nodes) == 0:
            core_labels = {}
        else:
            G_core = G_nx.subgraph(core_nodes).copy()
            
            if len(G_core.edges()) == 0:
                # 엣지가 없으면 각 노드를 별도 커뮤니티로
                core_labels = {node: i for i, node in enumerate(core_nodes)}
            else:
                # igraph 변환 및 Leiden 실행
                G_core_ig = ig.Graph.TupleList(G_core.edges(), directed=False)
                
                if self.seed is not None:
                    partition = find_partition(G_core_ig, ModularityVertexPartition, seed=self.seed)
                else:
                    partition = find_partition(G_core_ig, ModularityVertexPartition)
                
                core_labels = {v["name"]: partition.membership[i] 
                             for i, v in enumerate(G_core_ig.vs)}
        
        self.stats['leiden_time'] = time.time() - leiden_start
        
        # 3. 전체 라벨 초기화
        labels = {}
        for node in G_nx.nodes():
            if node in core_labels:
                labels[node] = core_labels[node]
            else:
                labels[node] = None  # 아직 할당되지 않음
        
        # 4. 주변 노드에 라벨 전파
        lpa_start = time.time()
        
        if self.anchor_fixed:
            # 앵커 고정: 한 번만 전파
            self._propagate_labels_fixed(G_nx, labels, periphery_nodes)
        else:
            # 앵커 비고정: 반복적 전파
            self._propagate_labels_dynamic(G_nx, labels, core_nodes, periphery_nodes)
        
        self.stats['lpa_time'] = time.time() - lpa_start
        labels = self._clean_labels(labels)
        
        return labels
    
    def _propagate_labels_fixed(self, G_nx: nx.Graph, labels: Dict[str, int], 
                            periphery_nodes: List[str]):
        """앵커 고정 라벨 전파 - None 값 처리 수정"""
        for v in periphery_nodes:
            labeled_neighbors = [labels[n] for n in G_nx.neighbors(v) 
                            if labels[n] is not None]
            if labeled_neighbors:
                most_common = Counter(labeled_neighbors).most_common(1)[0][0]
                labels[v] = most_common
            else:
                # 라벨된 이웃이 없으면 새로운 커뮤니티 생성
                # None이 아닌 값들만 고려해서 최대값 찾기
                existing_labels = [l for l in labels.values() if l is not None]
                if existing_labels:
                    max_label = max(existing_labels)
                else:
                    max_label = -1
                labels[v] = max_label + 1
        
        self.stats['lpa_iterations'] = 1

    def _propagate_labels_dynamic(self, G_nx: nx.Graph, labels: Dict[str, int],
                                core_nodes: List[str], periphery_nodes: List[str]):
        """앵커 비고정 반복적 라벨 전파 - None 값 처리 수정"""
        for iteration in range(self.max_lpa_iterations):
            updated = False
            
            # 주변 노드 업데이트
            for v in periphery_nodes:
                neighbor_labels = [labels[n] for n in G_nx.neighbors(v) 
                                if labels[n] is not None]
                if neighbor_labels:
                    most_common = Counter(neighbor_labels).most_common(1)[0][0]
                    if labels[v] != most_common:
                        labels[v] = most_common
                        updated = True
                elif labels[v] is None:
                    # 새로운 커뮤니티 생성 - None 값 안전 처리
                    existing_labels = [l for l in labels.values() if l is not None]
                    if existing_labels:
                        max_label = max(existing_labels)
                    else:
                        max_label = -1
                    labels[v] = max_label + 1
                    updated = True
            
            # 핵심 노드도 업데이트 (선택적)
            for v in core_nodes:
                neighbor_labels = [labels[n] for n in G_nx.neighbors(v) 
                                if labels[n] is not None]
                if neighbor_labels:
                    most_common = Counter(neighbor_labels).most_common(1)[0][0]
                    if labels[v] != most_common:
                        labels[v] = most_common
                        updated = True
            
            if not updated:
                break
        
        self.stats['lpa_iterations'] = iteration + 1

    # 추가로 라벨 후처리 함수
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
            'anchor_fixed': self.anchor_fixed,
            'max_lpa_iterations': self.max_lpa_iterations,
            'seed': self.seed
        }

# 편의 함수들
def leiden_lpa_hybrid(G_nx: nx.Graph, 
                     core_ratio: float = 0.4,
                     centrality_method: str = 'pagerank',
                     anchor_fixed: bool = True,
                     seed: Optional[int] = None) -> Dict[str, int]:
    """
    간단한 인터페이스 함수 (기존 호환성 유지)
    """
    alg = LeidenLPAHybrid(
        core_ratio=core_ratio,
        centrality_method=centrality_method,
        anchor_fixed=anchor_fixed,
        seed=seed
    )
    return alg.fit_predict(G_nx)

def run_baseline_comparison(G_nx: nx.Graph, 
                          seed: Optional[int] = None) -> Dict[str, Dict]:
    """
    기준선 알고리즘들과 비교 실행
    
    Returns:
    --------
    results : dict
        각 알고리즘별 결과와 실행 시간
    """
    results = {}
    
    # Pure LPA
    alg_lpa = LeidenLPAHybrid(core_ratio=0.0, seed=seed)
    start_time = time.time()
    labels_lpa = alg_lpa.fit_predict(G_nx)
    lpa_time = time.time() - start_time
    
    results['Pure_LPA'] = {
        'labels': labels_lpa,
        'runtime': lpa_time,
        'stats': alg_lpa.get_stats()
    }
    
    # Pure Leiden
    alg_leiden = LeidenLPAHybrid(core_ratio=1.0, seed=seed)
    start_time = time.time()
    labels_leiden = alg_leiden.fit_predict(G_nx)
    leiden_time = time.time() - start_time
    
    results['Pure_Leiden'] = {
        'labels': labels_leiden,
        'runtime': leiden_time,
        'stats': alg_leiden.get_stats()
    }
    
    # Hybrid (default)
    alg_hybrid = LeidenLPAHybrid(core_ratio=0.4, seed=seed)
    start_time = time.time()
    labels_hybrid = alg_hybrid.fit_predict(G_nx)
    hybrid_time = time.time() - start_time
    
    results['Leiden_LPA_Hybrid'] = {
        'labels': labels_hybrid,
        'runtime': hybrid_time,
        'stats': alg_hybrid.get_stats()
    }
    
    return results