"""
통합 커뮤니티 탐지 평가 시스템 (Overlapping NMI 지원)
"""
import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score
from collections import Counter, defaultdict
import time
import psutil
import os
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings

# Overlapping NMI를 위한 라이브러리
try:
    from cdlib.evaluation import overlapping_normalized_mutual_information_LFK
    from cdlib import NodeClustering
    OVERLAPPING_NMI_AVAILABLE = True
except ImportError:
    OVERLAPPING_NMI_AVAILABLE = False
    warnings.warn("cdlib not available. Overlapping NMI will be skipped.")

class CommunityEvaluator:
    """
    커뮤니티 탐지 결과 평가 클래스
    """
    
    def __init__(self, G_nx: nx.Graph, 
                 true_labels: Optional[Dict[str, Union[int, List[int]]]] = None,
                 is_overlapping: bool = False):
        """
        Parameters:
        -----------
        G_nx : networkx.Graph
            원본 그래프
        true_labels : dict, optional
            정답 커뮤니티 레이블 
            - Non-overlapping: {node_id: community_id}
            - Overlapping: {node_id: [community_id_list]}
        is_overlapping : bool
            True면 overlapping communities로 처리
        """
        self.G_nx = G_nx
        self.true_labels = true_labels
        self.is_overlapping = is_overlapping
        self.has_ground_truth = true_labels is not None
        
        # 그래프 기본 정보
        self.num_nodes = G_nx.number_of_nodes()
        self.num_edges = G_nx.number_of_edges()
        self.density = nx.density(G_nx)
        
    def evaluate_clustering(self, 
                          pred_labels: Dict[str, int],
                          runtime: float = 0.0,
                          memory_usage: float = 0.0,
                          algorithm_stats: Optional[Dict] = None) -> Dict[str, Any]:
        """
        클러스터링 결과를 종합적으로 평가
        
        Parameters:
        -----------
        pred_labels : dict
            예측된 커뮤니티 레이블 {node_id: community_id}
        runtime : float
            알고리즘 실행 시간 (초)
        memory_usage : float
            메모리 사용량 (MB)
        algorithm_stats : dict, optional
            알고리즘별 상세 통계
            
        Returns:
        --------
        evaluation_results : dict
            모든 평가 지표들
        """
        results = {
            # 기본 정보
            'graph_info': {
                'num_nodes': self.num_nodes,
                'num_edges': self.num_edges,
                'density': self.density
            },
            
            # 성능 지표
            'performance': {
                'runtime_seconds': runtime,
                'memory_mb': memory_usage
            },
            
            # 알고리즘 통계
            'algorithm_stats': algorithm_stats or {}
        }
        
        # 클러스터링 기본 정보
        cluster_info = self._analyze_clustering(pred_labels)
        results['clustering_info'] = cluster_info
        
        # 구조적 품질 지표 (ground truth 불필요)
        structural_quality = self._compute_structural_quality(pred_labels)
        results['structural_quality'] = structural_quality
        
        # Ground truth 기반 지표 (있는 경우)
        if self.has_ground_truth:
            ground_truth_metrics = self._compute_ground_truth_metrics(pred_labels)
            results['ground_truth_metrics'] = ground_truth_metrics
        
        return results
    
    def _analyze_clustering(self, pred_labels: Dict[str, int]) -> Dict[str, Any]:
        """클러스터링 기본 분석"""
        labels_list = list(pred_labels.values())
        unique_labels = set(labels_list)
        
        # 커뮤니티별 크기
        community_sizes = Counter(labels_list)
        sizes = list(community_sizes.values())
        
        return {
            'num_communities': len(unique_labels),
            'community_sizes': dict(community_sizes),
            'size_statistics': {
                'min_size': min(sizes) if sizes else 0,
                'max_size': max(sizes) if sizes else 0,
                'mean_size': np.mean(sizes) if sizes else 0,
                'std_size': np.std(sizes) if sizes else 0
            },
            'singleton_communities': sum(1 for size in sizes if size == 1),
            'largest_community_ratio': max(sizes) / len(labels_list) if sizes else 0
        }
    
    def _compute_structural_quality(self, pred_labels: Dict[str, int]) -> Dict[str, float]:
        """구조적 품질 지표 계산"""
        results = {}
        
        # Modularity 계산
        try:
            modularity = self._compute_modularity(pred_labels)
            results['modularity'] = modularity
        except Exception as e:
            warnings.warn(f"Modularity calculation failed: {e}")
            results['modularity'] = np.nan
        
        # Conductance 계산
        try:
            conductance = self._compute_conductance(pred_labels)
            results['conductance'] = conductance
        except Exception as e:
            warnings.warn(f"Conductance calculation failed: {e}")
            results['conductance'] = np.nan
        
        # Coverage 계산
        try:
            coverage = self._compute_coverage(pred_labels)
            results['coverage'] = coverage
        except Exception as e:
            warnings.warn(f"Coverage calculation failed: {e}")
            results['coverage'] = np.nan
        
        return results
    
    def _compute_modularity(self, pred_labels: Dict[str, int]) -> float:
        """Modularity 계산 (igraph 사용)"""
        # NetworkX를 igraph로 변환
        G_ig = ig.Graph.TupleList(self.G_nx.edges(), directed=False)
        
        # 레이블 리스트 생성 (igraph 노드 순서에 맞춤)
        label_list = []
        for v in G_ig.vs:
            node_name = str(v["name"])
            if node_name in pred_labels:
                label_list.append(int(pred_labels[node_name]))
            else:
                label_list.append(0)  # 기본값
        
        return G_ig.modularity(label_list)
    
    def _compute_conductance(self, pred_labels: Dict[str, int]) -> float:
        """평균 Conductance 계산"""
        communities = defaultdict(list)
        for node, label in pred_labels.items():
            communities[label].append(node)
        
        conductances = []
        for community_nodes in communities.values():
            if len(community_nodes) <= 1:
                continue
                
            # 내부 엣지와 외부 엣지 카운트
            internal_edges = 0
            external_edges = 0
            
            for node in community_nodes:
                for neighbor in self.G_nx.neighbors(node):
                    if neighbor in community_nodes:
                        internal_edges += 1
                    else:
                        external_edges += 1
            
            internal_edges //= 2  # 중복 카운트 제거
            total_edges = internal_edges + external_edges
            
            if total_edges > 0:
                conductance = external_edges / total_edges
                conductances.append(conductance)
        
        return np.mean(conductances) if conductances else 1.0
    
    def _compute_coverage(self, pred_labels: Dict[str, int]) -> float:
        """Coverage 계산 (내부 엣지 비율)"""
        communities = defaultdict(set)
        for node, label in pred_labels.items():
            communities[label].add(node)
        
        internal_edges = 0
        total_edges = self.G_nx.number_of_edges()
        
        for u, v in self.G_nx.edges():
            u_label = pred_labels.get(u)
            v_label = pred_labels.get(v)
            if u_label is not None and v_label is not None and u_label == v_label:
                internal_edges += 1
        
        return internal_edges / total_edges if total_edges > 0 else 0.0
    
    def _compute_ground_truth_metrics(self, pred_labels: Dict[str, int]) -> Dict[str, float]:
        """Ground truth 기반 지표 계산"""
        results = {}
        
        # 공통 노드만 추출
        common_nodes = set(pred_labels.keys()) & set(self.true_labels.keys())
        
        if len(common_nodes) == 0:
            warnings.warn("No common nodes between predictions and ground truth")
            return {
                'nmi': np.nan, 'ami': np.nan, 'ari': np.nan, 
                'accuracy': np.nan, 'f1_score': np.nan, 'overlapping_nmi': np.nan
            }
        
        if self.is_overlapping:
            # Overlapping 케이스
            try:
                onmi_score = self._compute_overlapping_nmi(pred_labels, common_nodes)
                results['overlapping_nmi'] = onmi_score
            except Exception as e:
                warnings.warn(f"Overlapping NMI calculation failed: {e}")
                results['overlapping_nmi'] = np.nan
            
            # Overlapping의 경우 다른 지표들은 계산하기 어려움
            results.update({
                'nmi': np.nan, 'ami': np.nan, 'ari': np.nan,
                'accuracy': np.nan, 'f1_score': np.nan
            })
            
        else:
            # Non-overlapping 케이스
            pred_list = [pred_labels[node] for node in common_nodes]
            true_list = [self.true_labels[node] for node in common_nodes]
            
            # NMI 계산
            try:
                nmi = normalized_mutual_info_score(true_list, pred_list)
                results['nmi'] = nmi
            except Exception as e:
                warnings.warn(f"NMI calculation failed: {e}")
                results['nmi'] = np.nan
            
            # AMI 계산
            try:
                from sklearn.metrics import adjusted_mutual_info_score
                ami = adjusted_mutual_info_score(true_list, pred_list)
                results['ami'] = ami
            except Exception as e:
                warnings.warn(f"AMI calculation failed: {e}")
                results['ami'] = np.nan
            
            # ARI 계산
            try:
                ari = adjusted_rand_score(true_list, pred_list)
                results['ari'] = ari
            except Exception as e:
                warnings.warn(f"ARI calculation failed: {e}")
                results['ari'] = np.nan
            
            # 정확도 계산
            try:
                accuracy = self._compute_accuracy(pred_list, true_list)
                results['accuracy'] = accuracy
            except Exception as e:
                warnings.warn(f"Accuracy calculation failed: {e}")
                results['accuracy'] = np.nan
            
            # F1-score 계산
            try:
                f1_score_val = self._compute_f1_score(pred_list, true_list)
                results['f1_score'] = f1_score_val
            except Exception as e:
                warnings.warn(f"F1-score calculation failed: {e}")
                results['f1_score'] = np.nan
            
            # Overlapping NMI (non-overlapping 데이터에서도 계산 가능)
            results['overlapping_nmi'] = np.nan  # 일단 생략
        
        return results
    
    def _compute_overlapping_nmi(self, pred_labels: Dict[str, int], 
                               common_nodes: set) -> float:
        """Overlapping NMI 계산"""
        if not OVERLAPPING_NMI_AVAILABLE:
            return np.nan
        
        try:
            # Predicted communities (non-overlapping을 overlapping 형식으로)
            pred_communities = defaultdict(list)
            for node in common_nodes:
                if node in pred_labels:
                    comm = pred_labels[node]
                    pred_communities[comm].append(node)
            
            # True communities (overlapping 형식)
            true_communities = defaultdict(list)
            for node in common_nodes:
                if node in self.true_labels:
                    if isinstance(self.true_labels[node], list):
                        # Overlapping 케이스
                        for comm_id in self.true_labels[node]:
                            true_communities[comm_id].append(node)
                    else:
                        # Non-overlapping 케이스
                        comm_id = self.true_labels[node]
                        true_communities[comm_id].append(node)
            
            # cdlib 형식으로 변환
            pred_clustering = NodeClustering(
                list(pred_communities.values()),
                graph=self.G_nx,
                method_name="prediction"
            )
            
            true_clustering = NodeClustering(
                list(true_communities.values()),
                graph=self.G_nx,
                method_name="ground_truth"
            )
            
            # Overlapping NMI 계산
            onmi = overlapping_normalized_mutual_information_LFK(true_clustering, pred_clustering)
            return onmi.score
            
        except Exception as e:
            warnings.warn(f"Overlapping NMI computation error: {e}")
            return np.nan
    
    def _compute_accuracy(self, pred_list: List[int], true_list: List[int]) -> float:
        """최적 매칭 기준 정확도 계산"""
        from scipy.optimize import linear_sum_assignment
        
        pred_labels = sorted(set(pred_list))
        true_labels = sorted(set(true_list))
        
        # 혼동 행렬 생성
        confusion_matrix = np.zeros((len(true_labels), len(pred_labels)))
        
        for true_label, pred_label in zip(true_list, pred_list):
            true_idx = true_labels.index(true_label)
            pred_idx = pred_labels.index(pred_label)
            confusion_matrix[true_idx, pred_idx] += 1
        
        # 헝가리안 알고리즘으로 최적 매칭
        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
        
        # 매칭된 항목들의 합 / 전체
        matched_count = confusion_matrix[row_ind, col_ind].sum()
        total_count = len(pred_list)
        
        return matched_count / total_count if total_count > 0 else 0.0

    def _compute_f1_score(self, pred_list: List[int], true_list: List[int]) -> float:
        """F1-score 계산 (macro-averaged)"""
        from scipy.optimize import linear_sum_assignment
        
        pred_labels = sorted(set(pred_list))
        true_labels = sorted(set(true_list))
        
        # 혼동 행렬 생성
        confusion_matrix = np.zeros((len(true_labels), len(pred_labels)))
        
        for true_label, pred_label in zip(true_list, pred_list):
            true_idx = true_labels.index(true_label)
            pred_idx = pred_labels.index(pred_label)
            confusion_matrix[true_idx, pred_idx] += 1
        
        # 헝가리안 알고리즘으로 최적 매칭
        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
        
        # 매칭 결과에 따라 예측 레이블 재매핑
        label_mapping = {}
        for true_idx, pred_idx in zip(row_ind, col_ind):
            if pred_idx < len(pred_labels):
                label_mapping[pred_labels[pred_idx]] = true_labels[true_idx]
        
        # 재매핑된 예측 레이블 생성
        remapped_pred = [label_mapping.get(pred, pred) for pred in pred_list]
        
        # F1-score 계산 (macro average)
        return f1_score(true_list, remapped_pred, average='macro')

def load_overlapping_labels(label_file_path: str) -> Dict[str, List[int]]:
    """
    Overlapping 형식의 라벨 로드
    
    각 라인이 하나의 커뮤니티를 나타내고, 라인의 숫자들이 해당 커뮤니티에 속하는 노드들
    
    Returns:
    --------
    node_to_communities : dict
        {node_id: [community_list]}
    """
    node_to_communities = defaultdict(list)
    
    with open(label_file_path, 'r') as f:
        for community_id, line in enumerate(f):
            nodes = line.strip().split()
            for node in nodes:
                node_to_communities[node].append(community_id)
    
    return dict(node_to_communities)

def run_timed_evaluation(evaluator_func, *args, **kwargs) -> Tuple[Any, float, float]:
    """
    함수 실행 시간과 메모리 사용량을 측정하면서 평가 실행
    """
    # 메모리 사용량 측정 시작
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # 실행 시간 측정
    start_time = time.time()
    result = evaluator_func(*args, **kwargs)
    runtime = time.time() - start_time
    
    # 메모리 사용량 측정 종료
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_usage = memory_after - memory_before
    
    return result, runtime, memory_usage

def compare_algorithms(G_nx: nx.Graph,
                      algorithms: Dict[str, callable],
                      true_labels: Optional[Dict[str, Union[int, List[int]]]] = None,
                      is_overlapping: bool = False,
                      repeat: int = 1,
                      seed: int = 42) -> pd.DataFrame:
    """여러 알고리즘을 비교 평가"""
    evaluator = CommunityEvaluator(G_nx, true_labels, is_overlapping)
    results = []
    
    for algorithm_name, algorithm_func in algorithms.items():
        print(f"Running {algorithm_name}...")
        
        for run_id in range(repeat):
            try:
                # 알고리즘 실행
                labels, runtime, memory_usage = run_timed_evaluation(
                    algorithm_func, G_nx, seed + run_id
                )
                
                # 평가 실행
                evaluation = evaluator.evaluate_clustering(
                    labels, runtime, memory_usage
                )
                
                # 결과 정리
                result_row = {
                    'algorithm': algorithm_name,
                    'run_id': run_id,
                    'runtime': runtime,
                    'memory_mb': memory_usage,
                    'num_communities': evaluation['clustering_info']['num_communities'],
                    'modularity': evaluation['structural_quality']['modularity'],
                    'conductance': evaluation['structural_quality']['conductance'],
                    'coverage': evaluation['structural_quality']['coverage']
                }
                
                # Ground truth 지표 추가
                if evaluator.has_ground_truth:
                    gt_metrics = evaluation['ground_truth_metrics']
                    result_row.update({
                        'nmi': gt_metrics['nmi'],
                        'ami': gt_metrics['ami'],
                        'ari': gt_metrics['ari'],
                        'accuracy': gt_metrics['accuracy'],
                        'f1_score': gt_metrics['f1_score'],
                        'overlapping_nmi': gt_metrics['overlapping_nmi']
                    })
                
                results.append(result_row)
                
            except Exception as e:
                warnings.warn(f"Error running {algorithm_name} (run {run_id}): {e}")
                continue
    
    return pd.DataFrame(results)

def save_evaluation_results(results_df: pd.DataFrame, 
                          filepath: str,
                          experiment_metadata: Optional[Dict] = None):
    """평가 결과를 파일로 저장"""
    # 메타데이터를 CSV 주석으로 추가
    with open(filepath, 'w') as f:
        if experiment_metadata:
            f.write(f"# Experiment Metadata\n")
            for key, value in experiment_metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write(f"# \n")
        
        # DataFrame 저장
        results_df.to_csv(f, index=False)
    
    print(f"Results saved to: {filepath}")

def quick_evaluate(G_nx: nx.Graph, 
                  pred_labels: Dict[str, int],
                  true_labels: Optional[Dict[str, Union[int, List[int]]]] = None,
                  is_overlapping: bool = False) -> Dict[str, float]:
    """빠른 평가 (주요 지표만)"""
    evaluator = CommunityEvaluator(G_nx, true_labels, is_overlapping)
    full_results = evaluator.evaluate_clustering(pred_labels)
    
    # 주요 지표만 추출
    quick_results = {
        'modularity': full_results['structural_quality']['modularity'],
        'num_communities': full_results['clustering_info']['num_communities']
    }
    
    if evaluator.has_ground_truth:
        if is_overlapping:
            quick_results['overlapping_nmi'] = full_results['ground_truth_metrics']['overlapping_nmi']
        else:
            quick_results['nmi'] = full_results['ground_truth_metrics']['nmi']
            quick_results['f1_score'] = full_results['ground_truth_metrics']['f1_score']
    
    return quick_results