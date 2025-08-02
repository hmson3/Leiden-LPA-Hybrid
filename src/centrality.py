"""
중심성 지표 계산 및 핵심 노드 선택 모듈
"""
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Union
import time
import warnings

def compute_centrality(G: nx.Graph, method: str = 'pagerank', **kwargs) -> Dict[str, float]:
    """
    다양한 중심성 지표를 계산하는 통합 함수
    
    Parameters:
    -----------
    G : networkx.Graph
        입력 그래프
    method : str
        중심성 지표 방법
        ['pagerank', 'degree', 'eigenvector', 'betweenness', 
         'closeness', 'clustering', 'katz', 'harmonic']
    **kwargs : dict
        각 방법별 추가 파라미터
        
    Returns:
    --------
    centrality_scores : dict
        {node_id: centrality_score}
    """
    
    start_time = time.time()
    
    try:
        if method == 'pagerank':
            alpha = kwargs.get('alpha', 0.85)
            max_iter = kwargs.get('max_iter', 100)
            scores = nx.pagerank(G, alpha=alpha, max_iter=max_iter)
            
        elif method == 'degree':
            scores = nx.degree_centrality(G)
            
        elif method == 'eigenvector':
            max_iter = kwargs.get('max_iter', 100)
            scores = nx.eigenvector_centrality(G, max_iter=max_iter)
            
        elif method == 'betweenness':
            normalized = kwargs.get('normalized', True)
            k = kwargs.get('k', None)  # 샘플링용
            scores = nx.betweenness_centrality(G, normalized=normalized, k=k)
            
        elif method == 'closeness':
            scores = nx.closeness_centrality(G)
            
        elif method == 'clustering':
            scores = nx.clustering(G)
            
        elif method == 'katz':
            alpha = kwargs.get('alpha', 0.1)
            max_iter = kwargs.get('max_iter', 100)
            scores = nx.katz_centrality(G, alpha=alpha, max_iter=max_iter)
            
        elif method == 'harmonic':
            scores = nx.harmonic_centrality(G)
            
        else:
            raise ValueError(f"Unknown centrality method: {method}")
            
    except Exception as e:
        warnings.warn(f"Error computing {method} centrality: {e}")
        # Fallback to degree centrality
        scores = nx.degree_centrality(G)
    
    computation_time = time.time() - start_time
    
    # 결과 검증
    if not scores or any(np.isnan(v) for v in scores.values()):
        warnings.warn(f"{method} centrality produced invalid results, falling back to degree")
        scores = nx.degree_centrality(G)
    
    print(f"  {method} centrality computed in {computation_time:.3f}s")
    return scores

def get_top_nodes(centrality_scores: Dict[str, float], 
                  ratio: float = 0.4) -> List[str]:
    """
    중심성 점수를 기반으로 상위 노드들을 선택
    
    Parameters:
    -----------
    centrality_scores : dict
        {node_id: centrality_score}
    ratio : float
        선택할 노드의 비율 (0.0 ~ 1.0)
        
    Returns:
    --------
    top_nodes : list
        선택된 상위 노드들의 리스트
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be between 0.0 and 1.0")
    
    if ratio == 0.0:
        return []
    
    if ratio == 1.0:
        return list(centrality_scores.keys())
    
    # 점수별로 정렬
    sorted_nodes = sorted(centrality_scores.items(), 
                         key=lambda x: x[1], reverse=True)
    
    # 상위 ratio만큼 선택
    num_top = max(1, int(len(sorted_nodes) * ratio))
    top_nodes = [node for node, score in sorted_nodes[:num_top]]
    
    return top_nodes

def get_centrality_stats(centrality_scores: Dict[str, float]) -> Dict[str, float]:
    """
    중심성 점수의 통계 정보 계산
    
    Returns:
    --------
    stats : dict
        통계 정보 (mean, std, min, max, etc.)
    """
    scores = list(centrality_scores.values())
    
    return {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        'median': np.median(scores),
        'q25': np.percentile(scores, 25),
        'q75': np.percentile(scores, 75)
    }

def compare_centralities(G: nx.Graph, 
                        methods: List[str] = None,
                        ratio: float = 0.4) -> Dict[str, Dict]:
    """
    여러 중심성 지표를 비교하는 함수
    
    Returns:
    --------
    comparison : dict
        각 방법별 결과와 통계
    """
    if methods is None:
        methods = ['pagerank', 'degree', 'eigenvector', 'clustering']
    
    results = {}
    
    for method in methods:
        print(f"Computing {method} centrality...")
        
        try:
            scores = compute_centrality(G, method)
            top_nodes = get_top_nodes(scores, ratio)
            stats = get_centrality_stats(scores)
            
            results[method] = {
                'scores': scores,
                'top_nodes': top_nodes,
                'stats': stats,
                'num_top_nodes': len(top_nodes)
            }
            
        except Exception as e:
            warnings.warn(f"Failed to compute {method}: {e}")
            continue
    
    return results

def analyze_top_node_overlap(comparison_results: Dict[str, Dict]) -> Dict[str, float]:
    """
    서로 다른 중심성 지표에서 선택된 상위 노드들의 겹침 정도 분석
    
    Returns:
    --------
    overlap_matrix : dict
        방법 간 Jaccard similarity
    """
    methods = list(comparison_results.keys())
    overlap_matrix = {}
    
    for i, method1 in enumerate(methods):
        overlap_matrix[method1] = {}
        for j, method2 in enumerate(methods):
            if i <= j:
                set1 = set(comparison_results[method1]['top_nodes'])
                set2 = set(comparison_results[method2]['top_nodes'])
                
                if len(set1) == 0 and len(set2) == 0:
                    jaccard = 1.0
                elif len(set1) == 0 or len(set2) == 0:
                    jaccard = 0.0
                else:
                    jaccard = len(set1 & set2) / len(set1 | set2)
                
                overlap_matrix[method1][method2] = jaccard
                if method1 != method2:
                    if method2 not in overlap_matrix:
                        overlap_matrix[method2] = {}
                    overlap_matrix[method2][method1] = jaccard
    
    return overlap_matrix

# 각 중심성 지표별 권장 사용 조건
CENTRALITY_RECOMMENDATIONS = {
    'pagerank': {
        'max_nodes': 100000,
        'description': 'Global influence, good for most networks',
        'time_complexity': 'O(V+E) * iterations'
    },
    'degree': {
        'max_nodes': float('inf'),
        'description': 'Local connectivity, fastest computation',
        'time_complexity': 'O(V)'
    },
    'eigenvector': {
        'max_nodes': 10000,
        'description': 'Quality of connections matters',
        'time_complexity': 'O(V^2) or iterative'
    },
    'betweenness': {
        'max_nodes': 1000,
        'description': 'Bridge nodes, expensive computation',
        'time_complexity': 'O(V*E)'
    },
    'closeness': {
        'max_nodes': 5000,
        'description': 'Accessibility to all nodes',
        'time_complexity': 'O(V*(V+E))'
    },
    'clustering': {
        'max_nodes': 50000,
        'description': 'Local cohesiveness',
        'time_complexity': 'O(V*d_avg)'
    }
}

def get_recommended_methods(num_nodes: int) -> List[str]:
    """
    그래프 크기에 따른 권장 중심성 지표들 반환
    """
    recommended = []
    
    for method, info in CENTRALITY_RECOMMENDATIONS.items():
        if num_nodes <= info['max_nodes']:
            recommended.append(method)
    
    return recommended