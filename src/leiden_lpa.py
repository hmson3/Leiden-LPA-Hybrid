import networkx as nx
from collections import Counter
from leidenalg import find_partition, ModularityVertexPartition
import igraph as ig
import random

def compute_centrality(G, method='pagerank', **kwargs):
    """
    다양한 중심성 지표 계산
    
    Parameters:
    -----------
    G : networkx.Graph
        입력 그래프
    method : str
        중심성 방법 ('pagerank', 'degree', 'eigenvector', 'betweenness', 'closeness')
    **kwargs : dict
        추가 파라미터들
        
    Returns:
    --------
    centrality_scores : dict
        {node_id: centrality_score}
    """
    
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
            k = kwargs.get('k', None)  # 샘플링용 (대형 그래프)
            scores = nx.betweenness_centrality(G, normalized=normalized, k=k)
            
        elif method == 'closeness':
            scores = nx.closeness_centrality(G)
            
        else:
            # 알 수 없는 방법이면 degree centrality로 폴백
            print(f"Warning: Unknown centrality method '{method}', using degree centrality")
            scores = nx.degree_centrality(G)
            
    except Exception as e:
        # 계산 실패 시 degree centrality로 폴백
        print(f"Warning: Error computing {method} centrality: {e}")
        print("Falling back to degree centrality")
        scores = nx.degree_centrality(G)
    
    return scores

def get_top_nodes(centrality_scores, ratio):
    """
    중심성 점수를 기준으로 상위 노드들 선택
    
    Parameters:
    -----------
    centrality_scores : dict
        {node_id: centrality_score}
    ratio : float
        선택할 노드 비율 (0.0 ~ 1.0)
        
    Returns:
    --------
    top_nodes : list
        상위 노드들의 리스트
    """
    if ratio <= 0.0:
        return []
    
    if ratio >= 1.0:
        return list(centrality_scores.keys())
    
    # 중심성 점수 기준으로 정렬
    sorted_nodes = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 상위 ratio 비율만큼 선택
    num_nodes = max(1, int(len(sorted_nodes) * ratio))
    top_nodes = [node for node, score in sorted_nodes[:num_nodes]]
    
    return top_nodes

def leiden_lpa_hybrid(G_nx, 
                     core_ratio=0.4, 
                     centrality_method='pagerank', 
                     seed=None, 
                     max_iter=10):
    """
    Leiden-LPA 하이브리드 알고리즘 (중심성 지원 버전)
    
    Parameters:
    -----------
    G_nx : networkx.Graph
        입력 그래프
    core_ratio : float
        핵심 노드 비율 (0.0 ~ 1.0)
    centrality_method : str
        중심성 계산 방법 ('pagerank', 'degree', 'eigenvector', 'betweenness', 'closeness')
    seed : int, optional
        랜덤 시드
    max_iter : int
        LPA 최대 반복 횟수
        
    Returns:
    --------
    labels : dict
        {node_id: community_label}
    """
    
    # 시드 설정
    if seed is not None:
        random.seed(seed)
    
    # 코어 비율 0.0 → LPA만 수행
    if core_ratio <= 0.0:
        labels = {v: i for i, v in enumerate(G_nx.nodes())}  # 초기 라벨
        updated = True
        iteration = 0
        
        while updated and iteration < max_iter:
            updated = False
            nodes_list = list(G_nx.nodes())
            if seed is not None:
                random.shuffle(nodes_list)
                
            for v in nodes_list:
                neighbor_labels = [labels[n] for n in G_nx.neighbors(v)]
                if neighbor_labels:
                    most_common = Counter(neighbor_labels).most_common(1)[0][0]
                    if labels[v] != most_common:
                        labels[v] = most_common
                        updated = True
            iteration += 1
        return labels

    # 코어 비율 1.0 → Leiden만 수행
    if core_ratio >= 1.0:
        G_ig = ig.Graph.TupleList(G_nx.edges(), directed=False)
        if seed is not None:
            part = find_partition(G_ig, ModularityVertexPartition, seed=seed)
        else:
            part = find_partition(G_ig, ModularityVertexPartition)
        return {v["name"]: part.membership[i] for i, v in enumerate(G_ig.vs)}

    # 하이브리드 모드 (0.0 < core_ratio < 1.0)
    
    # 1. 중심성에 따라 핵심 노드 선별
    centrality_scores = compute_centrality(G_nx, centrality_method)
    V_core = get_top_nodes(centrality_scores, core_ratio)
    V_periphery = [node for node in G_nx.nodes() if node not in V_core]

    # 2. 핵심 노드에 Leiden 적용
    if len(V_core) == 0:
        core_labels = {}
    elif len(V_core) == 1:
        # 핵심 노드가 1개면 바로 라벨 부여
        core_labels = {V_core[0]: 0}
    else:
        G_core_nx = G_nx.subgraph(V_core).copy()
        
        # 연결된 컴포넌트가 있는지 확인
        if G_core_nx.number_of_edges() == 0:
            # 엣지가 없으면 각각 별도 커뮤니티
            core_labels = {node: i for i, node in enumerate(V_core)}
        else:
            # 연결된 컴포넌트들에 각각 Leiden 적용
            core_labels = {}
            community_id = 0
            
            for component in nx.connected_components(G_core_nx):
                if len(component) == 1:
                    # 단일 노드는 별도 커뮤니티
                    node = list(component)[0]
                    core_labels[node] = community_id
                    community_id += 1
                else:
                    # 연결된 컴포넌트에 Leiden 적용
                    G_comp = G_core_nx.subgraph(component).copy()
                    G_comp_ig = ig.Graph.TupleList(G_comp.edges(), directed=False)
                    
                    if seed is not None:
                        part = find_partition(G_comp_ig, ModularityVertexPartition, seed=seed)
                    else:
                        part = find_partition(G_comp_ig, ModularityVertexPartition)
                    
                    # 라벨 매핑
                    for i, v in enumerate(G_comp_ig.vs):
                        original_node = v["name"]
                        core_labels[original_node] = part.membership[i] + community_id
                    
                    # 다음 컴포넌트를 위해 커뮤니티 ID 업데이트
                    community_id += max(part.membership) + 1

    # 3. 전체 라벨 초기화
    labels = {}
    max_core_label = -1
    
    # 핵심 노드 라벨 설정
    for node in V_core: 
        if node in core_labels:
            labels[node] = core_labels[node]
            max_core_label = max(max_core_label, core_labels[node])
        else: 
            # 연결이 없는 핵심 노드 처리
            max_core_label += 1
            labels[node] = max_core_label

    # 비핵심 노드 초기 라벨 (각각 고유 라벨)
    unique_label_start = max_core_label + 1
    for i, node in enumerate(V_periphery):
        labels[node] = unique_label_start + i

    # 4. 비핵심 노드에 LPA 수행 (핵심 노드 라벨은 고정)
    for iteration in range(max_iter):
        updates_made = False
        periphery_list = list(V_periphery)
        
        if seed is not None:
            random.shuffle(periphery_list)  # 편향을 막기 위해 순서 섞기
            
        for v in periphery_list:
            neighbor_labels = [labels[n] for n in G_nx.neighbors(v)]
            if not neighbor_labels:
                continue

            most_common = Counter(neighbor_labels).most_common(1)[0][0]
            if labels[v] != most_common:
                labels[v] = most_common
                updates_made = True

        if not updates_made:  # 변경이 없으면 수렴된 것으로 보고 조기 종료
            break

    return labels
