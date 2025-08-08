import random

def run_leiden(G, seed=None):
    """순수 Leiden 알고리즘 실행"""
    import igraph as ig
    import leidenalg
    G_ig = ig.Graph.TupleList(G.edges(), directed=False)
    if seed is not None:
        partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition, seed=seed)
    else:
        partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)
    return {v["name"]: partition.membership[i] for i, v in enumerate(G_ig.vs)}

def run_louvain(G, seed=None):
    """Louvain 알고리즘 실행"""
    import community as community_louvain  # python-louvain 패키지
    
    if seed is not None:
        random.seed(seed)
        
    # Louvain 알고리즘 실행
    partition = community_louvain.best_partition(G, random_state=seed)
    return partition

def run_pure_lpa(G, seed=None, max_iterations=10):
    """순수 LPA 알고리즘 실행 - Counter 제거한 최적화 버전"""
    
    def get_most_common_label(neighbor_labels):
        """가장 많이 나타나는 라벨 반환 (Counter 없이)"""
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
    
    # 초기 라벨 설정 (각 노드는 고유 라벨)
    labels = {v: i for i, v in enumerate(G.nodes())}
    
    # LPA 반복 수행
    for iteration in range(max_iterations):
        updated = False
        nodes_list = list(G.nodes())
        
        if seed is not None:
            random.seed(seed + iteration)
            random.shuffle(nodes_list)
        
        for v in nodes_list:
            neighbor_labels = [labels[n] for n in G.neighbors(v)]
            if neighbor_labels:
                most_common = get_most_common_label(neighbor_labels)
                if labels[v] != most_common:
                    labels[v] = most_common
                    updated = True
        
        if not updated:  # 수렴하면 종료
            break
    
    return labels