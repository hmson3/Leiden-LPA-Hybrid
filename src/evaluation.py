import igraph as ig
from sklearn.metrics import normalized_mutual_info_score

def compute_modularity(G_nx, labels):
    import igraph as ig
    G_ig = ig.Graph.TupleList(G_nx.edges(), directed=False)

    label_list = []
    for v in G_ig.vs:
        label = labels.get(str(v["name"]))
        if label is None:
            label = 0  # fallback to cluster 0
        label_list.append(int(label))

    return G_ig.modularity(label_list)

def compute_nmi(pred_labels, true_labels):
    # 공통 노드만 추출
    common_nodes = set(pred_labels.keys()) & set(true_labels.keys())

    # 리스트 만들기 (None 제거 or 기본값 0 대체)
    pred = [int(pred_labels[n]) if pred_labels[n] is not None else 0 for n in common_nodes]
    true = [int(true_labels[n]) for n in common_nodes]

    return normalized_mutual_info_score(true, pred)
