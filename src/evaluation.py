import igraph as ig
from sklearn.metrics import normalized_mutual_info_score, f1_score, adjusted_rand_score
import numpy as np
from scipy.optimize import linear_sum_assignment

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


def compute_f1_score(pred_labels, true_labels):
    """F1-score 계산 (macro-averaged with optimal mapping)"""
    # 라벨을 리스트로 변환
    pred_list = [pred_labels[node] for node in sorted(pred_labels.keys())]
    true_list = [true_labels[node] for node in sorted(true_labels.keys()) if node in pred_labels]
    
    pred_set = sorted(set(pred_list))
    true_set = sorted(set(true_list))
    
    # 혼동 행렬 생성
    confusion_matrix = np.zeros((len(true_set), len(pred_set)))
    
    for true_label, pred_label in zip(true_list, pred_list):
        true_idx = true_set.index(true_label)
        pred_idx = pred_set.index(pred_label)
        confusion_matrix[true_idx, pred_idx] += 1
    
    # 헝가리안 알고리즘으로 최적 매칭
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    
    # 매칭 결과에 따라 예측 레이블 재매핑
    label_mapping = {}
    for true_idx, pred_idx in zip(row_ind, col_ind):
        if pred_idx < len(pred_set):
            label_mapping[pred_set[pred_idx]] = true_set[true_idx]
    
    # 재매핑된 예측 레이블 생성
    remapped_pred = [label_mapping.get(pred, pred) for pred in pred_list]
    
    # F1-score 계산 (macro average)
    return f1_score(true_list, remapped_pred, average='macro')

def compute_ari(pred_labels, true_labels):
    """ARI (Adjusted Rand Index) 계산"""
    # 라벨을 리스트로 변환
    pred_list = [pred_labels[node] for node in sorted(pred_labels.keys())]
    true_list = [true_labels[node] for node in sorted(true_labels.keys()) if node in pred_labels]
    
    return adjusted_rand_score(true_list, pred_list)