import os
import time
import csv
import networkx as nx
import sys
import random
sys.path.append("../../src")
from leiden_lpa import leiden_lpa_hybrid
from evaluation import compute_modularity, compute_nmi

# 버전 이름을 명확히 설정
ALGORITHM_VERSION = "Leiden-LPA_Hybrid"

def run_leiden(G, seed=None):
    import igraph as ig
    import leidenalg
    G_ig = ig.Graph.TupleList(G.edges(), directed=False)
    if seed is not None:
        partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition, seed=seed)
    else:
        partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)
    return {v["name"]: partition.membership[i] for i, v in enumerate(G_ig.vs)}

def load_graph_and_labels(dataset_folder):
    graph_path = os.path.join(dataset_folder, "graph.edgelist")
    label_path = os.path.join(dataset_folder, "labels.txt")

    G = nx.read_edgelist(graph_path, nodetype=str)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    gt = {}
    with open(label_path, "r") as f:
        for line in f:
            node, label = line.strip().split()
            gt[node] = int(label)

    return G, gt

def run_experiment(dataset_base="../../data/data/processed", 
                   datasets=None,  # 새로 추가: 특정 데이터셋 리스트
                   repeat=10, 
                   output_csv=f"./results/results_{ALGORITHM_VERSION}.csv"):
    """
    실험 실행 함수
    
    Parameters:
    -----------
    dataset_base : str
        데이터셋이 있는 기본 폴더
    datasets : list or None
        실험할 데이터셋 이름 리스트. None이면 모든 데이터셋 사용
        예: ["karate", "email-Eu-core"]
    repeat : int
        각 데이터셋당 반복 횟수
    output_csv : str
        결과를 저장할 CSV 파일 경로
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = ["Graph", "Repeat", "Algorithm", "Time (s)", "Modularity", "NMI"]

    # 실험할 데이터셋 결정
    if datasets is None:
        # 모든 데이터셋 사용 (기존 방식)
        available_datasets = sorted(os.listdir(dataset_base))
        datasets_to_run = [d for d in available_datasets if os.path.isdir(os.path.join(dataset_base, d))]
    else:
        # 지정된 데이터셋만 사용
        datasets_to_run = datasets
        # 존재하지 않는 데이터셋 체크
        for dataset in datasets:
            if not os.path.isdir(os.path.join(dataset_base, dataset)):
                print(f"[WARNING] Dataset '{dataset}' not found in {dataset_base}")
        datasets_to_run = [d for d in datasets if os.path.isdir(os.path.join(dataset_base, d))]
    
    print(f"[INFO] Running experiments on datasets: {datasets_to_run}")
    
    append = os.path.exists(output_csv)
    with open(output_csv, 'a' if append else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append:
            writer.writeheader()

        for dataset_name in datasets_to_run:
            folder = os.path.join(dataset_base, dataset_name)
            
            print(f"[INFO] Running on {dataset_name}...")
            G, gt = load_graph_and_labels(folder)
            print(f"       Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

            for i in range(repeat):
                seed = i + 42

                # Leiden-LPA 하이브리드 알고리즘 실행
                start = time.time()
                hybrid_labels = leiden_lpa_hybrid(G, seed=seed)
                hybrid_time = time.time() - start
                hybrid_mod = compute_modularity(G, hybrid_labels)
                hybrid_nmi = compute_nmi(hybrid_labels, gt)

                writer.writerow({
                    "Graph": dataset_name,
                    "Repeat": i,
                    "Algorithm": ALGORITHM_VERSION,
                    "Time (s)": round(hybrid_time, 7),
                    "Modularity": round(hybrid_mod, 7),
                    "NMI": round(hybrid_nmi, 7)
                })

                # 기존 Leiden 알고리즘 실행 (비교용)
                start = time.time()
                leiden_labels = run_leiden(G, seed=seed)
                leiden_time = time.time() - start
                leiden_mod = compute_modularity(G, leiden_labels)
                leiden_nmi = compute_nmi(leiden_labels, gt)

                writer.writerow({
                    "Graph": dataset_name,
                    "Repeat": i,
                    "Algorithm": "Leiden",
                    "Time (s)": round(leiden_time, 7),
                    "Modularity": round(leiden_mod, 7),
                    "NMI": round(leiden_nmi, 7)
                })

                print(f"       Repeat {i}: Hybrid({hybrid_time:.4f}s, {hybrid_mod:.4f}, {hybrid_nmi:.4f}) vs Leiden({leiden_time:.4f}s, {leiden_mod:.4f}, {leiden_nmi:.4f})")

    print(f"[COMPLETED] Results saved to {output_csv}")

if __name__ == "__main__":
    # 사용 예시들:
    
    # 1. 모든 데이터셋에 대해 실험 (기존 방식)
    # run_experiment()
    
    # 2. 특정 데이터셋들만 실험
    # run_experiment(datasets=["karate", "email-Eu-core"])
    
    # 3. 하나의 데이터셋만 실험
    # run_experiment(datasets=["com-dblp"])
    
    # 4. 사용자가 직접 입력
    print("Available datasets:")
    dataset_base = "../../data/data/processed"  # 기본 데이터셋 폴더 경로
    if os.path.exists(dataset_base):
        available = [d for d in os.listdir(dataset_base) if os.path.isdir(os.path.join(dataset_base, d))]
        for i, dataset in enumerate(available, 1):
            print(f"  {i}. {dataset}")
        
        print("\nSelect datasets to run experiments:")
        print("  - Enter dataset names separated by comma (e.g., karate,email-Eu-core)")
        print("  - Enter 'all' to run on all datasets")
        print("  - Press Enter to use default selection")
        
        user_input = input("Selection: ").strip()
        
        if user_input.lower() == 'all' or user_input == '':
            selected_datasets = ["karate", "cora", "citeseer", "pubmed", "dolphin", "football", "mexican", "polblogs"] 
        else:
            selected_datasets = [name.strip() for name in user_input.split(',')]
        
        run_experiment(datasets=selected_datasets)
    else:
        print(f"Dataset folder '{dataset_base}' not found!")
        print("Running with default settings...")
        run_experiment()