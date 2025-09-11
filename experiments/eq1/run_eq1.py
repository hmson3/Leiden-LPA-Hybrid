import os
import time
import csv
import networkx as nx
import sys
import random
sys.path.append("../../src")
from exp4_leiden_lpa import leiden_lpa_hybrid
from evaluation import compute_modularity, compute_nmi, compute_f1_score, compute_ari
from baseline import run_leiden, run_louvain, run_pure_lpa

# 버전 이름을 명확히 설정
ALGORITHM_VERSION = "eq1_baseline"

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
                   datasets=None,
                   algorithms=None,  # 새로 추가: 실행할 알고리즘 선택
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
    algorithms : list or None
        실행할 알고리즘 리스트. None이면 모든 알고리즘 실행
        가능한 값: ["hybrid", "leiden", "louvain", "lpa"]
    repeat : int
        각 데이터셋당 반복 횟수
    output_csv : str
        결과를 저장할 CSV 파일 경로
    """
    
    # 기본 알고리즘 설정
    if algorithms is None:
        algorithms = ["hybrid", "leiden", "louvain", "lpa"]
    
    # 알고리즘별 함수 매핑
    algorithm_functions = {
        "hybrid": lambda G, seed: leiden_lpa_hybrid(G, seed=seed),
        "leiden": run_leiden,
        "louvain": run_louvain,
        "lpa": run_pure_lpa
    }
    
    algorithm_names = {
        "hybrid": "Leiden-LPA_Hybrid",
        "leiden": "Pure_Leiden", 
        "louvain": "Louvain",
        "lpa": "Pure_LPA"
    }
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = ["Graph", "Repeat", "Algorithm", "Time", "Modularity", "NMI", "F1_Score", "ARI", "Num_Communities"]

    # 실험할 데이터셋 결정
    if datasets is None:
        available_datasets = sorted(os.listdir(dataset_base))
        datasets_to_run = [d for d in available_datasets if os.path.isdir(os.path.join(dataset_base, d))]
    else:
        datasets_to_run = datasets
        for dataset in datasets:
            if not os.path.isdir(os.path.join(dataset_base, dataset)):
                print(f"[WARNING] Dataset '{dataset}' not found in {dataset_base}")
        datasets_to_run = [d for d in datasets if os.path.isdir(os.path.join(dataset_base, d))]
    
    print(f"[INFO] Running experiments on datasets: {datasets_to_run}")
    print(f"[INFO] Running algorithms: {[algorithm_names[alg] for alg in algorithms]}")
    
    total_experiments = len(datasets_to_run) * repeat * len(algorithms)
    print(f"[INFO] Total experiments: {total_experiments}")
    append = os.path.exists(output_csv)
    with open(output_csv, 'a' if append else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append:
            writer.writeheader()

        experiment_count = 0

        for dataset_name in datasets_to_run:
            folder = os.path.join(dataset_base, dataset_name)
            
            print(f"[INFO] Running on {dataset_name}...")
            G, gt = load_graph_and_labels(folder)
            print(f"       Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

            for i in range(repeat):
                
                seed = i + 42
                results_for_this_run = []

                # 각 알고리즘별 실행
                for alg_key in algorithms:
                    experiment_count += 1
                    alg_func = algorithm_functions[alg_key]
                    alg_name = algorithm_names[alg_key]
                    
                    try:
                        # 알고리즘 실행
                        start = time.time()
                        labels = alg_func(G, seed=seed)
                        exec_time = time.time() - start
                        
                        # 평가 지표 계산
                        modularity = compute_modularity(G, labels)
                        nmi = compute_nmi(labels, gt)
                        f1 = compute_f1_score(labels, gt)
                        ari = compute_ari(labels, gt)
                        num_communities = len(set(labels.values()))

                        result = {
                            "Graph": dataset_name,
                            "Repeat": i,
                            "Algorithm": alg_name,
                            "Time": round(exec_time, 7),
                            "Modularity": round(modularity, 7),
                            "NMI": round(nmi, 7),
                            "F1_Score": round(f1, 7),
                            "ARI": round(ari, 7),
                            "Num_Communities": num_communities
                        }
                        print(dataset_name, i, alg_name, round(exec_time, 7), round(modularity, 7), round(nmi, 7), round(f1, 7), round(ari, 7), num_communities)

                        writer.writerow(result)
                        results_for_this_run.append((alg_name, exec_time, modularity, nmi, f1, ari, num_communities))

                    except Exception as e:
                        print(f"       [ERROR] {alg_name} failed: {e}")
                        continue

                # 평균 결과 출력
                    if results_for_this_run:
                        avg_time = sum(r[1] for r in results_for_this_run) / len(results_for_this_run)
                        avg_mod = sum(r[2] for r in results_for_this_run) / len(results_for_this_run)
                        avg_nmi = sum(r[3] for r in results_for_this_run) / len(results_for_this_run)
                        avg_f1 = sum(r[4] for r in results_for_this_run) / len(results_for_this_run)
                        avg_ari = sum(r[5] for r in results_for_this_run) / len(results_for_this_run)
                        avg_communities = sum(r[6] for r in results_for_this_run) / len(results_for_this_run)

                        print(f"           Avg: {avg_time:.4f}s, Mod:{avg_mod:.3f}, NMI:{avg_nmi:.3f}, "
                              f"F1:{avg_f1:.3f}, ARI:{avg_ari:.3f}, Communities:{avg_communities:.1f}")
                        
                    # 진행률 표시
                    progress = (experiment_count / total_experiments) * 100
                    print(f"           Progress: {experiment_count}/{total_experiments} ({progress:.1f}%)")

    print(f"[COMPLETED] Results saved to {output_csv}")

if __name__ == "__main__":
    print("=== Leiden-LPA 하이브리드 실험 도구 ===")
    print()
    
    # 1. 데이터셋 선택
    print("📊 Available datasets:")
    dataset_base = "../../data/data/processed"
    if os.path.exists(dataset_base):
        available = [d for d in os.listdir(dataset_base) if os.path.isdir(os.path.join(dataset_base, d))]
        for i, dataset in enumerate(available, 1):
            print(f"  {i}. {dataset}")
        
        print("\nSelect datasets to run experiments:")
        print("  - Enter dataset names separated by comma (e.g., karate,email-Eu-core)")
        print("  - Enter 'all' to run on all datasets")
        print("  - Press Enter to use default selection")
        
        user_input = input("Dataset selection: ").strip()
        
        if user_input.lower() == 'all' or user_input == '':
            selected_datasets = ["karate", "cora", "citeseer", "pubmed","polblogs", "dolphin", "football", "mexican"] 
        else:
            selected_datasets = [name.strip() for name in user_input.split(',')]
    else:
        print(f"Dataset folder '{dataset_base}' not found!")
        selected_datasets = None
    
    # 2. 알고리즘 선택
    print("\n🔬 Available algorithms:")
    available_algorithms = {
        "1": ("hybrid", "Leiden-LPA Hybrid"),
        "2": ("leiden", "Pure Leiden"),
        "3": ("louvain", "Louvain"),  
        "4": ("lpa", "Pure LPA")
    }
    
    for key, (_, name) in available_algorithms.items():
        print(f"  {key}. {name}")
    
    print("\nSelect algorithms to run:")
    print("  - Enter numbers separated by comma (e.g., 1,2,3)")
    print("  - Enter 'all' to run all algorithms")
    print("  - Press Enter to run all algorithms")
    
    alg_input = input("Algorithm selection: ").strip()
    
    if alg_input.lower() == 'all' or alg_input == '':
        selected_algorithms = ["hybrid", "leiden"]
    else:
        selected_numbers = [num.strip() for num in alg_input.split(',')]
        selected_algorithms = []
        for num in selected_numbers:
            if num in available_algorithms:
                selected_algorithms.append(available_algorithms[num][0])
    
    # 3. 실행
    print(f"\n🚀 Starting experiments...")
    print(f"   Datasets: {selected_datasets}")
    print(f"   Algorithms: {[available_algorithms[k][1] for k in available_algorithms.keys() if available_algorithms[k][0] in selected_algorithms]}")
    
    run_experiment(
        datasets=selected_datasets,
        algorithms=selected_algorithms,
        repeat=5
    )