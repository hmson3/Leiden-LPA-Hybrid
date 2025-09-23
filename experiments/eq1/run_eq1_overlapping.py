import os
import time
import csv
import networkx as nx
import sys
import random
sys.path.append("../../src")
from llama import llama
from evaluation import compute_modularity, compute_nmi, compute_f1_score, compute_ari
from baseline import run_leiden, run_louvain, run_pure_lpa

# 버전 이름을 명확히 설정
ALGORITHM_VERSION = "eq1_baseline"

def load_graph_and_labels(dataset_folder):
    """그래프와 레이블 로드 (overlapping 지원하지만 간단하게)"""
    graph_path = os.path.join(dataset_folder, "graph.edgelist")
    label_path = os.path.join(dataset_folder, "labels.txt")

    G = nx.read_edgelist(graph_path, nodetype=str)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    gt = {}
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            node = parts[0]
            
            if len(parts) == 2:
                # Non-overlapping: node comm_id
                gt[node] = int(parts[1])
            else:
                # Overlapping: node comm_id1 comm_id2 ... (첫 번째 커뮤니티만 사용)
                gt[node] = int(parts[1])  # 첫 번째 커뮤니티 ID만 사용

    return G, gt

def is_overlapping_dataset(dataset_name):
    """데이터셋이 overlapping community인지 확인"""
    overlapping_datasets = ["com-youtube", "com-dblp", "dblp", "com-amazon"]
    return any(overlap_name in dataset_name.lower() for overlap_name in overlapping_datasets)

def run_experiment(dataset_base="../../data/data/processed", 
                   datasets=None,
                   algorithms=None,
                   repeat=10, 
                   output_csv=f"./results/results_{ALGORITHM_VERSION}.csv"):
    """
    실험 실행 함수 (간단한 버전)
    
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
        "hybrid": lambda G, seed: llama(G, seed=seed),
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
    fieldnames = ["Graph", "Repeat", "Algorithm", "Time", "Modularity", "NMI", "F1_Score", "ARI", "Num_Communities", "Is_Overlapping"]

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
            
            # Overlapping 데이터셋 여부 확인
            is_overlapping = is_overlapping_dataset(dataset_name)
            print(f"       Overlapping dataset: {is_overlapping}")
            if is_overlapping:
                print(f"       [NOTE] For overlapping datasets, only Modularity will be calculated")

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
                        
                        # 기본 지표 계산
                        modularity = compute_modularity(G, labels)
                        num_communities = len(set(labels.values()))
                        
                        # 평가 지표 계산
                        if is_overlapping:
                            # Overlapping 데이터셋: Modularity만 계산
                            nmi = None
                            f1 = None
                            ari = None
                            print(f"       {dataset_name} {i} {alg_name} T:{exec_time:.4f} Mod:{modularity:.3f} C:{num_communities}")
                        else:
                            # Non-overlapping 데이터셋: 모든 지표 계산
                            try:
                                nmi = compute_nmi(labels, gt)
                            except Exception as e:
                                print(f"[WARNING] NMI calculation failed: {e}")
                                nmi = None
                            
                            try:
                                f1 = compute_f1_score(labels, gt)
                            except Exception as e:
                                print(f"[WARNING] F1 calculation failed: {e}")
                                f1 = None
                                
                            try:
                                ari = compute_ari(labels, gt)
                            except Exception as e:
                                print(f"[WARNING] ARI calculation failed: {e}")
                                ari = None
                            
                            # 안전한 문자열 포맷팅
                            nmi_str = f"{nmi:.3f}" if nmi is not None else "N/A"
                            f1_str = f"{f1:.3f}" if f1 is not None else "N/A"
                            ari_str = f"{ari:.3f}" if ari is not None else "N/A"
                            print(f"       {dataset_name} {i} {alg_name} T:{exec_time:.4f} Mod:{modularity:.3f} NMI:{nmi_str} F1:{f1_str} ARI:{ari_str} C:{num_communities}")

                        result = {
                            "Graph": dataset_name,
                            "Repeat": i,
                            "Algorithm": alg_name,
                            "Time": round(exec_time, 7),
                            "Modularity": round(modularity, 7),
                            "NMI": round(nmi, 7) if nmi is not None else None,
                            "F1_Score": round(f1, 7) if f1 is not None else None,
                            "ARI": round(ari, 7) if ari is not None else None,
                            "Num_Communities": num_communities,
                            "Is_Overlapping": is_overlapping
                        }

                        writer.writerow(result)
                        results_for_this_run.append((alg_name, exec_time, modularity, nmi, f1, ari, num_communities))

                    except Exception as e:
                        print(f"       [ERROR] {alg_name} failed: {e}")
                        continue

                # 평균 결과 출력 (해당 repeat의 모든 알고리즘)
                if results_for_this_run:
                    avg_time = sum(r[1] for r in results_for_this_run) / len(results_for_this_run)
                    avg_mod = sum(r[2] for r in results_for_this_run) / len(results_for_this_run)
                    avg_communities = sum(r[6] for r in results_for_this_run) / len(results_for_this_run)
                    
                    if is_overlapping:
                        print(f"           Avg: {avg_time:.4f}s, Mod:{avg_mod:.3f}, Communities:{avg_communities:.1f}")
                    else:
                        # Non-overlapping 평균 계산 (None이 아닌 값들만)
                        non_none_nmi = [r[3] for r in results_for_this_run if r[3] is not None]
                        non_none_f1 = [r[4] for r in results_for_this_run if r[4] is not None]
                        non_none_ari = [r[5] for r in results_for_this_run if r[5] is not None]
                        
                        avg_nmi = sum(non_none_nmi) / len(non_none_nmi) if non_none_nmi else None
                        avg_f1 = sum(non_none_f1) / len(non_none_f1) if non_none_f1 else None
                        avg_ari = sum(non_none_ari) / len(non_none_ari) if non_none_ari else None
                        
                        # 안전한 문자열 포맷팅
                        avg_nmi_str = f"{avg_nmi:.3f}" if avg_nmi is not None else "N/A"
                        avg_f1_str = f"{avg_f1:.3f}" if avg_f1 is not None else "N/A"
                        avg_ari_str = f"{avg_ari:.3f}" if avg_ari is not None else "N/A"
                        print(f"           Avg: {avg_time:.4f}s, Mod:{avg_mod:.3f}, NMI:{avg_nmi_str}, F1:{avg_f1_str}, ARI:{avg_ari_str}, Communities:{avg_communities:.1f}")
                    
                # 진행률 표시
                progress = (experiment_count / total_experiments) * 100
                print(f"           Progress: {experiment_count}/{total_experiments} ({progress:.1f}%)")

    print(f"[COMPLETED] Results saved to {output_csv}")

if __name__ == "__main__":
    print("=== Leiden-LPA 하이브리드 실험 도구 (간단 버전) ===")
    print()
    
    # 1. 데이터셋 선택
    print("📊 Available datasets:")
    dataset_base = "../../data/data/processed"
    if os.path.exists(dataset_base):
        available = [d for d in os.listdir(dataset_base) if os.path.isdir(os.path.join(dataset_base, d))]
        for i, dataset in enumerate(available, 1):
            overlap_mark = " (overlapping - Modularity only)" if is_overlapping_dataset(dataset) else ""
            print(f"  {i}. {dataset}{overlap_mark}")
        
        print("\nSelect datasets to run experiments:")
        print("  - Enter dataset names separated by comma (e.g., karate,com-dblp)")
        print("  - Enter 'all' to run on all datasets")
        print("  - Press Enter to use default selection")
        
        user_input = input("Dataset selection: ").strip()
        
        if user_input.lower() == 'all':
            selected_datasets = available
        elif user_input == '':
            # 기본 선택: 다양한 데이터셋 포함
            default_datasets = ["karate", "cora", "citeseer", "pubmed", "polblogs", "dolphin", "football", "mexican"]
            selected_datasets = [d for d in default_datasets if d in available]
            # Overlapping 데이터셋도 포함 (Modularity만 계산)
            overlap_datasets = ["com-dblp", "com-amazon", "com-youtube"]
            selected_datasets.extend([d for d in overlap_datasets if d in available])
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
    print("  - Press Enter to run hybrid and leiden")
    
    alg_input = input("Algorithm selection: ").strip()
    
    if alg_input.lower() == 'all':
        selected_algorithms = ["hybrid", "leiden", "louvain", "lpa"]
    elif alg_input == '':
        selected_algorithms = ["hybrid", "leiden","lpa"]
    else:
        selected_numbers = [num.strip() for num in alg_input.split(',')]
        selected_algorithms = []
        for num in selected_numbers:
            if num in available_algorithms:
                selected_algorithms.append(available_algorithms[num][0])
    
    # 3. 반복 횟수 설정
    print("\n🔄 Number of repeats:")
    repeat_input = input("Enter number of repeats (default: 5): ").strip()
    repeat_count = int(repeat_input) if repeat_input.isdigit() else 5
    
    # 4. 실행
    print(f"\n🚀 Starting experiments...")
    print(f"   Datasets: {selected_datasets}")
    print(f"   Algorithms: {[available_algorithms[k][1] for k in available_algorithms.keys() if available_algorithms[k][0] in selected_algorithms]}")
    print(f"   Repeats: {repeat_count}")
    print(f"   Note: Overlapping datasets will only calculate Modularity")
    
    run_experiment(
        datasets=selected_datasets,
        algorithms=selected_algorithms,
        repeat=repeat_count
    )