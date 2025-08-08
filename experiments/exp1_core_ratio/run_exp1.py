import os
import time
import csv
import networkx as nx
import sys
import random
sys.path.append("../../src")
from leiden_lpa import leiden_lpa_hybrid
from evaluation import compute_modularity, compute_nmi, compute_f1_score, compute_ari

# 실험 이름
EXPERIMENT_NAME = "Core_Ratio"

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
                   core_ratios=None,
                   repeat=5, 
                   output_csv=f"./results/exp1_{EXPERIMENT_NAME}.csv"):
    """
    Core Ratio 실험 실행 함수
    
    Parameters:
    -----------
    dataset_base : str
        데이터셋이 있는 기본 폴더
    datasets : list or None
        실험할 데이터셋 이름 리스트
    core_ratios : list or None
        실험할 core ratio 값들. None이면 [0.0, 0.1, 0.2, ..., 1.0] 사용
    repeat : int
        각 설정당 반복 횟수
    output_csv : str
        결과를 저장할 CSV 파일 경로
    """
    
    # 기본 core ratio 설정 (0.0부터 1.0까지 0.1 단위)
    if core_ratios is None:
        core_ratios = [round(i * 0.1, 1) for i in range(11)]  # [0.0, 0.1, 0.2, ..., 1.0]
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = ["Graph", "Core_Ratio", "Repeat", "Time", "Modularity", 
                  "NMI", "F1_Score", "ARI", "Num_Communities"]

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
    print(f"[INFO] Core ratios: {core_ratios}")
    print(f"[INFO] Repeat count: {repeat}")
    
    # 전체 실험 횟수 계산
    total_experiments = len(datasets_to_run) * len(core_ratios) * repeat
    print(f"[INFO] Total experiments: {total_experiments}")
    
    append = os.path.exists(output_csv)
    with open(output_csv, 'a' if append else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append:
            writer.writeheader()

        experiment_count = 0
        
        for dataset_name in datasets_to_run:
            folder = os.path.join(dataset_base, dataset_name)
            
            print(f"\n[INFO] Running on {dataset_name}...")
            G, gt = load_graph_and_labels(folder)
            print(f"       Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
            
            for core_ratio in core_ratios:
                print(f"       Core ratio: {core_ratio}")
                
                # 각 core ratio별 결과 수집
                ratio_results = []
                
                for i in range(repeat):
                    experiment_count += 1
                    seed = i + 42
                    
                    try:
                        # Leiden-LPA 하이브리드 실행 (PageRank 중심성 기본 사용)
                        start = time.time()
                        labels = leiden_lpa_hybrid(G, core_ratio=core_ratio, seed=seed)
                        exec_time = time.time() - start
                        
                        # 평가 지표 계산
                        modularity = compute_modularity(G, labels)
                        nmi = compute_nmi(labels, gt)
                        f1 = compute_f1_score(labels, gt)
                        ari = compute_ari(labels, gt)
                        num_communities = len(set(labels.values()))

                        result = {
                            "Graph": dataset_name,
                            "Core_Ratio": core_ratio,
                            "Repeat": i,
                            "Time": round(exec_time, 7),
                            "Modularity": round(modularity, 7),
                            "NMI": round(nmi, 7),
                            "F1_Score": round(f1, 7),
                            "ARI": round(ari, 7),
                            "Num_Communities": num_communities
                        }
                        
                        writer.writerow(result)
                        ratio_results.append((exec_time, modularity, nmi, f1, ari, num_communities))
                        
                    except Exception as e:
                        print(f"         [ERROR] Core ratio {core_ratio}, Repeat {i} failed: {e}")
                        continue
                
                # 평균 결과 출력
                if ratio_results:
                    avg_time = sum(r[0] for r in ratio_results) / len(ratio_results)
                    avg_mod = sum(r[1] for r in ratio_results) / len(ratio_results)
                    avg_nmi = sum(r[2] for r in ratio_results) / len(ratio_results)
                    avg_f1 = sum(r[3] for r in ratio_results) / len(ratio_results)
                    avg_ari = sum(r[4] for r in ratio_results) / len(ratio_results)
                    avg_communities = sum(r[5] for r in ratio_results) / len(ratio_results)
                    
                    print(f"         Avg: {avg_time:.4f}s, Mod:{avg_mod:.3f}, NMI:{avg_nmi:.3f}, "
                          f"F1:{avg_f1:.3f}, ARI:{avg_ari:.3f}, Communities:{avg_communities:.1f}")
                    
                    # 진행률 표시
                    progress = (experiment_count / total_experiments) * 100
                    print(f"         Progress: {experiment_count}/{total_experiments} ({progress:.1f}%)")

    print(f"\n[COMPLETED] Results saved to {output_csv}")

if __name__ == "__main__":
    print("=== EXP1: Core Ratio 분석 실험 ===")
    print()
    
    # 1. 데이터셋 선택
    print("📊 Available datasets:")
    dataset_base = "../../data/data/processed"
    if os.path.exists(dataset_base):
        available = [d for d in os.listdir(dataset_base) if os.path.isdir(os.path.join(dataset_base, d))]
        for i, dataset in enumerate(available, 1):
            print(f"  {i}. {dataset}")
        
        print("\nSelect datasets to run experiments:")
        print("  - Enter dataset names separated by comma (e.g., karate,cora)")
        print("  - Enter 'all' to run on all datasets")
        print("  - Press Enter to use default selection")
        
        user_input = input("Dataset selection: ").strip()
        
        if user_input.lower() == 'all':
            selected_datasets = available
        elif user_input == '':
            selected_datasets = ["karate", "cora", "citeseer", "pubmed", "dolphin", "football", "mexican", "polblogs"] 
        else:
            selected_datasets = [name.strip() for name in user_input.split(',')]
    else:
        print(f"Dataset folder '{dataset_base}' not found!")
        selected_datasets = None
    
    # 2. Core ratio 설정
    print("\n📏 Core ratio settings:")
    print("  - Enter core ratios separated by comma (e.g., 0.0,0.2,0.4,0.6,1.0)")
    print("  - Press Enter to use default (0.0 to 1.0, step 0.1)")
    
    ratio_input = input("Core ratios: ").strip()
    if ratio_input:
        core_ratios = [float(r.strip()) for r in ratio_input.split(',')]
    else:
        core_ratios = [round(i * 0.1, 1) for i in range(11)]  # 0.0 ~ 1.0
    
    # 3. 반복 횟수 설정
    repeat_input = input("\nRepeat count (default: 5): ").strip()
    repeat = int(repeat_input) if repeat_input else 5
    
    # 4. 실행
    print(f"\n🚀 Starting Core Ratio Analysis...")
    print(f"   Datasets: {selected_datasets}")
    print(f"   Core ratios: {core_ratios}")
    print(f"   Centrality method: PageRank (fixed)")
    print(f"   Repeat: {repeat}")
    
    run_experiment(
        datasets=selected_datasets,
        core_ratios=core_ratios,
        repeat=repeat
    )
    
    print("\n✅ Core Ratio Analysis completed!")