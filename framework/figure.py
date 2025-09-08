import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import random
from matplotlib.patches import Patch
import time
import igraph as ig
import leidenalg

# 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False

def create_two_cluster_network():
    """2개 클러스터, 20개 노드의 명확한 네트워크 생성"""
    np.random.seed(42)
    random.seed(42)
    
    G = nx.Graph()
    
    # 클러스터 1 (노드 0-9): 10개 노드
    cluster1_nodes = list(range(10))
    hub1 = 0  # 허브 노드
    
    # 허브 중심 구조
    for node in range(1, 10):
        if node == 2: continue
        G.add_edge(hub1, node)
        
  
    # 클러스터 1 내부 연결 (높은 밀도)
    cluster1_edges = [
        (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8), (8, 9), # 링 구조
        (4, 6), (4, 8)  # 추가 연결
    ]
    G.add_edges_from(cluster1_edges)
    
    # 클러스터 2 (노드 10-19): 10개 노드
    cluster2_nodes = list(range(10, 20))
    hub2 = 10  # 허브 노드
    
    # 허브 중심 구조
    for node in range(11, 20):
        if node == 14: continue
        G.add_edge(hub2, node)
    
    # 클러스터 2 내부 연결 (높은 밀도)
    cluster2_edges = [
        (12, 13), (13, 14),
        (15, 16), (16, 17), (17, 18),
        (19, 11),  # 링 구조
        (13, 17)  # 추가 연결
    ]
    G.add_edges_from(cluster2_edges)
    
    # 클러스터 간 약한 연결 (bridge edges)
    bridge_edges = [(3, 13), (7, 17), (6, 18), (9, 15)]  # 2개의 bridge만
    G.add_edges_from(bridge_edges)
    
    print(f"Two-cluster network: {len(G.nodes())} nodes, {len(G.edges())} edges")
    return G, "Two-Cluster Network (20 nodes)"

def compute_pagerank_and_select_core(G, core_ratio=0.5):
    """PageRank 계산 및 핵심 노드 선별"""
    pagerank_scores = nx.pagerank(G, alpha=0.85)
    
    # 상위 core_ratio만큼 선택
    sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
    num_core = max(1, int(len(sorted_nodes) * core_ratio))
    
    core_nodes = [node for node, score in sorted_nodes[:num_core]]
    periphery_nodes = [node for node, score in sorted_nodes[num_core:]]
    
    return pagerank_scores, core_nodes, periphery_nodes

def apply_leiden_algorithm(G, core_nodes, resolution=1.0, seed=42):
    """실제 Leiden 알고리즘 적용"""
    if not core_nodes:
        return {}
    
    # NetworkX 서브그래프를 igraph로 변환
    core_subgraph = G.subgraph(core_nodes)
    
    # NetworkX 그래프를 igraph로 변환
    edge_list = [(u, v) for u, v in core_subgraph.edges()]
    
    # 노드 매핑 (igraph는 0부터 시작하는 연속 인덱스 필요)
    node_list = list(core_subgraph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    # igraph용 엣지 리스트 생성
    ig_edges = [(node_to_idx[u], node_to_idx[v]) for u, v in edge_list]
    
    # igraph 그래프 생성
    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(node_list))
    if ig_edges:  # 엣지가 있는 경우에만 추가
        ig_graph.add_edges(ig_edges)
    
    # Leiden 알고리즘 실행 (올바른 파라미터 사용)
    print(f"Applying Leiden algorithm to {len(core_nodes)} core nodes...")
    
    try:
        # resolution_parameter 대신 다른 방법 시도
        partition = leidenalg.find_partition(
            ig_graph, 
            leidenalg.ModularityVertexPartition,
            seed=seed
        )
    except Exception as e:
        print(f"Leiden with ModularityVertexPartition failed: {e}")
        # 대안: CPMVertexPartition 사용
        try:
            partition = leidenalg.find_partition(
                ig_graph, 
                leidenalg.CPMVertexPartition,
                resolution_parameter=resolution,
                seed=seed
            )
        except Exception as e2:
            print(f"Leiden with CPMVertexPartition also failed: {e2}")
            # 최종 폴백: 연결 컴포넌트 기반
            print("Using connected components as fallback...")
            core_labels = {}
            label_id = 0
            for component in nx.connected_components(core_subgraph):
                for node in component:
                    core_labels[node] = label_id
                label_id += 1
            return core_labels
    
    # 결과를 원래 노드 ID로 변환
    core_labels = {}
    for ig_node_idx, cluster_id in enumerate(partition.membership):
        original_node = idx_to_node[ig_node_idx]
        core_labels[original_node] = cluster_id
    
    # Leiden 결과 정보 출력
    print(f"Leiden clustering completed:")
    print(f"- Clusters found: {len(set(partition.membership))}")
    print(f"- Modularity: {partition.modularity:.4f}")
    try:
        print(f"- Quality: {partition.quality():.4f}")
    except:
        print("- Quality: (not available)")
    
    return core_labels

def propagate_labels(G, core_labels, periphery_nodes, max_iterations=10):
    """레이블 전파 알고리즘"""
    labels = core_labels.copy()
    
    # 주변 노드들에 고유 라벨 할당
    max_core_label = max(core_labels.values()) if core_labels else -1
    for i, node in enumerate(periphery_nodes):
        labels[node] = max_core_label + 1 + i
    
    print(f"Starting label propagation on {len(periphery_nodes)} periphery nodes...")
    
    # 라벨 전파 수행
    converged_iteration = 0
    for iteration in range(max_iterations):
        updated = False
        random.shuffle(periphery_nodes)
        
        for node in periphery_nodes:
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue
            
            neighbor_labels = [labels[neighbor] for neighbor in neighbors]
            if neighbor_labels:
                most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
                
                if labels[node] != most_common_label:
                    labels[node] = most_common_label
                    updated = True
        
        if not updated:
            converged_iteration = iteration + 1
            break
    
    print(f"Label propagation converged after {converged_iteration} iterations")
    return labels

def compute_metrics(G, labels):
    """클러스터링 품질 지표 계산"""
    try:
        # 커뮤니티 리스트 생성
        communities = {}
        for node, label in labels.items():
            if label not in communities:
                communities[label] = []
            communities[label].append(node)
        
        community_list = list(communities.values())
        
        # 모듈러리티 계산
        modularity = nx.algorithms.community.modularity(G, community_list)
        
        return {
            'modularity': modularity,
            'num_clusters': len(community_list),
            'cluster_sizes': [len(c) for c in community_list]
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {
            'modularity': 0.0,
            'num_clusters': len(set(labels.values())),
            'cluster_sizes': list(Counter(labels.values()).values())
        }

def get_two_cluster_layout(G):
    """2클러스터 구조가 잘 보이는 레이아웃"""
    pos = {}
    
    # 클러스터별 중심 위치
    cluster1_center = (-2, 0)
    cluster2_center = (2, 0)
    
    # 클러스터 1 (0-9) 배치
    cluster1_nodes = list(range(10))
    hub1 = 0
    pos[hub1] = cluster1_center
    
    # 나머지 노드들을 원형으로 배치
    other_nodes1 = [n for n in cluster1_nodes if n != hub1]
    for i, node in enumerate(other_nodes1):
        angle = 2 * np.pi * i / len(other_nodes1)
        radius = 1.2
        x = cluster1_center[0] + radius * np.cos(angle)
        y = cluster1_center[1] + radius * np.sin(angle)
        pos[node] = (x, y)
    
    # 클러스터 2 (10-19) 배치
    cluster2_nodes = list(range(10, 20))
    hub2 = 10
    pos[hub2] = cluster2_center
    
    # 나머지 노드들을 원형으로 배치
    other_nodes2 = [n for n in cluster2_nodes if n != hub2]
    for i, node in enumerate(other_nodes2):
        angle = 2 * np.pi * i / len(other_nodes2)
        radius = 1.2
        x = cluster2_center[0] + radius * np.cos(angle)
        y = cluster2_center[1] + radius * np.sin(angle)
        pos[node] = (x, y)
    
    return pos

def visualize_step(G, pos, title, filename, dataset_name,
                  core_nodes=None, periphery_nodes=None, 
                  pagerank_scores=None, labels=None,
                  metrics=None, step_description=""):
    """단계별 시각화"""
    plt.figure(figsize=(12, 8))
    
    # 노드 크기 계산
    if pagerank_scores:
        max_pr = max(pagerank_scores.values())
        min_pr = min(pagerank_scores.values())
        node_sizes = []
        for node in G.nodes():
            if max_pr != min_pr:
                normalized_pr = (pagerank_scores[node] - min_pr) / (max_pr - min_pr)
            else:
                normalized_pr = 0.5
            size = 100 + normalized_pr * 200  # 100-300 범위
            node_sizes.append(size)
    else:
        node_sizes = [150 for _ in G.nodes()]
    
    # 노드 색상 결정
    if labels:
        unique_labels = sorted(list(set(labels.values())))
        colors = ['red', 'blue', 'green', 'orange', 'purple']  # 명확한 색상
        label_to_color = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
        node_colors = [label_to_color[labels[node]] for node in G.nodes()]
    elif core_nodes and periphery_nodes:
        node_colors = []
        for node in G.nodes():
            if node in core_nodes:
                node_colors.append('red')
            else:
                node_colors.append('lightblue')
    else:
        node_colors = ['lightblue' for _ in G.nodes()]
    
    # 그래프 그리기
    nx.draw(G, pos, 
            node_color=node_colors,
            node_size=node_sizes,
            with_labels=True,  # 20개 노드라서 라벨 표시
            font_size=12,
            font_weight='bold',
            font_color='white',
            edge_color='gray',
            alpha=0.8,
            width=1.0)
    
    plt.title(f"{title}\n{dataset_name}", fontsize=16, fontweight='bold', pad=20)
    
    # 범례 추가
    if core_nodes and periphery_nodes and not labels:
        legend_elements = [
            Patch(facecolor='red', label=f'Core Nodes ({len(core_nodes)})'),
            Patch(facecolor='lightblue', label=f'Periphery Nodes ({len(periphery_nodes)})')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    elif labels:
        unique_labels = sorted(list(set(labels.values())))
        legend_elements = []
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, label in enumerate(unique_labels):
            color = colors[i % len(colors)]
            count = list(labels.values()).count(label)
            legend_elements.append(Patch(facecolor=color, label=f'Cluster {label} ({count} nodes)'))
        plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # 메트릭스 정보
    metrics_text = ""
    if metrics:
        metrics_text = f"Modularity: {metrics['modularity']:.3f} | Clusters: {metrics['num_clusters']}"
    
    # 설명 텍스트
    full_description = step_description
    if metrics_text:
        full_description += f"\n{metrics_text}"
    
    if full_description:
        plt.figtext(0.02, 0.02, full_description, fontsize=11, 
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

def run_leiden_lpa_demo():
    """실제 Leiden 알고리즘을 사용한 Leiden-LPA 하이브리드 데모"""
    print("=== Leiden-LPA Hybrid with Real Leiden Algorithm ===\n")
    
    # 라이브러리 확인
    print("Required libraries:")
    print(f"- networkx: {nx.__version__}")
    print(f"- igraph: {ig.__version__}")
    try:
        print(f"- leidenalg: {leidenalg.__version__}")
    except AttributeError:
        print("- leidenalg: installed (version info not available)")
    print()
    
    # 데이터셋 생성
    G, dataset_title = create_two_cluster_network()
    
    # 기본 통계
    print(f"Dataset: {dataset_title}")
    print(f"Nodes: {len(G.nodes())} (2 clusters of 10 nodes each)")
    print(f"Edges: {len(G.edges())}")
    print(f"Average degree: {np.mean([d for n, d in G.degree()]):.2f}")
    print(f"Density: {nx.density(G):.3f}")
    print()
    
    # 레이아웃 계산
    pos = get_two_cluster_layout(G)
    core_ratio = 0.5  # 50%
    
    # Step 0: 원본 네트워크
    visualize_step(G, pos, 
                  "Step 0: Original Network", 
                  "leiden_step0.png",
                  dataset_title,
                  step_description="Original two-cluster network\nCluster 1: nodes 0-9 (left) | Cluster 2: nodes 10-19 (right)")
    
    # Step 1: PageRank 및 핵심 노드 선별
    print(f"Step 1: PageRank analysis...")
    pagerank_scores, core_nodes, periphery_nodes = compute_pagerank_and_select_core(G, core_ratio)
    
    print(f"Core ratio: {core_ratio} ({len(core_nodes)} core nodes)")
    print(f"Core nodes: {sorted(core_nodes)}")
    print(f"Periphery nodes: {sorted(periphery_nodes)}")
    print(f"Top PageRank nodes: {sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:5]}")
    print()
    
    visualize_step(G, pos, 
                  f"Step 1: Core Node Selection (50%)", 
                  "leiden_step1.png",
                  dataset_title,
                  core_nodes=core_nodes, 
                  periphery_nodes=periphery_nodes,
                  pagerank_scores=pagerank_scores,
                  step_description=f"PageRank-based core selection (50%)\nRed: {len(core_nodes)} core nodes | Blue: {len(periphery_nodes)} periphery nodes\nNode size ∝ PageRank score")
    
    # Step 2: 실제 Leiden 알고리즘 적용
    print(f"Step 2: Applying real Leiden algorithm to core nodes...")
    start_time = time.time()
    core_labels = apply_leiden_algorithm(G, core_nodes, resolution=1.0, seed=42)
    core_time = time.time() - start_time
    
    core_metrics = compute_metrics(G.subgraph(core_nodes), core_labels)
    print(f"Core clustering time: {core_time:.4f}s")
    print(f"NetworkX modularity: {core_metrics['modularity']:.3f}")
    print()
    
    if core_nodes:
        core_subgraph = G.subgraph(core_nodes)
        core_pos = {node: pos[node] for node in core_nodes}
        visualize_step(core_subgraph, core_pos, 
                      "Step 2: Leiden Algorithm on Core Nodes", 
                      "leiden_step2.png",
                      f"{dataset_title} (Core Only - Real Leiden)",
                      labels=core_labels,
                      metrics=core_metrics,
                      step_description=f"Real Leiden algorithm applied to {len(core_nodes)} core nodes\nOptimal modularity-based clustering")
    
    # Step 3: 레이블 전파
    print(f"Step 3: Label propagation to periphery nodes...")
    start_time = time.time()
    final_labels = propagate_labels(G, core_labels, periphery_nodes)
    total_time = core_time + (time.time() - start_time)
    
    final_metrics = compute_metrics(G, final_labels)
    print(f"Total hybrid time: {total_time:.4f}s")
    print()
    
    visualize_step(G, pos, 
                  "Step 3: Final Hybrid Result", 
                  "leiden_step3.png",
                  dataset_title,
                  labels=final_labels,
                  metrics=final_metrics,
                  step_description=f"Leiden-LPA Hybrid final result\n{final_metrics['num_clusters']} clusters found\nColors show final cluster assignments")
    
    # Step 4: 비교 - 전체 그래프에 Leiden 적용
    print(f"Step 4: Full Leiden algorithm for comparison...")
    start_time = time.time()
    full_leiden_labels = apply_leiden_algorithm(G, list(G.nodes()), resolution=1.0, seed=42)
    full_leiden_time = time.time() - start_time
    
    full_leiden_metrics = compute_metrics(G, full_leiden_labels)
    print(f"Full Leiden time: {full_leiden_time:.4f}s")
    print()
    
    visualize_step(G, pos, 
                  "Comparison: Full Leiden Result", 
                  "leiden_comparison_full.png",
                  dataset_title,
                  labels=full_leiden_labels,
                  metrics=full_leiden_metrics,
                  step_description=f"Full Leiden algorithm on entire network\n{full_leiden_metrics['num_clusters']} clusters found\nOptimal but computationally expensive")
    
    # Step 5: 비교 - 순수 LPA
    print(f"Step 5: Pure LPA for comparison...")
    start_time = time.time()
    pure_lpa_labels = propagate_labels(G, {}, list(G.nodes()), max_iterations=20)
    lpa_time = time.time() - start_time
    
    lpa_metrics = compute_metrics(G, pure_lpa_labels)
    print(f"Pure LPA time: {lpa_time:.4f}s")
    print()
    
    visualize_step(G, pos, 
                  "Comparison: Pure LPA Result", 
                  "leiden_comparison_lpa.png",
                  dataset_title,
                  labels=pure_lpa_labels,
                  metrics=lpa_metrics,
                  step_description=f"Pure Label Propagation result\n{lpa_metrics['num_clusters']} clusters found\nFast but lower quality")
    
    # 결과 요약
    print("="*60)
    print("COMPREHENSIVE RESULTS COMPARISON")
    print("="*60)
    print(f"Dataset: {dataset_title}")
    print(f"Network: {len(G.nodes())} nodes, {len(G.edges())} edges")
    print(f"Ground truth: 2 clusters (nodes 0-9, 10-19)")
    print(f"Core ratio: {core_ratio} ({len(core_nodes)} core nodes)")
    print()
    
    print("CLUSTERING QUALITY COMPARISON:")
    print(f"  1. Full Leiden:     {full_leiden_metrics['modularity']:.3f} modularity, {full_leiden_metrics['num_clusters']} clusters {full_leiden_metrics['cluster_sizes']}")
    print(f"  2. Hybrid (Ours):   {final_metrics['modularity']:.3f} modularity, {final_metrics['num_clusters']} clusters {final_metrics['cluster_sizes']}")
    print(f"  3. Pure LPA:        {lpa_metrics['modularity']:.3f} modularity, {lpa_metrics['num_clusters']} clusters {lpa_metrics['cluster_sizes']}")
    print()
    
    print("COMPUTATION TIME COMPARISON:")
    print(f"  1. Full Leiden:     {full_leiden_time:.4f}s")
    print(f"  2. Hybrid (Ours):   {total_time:.4f}s ({core_time:.4f}s Leiden + {total_time-core_time:.4f}s LPA)")
    print(f"  3. Pure LPA:        {lpa_time:.4f}s")
    print()
    
    print("EFFICIENCY ANALYSIS:")
    if full_leiden_time > 0:
        speedup_vs_full = full_leiden_time / total_time
        print(f"  - Hybrid vs Full Leiden: {speedup_vs_full:.2f}x speedup")
    if lpa_metrics['modularity'] > 0:
        quality_vs_lpa = final_metrics['modularity'] / lpa_metrics['modularity']
        print(f"  - Hybrid vs Pure LPA: {quality_vs_lpa:.2f}x quality improvement")
    if full_leiden_metrics['modularity'] > 0:
        quality_retention = final_metrics['modularity'] / full_leiden_metrics['modularity']
        print(f"  - Quality retention: {quality_retention:.1%} of full Leiden quality")
    
    print(f"\nGENERATED FILES:")
    print("- leiden_step0.png (original network)")
    print("- leiden_step1.png (core selection)")  
    print("- leiden_step2.png (Leiden on core)")
    print("- leiden_step3.png (hybrid final result)")
    print("- leiden_comparison_full.png (full Leiden)")
    print("- leiden_comparison_lpa.png (pure LPA)")

if __name__ == "__main__":
    # 필요한 라이브러리 설치 안내
    print("Required installation:")
    print("pip install igraph leidenalg networkx matplotlib numpy")
    print()
    
    try:
        run_leiden_lpa_demo()
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nPlease install required packages:")
        print("pip install igraph leidenalg")
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your installation and try again.")