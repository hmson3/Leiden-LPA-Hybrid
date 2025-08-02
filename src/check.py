# 테스트해보기
import networkx as nx
from evaluation_system import CommunityEvaluator, compare_algorithms, quick_evaluate
from improved_leiden_lpa import LeidenLPAHybrid

# Karate club 테스트
G = nx.karate_club_graph()

# Ground truth (Zachary's original split)
true_labels = {}
for node in G.nodes():
    true_labels[node] = 0 if G.nodes[node]['club'] == 'Mr. Hi' else 1

# 1. 빠른 평가 테스트
alg = LeidenLPAHybrid(core_ratio=0.4, centrality_method='pagerank')
pred_labels = alg.fit_predict(G)
quick_results = quick_evaluate(G, pred_labels, true_labels)
print("Quick evaluation:", quick_results)

# 2. 상세 평가 테스트
evaluator = CommunityEvaluator(G, true_labels)
detailed_results = evaluator.evaluate_clustering(pred_labels, runtime=0.1)
print("Detailed results keys:", list(detailed_results.keys()))

# 3. 알고리즘 비교 테스트
algorithms = {
    'Pure_LPA': lambda g, s: LeidenLPAHybrid(core_ratio=0.0, seed=s).fit_predict(g),
    'Pure_Leiden': lambda g, s: LeidenLPAHybrid(core_ratio=1.0, seed=s).fit_predict(g),
    'Hybrid_PageRank': lambda g, s: LeidenLPAHybrid(core_ratio=0.4, centrality_method='pagerank', seed=s).fit_predict(g),
    'Hybrid_Degree': lambda g, s: LeidenLPAHybrid(core_ratio=0.4, centrality_method='degree', seed=s).fit_predict(g)
}

comparison_df = compare_algorithms(G, algorithms, true_labels, repeat=3)
print(comparison_df.head())