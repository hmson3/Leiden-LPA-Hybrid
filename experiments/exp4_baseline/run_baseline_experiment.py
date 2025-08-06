#!/usr/bin/env python3
"""
실험 4: 기존 커뮤니티 탐지 방법과의 성능 비교
Core 4개 알고리즘: Pure_LPA, Pure_Leiden, Optimal_Hybrid, Louvain
"""

import os
import sys
import pandas as pd
import networkx as nx
import igraph as ig
import numpy as np
from pathlib import Path
import time
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 설정
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from improved_leiden_lpa import LeidenLPAHybrid
from evaluation_system import CommunityEvaluator, save_evaluation_results

# Louvain 알고리즘을 위한 라이브러리
try:
    import community as community_louvain  # python-louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    print("⚠️ python-louvain 라이브러리가 설치되지 않았습니다.")
    print("설치: pip install python-louvain")
    LOUVAIN_AVAILABLE = False

class BaselineExperiment:
    """기존 방법과의 성능 비교 실험 클래스"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / 'data' / 'data' / 'processed'
        self.results_dir = self.project_root / 'results'
        self.exp_dir = self.project_root / 'experiments' / 'exp4_baseline'
        
        # 실험 설정
        self.datasets = ['karate', 'cora', 'citeseer', 'pubmed', 'dolphin', 'football', 'polblog', 'mexican',
                        'com-amazon', 'com-youtube', 'com-dblp']
        self.repeat_count = 5
        
        # Core 4개 알고리즘만 사용
        self.algorithms = ['Pure_LPA', 'Pure_Leiden', 'Optimal_Hybrid', 'Louvain']
        
        self._create_directories()
    
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'raw').mkdir(exist_ok=True)
        (self.results_dir / 'figures').mkdir(exist_ok=True)
    
    def load_dataset(self, dataset_name: str) -> Tuple[nx.Graph, Dict[str, int], str]:
        """데이터셋 로드 (기존 실험과 동일)"""
        dataset_path = self.data_dir / dataset_name
        
        # 그래프 로드
        edge_file = dataset_path / 'graph.edgelist'
        if not edge_file.exists():
            raise FileNotFoundError(f"Dataset {dataset_name} not found")
        
        G = nx.read_edgelist(str(edge_file))
        
        # 레이블 로드
        label_file = dataset_path / 'labels.txt'
        labels = {}
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    node, label = parts[0], int(parts[1])
                    labels[node] = label
        
        # 노드 매칭 및 정리
        graph_nodes = set(G.nodes())
        label_nodes = set(labels.keys())
        common_nodes = graph_nodes & label_nodes
        
        if len(common_nodes) == 0:
            # 정수 변환 시도
            try:
                int_mapping = {node: int(node) for node in G.nodes()}
                G = nx.relabel_nodes(G, int_mapping)
                
                int_labels = {}
                for node, label in labels.items():
                    try:
                        int_node = int(node)
                        int_labels[int_node] = label
                    except:
                        continue
                
                labels = int_labels
                common_nodes = set(G.nodes()) & set(labels.keys())
                
            except Exception as e:
                raise ValueError(f"노드 ID 매칭 실패: {e}")
        
        # 공통 노드만 유지
        if len(common_nodes) > 0:
            G = G.subgraph(common_nodes).copy()
            labels = {node: labels[node] for node in common_nodes}
        else:
            raise ValueError("공통 노드가 없습니다!")
        
        # 크기 카테고리 결정
        n_nodes = G.number_of_nodes()
        if n_nodes < 100:
            size_category = 'small'
        elif n_nodes < 5000:
            size_category = 'medium'
        else:
            size_category = 'large'
        
        return G, labels, size_category
    
    def _run_pure_lpa(self, G: nx.Graph, seed: int) -> Dict[str, int]:
        """Pure LPA 실행"""
        alg = LeidenLPAHybrid(core_ratio=0.0, seed=seed)
        return alg.fit_predict(G)
    
    def _run_pure_leiden(self, G: nx.Graph, seed: int) -> Dict[str, int]:
        """Pure Leiden 실행"""
        alg = LeidenLPAHybrid(core_ratio=1.0, seed=seed)
        return alg.fit_predict(G)
    
    def _run_optimal_hybrid(self, G: nx.Graph, seed: int) -> Dict[str, int]:
        """Optimal Hybrid 실행 (실험 1,2,3 결과 기반)"""
        alg = LeidenLPAHybrid(
            core_ratio=0.4,  # 실험 1 결과
            centrality_method='pagerank',  # 실험 2 결과
            anchor_strategy='dynamic_iterative',  # 실험 3 결과 (Fixed_Iterative)
            seed=seed
        )
        return alg.fit_predict(G)
    
    def _run_louvain(self, G: nx.Graph, seed: int) -> Dict[str, int]:
        """Louvain 알고리즘 실행"""
        if not LOUVAIN_AVAILABLE:
            raise ImportError("python-louvain 라이브러리가 필요합니다")
        
        # NetworkX 그래프에서 연속된 정수 노드로 변환
        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        reverse_mapping = {i: node for node, i in node_mapping.items()}
        
        G_mapped = nx.relabel_nodes(G, node_mapping)
        
        # Louvain 실행 (seed 설정)
        np.random.seed(seed)
        partition = community_louvain.best_partition(G_mapped, random_state=seed)
        
        # 원래 노드 ID로 복원
        result = {reverse_mapping[mapped_node]: community_id 
                 for mapped_node, community_id in partition.items()}
        
        return result
    
    def _get_algorithm_function(self, algorithm_name: str):
        """알고리즘 이름으로 함수 반환"""
        algorithm_map = {
            'Pure_LPA': self._run_pure_lpa,
            'Pure_Leiden': self._run_pure_leiden,
            'Optimal_Hybrid': self._run_optimal_hybrid,
            'Louvain': self._run_louvain
        }
        
        if algorithm_name not in algorithm_map:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        return algorithm_map[algorithm_name]
    
    def run_single_experiment(self, dataset_name: str, algorithm_name: str,
                            G: nx.Graph, true_labels: Dict[str, int], 
                            run_id: int) -> Dict[str, Any]:
        """단일 실험 실행"""
        try:
            # 알고리즘 실행
            start_time = time.time()
            algorithm_func = self._get_algorithm_function(algorithm_name)
            pred_labels = algorithm_func(G, seed=42 + run_id)
            runtime = time.time() - start_time
            
            # 결과 검증
            if not pred_labels or len(pred_labels) == 0:
                print(f"      ⚠️ 빈 결과: {algorithm_name}")
                return None
            
            num_communities = len(set(pred_labels.values()))
            if num_communities <= 1:
                print(f"      ⚠️ 커뮤니티 부족: {algorithm_name}")
                return None
            
            # 평가
            evaluator = CommunityEvaluator(G, true_labels)
            evaluation = evaluator.evaluate_clustering(pred_labels, runtime)
            
            # NaN 처리
            def safe_get(d, key, default=-1.0):
                value = d.get(key, default)
                return default if np.isnan(value) else value
            
            # 결과 정리
            result = {
                'dataset': dataset_name,
                'algorithm': algorithm_name,
                'run_id': run_id,
                'runtime': runtime,
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'num_communities': evaluation['clustering_info']['num_communities'],
                'modularity': safe_get(evaluation['structural_quality'], 'modularity'),
                'conductance': safe_get(evaluation['structural_quality'], 'conductance'),
                'coverage': safe_get(evaluation['structural_quality'], 'coverage'),
                'nmi': safe_get(evaluation['ground_truth_metrics'], 'nmi'),
                'ami': safe_get(evaluation['ground_truth_metrics'], 'ami'),
                'ari': safe_get(evaluation['ground_truth_metrics'], 'ari'),
                'accuracy': safe_get(evaluation['ground_truth_metrics'], 'accuracy'),
                'f1_score': safe_get(evaluation['ground_truth_metrics'], 'f1_score')
            }
            
            # 하이브리드 알고리즘인 경우 추가 통계
            if algorithm_name in ['Pure_LPA', 'Pure_Leiden', 'Optimal_Hybrid']:
                if hasattr(self, '_last_algorithm_stats'):
                    stats = getattr(self, '_last_algorithm_stats', {})
                    result.update({
                        'core_nodes_count': stats.get('core_nodes_count', 0),
                        'periphery_nodes_count': stats.get('periphery_nodes_count', 0),
                        'leiden_time': stats.get('leiden_time', 0),
                        'lpa_time': stats.get('lpa_time', 0),
                        'centrality_time': stats.get('centrality_time', 0)
                    })
            
            return result
            
        except Exception as e:
            print(f"   ❌ 오류: {algorithm_name} - {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_dataset_experiment(self, dataset_name: str) -> List[Dict[str, Any]]:
        """단일 데이터셋에 대한 모든 알고리즘 실험"""
        print(f"\n🔬 {dataset_name.upper()} 데이터셋 실험 시작")
        print("=" * 50)
        
        # 데이터 로드
        try:
            G, true_labels, size_category = self.load_dataset(dataset_name)
            print(f"   📊 로드됨: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
            print(f"   📏 크기: {size_category}")
            print(f"   🏷️  클래스: {len(set(true_labels.values()))}개")
        except Exception as e:
            print(f"   ❌ 데이터 로드 실패: {e}")
            return []
        
        print(f"   🎯 테스트할 알고리즘: {len(self.algorithms)}개")
        
        results = []
        
        # 각 알고리즘별로 실험
        for algorithm_name in self.algorithms:
            print(f"\n   🔧 {algorithm_name} 실험 중...")
            
            # Louvain 라이브러리 체크
            if algorithm_name == 'Louvain' and not LOUVAIN_AVAILABLE:
                print(f"      ⚠️ Louvain 라이브러리 없음, 건너뜀")
                continue
            
            algorithm_results = []
            
            # 반복 실험
            for run_id in range(self.repeat_count):
                result = self.run_single_experiment(
                    dataset_name, algorithm_name, G, true_labels, run_id
                )
                if result:
                    algorithm_results.append(result)
                    results.append(result)
            
            # 해당 알고리즘 결과 요약
            if algorithm_results:
                avg_ami = np.mean([r['ami'] for r in algorithm_results if r['ami'] != -1])
                avg_modularity = np.mean([r['modularity'] for r in algorithm_results if r['modularity'] != -1])
                avg_runtime = np.mean([r['runtime'] for r in algorithm_results])
                avg_communities = np.mean([r['num_communities'] for r in algorithm_results])
                
                print(f"      평균 AMI: {avg_ami:.3f}, 모듈러리티: {avg_modularity:.3f}")
                print(f"      실행시간: {avg_runtime:.3f}초, 커뮤니티: {avg_communities:.1f}개")
        
        print(f"\n   ✅ {dataset_name} 완료: {len(results)}개 결과")
        return results
    
    def run_all_experiments(self) -> pd.DataFrame:
        """모든 데이터셋에 대한 실험 실행"""
        print("🚀 기존 방법 비교 실험 시작")
        print("=" * 60)
        
        all_results = []
        
        # 각 데이터셋별 실험
        for dataset_name in self.datasets:
            dataset_results = self.run_dataset_experiment(dataset_name)
            all_results.extend(dataset_results)
        
        # DataFrame으로 변환
        if all_results:
            results_df = pd.DataFrame(all_results)
            
            # 실험 메타데이터
            metadata = {
                'experiment': 'baseline_comparison',
                'algorithms': self.algorithms,
                'repeat_count': self.repeat_count,
                'datasets': self.datasets,
                'timestamp': pd.Timestamp.now().isoformat(),
                'optimal_hybrid_config': {
                    'core_ratio': 0.5,
                    'centrality_method': 'pagerank',
                    'anchor_strategy': 'dynamic_iterative'
                }
            }
            
            # 결과 저장
            output_file = self.results_dir / 'raw' / 'exp4_baseline_results.csv'
            save_evaluation_results(results_df, str(output_file), metadata)
            
            # 요약 분석 저장
            self._save_analysis(results_df)
            
            return results_df
        else:
            print("❌ 실험 결과가 없습니다.")
            return pd.DataFrame()
    
    def _save_analysis(self, results_df: pd.DataFrame):
        """실험 결과 분석 및 저장"""
        print(f"\n📊 결과 분석 생성 중...")
        
        # 알고리즘별 평균 성능
        avg_performance = results_df.groupby('algorithm').agg({
            'ami': 'mean',
            'modularity': 'mean', 
            'runtime': 'mean',
            'accuracy': 'mean',
            'num_communities': 'mean'
        }).round(4)
        
        # 데이터셋별 성능
        dataset_performance = results_df.groupby(['dataset', 'algorithm']).agg({
            'ami': 'mean',
            'modularity': 'mean',
            'runtime': 'mean'
        }).round(4)
        
        # 상대적 성능 (Pure_Leiden 대비)
        relative_performance = {}
        for dataset in self.datasets:
            if dataset in results_df['dataset'].values:
                dataset_data = results_df[results_df['dataset'] == dataset]
                
                leiden_ami = dataset_data[dataset_data['algorithm'] == 'Pure_Leiden']['ami'].mean()
                leiden_runtime = dataset_data[dataset_data['algorithm'] == 'Pure_Leiden']['runtime'].mean()
                
                for algorithm in self.algorithms:
                    if algorithm in dataset_data['algorithm'].values:
                        alg_data = dataset_data[dataset_data['algorithm'] == algorithm]
                        alg_ami = alg_data['ami'].mean()
                        alg_runtime = alg_data['runtime'].mean()
                        
                        key = f"{dataset}_{algorithm}"
                        relative_performance[key] = {
                            'ami_ratio': alg_ami / leiden_ami if leiden_ami > 0 else 0,
                            'speedup': leiden_runtime / alg_runtime if alg_runtime > 0 else 0
                        }
        
        # 분석 리포트 저장
        analysis_file = self.exp_dir / 'baseline_analysis.txt'
        with open(analysis_file, 'w') as f:
            f.write("기존 방법 비교 실험 분석\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. 알고리즘별 평균 성능:\n")
            f.write("-" * 30 + "\n")
            f.write(str(avg_performance))
            f.write("\n\n")
            
            f.write("2. 데이터셋별 성능:\n")
            f.write("-" * 30 + "\n")
            f.write(str(dataset_performance))
            f.write("\n\n")
            
            f.write("3. Optimal_Hybrid vs Pure_Leiden 비교:\n")
            f.write("-" * 30 + "\n")
            for dataset in self.datasets:
                if f"{dataset}_Optimal_Hybrid" in relative_performance:
                    perf = relative_performance[f"{dataset}_Optimal_Hybrid"]
                    f.write(f"{dataset}: AMI 유지율 {perf['ami_ratio']:.3f}, "
                           f"속도 향상 {perf['speedup']:.1f}x\n")
            
            f.write("\n4. 주요 발견사항:\n")
            f.write("-" * 30 + "\n")
            f.write(self._generate_insights(results_df, relative_performance))
        
        print(f"   💾 분석 저장: {analysis_file}")
    
    def _generate_insights(self, results_df: pd.DataFrame, 
                          relative_performance: Dict) -> str:
        """주요 발견사항 생성"""
        insights = []
        
        # Optimal_Hybrid의 성능
        hybrid_data = results_df[results_df['algorithm'] == 'Optimal_Hybrid']
        if len(hybrid_data) > 0:
            avg_ami_retention = np.mean([
                relative_performance[f"{dataset}_Optimal_Hybrid"]['ami_ratio']
                for dataset in self.datasets
                if f"{dataset}_Optimal_Hybrid" in relative_performance
            ])
            
            avg_speedup = np.mean([
                relative_performance[f"{dataset}_Optimal_Hybrid"]['speedup']
                for dataset in self.datasets
                if f"{dataset}_Optimal_Hybrid" in relative_performance
            ])
            
            insights.append(f"- Optimal_Hybrid: 평균 AMI {avg_ami_retention:.1%} 유지, {avg_speedup:.1f}x 속도 향상")
        
        # 알고리즘 순위
        avg_perf = results_df.groupby('algorithm')[['ami', 'runtime']].mean()
        ami_ranking = avg_perf.sort_values('ami', ascending=False).index.tolist()
        speed_ranking = avg_perf.sort_values('runtime').index.tolist()
        
        insights.append(f"- AMI 순위: {' > '.join(ami_ranking)}")
        insights.append(f"- 속도 순위: {' > '.join(speed_ranking)}")
        
        # 목표 달성 여부
        if avg_ami_retention > 0.9 and avg_speedup > 1.5:
            insights.append("- 🎯 목표 달성: 품질 유지(90%+) + 속도 향상(1.5x+)")
        else:
            insights.append("- ⚠️ 목표 미달성: 파라미터 조정 필요")
        
        return "\n".join(insights)

def main():
    """메인 실행 함수"""
    experiment = BaselineExperiment()
    
    print("🎯 실험 설정:")
    print(f"   데이터셋: {experiment.datasets}")
    print(f"   알고리즘: {experiment.algorithms}")
    print(f"   반복 횟수: {experiment.repeat_count}")
    
    # 라이브러리 체크
    if not LOUVAIN_AVAILABLE:
        print(f"   ⚠️ Louvain 제외하고 진행")
        experiment.algorithms = [alg for alg in experiment.algorithms if alg != 'Louvain']
    
    # 실험 실행
    results_df = experiment.run_all_experiments()
    
    if not results_df.empty:
        print(f"\n🎉 실험 완료!")
        print(f"   총 결과: {len(results_df)}개")
        print(f"   데이터셋: {results_df['dataset'].nunique()}개")
        print(f"   알고리즘: {results_df['algorithm'].nunique()}개")
        
        # 간단한 결과 미리보기
        print(f"\n📈 알고리즘별 평균 성능:")
        preview = results_df.groupby('algorithm')[['ami', 'modularity', 'runtime']].mean().round(3)
        print(preview)
        
        print(f"\n📁 결과 파일:")
        print(f"   상세: results/raw/exp4_baseline_results.csv")
        print(f"   분석: experiments/exp4_baseline/baseline_analysis.txt")
        
        return True
    else:
        print("❌ 실험 실패")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎊 실험4 완료!")
    else:
        print(f"\n🔧 실험을 다시 확인해보세요.")