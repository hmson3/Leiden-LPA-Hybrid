#!/usr/bin/env python3
"""
실험 2: 중심성 지표 비교
6가지 중심성 지표의 효과를 4개 레이블 데이터셋에서 비교
"""

import os
import sys
import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
import time
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from improved_leiden_lpa import LeidenLPAHybrid
from evaluation_system import CommunityEvaluator, compare_algorithms, save_evaluation_results

class CentralityExperiment:
    """중심성 지표 비교 실험 클래스"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent  # experiments/exp2_centrality에서 3단계 위로
        self.data_dir = self.project_root / 'data' / 'data' / 'processed'
        self.results_dir = self.project_root / 'results'
        self.exp_dir = self.project_root / 'experiments' / 'exp2_centrality'
        
        # 실험 설정
        self.datasets = ['karate', 'cora', 'citeseer', 'pubmed']
        self.centrality_methods = {
            'pagerank': 'PageRank (기본)',
            'degree': 'Degree Centrality', 
            'eigenvector': 'Eigenvector Centrality',
            'betweenness': 'Betweenness Centrality',
            'closeness': 'Closeness Centrality'
        }
        
        # 크기별 사용할 중심성 지표 (계산 시간 고려)
        self.size_based_methods = {
            'tiny': ['pagerank', 'degree', 'eigenvector', 'betweenness', 'closeness'],
            'small': ['pagerank', 'degree', 'eigenvector', 'betweenness', 'closeness'],
            'medium': ['pagerank', 'degree', 'eigenvector', 'betweenness', 'closeness'], 
            'large': ['pagerank', 'degree', 'eigenvector', 'betweenness', 'closeness']
        }
        
        self.core_ratio = 0.4  # 고정값 사용
        self.repeat_count = 5  # 반복 횟수
        
        self._create_directories()
    
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'raw').mkdir(exist_ok=True)
        (self.results_dir / 'figures').mkdir(exist_ok=True)
    
    def load_dataset(self, dataset_name: str) -> Tuple[nx.Graph, Dict[str, int], str]:
        """데이터셋 로드 - 간단한 방법으로 수정"""
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
        
        # 공통 노드 확인
        graph_nodes = set(G.nodes())
        label_nodes = set(labels.keys())
        common_nodes = graph_nodes & label_nodes
        
        print(f"   🔍 그래프 노드: {len(graph_nodes)}, 레이블 노드: {len(label_nodes)}, 공통: {len(common_nodes)}")
        
        if len(common_nodes) == 0:
            print(f"   ⚠️ 노드 ID 타입 불일치 감지, 정수로 변환 시도...")
            
            # 그래프 노드를 정수로 변환 시도
            try:
                int_mapping = {node: int(node) for node in G.nodes()}
                G = nx.relabel_nodes(G, int_mapping)
                
                # 레이블도 정수 키로 변환
                int_labels = {}
                for node, label in labels.items():
                    try:
                        int_node = int(node)
                        int_labels[int_node] = label
                    except:
                        continue
                
                labels = int_labels
                common_nodes = set(G.nodes()) & set(labels.keys())
                print(f"   ✅ 정수 변환 후 공통 노드: {len(common_nodes)}")
                
            except Exception as e:
                print(f"   ❌ 정수 변환 실패: {e}")
                raise ValueError(f"노드 ID 매칭 실패: 그래프={list(G.nodes())[:3]}, 레이블={list(labels.keys())[:3]}")
        
        # 공통 노드만 유지
        if len(common_nodes) > 0:
            G = G.subgraph(common_nodes).copy()
            labels = {node: labels[node] for node in common_nodes}
        else:
            raise ValueError("공통 노드가 없습니다!")
        
        # 크기 카테고리 결정
        n_nodes = G.number_of_nodes()
        if n_nodes < 100:
            size_category = 'tiny'
        elif n_nodes < 1000:
            size_category = 'small'
        elif n_nodes < 10000:
            size_category = 'medium'
        else:
            size_category = 'large'
        
        return G, labels, size_category
    
    def run_single_experiment(self, dataset_name: str, centrality_method: str, 
                            G: nx.Graph, true_labels: Dict[str, int], 
                            run_id: int) -> Dict[str, Any]:
        """단일 실험 실행"""
        try:
            # 알고리즘 실행
            start_time = time.time()
            alg = LeidenLPAHybrid(
                core_ratio=self.core_ratio,
                centrality_method=centrality_method,
                seed=42 + run_id  # 재현 가능한 시드
            )
            pred_labels = alg.fit_predict(G)
            runtime = time.time() - start_time
            
            # 디버깅: 결과 확인
            if not pred_labels or len(pred_labels) == 0:
                print(f"      ⚠️ 빈 결과: {centrality_method}")
                return None
            
            num_communities = len(set(pred_labels.values()))
            if num_communities <= 1:
                print(f"      ⚠️ 커뮤니티 부족 ({num_communities}개): {centrality_method}")
                return None
            
            # 과도한 세분화 체크
            if num_communities > len(pred_labels) / 10:  # 노드 10개당 1개 커뮤니티 이상이면 과도함
                print(f"      ⚠️ 과도한 세분화 ({num_communities}개 커뮤니티): {centrality_method}")
                return None
            
            # 평가
            evaluator = CommunityEvaluator(G, true_labels)
            evaluation = evaluator.evaluate_clustering(pred_labels, runtime)
            
            # NaN 체크 - 하지만 계속 진행
            modularity = evaluation['structural_quality']['modularity']
            nmi = evaluation['ground_truth_metrics']['nmi']
            ami = evaluation['ground_truth_metrics'].get('ami', np.nan)  # AMI 안전하게 가져오기
            
            if np.isnan(modularity) or np.isnan(nmi):
                print(f"      ⚠️ NaN 결과 (하지만 저장): {centrality_method}, mod={modularity}, nmi={nmi}")
                print(f"         커뮤니티 수: {num_communities}, 노드 수: {len(pred_labels)}")
                # NaN 값을 -1로 대체해서 저장
                if np.isnan(modularity):
                    evaluation['structural_quality']['modularity'] = -1.0
                if np.isnan(nmi):
                    evaluation['ground_truth_metrics']['nmi'] = -1.0
                if np.isnan(evaluation['ground_truth_metrics']['ari']):
                    evaluation['ground_truth_metrics']['ari'] = -1.0
                if np.isnan(evaluation['ground_truth_metrics']['accuracy']):
                    evaluation['ground_truth_metrics']['accuracy'] = -1.0
                if np.isnan(ami):
                    evaluation['ground_truth_metrics']['ami'] = -1.0
            
            # 결과 정리
            result = {
                'dataset': dataset_name,
                'centrality_method': centrality_method,
                'run_id': run_id,
                'runtime': runtime,
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'num_communities': evaluation['clustering_info']['num_communities'],
                'modularity': evaluation['structural_quality']['modularity'],
                'conductance': evaluation['structural_quality']['conductance'],
                'coverage': evaluation['structural_quality']['coverage'],
                'nmi': evaluation['ground_truth_metrics']['nmi'],
                'ami': evaluation['ground_truth_metrics'].get('ami', -1.0),  # AMI 안전하게 추가
                'ari': evaluation['ground_truth_metrics']['ari'],
                'accuracy': evaluation['ground_truth_metrics']['accuracy'],
                'core_nodes_count': alg.get_stats().get('core_nodes_count', 0),
                'periphery_nodes_count': alg.get_stats().get('periphery_nodes_count', 0),
                'leiden_time': alg.get_stats().get('leiden_time', 0),
                'lpa_time': alg.get_stats().get('lpa_time', 0),
                'centrality_time': alg.get_stats().get('centrality_time', 0)
            }
            
            return result
            
        except Exception as e:
            print(f"   ❌ 오류: {centrality_method} - {e}")
            import traceback
            traceback.print_exc()  # 상세한 오류 정보
            return None
    
    def run_dataset_experiment(self, dataset_name: str) -> List[Dict[str, Any]]:
        """단일 데이터셋에 대한 모든 중심성 지표 실험"""
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
        
        # 해당 크기에서 사용할 중심성 지표
        available_methods = self.size_based_methods[size_category]
        print(f"   🎯 사용할 중심성 지표 ({len(available_methods)}개): {', '.join(available_methods)}")
        
        results = []
        
        # 각 중심성 지표별로 실험
        for method in available_methods:
            print(f"\n   🔧 {method} 실험 중...")
            method_results = []
            
            # 반복 실험
            for run_id in range(self.repeat_count):
                result = self.run_single_experiment(
                    dataset_name, method, G, true_labels, run_id
                )
                if result:
                    method_results.append(result)
                    results.append(result)
            
            # 해당 중심성 지표 결과 요약
            if method_results:
                avg_nmi = np.mean([r['nmi'] for r in method_results])
                avg_modularity = np.mean([r['modularity'] for r in method_results])
                avg_runtime = np.mean([r['runtime'] for r in method_results])
                
                print(f"      평균 NMI: {avg_nmi:.3f}, 모듈러리티: {avg_modularity:.3f}, "
                      f"실행시간: {avg_runtime:.3f}초")
        
        print(f"\n   ✅ {dataset_name} 완료: {len(results)}개 결과")
        return results
    
    def run_all_experiments(self) -> pd.DataFrame:
        """모든 데이터셋에 대한 실험 실행"""
        print("🚀 중심성 지표 비교 실험 시작")
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
                'experiment': 'centrality_comparison',
                'core_ratio': self.core_ratio,
                'repeat_count': self.repeat_count,
                'datasets': self.datasets,
                'centrality_methods': list(self.centrality_methods.keys()),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # 결과 저장
            output_file = self.results_dir / 'raw' / 'exp2_centrality_results.csv'
            save_evaluation_results(results_df, str(output_file), metadata)
            
            # 요약 저장
            self._save_summary(results_df)
            
            return results_df
        else:
            print("❌ 실험 결과가 없습니다.")
            return pd.DataFrame()
    
    def _save_summary(self, results_df: pd.DataFrame):
        """실험 결과 요약 저장"""
        print(f"\n📊 실험 결과 요약 생성 중...")
        
        # 데이터셋별 평균 결과
        summary_by_dataset = results_df.groupby(['dataset', 'centrality_method']).agg({
            'nmi': ['mean', 'std'],
            'modularity': ['mean', 'std'], 
            'runtime': ['mean', 'std'],
            'accuracy': ['mean', 'std']
        }).round(4)
        
        # 중심성 지표별 평균 결과
        summary_by_method = results_df.groupby('centrality_method').agg({
            'nmi': ['mean', 'std'],
            'modularity': ['mean', 'std'],
            'runtime': ['mean', 'std'],
            'accuracy': ['mean', 'std']
        }).round(4)
        
        # 요약 파일 저장
        summary_file = self.exp_dir / 'summary_results.txt'
        with open(summary_file, 'w') as f:
            f.write("중심성 지표 비교 실험 결과 요약\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. 데이터셋별 결과:\n")
            f.write(str(summary_by_dataset))
            f.write("\n\n")
            
            f.write("2. 중심성 지표별 전체 평균:\n")
            f.write(str(summary_by_method))
            f.write("\n\n")
            
            f.write("3. 주요 발견사항:\n")
            f.write(self._generate_insights(results_df))
        
        print(f"   💾 요약 저장: {summary_file}")
    
    def _generate_insights(self, results_df: pd.DataFrame) -> str:
        """주요 발견사항 생성"""
        insights = []
        
        # 최고 성능 중심성 지표
        avg_nmi = results_df.groupby('centrality_method')['nmi'].mean()
        best_centrality = avg_nmi.idxmax()
        best_nmi = avg_nmi.max()
        
        insights.append(f"- 전체 평균 NMI가 가장 높은 중심성 지표: {best_centrality} ({best_nmi:.3f})")
        
        # 가장 빠른 중심성 지표
        avg_runtime = results_df.groupby('centrality_method')['runtime'].mean()
        fastest_centrality = avg_runtime.idxmin()
        fastest_time = avg_runtime.min()
        
        insights.append(f"- 가장 빠른 중심성 지표: {fastest_centrality} ({fastest_time:.3f}초)")
        
        # 데이터셋별 최적 중심성
        for dataset in self.datasets:
            if dataset in results_df['dataset'].values:
                dataset_data = results_df[results_df['dataset'] == dataset]
                dataset_best = dataset_data.groupby('centrality_method')['nmi'].mean().idxmax()
                insights.append(f"- {dataset}에서 최고 성능: {dataset_best}")
        
        return "\n".join(insights)

def main():
    """메인 실행 함수"""
    experiment = CentralityExperiment()
    
    print("🎯 실험 설정:")
    print(f"   데이터셋: {experiment.datasets}")
    print(f"   중심성 지표: {list(experiment.centrality_methods.keys())}")
    print(f"   Core ratio: {experiment.core_ratio}")
    print(f"   반복 횟수: {experiment.repeat_count}")
    
    # 실험 실행
    results_df = experiment.run_all_experiments()
    
    if not results_df.empty:
        print(f"\n🎉 실험 완료!")
        print(f"   총 결과: {len(results_df)}개")
        print(f"   데이터셋: {results_df['dataset'].nunique()}개")
        print(f"   중심성 지표: {results_df['centrality_method'].nunique()}개")
        
        # 간단한 결과 미리보기
        print(f"\n📈 평균 성능 미리보기:")
        preview = results_df.groupby('centrality_method')[['nmi', 'modularity', 'runtime']].mean().round(3)
        print(preview)
        
        print(f"\n📁 결과 파일:")
        print(f"   상세: results/raw/exp2_centrality_results.csv")
        print(f"   요약: experiments/exp2_centrality/summary_results.txt")
        
        return True
    else:
        print("❌ 실험 실패")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🚀 실험2 완료!")
    else:
        print(f"\n🔧 데이터나 코드를 확인해보세요.")