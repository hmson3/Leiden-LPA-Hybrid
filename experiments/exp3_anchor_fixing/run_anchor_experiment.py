#!/usr/bin/env python3
"""
실험 3: 3-Way 앵커 전략 비교
- Fixed_Single: 앵커 고정 + 1회 전파
- Fixed_Iterative: 앵커 고정 + 반복 전파  
- Dynamic_Iterative: 앵커 비고정 + 반복 전파
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

from improved_leiden_lpa import LeidenLPAHybrid  # v2 사용!
from evaluation_system import CommunityEvaluator, save_evaluation_results

class AnchorStrategyExperiment:
    """3-Way 앵커 전략 비교 실험 클래스"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / 'data' / 'data' / 'processed'
        self.results_dir = self.project_root / 'results'
        self.exp_dir = self.project_root / 'experiments' / 'exp3_anchor_fixing'
        
        # 실험 설정
        self.datasets = ['karate', 'cora', 'citeseer', 'pubmed']
        self.core_ratios = [0.4, 0.5, 0.6]  # 실험1 최적 범위
        self.anchor_strategies = {
            'Fixed_Single': 'fixed_single',       # 고정 + 1회
            'Fixed_Iterative': 'fixed_iterative', # 고정 + 반복  
            'Dynamic_Iterative': 'dynamic_iterative'  # 비고정 + 반복
        }
        self.centrality_method = 'pagerank'  # 실험2 최적
        self.repeat_count = 5
        
        self._create_directories()
    
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'raw').mkdir(exist_ok=True)
        (self.results_dir / 'figures').mkdir(exist_ok=True)
    
    def load_dataset(self, dataset_name: str) -> Tuple[nx.Graph, Dict[str, int], str]:
        """데이터셋 로드 (기존과 동일)"""
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
        
        # 공통 노드 확인 및 타입 변환
        graph_nodes = set(G.nodes())
        label_nodes = set(labels.keys())
        common_nodes = graph_nodes & label_nodes
        
        if len(common_nodes) == 0:
            # 노드 ID 타입 변환 시도
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
            size_category = 'tiny'
        elif n_nodes < 1000:
            size_category = 'small'
        elif n_nodes < 10000:
            size_category = 'medium'
        else:
            size_category = 'large'
        
        return G, labels, size_category
    
    def run_single_experiment(self, dataset_name: str, core_ratio: float,
                            anchor_strategy: str, strategy_name: str,
                            G: nx.Graph, true_labels: Dict[str, int], 
                            run_id: int) -> Dict[str, Any]:
        """단일 실험 실행"""
        try:
            # 알고리즘 실행
            start_time = time.time()
            alg = LeidenLPAHybrid(
                core_ratio=core_ratio,
                centrality_method=self.centrality_method,
                anchor_strategy=anchor_strategy,
                seed=42 + run_id
            )
            pred_labels = alg.fit_predict(G)
            runtime = time.time() - start_time
            
            # 결과 검증
            if not pred_labels or len(pred_labels) == 0:
                print(f"      ⚠️ 빈 결과: {strategy_name}")
                return None
            
            num_communities = len(set(pred_labels.values()))
            if num_communities <= 1:
                print(f"      ⚠️ 커뮤니티 부족: {strategy_name}")
                return None
            
            # 평가
            evaluator = CommunityEvaluator(G, true_labels)
            evaluation = evaluator.evaluate_clustering(pred_labels, runtime)
            
            # NaN 처리
            for metric in ['modularity', 'conductance', 'coverage']:
                if np.isnan(evaluation['structural_quality'][metric]):
                    evaluation['structural_quality'][metric] = -1.0
            
            for metric in ['nmi', 'ami', 'ari', 'accuracy']:
                if np.isnan(evaluation['ground_truth_metrics'].get(metric, np.nan)):
                    evaluation['ground_truth_metrics'][metric] = -1.0
            
            # 결과 정리
            result = {
                'dataset': dataset_name,
                'core_ratio': core_ratio,
                'anchor_strategy': strategy_name,
                'run_id': run_id,
                'runtime': runtime,
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'num_communities': evaluation['clustering_info']['num_communities'],
                'modularity': evaluation['structural_quality']['modularity'],
                'conductance': evaluation['structural_quality']['conductance'],
                'coverage': evaluation['structural_quality']['coverage'],
                'nmi': evaluation['ground_truth_metrics']['nmi'],
                'ami': evaluation['ground_truth_metrics'].get('ami', -1.0),
                'ari': evaluation['ground_truth_metrics']['ari'],
                'accuracy': evaluation['ground_truth_metrics']['accuracy'],
                'core_nodes_count': alg.get_stats().get('core_nodes_count', 0),
                'periphery_nodes_count': alg.get_stats().get('periphery_nodes_count', 0),
                'leiden_time': alg.get_stats().get('leiden_time', 0),
                'lpa_time': alg.get_stats().get('lpa_time', 0),
                'lpa_iterations': alg.get_stats().get('lpa_iterations', 0),
                'centrality_time': alg.get_stats().get('centrality_time', 0)
            }
            
            return result
            
        except Exception as e:
            print(f"   ❌ 오류: {strategy_name} - {e}")
            return None
    
    def run_dataset_experiment(self, dataset_name: str) -> List[Dict[str, Any]]:
        """단일 데이터셋에 대한 모든 앵커 전략 실험"""
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
        
        print(f"   🎯 테스트할 조합: {len(self.core_ratios)} ratios × {len(self.anchor_strategies)} strategies")
        
        results = []
        
        # 각 core_ratio별로 실험
        for core_ratio in self.core_ratios:
            print(f"\n   🔧 Core Ratio = {core_ratio:.1f}")
            
            # 각 앵커 전략별로 실험
            for strategy_name, anchor_strategy in self.anchor_strategies.items():
                print(f"     📌 {strategy_name} 실험 중...")
                strategy_results = []
                
                # 반복 실험
                for run_id in range(self.repeat_count):
                    result = self.run_single_experiment(
                        dataset_name, core_ratio, anchor_strategy, strategy_name,
                        G, true_labels, run_id
                    )
                    if result:
                        strategy_results.append(result)
                        results.append(result)
                
                # 해당 전략 결과 요약
                if strategy_results:
                    avg_ami = np.mean([r['ami'] for r in strategy_results if r['ami'] != -1])
                    avg_runtime = np.mean([r['runtime'] for r in strategy_results])
                    avg_iterations = np.mean([r['lpa_iterations'] for r in strategy_results])
                    
                    print(f"        AMI: {avg_ami:.3f}, 시간: {avg_runtime:.3f}초, "
                          f"반복: {avg_iterations:.1f}회")
        
        print(f"\n   ✅ {dataset_name} 완료: {len(results)}개 결과")
        return results
    
    def run_all_experiments(self) -> pd.DataFrame:
        """모든 데이터셋에 대한 실험 실행"""
        print("🚀 3-Way 앵커 전략 비교 실험 시작")
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
                'experiment': '3way_anchor_strategy_comparison',
                'core_ratios': self.core_ratios,
                'anchor_strategies': list(self.anchor_strategies.keys()),
                'centrality_method': self.centrality_method,
                'repeat_count': self.repeat_count,
                'datasets': self.datasets,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # 결과 저장
            output_file = self.results_dir / 'raw' / 'exp3_anchor_results.csv'
            save_evaluation_results(results_df, str(output_file), metadata)
            
            # 요약 및 분석 저장
            self._save_analysis(results_df)
            
            return results_df
        else:
            print("❌ 실험 결과가 없습니다.")
            return pd.DataFrame()
    
    def _save_analysis(self, results_df: pd.DataFrame):
        """실험 결과 분석 및 저장"""
        print(f"\n📊 결과 분석 생성 중...")
        
        # 전략별 평균 성능
        strategy_summary = results_df.groupby('anchor_strategy').agg({
            'ami': ['mean', 'std'],
            'runtime': ['mean', 'std'],
            'lpa_iterations': ['mean', 'std'],
            'modularity': ['mean', 'std']
        }).round(4)
        
        # 데이터셋별 최적 전략
        best_strategies = {}
        for dataset in self.datasets:
            if dataset in results_df['dataset'].values:
                dataset_data = results_df[results_df['dataset'] == dataset]
                
                # AMI 기준 최적 전략
                avg_ami_by_strategy = dataset_data.groupby('anchor_strategy')['ami'].mean()
                valid_ami = avg_ami_by_strategy[avg_ami_by_strategy != -1]
                
                if len(valid_ami) > 0:
                    best_strategy = valid_ami.idxmax()
                    best_score = valid_ami.max()
                    
                    best_strategies[dataset] = {
                        'best_strategy': best_strategy,
                        'ami_score': best_score,
                        'all_scores': dict(valid_ami)
                    }
        
        # Core Ratio × Strategy 상호작용 분석
        interaction_analysis = {}
        for core_ratio in self.core_ratios:
            ratio_data = results_df[results_df['core_ratio'] == core_ratio]
            if len(ratio_data) > 0:
                avg_by_strategy = ratio_data.groupby('anchor_strategy')['ami'].mean()
                valid_scores = avg_by_strategy[avg_by_strategy != -1]
                if len(valid_scores) > 0:
                    interaction_analysis[core_ratio] = {
                        'best_strategy': valid_scores.idxmax(),
                        'scores': dict(valid_scores)
                    }
        
        # 수렴 분석
        convergence_analysis = {}
        for strategy in self.anchor_strategies.keys():
            strategy_data = results_df[results_df['anchor_strategy'] == strategy]
            if len(strategy_data) > 0:
                convergence_analysis[strategy] = {
                    'avg_iterations': strategy_data['lpa_iterations'].mean(),
                    'max_iterations': strategy_data['lpa_iterations'].max(),
                    'convergence_rate': len(strategy_data[strategy_data['lpa_iterations'] < 10]) / len(strategy_data) * 100
                }
        
        # 분석 리포트 저장
        analysis_file = self.exp_dir / 'analysis_results.txt'
        with open(analysis_file, 'w') as f:
            f.write("3-Way 앵커 전략 비교 실험 분석\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. 전략별 전체 평균 성능:\n")
            f.write("-" * 30 + "\n")
            f.write(str(strategy_summary))
            f.write("\n\n")
            
            f.write("2. 데이터셋별 최적 전략 (AMI 기준):\n")
            f.write("-" * 30 + "\n")
            for dataset, info in best_strategies.items():
                f.write(f"{dataset}:\n")
                f.write(f"  최적: {info['best_strategy']} (AMI: {info['ami_score']:.3f})\n")
                f.write(f"  전체: {info['all_scores']}\n\n")
            
            f.write("3. Core Ratio별 최적 전략:\n")
            f.write("-" * 30 + "\n")
            for ratio, info in interaction_analysis.items():
                f.write(f"Core Ratio {ratio}:\n")
                f.write(f"  최적: {info['best_strategy']}\n")
                f.write(f"  점수: {info['scores']}\n\n")
            
            f.write("4. 수렴 분석:\n")
            f.write("-" * 30 + "\n")
            for strategy, info in convergence_analysis.items():
                f.write(f"{strategy}:\n")
                f.write(f"  평균 반복: {info['avg_iterations']:.1f}회\n")
                f.write(f"  최대 반복: {info['max_iterations']}회\n")
                f.write(f"  수렴률: {info['convergence_rate']:.1f}%\n\n")
            
            f.write("5. 주요 발견사항:\n")
            f.write("-" * 30 + "\n")
            f.write(self._generate_insights(results_df, best_strategies, interaction_analysis))
        
        print(f"   💾 분석 저장: {analysis_file}")
    
    def _generate_insights(self, results_df: pd.DataFrame, 
                          best_strategies: Dict, interaction_analysis: Dict) -> str:
        """주요 발견사항 생성"""
        insights = []
        
        # 전반적인 우승 전략
        overall_ami = results_df.groupby('anchor_strategy')['ami'].mean()
        overall_runtime = results_df.groupby('anchor_strategy')['runtime'].mean()
        
        best_quality = overall_ami[overall_ami != -1].idxmax()
        best_speed = overall_runtime.idxmin()
        
        insights.append(f"- 전체 평균 품질 최고: {best_quality}")
        insights.append(f"- 전체 평균 속도 최고: {best_speed}")
        
        # 데이터셋 크기별 패턴
        small_datasets = ['karate']  # < 100 nodes
        medium_datasets = ['cora', 'citeseer']  # 1K-10K nodes  
        large_datasets = ['pubmed']  # > 10K nodes
        
        def get_pattern(dataset_list, label):
            if len(dataset_list) > 0:
                winners = [best_strategies.get(d, {}).get('best_strategy', 'N/A') 
                          for d in dataset_list if d in best_strategies]
                if winners:
                    most_common = max(set(winners), key=winners.count) if winners else 'N/A'
                    insights.append(f"- {label} 선호 전략: {most_common}")
        
        get_pattern(small_datasets, "소규모 그래프")
        get_pattern(medium_datasets, "중간 규모 그래프")  
        get_pattern(large_datasets, "대규모 그래프")
        
        # Core Ratio 패턴
        if interaction_analysis:
            low_ratio_strategy = interaction_analysis.get(0.4, {}).get('best_strategy', 'N/A')
            high_ratio_strategy = interaction_analysis.get(0.6, {}).get('best_strategy', 'N/A')
            
            insights.append(f"- 낮은 Core Ratio(0.4) 최적: {low_ratio_strategy}")
            insights.append(f"- 높은 Core Ratio(0.6) 최적: {high_ratio_strategy}")
        
        # 반복 vs 품질 트레이드오프
        iterative_strategies = ['Fixed_Iterative', 'Dynamic_Iterative']
        single_strategy = ['Fixed_Single']
        
        iter_ami = results_df[results_df['anchor_strategy'].isin(iterative_strategies)]['ami'].mean()
        single_ami = results_df[results_df['anchor_strategy'].isin(single_strategy)]['ami'].mean()
        
        if iter_ami > single_ami:
            insights.append(f"- 반복 전파가 품질 향상: +{(iter_ami-single_ami):.3f} AMI")
        else:
            insights.append(f"- 1회 전파도 충분한 품질 확보")
        
        return "\n".join(insights)

def main():
    """메인 실행 함수"""
    experiment = AnchorStrategyExperiment()
    
    print("🎯 실험 3 설정:")
    print(f"   데이터셋: {experiment.datasets}")
    print(f"   Core Ratios: {experiment.core_ratios}")
    print(f"   앵커 전략: {list(experiment.anchor_strategies.keys())}")
    print(f"   중심성 지표: {experiment.centrality_method}")
    print(f"   반복 횟수: {experiment.repeat_count}")
    
    # 실험 실행
    results_df = experiment.run_all_experiments()
    
    if not results_df.empty:
        print(f"\n🎉 실험 완료!")
        print(f"   총 결과: {len(results_df)}개")
        print(f"   데이터셋: {results_df['dataset'].nunique()}개")
        print(f"   앵커 전략: {results_df['anchor_strategy'].nunique()}개")
        
        # 간단한 결과 미리보기
        print(f"\n📈 앵커 전략별 평균 성능:")
        preview = results_df.groupby('anchor_strategy')[['ami', 'runtime', 'lpa_iterations']].mean().round(3)
        print(preview)
        
        print(f"\n📁 결과 파일:")
        print(f"   상세: results/raw/exp3_anchor_results.csv")
        print(f"   분석: experiments/exp3_anchor_fixing/analysis_results.txt")
        
        return True
    else:
        print("❌ 실험 실패")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🚀 다음 단계: 실험 4 (기존 방법 비교) 또는 통합 분석!")
    else:
        print(f"\n🔧 실험을 다시 확인해보세요.")