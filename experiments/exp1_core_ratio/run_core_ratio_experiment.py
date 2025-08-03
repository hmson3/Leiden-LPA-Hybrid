#!/usr/bin/env python3
"""
실험 1: Core Ratio 최적화
다양한 core_ratio 값에서 최적 성능을 찾는 실험
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
from evaluation_system import CommunityEvaluator, save_evaluation_results

class CoreRatioExperiment:
    """Core Ratio 최적화 실험 클래스"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / 'data' / 'data' / 'processed'
        self.results_dir = self.project_root / 'results'
        self.exp_dir = self.project_root / 'experiments' / 'exp1_core_ratio'
        
        # 실험 설정
        self.datasets = ['karate', 'cora', 'citeseer', 'pubmed']
        self.core_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 11개 값
        self.centrality_method = 'pagerank'  # 실험 2 결과 기반으로 PageRank 고정
        self.repeat_count = 5  # 반복 횟수
        
        self._create_directories()
    
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'raw').mkdir(exist_ok=True)
        (self.results_dir / 'figures').mkdir(exist_ok=True)
    
    def load_dataset(self, dataset_name: str) -> Tuple[nx.Graph, Dict[str, int], str]:
        """데이터셋 로드 (실험 2와 동일)"""
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
                            G: nx.Graph, true_labels: Dict[str, int], 
                            run_id: int) -> Dict[str, Any]:
        """단일 실험 실행"""
        try:
            # 알고리즘 실행
            start_time = time.time()
            alg = LeidenLPAHybrid(
                core_ratio=core_ratio,
                centrality_method=self.centrality_method,
                seed=42 + run_id  # 재현 가능한 시드
            )
            pred_labels = alg.fit_predict(G)
            runtime = time.time() - start_time
            
            # 결과 검증
            if not pred_labels or len(pred_labels) == 0:
                print(f"      ⚠️ 빈 결과: core_ratio={core_ratio}")
                return None
            
            num_communities = len(set(pred_labels.values()))
            if num_communities <= 1:
                print(f"      ⚠️ 커뮤니티 부족: core_ratio={core_ratio}")
                return None
            
            # 평가
            evaluator = CommunityEvaluator(G, true_labels)
            evaluation = evaluator.evaluate_clustering(pred_labels, runtime)
            
            # NaN 처리
            modularity = evaluation['structural_quality']['modularity']
            nmi = evaluation['ground_truth_metrics']['nmi']
            ami = evaluation['ground_truth_metrics'].get('ami', np.nan)
            
            if np.isnan(modularity) or np.isnan(nmi) or np.isnan(ami):
                # NaN 값을 -1로 대체
                if np.isnan(modularity):
                    evaluation['structural_quality']['modularity'] = -1.0
                if np.isnan(nmi):
                    evaluation['ground_truth_metrics']['nmi'] = -1.0
                if np.isnan(ami):
                    evaluation['ground_truth_metrics']['ami'] = -1.0
                if np.isnan(evaluation['ground_truth_metrics']['ari']):
                    evaluation['ground_truth_metrics']['ari'] = -1.0
                if np.isnan(evaluation['ground_truth_metrics']['accuracy']):
                    evaluation['ground_truth_metrics']['accuracy'] = -1.0
            
            # 결과 정리
            result = {
                'dataset': dataset_name,
                'core_ratio': core_ratio,
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
                'centrality_time': alg.get_stats().get('centrality_time', 0)
            }
            
            return result
            
        except Exception as e:
            print(f"   ❌ 오류: core_ratio={core_ratio} - {e}")
            return None
    
    def run_dataset_experiment(self, dataset_name: str) -> List[Dict[str, Any]]:
        """단일 데이터셋에 대한 모든 core_ratio 실험"""
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
        
        print(f"   🎯 테스트할 Core Ratio: {len(self.core_ratios)}개")
        
        results = []
        
        # 각 core_ratio별로 실험
        for core_ratio in self.core_ratios:
            print(f"\n   🔧 Core Ratio = {core_ratio:.1f} 실험 중...")
            ratio_results = []
            
            # 반복 실험
            for run_id in range(self.repeat_count):
                result = self.run_single_experiment(
                    dataset_name, core_ratio, G, true_labels, run_id
                )
                if result:
                    ratio_results.append(result)
                    results.append(result)
            
            # 해당 core_ratio 결과 요약
            if ratio_results:
                avg_ami = np.mean([r['ami'] for r in ratio_results if r['ami'] != -1])
                avg_modularity = np.mean([r['modularity'] for r in ratio_results if r['modularity'] != -1])
                avg_runtime = np.mean([r['runtime'] for r in ratio_results])
                avg_communities = np.mean([r['num_communities'] for r in ratio_results])
                
                print(f"      평균 AMI: {avg_ami:.3f}, 모듈러리티: {avg_modularity:.3f}")
                print(f"      실행시간: {avg_runtime:.3f}초, 커뮤니티: {avg_communities:.1f}개")
        
        print(f"\n   ✅ {dataset_name} 완료: {len(results)}개 결과")
        return results
    
    def run_all_experiments(self) -> pd.DataFrame:
        """모든 데이터셋에 대한 실험 실행"""
        print("🚀 Core Ratio 최적화 실험 시작")
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
                'experiment': 'core_ratio_optimization',
                'centrality_method': self.centrality_method,
                'core_ratios': self.core_ratios,
                'repeat_count': self.repeat_count,
                'datasets': self.datasets,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # 결과 저장
            output_file = self.results_dir / 'raw' / 'exp1_core_ratio_results.csv'
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
        
        # 데이터셋별 최적 core_ratio 찾기
        optimal_ratios = {}
        
        for dataset in self.datasets:
            if dataset in results_df['dataset'].values:
                dataset_data = results_df[results_df['dataset'] == dataset]
                
                # AMI 기준 최적값
                avg_ami_by_ratio = dataset_data.groupby('core_ratio')['ami'].mean()
                valid_ami = avg_ami_by_ratio[avg_ami_by_ratio != -1]
                
                if len(valid_ami) > 0:
                    optimal_ami_ratio = valid_ami.idxmax()
                    optimal_ami_score = valid_ami.max()
                    
                    # Modularity 기준 최적값
                    avg_mod_by_ratio = dataset_data.groupby('core_ratio')['modularity'].mean()
                    valid_mod = avg_mod_by_ratio[avg_mod_by_ratio != -1]
                    optimal_mod_ratio = valid_mod.idxmax()
                    optimal_mod_score = valid_mod.max()
                    
                    optimal_ratios[dataset] = {
                        'ami_ratio': optimal_ami_ratio,
                        'ami_score': optimal_ami_score,
                        'mod_ratio': optimal_mod_ratio,
                        'mod_score': optimal_mod_score
                    }
        
        # 전체 평균 최적값
        overall_ami = results_df.groupby('core_ratio')['ami'].mean()
        overall_mod = results_df.groupby('core_ratio')['modularity'].mean()
        
        valid_overall_ami = overall_ami[overall_ami != -1]
        valid_overall_mod = overall_mod[overall_mod != -1]
        
        # 분석 리포트 저장
        analysis_file = self.exp_dir / 'analysis_results.txt'
        with open(analysis_file, 'w') as f:
            f.write("Core Ratio 최적화 실험 분석\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. 데이터셋별 최적 Core Ratio:\n")
            f.write("-" * 30 + "\n")
            for dataset, ratios in optimal_ratios.items():
                f.write(f"{dataset}:\n")
                f.write(f"  AMI 최적: {ratios['ami_ratio']:.1f} (점수: {ratios['ami_score']:.3f})\n")
                f.write(f"  Modularity 최적: {ratios['mod_ratio']:.1f} (점수: {ratios['mod_score']:.3f})\n\n")
            
            f.write("2. 전체 평균 최적 Core Ratio:\n")
            f.write("-" * 30 + "\n")
            if len(valid_overall_ami) > 0:
                f.write(f"AMI 기준: {valid_overall_ami.idxmax():.1f} (점수: {valid_overall_ami.max():.3f})\n")
            if len(valid_overall_mod) > 0:
                f.write(f"Modularity 기준: {valid_overall_mod.idxmax():.1f} (점수: {valid_overall_mod.max():.3f})\n\n")
            
            f.write("3. 주요 발견사항:\n")
            f.write("-" * 30 + "\n")
            f.write(self._generate_insights(results_df, optimal_ratios))
        
        print(f"   💾 분석 저장: {analysis_file}")
    
    def _generate_insights(self, results_df: pd.DataFrame, optimal_ratios: Dict) -> str:
        """주요 발견사항 생성"""
        insights = []
        
        # 일반적인 패턴 분석
        ami_ratios = [r['ami_ratio'] for r in optimal_ratios.values()]
        mod_ratios = [r['mod_ratio'] for r in optimal_ratios.values()]
        
        if ami_ratios:
            avg_ami_ratio = np.mean(ami_ratios)
            insights.append(f"- 평균 최적 AMI Core Ratio: {avg_ami_ratio:.2f}")
        
        if mod_ratios:
            avg_mod_ratio = np.mean(mod_ratios)
            insights.append(f"- 평균 최적 Modularity Core Ratio: {avg_mod_ratio:.2f}")
        
        # 극단값 분석
        extreme_low = sum(1 for r in ami_ratios if r <= 0.2)
        extreme_high = sum(1 for r in ami_ratios if r >= 0.8)
        
        if extreme_low > 0:
            insights.append(f"- {extreme_low}개 데이터셋에서 낮은 Core Ratio(≤0.2) 선호")
        if extreme_high > 0:
            insights.append(f"- {extreme_high}개 데이터셋에서 높은 Core Ratio(≥0.8) 선호")
        
        # 크기별 패턴 (간단히)
        insights.append("- 데이터셋 크기별 최적 전략 확인 필요")
        insights.append("- 품질 vs 속도 트레이드오프 고려")
        
        return "\n".join(insights)

def main():
    """메인 실행 함수"""
    experiment = CoreRatioExperiment()
    
    print("🎯 실험 설정:")
    print(f"   데이터셋: {experiment.datasets}")
    print(f"   Core Ratios: {experiment.core_ratios}")
    print(f"   중심성 지표: {experiment.centrality_method}")
    print(f"   반복 횟수: {experiment.repeat_count}")
    
    # 실험 실행
    results_df = experiment.run_all_experiments()
    
    if not results_df.empty:
        print(f"\n🎉 실험 완료!")
        print(f"   총 결과: {len(results_df)}개")
        print(f"   데이터셋: {results_df['dataset'].nunique()}개")
        print(f"   Core Ratios: {results_df['core_ratio'].nunique()}개")
        
        # 간단한 결과 미리보기
        print(f"\n📈 Core Ratio별 평균 성능:")
        preview = results_df.groupby('core_ratio')[['ami', 'modularity', 'runtime']].mean().round(3)
        print(preview)
        
        print(f"\n📁 결과 파일:")
        print(f"   상세: results/raw/exp1_core_ratio_results.csv")
        print(f"   분석: experiments/exp1_core_ratio/analysis_results.txt")
        
        return True
    else:
        print("❌ 실험 실패")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🚀 다음 단계: 실험 3 (앵커 고정 효과) 또는 결과 분석!")
    else:
        print(f"\n🔧 실험을 다시 확인해보세요.")