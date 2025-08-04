"""
실험 1: Core Ratio 최적화
목적: 각 데이터셋에서 최적 core_ratio 값을 찾아 AMI, NMI, ARI, Accuracy 4개 지표로 종합 분석
"""

import os
import sys
import time
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
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
        self.project_root = Path(__file__).parent.parent.parent  # experiments/exp1_core_ratio에서 3단계 위로
        self.data_dir = self.project_root / 'data' / 'data' / 'processed'
        self.results_dir = self.project_root / 'results'
        self.exp_dir = self.project_root / 'experiments' / 'exp1_core_ratio'
        
        # 실험 설정
        self.core_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.datasets = ['karate', 'cora', 'citeseer', 'pubmed']
        self.centrality_method = 'pagerank'  # 실험2에서 최적으로 확인된 지표
        self.anchor_fixed = True
        self.repeat_runs = 5  # 통계적 신뢰성을 위한 반복 실험
        
        # 결과 저장용
        self.results = []
        
        self._create_directories()
    
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'raw').mkdir(exist_ok=True)
        (self.results_dir / 'figures').mkdir(exist_ok=True)
    
    def load_dataset(self, dataset_name: str) -> Tuple[nx.Graph, Dict[str, int], str]:
        """데이터셋 로드 (실험2와 동일한 방식)"""
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
        
    def run_single_experiment(self, dataset_name, core_ratio, run_id, G, true_labels):
        """단일 실험 실행 (실험2 스타일로 수정)"""
        try:
            # 알고리즘 실행
            start_time = time.time()
            alg = LeidenLPAHybrid(
                core_ratio=core_ratio,
                centrality_method=self.centrality_method,
                anchor_strategy='fixed_iterative',
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
            
            # 알고리즘 통계
            stats = alg.get_stats()
            
            # 결과 정리
            result = {
                'dataset': dataset_name,
                'core_ratio': core_ratio,
                'run_id': run_id,
                'runtime': runtime,
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'num_communities': evaluation['clustering_info']['num_communities'],
                
                # 구조적 품질 지표
                'modularity': evaluation['structural_quality']['modularity'],
                'conductance': evaluation['structural_quality']['conductance'],
                'coverage': evaluation['structural_quality']['coverage'],
                
                # Ground truth 비교 지표 (4개 핵심 지표)
                'nmi': evaluation['ground_truth_metrics']['nmi'],
                'ami': evaluation['ground_truth_metrics']['ami'],  # 주요 지표
                'ari': evaluation['ground_truth_metrics']['ari'],
                'accuracy': evaluation['ground_truth_metrics']['accuracy'],
                
                # 알고리즘 분석 정보
                'core_nodes_count': stats.get('core_nodes_count', 0),
                'periphery_nodes_count': stats.get('periphery_nodes_count', 0),
                'leiden_time': stats.get('leiden_time', 0),
                'lpa_time': stats.get('lpa_time', 0),
                'centrality_time': stats.get('centrality_time', 0)
            }
            
            return result
            
        except Exception as e:
            print(f"    ❌ 오류: core_ratio={core_ratio} - {e}")
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
            for run_id in range(self.repeat_runs):
                result = self.run_single_experiment(
                    dataset_name, core_ratio, run_id, G, true_labels
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
                'repeat_count': self.repeat_runs,
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
        """실험 결과 분석 및 저장 (실험2 스타일)"""
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
    
    def create_visualizations(self, df):
        """시각화 생성"""
        print("🎨 시각화 생성 중...")
        
        # 1. 4개 주요 지표 종합 비교
        self.plot_four_metrics_comparison(df)
        
        # 2. 데이터셋별 상세 분석
        self.plot_dataset_detailed_analysis(df)
        
        # 3. Runtime vs Quality 트레이드오프
        self.plot_runtime_quality_tradeoff(df)
        
        # 4. 최적 core_ratio 요약
        self.plot_optimal_ratios_summary(df)
    
    def plot_four_metrics_comparison(self, df):
        """4개 주요 지표 종합 비교 (AMI, NMI, ARI, Accuracy)"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        metrics = ['ami', 'nmi', 'ari', 'accuracy']
        titles = ['AMI (주요 지표)', 'NMI (전통적 지표)', 'ARI (클러스터 일치도)', 'Accuracy (직관적 정확도)']
        
        # 평균값 계산
        df_mean = df.groupby(['dataset', 'core_ratio'])[metrics].mean().reset_index()
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            
            # 데이터셋별 라인 플롯
            for dataset in self.datasets:
                data = df_mean[df_mean['dataset'] == dataset]
                ax.plot(data['core_ratio'], data[metric], 
                       marker='o', linewidth=2, markersize=6, 
                       label=dataset, alpha=0.8)
            
            ax.set_xlabel('Core Ratio')
            ax.set_ylabel(metric.upper())
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'four_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'four_metrics_comparison.pdf', bbox_inches='tight')
        print(f"📊 저장: four_metrics_comparison.png")
        plt.show()
    
    def plot_dataset_detailed_analysis(self, df):
        """데이터셋별 상세 분석"""
        for dataset in self.datasets:
            dataset_df = df[df['dataset'] == dataset]
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'{dataset.upper()} Dataset - Core Ratio Analysis', fontsize=16, fontweight='bold')
            
            # 평균값 계산
            df_mean = dataset_df.groupby('core_ratio').agg({
                'ami': ['mean', 'std'],
                'nmi': ['mean', 'std'], 
                'ari': ['mean', 'std'],
                'accuracy': ['mean', 'std'],
                'runtime': ['mean', 'std'],
                'modularity': ['mean', 'std']
            }).reset_index()
            
            df_mean.columns = ['core_ratio'] + [f'{col[0]}_{col[1]}' for col in df_mean.columns[1:]]
            
            # 6개 서브플롯
            metrics = [
                ('ami', 'AMI (주요 지표)'),
                ('nmi', 'NMI'), 
                ('ari', 'ARI'),
                ('accuracy', 'Accuracy'),
                ('runtime', 'Runtime (초)'),
                ('modularity', 'Modularity')
            ]
            
            for i, (metric, title) in enumerate(metrics):
                ax = axes[i//3, i%3]
                
                # 에러바와 함께 플롯
                ax.errorbar(df_mean['core_ratio'], df_mean[f'{metric}_mean'], 
                           yerr=df_mean[f'{metric}_std'], 
                           marker='o', linewidth=2, markersize=8, 
                           capsize=5, capthick=2, alpha=0.8)
                
                ax.set_xlabel('Core Ratio')
                ax.set_ylabel(title)
                ax.set_title(title, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(-0.05, 1.05)
                
                # 최적점 표시 (AMI 기준)
                if metric == 'ami':
                    best_idx = df_mean[f'{metric}_mean'].idxmax()
                    best_ratio = df_mean.loc[best_idx, 'core_ratio']
                    best_value = df_mean.loc[best_idx, f'{metric}_mean']
                    ax.scatter(best_ratio, best_value, color='red', s=100, zorder=5)
                    ax.annotate(f'Best: {best_ratio:.1f}', 
                               xy=(best_ratio, best_value),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                               fontweight='bold', color='white')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{dataset}_detailed_analysis.png', dpi=300, bbox_inches='tight')
            print(f"📊 저장: {dataset}_detailed_analysis.png")
            plt.show()
    
    def plot_runtime_quality_tradeoff(self, df):
        """Runtime vs Quality 트레이드오프 분석"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Runtime vs Quality Tradeoff Analysis', fontsize=16, fontweight='bold')
        
        # 평균값 계산
        df_mean = df.groupby(['dataset', 'core_ratio']).agg({
            'runtime': 'mean',
            'ami': 'mean', 
            'nmi': 'mean',
            'ari': 'mean',
            'accuracy': 'mean'
        }).reset_index()
        
        metrics = ['ami', 'nmi', 'ari', 'accuracy']
        titles = ['Runtime vs AMI', 'Runtime vs NMI', 'Runtime vs ARI', 'Runtime vs Accuracy']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            
            for dataset in self.datasets:
                data = df_mean[df_mean['dataset'] == dataset]
                
                # 산점도
                scatter = ax.scatter(data['runtime'], data[metric], 
                                   s=100, alpha=0.7, label=dataset)
                
                # core_ratio 값을 텍스트로 표시
                for _, row in data.iterrows():
                    ax.annotate(f"{row['core_ratio']:.1f}", 
                               (row['runtime'], row[metric]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
            
            ax.set_xlabel('Runtime (초)')
            ax.set_ylabel(metric.upper())
            ax.set_title(title, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'runtime_quality_tradeoff.png', dpi=300, bbox_inches='tight')
        print(f"📊 저장: runtime_quality_tradeoff.png")
        plt.show()
    
    def plot_optimal_ratios_summary(self, df):
        """최적 core_ratio 요약"""
        # 각 데이터셋별 최적 core_ratio 찾기 (AMI 기준)
        df_mean = df.groupby(['dataset', 'core_ratio'])['ami'].mean().reset_index()
        optimal_ratios = df_mean.loc[df_mean.groupby('dataset')['ami'].idxmax()]
        
        # 바 차트
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 최적 core_ratio
        bars1 = ax1.bar(optimal_ratios['dataset'], optimal_ratios['core_ratio'], 
                        color=['skyblue', 'lightgreen', 'lightcoral', 'gold'], alpha=0.8)
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Optimal Core Ratio')
        ax1.set_title('Optimal Core Ratio by Dataset (AMI 기준)', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 값 표시
        for bar, ratio in zip(bars1, optimal_ratios['core_ratio']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{ratio:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 최적점에서의 AMI 값
        bars2 = ax2.bar(optimal_ratios['dataset'], optimal_ratios['ami'],
                        color=['skyblue', 'lightgreen', 'lightcoral', 'gold'], alpha=0.8)
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('AMI Score')
        ax2.set_title('AMI Score at Optimal Core Ratio', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 값 표시
        for bar, ami in zip(bars2, optimal_ratios['ami']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{ami:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'optimal_ratios_summary.png', dpi=300, bbox_inches='tight')
        print(f"📊 저장: optimal_ratios_summary.png")
        plt.show()
        
        return optimal_ratios
    
    def generate_report(self, df, optimal_ratios):
        """실험 결과 리포트 생성"""
        report_path = self.output_dir / 'experiment_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("🎯 실험 1: Core Ratio 최적화 결과 리포트\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("📊 실험 설정:\n")
            f.write(f"  - 데이터셋: {', '.join(self.datasets)}\n")
            f.write(f"  - Core Ratios: {self.core_ratios}\n")
            f.write(f"  - 중심성 지표: {self.centrality_method}\n")
            f.write(f"  - 반복 실험: {self.repeat_runs}회\n")
            f.write(f"  - 총 실험 수: {len(df)}\n\n")
            
            f.write("🏆 최적 Core Ratio (AMI 기준):\n")
            for _, row in optimal_ratios.iterrows():
                f.write(f"  - {row['dataset']}: {row['core_ratio']:.1f} (AMI: {row['ami']:.4f})\n")
            f.write("\n")
            
            # 데이터셋별 요약
            for dataset in self.datasets:
                dataset_df = df[df['dataset'] == dataset]
                f.write(f"📈 {dataset.upper()} 분석:\n")
                
                # 네트워크 정보
                nodes = dataset_df['nodes'].iloc[0]
                edges = dataset_df['edges'].iloc[0] 
                f.write(f"  - 노드 수: {nodes:,}, 엣지 수: {edges:,}\n")
                
                # 최적 core_ratio에서의 성능
                optimal_ratio = optimal_ratios[optimal_ratios['dataset'] == dataset]['core_ratio'].iloc[0]
                optimal_data = dataset_df[dataset_df['core_ratio'] == optimal_ratio]
                
                f.write(f"  - 최적 Core Ratio: {optimal_ratio:.1f}\n")
                f.write(f"  - AMI: {optimal_data['ami'].mean():.4f} ± {optimal_data['ami'].std():.4f}\n")
                f.write(f"  - NMI: {optimal_data['nmi'].mean():.4f} ± {optimal_data['nmi'].std():.4f}\n")
                f.write(f"  - ARI: {optimal_data['ari'].mean():.4f} ± {optimal_data['ari'].std():.4f}\n")
                f.write(f"  - Accuracy: {optimal_data['accuracy'].mean():.4f} ± {optimal_data['accuracy'].std():.4f}\n")
                f.write(f"  - Runtime: {optimal_data['runtime'].mean():.4f} ± {optimal_data['runtime'].std():.4f} 초\n")
                f.write(f"  - Modularity: {optimal_data['modularity'].mean():.4f} ± {optimal_data['modularity'].std():.4f}\n")
                f.write("\n")
        
        print(f"📝 리포트 저장: {report_path}")

def main():
    """메인 실행 함수 (실험2 스타일)"""
    experiment = CoreRatioExperiment()
    
    print("🎯 실험 설정:")
    print(f"   데이터셋: {experiment.datasets}")
    print(f"   Core Ratios: {experiment.core_ratios}")
    print(f"   중심성 지표: {experiment.centrality_method}")
    print(f"   반복 횟수: {experiment.repeat_runs}")
    
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
        print(f"\n🚀 실험1 완료!")
    else:
        print(f"\n🔧 실험을 다시 확인해보세요.")