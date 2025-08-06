#!/usr/bin/env python3
"""
실험 1: Core Ratio 최적화 (단순 체크포인트만)
"""

import os
import sys
import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
import time
from typing import Dict, List, Tuple, Any, Union
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from improved_leiden_lpa import LeidenLPAHybrid
from evaluation_system import CommunityEvaluator, load_overlapping_labels
from experiment_utils import ExperimentCheckpoint, create_task_id, ProgressReporter

class CoreRatioExperiment:
    """Core Ratio 최적화 실험 클래스"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / 'data' / 'data' / 'processed'
        self.results_dir = self.project_root / 'results'
        self.exp_dir = self.project_root / 'experiments' / 'exp1_core_ratio'
        
        # 실험 설정
        self.datasets = ['karate', 'cora', 'citeseer', 'pubmed','com-amazon', 'com-dblp', 'com-youtube']
        self.core_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.centrality_method = 'pagerank'
        self.repeat_count = 1
        
        # 데이터셋별 overlapping 여부 (사용자 수정 필요)
        self.dataset_overlapping = {
            'karate': False,     # 수정하세요
            'cora': False,       # 수정하세요  
            'citeseer': False,   # 수정하세요
            'pubmed': False       # 수정하세요
        }
        
        # 체크포인트 설정
        self.checkpoint_interval = 1  # 5개마다 저장
        
        self._create_directories()
    
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'raw').mkdir(exist_ok=True)
        (self.exp_dir / 'checkpoints').mkdir(exist_ok=True)
    
    def load_dataset(self, dataset_name: str) -> Tuple[nx.Graph, Dict[str, Union[int, List[int]]], str, bool]:
        """데이터셋 로드 (전체 그래프 사용, 레이블 없는 노드는 고유 레이블)"""
        dataset_path = self.data_dir / dataset_name
        
        # 그래프 로드 (전체)
        edge_file = dataset_path / 'graph.edgelist'
        if not edge_file.exists():
            raise FileNotFoundError(f"Dataset {dataset_name} not found")
        
        G = nx.read_edgelist(str(edge_file))
        print(f"   🔍 원본 그래프: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        
        # 레이블 로드
        is_overlapping = self.dataset_overlapping.get(dataset_name, False)
        label_file = dataset_path / 'labels.txt'
        
        if is_overlapping:
            # Overlapping 형식
            partial_labels = load_overlapping_labels(str(label_file))
        else:
            # Non-overlapping 형식
            partial_labels = {}
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        node, label = parts[0], int(parts[1])
                        partial_labels[node] = label
        
        print(f"   🔍 레이블 파일의 노드: {len(partial_labels):,}개")
        
        # 노드 타입 매칭 및 변환
        graph_nodes = set(G.nodes())
        label_nodes = set(partial_labels.keys())
        
        # 타입이 다르면 정수로 통일
        if len(graph_nodes & label_nodes) == 0:
            print(f"   ⚙️ 노드 타입 변환 시도...")
            try:
                # 그래프 노드를 정수로
                int_mapping = {node: int(node) for node in G.nodes()}
                G = nx.relabel_nodes(G, int_mapping)
                
                # 레이블도 정수 키로
                if is_overlapping:
                    int_labels = {}
                    for node, comm_list in partial_labels.items():
                        try:
                            int_node = int(node)
                            int_labels[int_node] = comm_list
                        except:
                            continue
                else:
                    int_labels = {}
                    for node, label in partial_labels.items():
                        try:
                            int_node = int(node)
                            int_labels[int_node] = label
                        except:
                            continue
                
                partial_labels = int_labels
                print(f"   ✅ 타입 변환 완료")
                
            except Exception as e:
                raise ValueError(f"노드 ID 매칭 실패: {e}")
        
        # 전체 그래프의 모든 노드에 대해 레이블 생성
        all_nodes = set(G.nodes())
        labeled_nodes = set(partial_labels.keys()) & all_nodes
        unlabeled_nodes = all_nodes - labeled_nodes
        
        print(f"   📊 노드 분포:")
        print(f"      레이블 있는 노드: {len(labeled_nodes):,}개")
        print(f"      레이블 없는 노드: {len(unlabeled_nodes):,}개")
        
        # 완전한 레이블 딕셔너리 생성
        complete_labels = {}
        
        # 1. 기존 레이블 추가
        for node in labeled_nodes:
            complete_labels[node] = partial_labels[node]
        
        # 2. 레이블 없는 노드들은 고유 레이블 할당
        if is_overlapping:
            # Overlapping의 경우: 기존 커뮤니티 ID의 최대값 찾기
            existing_communities = set()
            for comm_list in partial_labels.values():
                existing_communities.update(comm_list)
            next_community_id = max(existing_communities) + 1 if existing_communities else 0
            
            for node in unlabeled_nodes:
                complete_labels[node] = [next_community_id]  # 리스트 형태
                next_community_id += 1
        else:
            # Non-overlapping의 경우: 기존 레이블의 최대값 찾기
            existing_labels = set(partial_labels.values()) if partial_labels else set()
            next_label = max(existing_labels) + 1 if existing_labels else 0
            
            for node in unlabeled_nodes:
                complete_labels[node] = next_label  # 정수 형태
                next_label += 1
        
        print(f"   ✅ 완전한 레이블 생성: {len(complete_labels):,}개 노드")
        
        # 크기 카테고리
        n_nodes = G.number_of_nodes()
        if n_nodes < 100:
            size_category = 'tiny'
        elif n_nodes < 1000:
            size_category = 'small'
        elif n_nodes < 10000:
            size_category = 'medium'
        else:
            size_category = 'large'
        
        return G, complete_labels, size_category, is_overlapping
    
    def run_single_experiment(self, dataset_name: str, core_ratio: float,
                            G: nx.Graph, true_labels: Dict[str, Union[int, List[int]]], 
                            is_overlapping: bool, run_id: int) -> Dict[str, Any]:
        """단일 실험 실행"""
        try:
            # 알고리즘 실행
            start_time = time.time()
            alg = LeidenLPAHybrid(
                core_ratio=core_ratio,
                centrality_method=self.centrality_method,
                seed=42 + run_id
            )
            pred_labels = alg.fit_predict(G)
            runtime = time.time() - start_time
            
            # 결과 검증
            if not pred_labels or len(pred_labels) == 0:
                return None
            
            num_communities = len(set(pred_labels.values()))
            if num_communities <= 1:
                return None
            
            # 평가
            evaluator = CommunityEvaluator(G, true_labels, is_overlapping)
            evaluation = evaluator.evaluate_clustering(pred_labels, runtime)
            
            # 결과 정리
            result = {
                'dataset': dataset_name,
                'core_ratio': core_ratio,
                'run_id': run_id,
                'runtime': runtime,
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'is_overlapping': is_overlapping,
                'num_communities': evaluation['clustering_info']['num_communities'],
                'modularity': evaluation['structural_quality'].get('modularity', -1.0),
                'conductance': evaluation['structural_quality'].get('conductance', -1.0),
                'coverage': evaluation['structural_quality'].get('coverage', -1.0),
            }
            
            # Ground truth 지표 추가
            if evaluation.get('ground_truth_metrics'):
                gt = evaluation['ground_truth_metrics']
                result.update({
                    'nmi': gt.get('nmi', -1.0),
                    'ami': gt.get('ami', -1.0),
                    'ari': gt.get('ari', -1.0),
                    'accuracy': gt.get('accuracy', -1.0),
                    'f1_score': gt.get('f1_score', -1.0),
                    'overlapping_nmi': gt.get('overlapping_nmi', -1.0)
                })
            
            # 알고리즘 통계
            stats = alg.get_stats()
            result.update({
                'core_nodes_count': stats.get('core_nodes_count', 0),
                'periphery_nodes_count': stats.get('periphery_nodes_count', 0),
                'leiden_time': stats.get('leiden_time', 0),
                'lpa_time': stats.get('lpa_time', 0),
                'centrality_time': stats.get('centrality_time', 0)
            })
            
            return result
            
        except Exception as e:
            print(f"❌ 실험 실패: {dataset_name} ratio={core_ratio} run={run_id} - {e}")
            return None
    
    def generate_all_tasks(self) -> List[str]:
        """모든 작업 ID 생성"""
        tasks = []
        for dataset in self.datasets:
            for core_ratio in self.core_ratios:
                for run_id in range(self.repeat_count):
                    task_id = create_task_id(
                        dataset=dataset,
                        method="core_ratio",
                        run_id=run_id,
                        ratio=f"{core_ratio:.1f}"
                    )
                    tasks.append(task_id)
        return tasks
    
    def run_with_checkpoint(self, checkpoint: ExperimentCheckpoint) -> None:
        """체크포인트를 사용한 실험 실행"""
        
        # 모든 작업 목록
        all_tasks = self.generate_all_tasks()
        total_tasks = len(all_tasks)
        remaining_tasks = checkpoint.get_remaining_tasks(all_tasks)
        
        print(f"🎯 전체 작업: {total_tasks}개")
        print(f"✅ 완료된 작업: {checkpoint.get_completed_count()}개")
        print(f"⏳ 남은 작업: {len(remaining_tasks)}개")
        
        if len(remaining_tasks) == 0:
            print("🎉 모든 작업이 완료되었습니다!")
            return
        
        # 진행 상황 리포터
        progress = ProgressReporter(total_tasks, report_interval=10)
        
        # 데이터셋 로드 (캐시)
        loaded_datasets = {}
        
        batch_results = []
        task_count = 0
        
        for task_id in remaining_tasks:
            # 작업 정보 파싱
            parts = task_id.split('_')
            dataset_name = parts[0]
            run_id = int(parts[3].replace('run', ''))
            core_ratio = float(parts[4].replace('ratio', ''))
            
            # 데이터셋 로드 (캐시 사용)
            if dataset_name not in loaded_datasets:
                try:
                    print(f"\n📊 {dataset_name} 데이터셋 로드 중...")
                    G, true_labels, size_category, is_overlapping = self.load_dataset(dataset_name)
                    loaded_datasets[dataset_name] = (G, true_labels, size_category, is_overlapping)
                    print(f"   ✅ 로드 완료: {G.number_of_nodes():,} nodes, overlapping={is_overlapping}")
                except Exception as e:
                    print(f"   ❌ 로드 실패: {e}")
                    continue
            
            G, true_labels, size_category, is_overlapping = loaded_datasets[dataset_name]
            
            # 실험 실행
            result = self.run_single_experiment(
                dataset_name, core_ratio, G, true_labels, is_overlapping, run_id
            )
            
            if result:
                batch_results.append(result)
            
            # 완료 표시
            checkpoint.completed_tasks.add(task_id)
            task_count += 1
            
            # 진행 상황 보고
            completed_count = checkpoint.get_completed_count()
            if task_count % 5 == 0:  # 5개마다 보고
                progress.report(completed_count, f"{dataset_name} ratio={core_ratio:.1f}")
            
            # 체크포인트 저장 (5개마다)
            if task_count % self.checkpoint_interval == 0 or task_count == len(remaining_tasks):
                print(f"💾 체크포인트 저장 중... ({task_count}/{len(remaining_tasks)})")
                checkpoint.save_checkpoint(results=batch_results)
                batch_results = []  # 배치 초기화
    
    def run_all_experiments(self) -> pd.DataFrame:
        """모든 실험 실행"""
        print("🚀 Core Ratio 최적화 실험 시작")
        print("=" * 60)
        
        # 체크포인트 초기화
        checkpoint = ExperimentCheckpoint(
            "core_ratio_experiment",
            checkpoint_dir=str(self.exp_dir / 'checkpoints')
        )
        
        # 메타데이터 설정
        experiment_metadata = {
            'experiment': 'core_ratio_optimization',
            'centrality_method': self.centrality_method,
            'core_ratios': self.core_ratios,
            'repeat_count': self.repeat_count,
            'datasets': self.datasets,
            'dataset_overlapping': self.dataset_overlapping,
            'checkpoint_interval': self.checkpoint_interval,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        checkpoint.save_checkpoint(metadata=experiment_metadata)
        
        try:
            # 실험 실행
            self.run_with_checkpoint(checkpoint)
            
            # 최종 결과
            results_df = checkpoint.get_results_dataframe()
            
            if not results_df.empty:
                # 최종 결과 저장
                output_file = self.results_dir / 'raw' / 'over_exp1_core_ratio_results.csv'
                checkpoint.export_final_results(str(output_file), experiment_metadata)
                
                print(f"✅ 실험 완료: {len(results_df)}개 결과")
                return results_df
            else:
                print("❌ 실험 결과가 없습니다.")
                return pd.DataFrame()
                
        except KeyboardInterrupt:
            print(f"\n⏸️ 실험 중단됨")
            print(f"📁 진행 상황 저장됨 - 나중에 이어서 진행 가능")
            return checkpoint.get_results_dataframe()

def main():
    """메인 실행 함수"""
    experiment = CoreRatioExperiment()
    
    print("🎯 실험 설정:")
    print(f"   데이터셋: {experiment.datasets}")
    print(f"   Overlapping: {experiment.dataset_overlapping}")
    print(f"   Core Ratios: {len(experiment.core_ratios)}개")
    print(f"   반복 횟수: {experiment.repeat_count}")
    print(f"   체크포인트: {experiment.checkpoint_interval}개마다 저장")
    
    # 실험 실행
    results_df = experiment.run_all_experiments()
    
    if not results_df.empty:
        print(f"🎉 완료: {len(results_df)}개 결과")
        print(f"📁 결과: results/raw/exp1_core_ratio_results.csv")
        return True
    else:
        print("🔄 체크포인트에서 이어서 진행하려면 다시 실행하세요.")
        return False

if __name__ == "__main__":
    # 🔧 데이터셋별 overlapping 여부 설정 (여기서 수정하세요!)
    experiment = CoreRatioExperiment()
    
    experiment.dataset_overlapping = {
        'karate': False,     # ← True/False로 수정
        'cora': False,      
        'citeseer': False,  
        'pubmed': False,      
        'com-dblp': True,  
        'com-youtube': True,  
        'com-amazon': True  
    }
    
    success = main()
    if success:
        print(f"🚀 다음: 실험 2 또는 결과 분석!")