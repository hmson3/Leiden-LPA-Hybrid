"""
실험 중간 저장 및 재시작 유틸리티
"""
import os
import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time
from datetime import datetime

class ExperimentCheckpoint:
    """실험 중간 저장 및 재시작 관리 클래스"""
    
    def __init__(self, experiment_name: str, checkpoint_dir: str = "checkpoints"):
        self.experiment_name = experiment_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 체크포인트 파일 경로들
        self.progress_file = self.checkpoint_dir / f"{experiment_name}_progress.json"
        self.results_file = self.checkpoint_dir / f"{experiment_name}_results.pkl"
        self.metadata_file = self.checkpoint_dir / f"{experiment_name}_metadata.json"
        
        # 진행 상태
        self.completed_tasks = set()
        self.results = []
        self.metadata = {}
        
        # 기존 체크포인트 로드
        self.load_checkpoint()
    
    def save_checkpoint(self, current_task: str = None, 
                       results: List[Dict] = None, 
                       metadata: Dict = None):
        """현재 진행 상태 저장"""
        try:
            # 진행 상태 저장
            if current_task:
                self.completed_tasks.add(current_task)
            
            progress_data = {
                'experiment_name': self.experiment_name,
                'completed_tasks': list(self.completed_tasks),
                'total_results': len(self.results),
                'last_saved': datetime.now().isoformat(),
                'checkpoint_version': '1.0'
            }
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            
            # 결과 저장
            if results is not None:
                self.results.extend(results)
            
            with open(self.results_file, 'wb') as f:
                pickle.dump(self.results, f)
            
            # 메타데이터 저장
            if metadata is not None:
                self.metadata.update(metadata)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            print(f"📁 체크포인트 저장됨: {len(self.completed_tasks)}개 작업 완료, {len(self.results)}개 결과")
            
        except Exception as e:
            print(f"⚠️ 체크포인트 저장 실패: {e}")
    
    def load_checkpoint(self) -> bool:
        """기존 체크포인트 로드"""
        try:
            # 진행 상태 로드
            if self.progress_file.exists():
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                self.completed_tasks = set(progress_data.get('completed_tasks', []))
                print(f"📂 기존 진행 상태 로드: {len(self.completed_tasks)}개 작업 완료")
            
            # 결과 로드
            if self.results_file.exists():
                with open(self.results_file, 'rb') as f:
                    self.results = pickle.load(f)
                print(f"📊 기존 결과 로드: {len(self.results)}개 결과")
            
            # 메타데이터 로드
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                print(f"📋 메타데이터 로드됨")
            
            return len(self.completed_tasks) > 0
            
        except Exception as e:
            print(f"⚠️ 체크포인트 로드 실패: {e}")
            return False
    
    def is_completed(self, task_id: str) -> bool:
        """특정 작업이 완료되었는지 확인"""
        return task_id in self.completed_tasks
    
    def get_remaining_tasks(self, all_tasks: List[str]) -> List[str]:
        """남은 작업 목록 반환"""
        return [task for task in all_tasks if task not in self.completed_tasks]
    
    def get_completed_count(self) -> int:
        """완료된 작업 수 반환"""
        return len(self.completed_tasks)
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """결과를 DataFrame으로 반환"""
        if self.results:
            return pd.DataFrame(self.results)
        else:
            return pd.DataFrame()
    
    def clear_checkpoint(self):
        """체크포인트 파일들 삭제"""
        try:
            for file_path in [self.progress_file, self.results_file, self.metadata_file]:
                if file_path.exists():
                    file_path.unlink()
            print(f"🗑️ 체크포인트 파일들 삭제됨")
        except Exception as e:
            print(f"⚠️ 체크포인트 삭제 실패: {e}")
    
    def export_final_results(self, output_path: str, 
                           experiment_metadata: Optional[Dict] = None):
        """최종 결과를 CSV로 내보내기"""
        try:
            results_df = self.get_results_dataframe()
            
            if results_df.empty:
                print("❌ 내보낼 결과가 없습니다.")
                return
            
            # 최종 메타데이터 준비
            final_metadata = self.metadata.copy()
            if experiment_metadata:
                final_metadata.update(experiment_metadata)
            
            final_metadata.update({
                'total_results': len(results_df),
                'completed_tasks': len(self.completed_tasks),
                'export_time': datetime.now().isoformat()
            })
            
            # CSV로 저장 (메타데이터 포함)
            with open(output_path, 'w') as f:
                # 메타데이터를 주석으로 추가
                f.write("# Final Experiment Results\n")
                for key, value in final_metadata.items():
                    f.write(f"# {key}: {value}\n")
                f.write("# \n")
                
                # DataFrame 저장
                results_df.to_csv(f, index=False)
            
            print(f"✅ 최종 결과 저장: {output_path}")
            print(f"   총 {len(results_df)}개 결과, {len(self.completed_tasks)}개 작업 완료")
            
        except Exception as e:
            print(f"❌ 결과 내보내기 실패: {e}")

def create_task_id(dataset: str, method: str, run_id: int, **kwargs) -> str:
    """작업 고유 ID 생성"""
    base_id = f"{dataset}_{method}_run{run_id}"
    
    # 추가 파라미터가 있으면 포함
    for key, value in kwargs.items():
        base_id += f"_{key}{value}"
    
    return base_id

def estimate_remaining_time(completed_count: int, total_count: int, 
                          elapsed_time: float) -> str:
    """남은 시간 추정"""
    if completed_count == 0:
        return "알 수 없음"
    
    avg_time_per_task = elapsed_time / completed_count
    remaining_tasks = total_count - completed_count
    remaining_seconds = avg_time_per_task * remaining_tasks
    
    hours = int(remaining_seconds // 3600)
    minutes = int((remaining_seconds % 3600) // 60)
    
    if hours > 0:
        return f"약 {hours}시간 {minutes}분"
    else:
        return f"약 {minutes}분"

class ProgressReporter:
    """실험 진행 상황 리포터"""
    
    def __init__(self, total_tasks: int, report_interval: int = 10):
        self.total_tasks = total_tasks
        self.report_interval = report_interval
        self.start_time = time.time()
        self.last_report_time = self.start_time
    
    def report(self, completed_tasks: int, current_task: str = ""):
        """진행 상황 보고"""
        current_time = time.time()
        
        # 일정 간격으로만 보고
        if (completed_tasks % self.report_interval == 0 or 
            completed_tasks == self.total_tasks or
            current_time - self.last_report_time > 300):  # 5분마다
            
            progress_pct = (completed_tasks / self.total_tasks) * 100
            elapsed_time = current_time - self.start_time
            
            remaining_time = estimate_remaining_time(
                completed_tasks, self.total_tasks, elapsed_time
            )
            
            print(f"📈 진행 상황: {completed_tasks}/{self.total_tasks} ({progress_pct:.1f}%)")
            print(f"   현재 작업: {current_task}")
            print(f"   경과 시간: {elapsed_time/60:.1f}분")
            print(f"   남은 시간: {remaining_time}")
            print()
            
            self.last_report_time = current_time

# 실험별 체크포인트 래퍼 함수들
def safe_experiment_run(experiment_func, experiment_name: str, 
                       checkpoint_interval: int = 5, **kwargs):
    """안전한 실험 실행 (자동 체크포인트 포함)"""
    checkpoint = ExperimentCheckpoint(experiment_name)
    
    try:
        # 이전 실행이 있었는지 확인
        if checkpoint.get_completed_count() > 0:
            print(f"🔄 이전 실험 이어서 진행: {checkpoint.get_completed_count()}개 작업 완료됨")
            resume = input("이어서 진행하시겠습니까? (y/n): ").lower() == 'y'
            
            if not resume:
                print("🗑️ 기존 진행 상황을 삭제하고 새로 시작합니다.")
                checkpoint.clear_checkpoint()
                checkpoint = ExperimentCheckpoint(experiment_name)
        
        # 실험 실행
        results = experiment_func(checkpoint=checkpoint, **kwargs)
        
        # 최종 저장
        if results:
            checkpoint.save_checkpoint(results=results)
        
        return checkpoint.get_results_dataframe()
        
    except KeyboardInterrupt:
        print(f"\n⏸️ 실험이 중단되었습니다.")
        print(f"📁 진행 상황이 저장되었습니다. 나중에 이어서 진행할 수 있습니다.")
        return checkpoint.get_results_dataframe()
        
    except Exception as e:
        print(f"❌ 실험 중 오류 발생: {e}")
        print(f"📁 현재까지의 진행 상황은 저장되었습니다.")
        return checkpoint.get_results_dataframe()