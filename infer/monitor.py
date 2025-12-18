import os
import json
import time
import tempfile
import threading
import glob
import shutil
from datetime import datetime

# 尝试导入 rich，如果环境没有安装，则提供 Dummy 实现
try:
    from rich.live import Live
    from rich.table import Table
    from rich.console import Console
    from rich.layout import Layout
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

class StatusMonitor:
    def __init__(self, world_size, experiment_id):
        self.world_size = world_size
        self.experiment_id = experiment_id
        self.running = False
        self.thread = None
        self.live = None
        # 使用临时目录存储状态文件，避免污染项目目录
        self.status_dir = os.path.join('./tmp', f"epba_status_{experiment_id}")
        
        # 清理旧的状态文件
        if os.path.exists(self.status_dir):
            shutil.rmtree(self.status_dir)
        os.makedirs(self.status_dir, exist_ok=True)
        
        if HAS_RICH:
            self.console = Console()

    def start(self):
        if not HAS_RICH:
            return
        self.running = True
        self.live = Live(self.generate_table(), console=self.console, refresh_per_second=10)
        self.live.start()
        self.thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.live:
            self.live.stop()
        # 清理临时目录
        try:
            if os.path.exists(self.status_dir):
                shutil.rmtree(self.status_dir)
        except:
            pass

    def log(self, message):
        """
        在 Live 模式下安全打印日志，避免打乱表格
        """
        if self.live:
            self.live.console.print(message)
        else:
            print(message)

    def generate_table(self):
        table = Table(box=box.ROUNDED, title=f"EPBA Inference Monitor (Exp: {self.experiment_id})", width=None)
        table.add_column("Rank", justify="center", style="cyan", no_wrap=True, width=4)
        table.add_column("Current Task", style="magenta", width=12)
        table.add_column("Progress", justify="right", style="green", width=8)
        table.add_column("Level", justify="right", style="yellow", width=10)
        table.add_column("Current Step", style="blue")
        table.add_column("Last Update", style="dim", width=10)

        # 收集所有 Rank 的数据
        rows = []
        for rank in range(self.world_size):
            file_path = os.path.join(self.status_dir, f"rank_{rank}.json")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        rows.append(data)
                except:
                    # 读取冲突时的容错
                    rows.append({"rank": rank, "current_task": "Reading...", "progress": "-", "level": "-", "current_step": "-", "timestamp": 0})
            else:
                rows.append({"rank": rank, "current_task": "Waiting...", "progress": "-", "level": "-", "current_step": "-", "timestamp": 0})
        
        rows.sort(key=lambda x: x.get("rank", -1))
        
        for row in rows:
            ts = row.get("timestamp", 0)
            dt_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts > 0 else "-"
            table.add_row(
                str(row.get("rank", "-")),
                str(row.get("current_task", "-")),
                str(row.get("progress", "-")),
                str(row.get("level", "-")),
                str(row.get("current_step", "-")),
                dt_str
            )
        return table

    def _refresh_loop(self):
        while self.running:
            try:
                self.live.update(self.generate_table())
            except Exception:
                pass
            time.sleep(0.1)

class StatusReporter:
    def __init__(self, rank, world_size, experiment_id, monitor=None):
        self.rank = rank
        self.monitor = monitor # 只有 Rank 0 持有 monitor 实例
        self.status_dir = os.path.join(tempfile.gettempdir(), f"epba_status_{experiment_id}")
        os.makedirs(self.status_dir, exist_ok=True)
        self.file_path = os.path.join(self.status_dir, f"rank_{rank}.json")
        
        self.state = {
            "rank": rank,
            "current_task": "Initializing",
            "progress": "0/0",
            "level": "-",
            "current_step": "Startup",
            "timestamp": time.time()
        }
        self.last_write_time = 0
        self.flush()

    def update(self, **kwargs):
        """
        更新状态并写入文件。
        为了避免过高的 I/O 频率，可以根据 key 的重要性或时间间隔来决定是否立即写入。
        """
        updated = False
        for k, v in kwargs.items():
            if self.state.get(k) != v:
                self.state[k] = v
                updated = True
        
        self.state["timestamp"] = time.time()
        
        # 策略：如果 task, progress, level 变化，或者距离上次写入超过0.1s，则写入
        force_keys = ['current_task', 'progress', 'level']
        is_force = any(k in kwargs for k in force_keys)
        
        if updated:
            now = time.time()
            if is_force or (now - self.last_write_time > 0.1):
                self.flush()

    def flush(self):
        try:
            # 原子写入：写临时文件 -> 重命名
            tmp_file = self.file_path + ".tmp"
            with open(tmp_file, 'w') as f:
                json.dump(self.state, f)
            os.replace(tmp_file, self.file_path)
            self.last_write_time = time.time()
        except Exception:
            pass 

    def log(self, message):
        """
        统一日志接口
        """
        msg = f"[Rank {self.rank}] {message}"
        if self.monitor:
            # Rank 0 通过 monitor 在 Live 界面上方输出
            self.monitor.log(msg)
        else:
            # 其他 Rank 正常输出，通常会被重定向或忽略，但不会干扰 Rank 0 的界面
            print(msg)