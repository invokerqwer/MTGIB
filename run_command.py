import subprocess
import datetime
import os

# 获取当前时间戳
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# 定义命令和对应的日志文件名
commands = [
    "python Experiments/Training.py --select_task_list 'CYP2D6', 'CYP2C9', 'PPB', 'ESOL'",
    "python Experiments/Training.py --select_task_list 'CYP2D6', 'CYP2C9', 'PPB', 'ESOL' --use_uncertainty False",
    "python Experiments/Training.py --select_task_list 'CYP2D6', 'CYP2C9', 'PPB', 'ESOL' --use_gib False",
    "python Experiments/Training.py --select_task_list 'CYP2D6', 'CYP2C9', 'PPB', 'ESOL' --use_uncertainty False --use_gib False"
]

log_files = [
    f"Logs/command1_{timestamp}.txt",
    f"Logs/command2_{timestamp}.txt",
    f"Logs/command3_{timestamp}.txt",
    f"Logs/command4_{timestamp}.txt"
]

# 创建日志目录
if not os.path.exists('Logs'):
    os.makedirs('Logs')

# 同时执行命令并保存日志
processes = []
for command, log_file in zip(commands, log_files):
    with open(log_file, 'w') as f:
        process = subprocess.Popen(command, shell=True, stdout=f, stderr=subprocess.STDOUT)
        processes.append(process)

# 等待所有进程完成
for process in processes:
    process.wait()
