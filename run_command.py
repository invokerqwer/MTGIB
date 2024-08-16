import subprocess
import datetime
import os

# ��ȡ��ǰʱ���
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# ��������Ͷ�Ӧ����־�ļ���
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

# ������־Ŀ¼
if not os.path.exists('Logs'):
    os.makedirs('Logs')

# ͬʱִ�����������־
processes = []
for command, log_file in zip(commands, log_files):
    with open(log_file, 'w') as f:
        process = subprocess.Popen(command, shell=True, stdout=f, stderr=subprocess.STDOUT)
        processes.append(process)

# �ȴ����н������
for process in processes:
    process.wait()
