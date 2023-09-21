#!/bin/bash

# 获取包含"plan"的进程的PID列表
pid_list=$(pgrep -f ".*nnUNetv2_plan.*")

# 打印包含"plan"的进程
echo "包含 'nnUNetv2_plan' 的进程："
ps -ef | grep "nnUNetv2_plan"

# 检查进程是否存在的函数
function is_process_running {
  if ps -p $1 > /dev/null; then
    return 0
  else
    return 1
  fi
}

# 循环检查进程是否存在，如果不存在则执行test.py
for pid in $pid_list
do
  echo "等待进程 $pid 结束"
  while is_process_running $pid; do
    sleep 10
  done
  echo "进程 $pid 已经结束"
done

# 执行run_training_Flare.py
echo "所有进程已经结束，执行 run_training_Flare.py fold0"
python run_training_Flare.py 2 3d_fullres 0 -tr nnUNetTrainerFlare

exit 0
