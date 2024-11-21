import torch
import time

# 检查是否有可用的GPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # 指定设备为GPU
else:
    raise ValueError("No GPU available, please check your installation.")

# 定义矩阵的大小
matrix_size = 1000

# 在GPU上创建随机矩阵
matrix_a = torch.rand(matrix_size, matrix_size, device=device)
matrix_b = torch.rand(matrix_size, matrix_size, device=device)

# 将CUDA事件用于计时cond
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# 记录开始时间
start_event.record()

# 在GPU上执行矩阵乘法
result = torch.matmul(matrix_a, matrix_b)

# 记录结束时间
end_event.record()

# 等待CUDA事件完成
torch.cuda.synchronize()  # 确保所有CUDA操作完成

# 计算并输出所需的时间
elapsed_time = start_event.elapsed_time(end_event)  # 以毫秒为单位
print(f"Matrix multiplication on GPU took {elapsed_time} ms")

# 如果需要，可以将结果转移到CPU
result_cpu = result.cpu()