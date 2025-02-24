import os
import psutil
import platform
import resource
import socket
from sys import getsizeof

def get_model_size_gb(model):
    return getsizeof(model) / (1024 )


def get_memory_usage(device='cpu'):
    """
    Function to get the memory usage of the given device.
    It works for both CPU and GPU (using CUDA).
    
    Parameters:
        device (str): Either 'cpu' or 'cuda' for GPU.
        
    Returns:
        memory_usage (str): Memory usage in a readable format.
    """


    
    if device == 'cuda':
        import torch
        # Track memory usage for GPU
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Memory in MB
        memory_cached = torch.cuda.memory_reserved() / (1024 ** 2)  # Memory in MB
        return f"Memory Allocated (GPU): {memory_allocated:.2f} MB, Memory Cached (GPU): {memory_cached:.2f} MB"
    
    # Track memory usage for CPU
    import psutil
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Memory usage in MB
    return f"Memory Usage (CPU): {memory_usage:.2f} MB"

def get_system_and_usage_data():
    # 获取当前进程的资源使用情况, usage.ru_maxrss, not correct
    # usage = resource.getrusage(resource.RUSAGE_SELF)
    # memory_gb = usage.ru_maxrss / (1024 * 1024)  # 转换为 GB
    # memory_gb = round(memory_gb, 2)
    process = psutil.Process()
    # Get memory information
    memory_info = process.memory_info()

    # The RSS (Resident Set Size) is the non-swapped physical memory the process is using
    memory_gb = memory_info.rss / (1024 * 1024 * 1024)  # Convert to GB
    memory_gb = round(memory_gb, 2)
    # memory_vms = memory_info.vms / (1024 * 1024*1024)  # Virtual memory size in GB

    # 获取 CPU 使用情况
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count(logical=True)

    # 获取内存使用情况
    virtual_memory = psutil.virtual_memory()
    total_memory_gb = round(virtual_memory.total / (1024**3), 2)
    available_memory_gb = round(virtual_memory.available / (1024**3), 2)
    memory_usage_percent = virtual_memory.percent

    # 获取磁盘使用情况
    disk_usage = psutil.disk_usage("/")
    total_disk_gb = round(disk_usage.total / (1024**3), 2)
    used_disk_gb = round(disk_usage.used / (1024**3), 2)
    free_disk_gb = round(disk_usage.free / (1024**3), 2)
    disk_usage_percent = disk_usage.percent

    # 获取网络信息
    net_io = psutil.net_io_counters()
    bytes_sent_mb = round(net_io.bytes_sent / (1024**2), 2)
    bytes_recv_mb = round(net_io.bytes_recv / (1024**2), 2)

    # 获取操作系统信息
    # system_info = {
    #     "system": platform.system(),
    #     "node_name": platform.node(),
    #     "release": platform.release(),
    #     "version": platform.version(),
    #     "machine": platform.machine(),
    #     "processor": platform.processor(),
    #     "hostname": socket.gethostname(),
    #     "ip_address": socket.gethostbyname(socket.gethostname()),
    # }

    # 组装系统信息
    system_data = {
        "total_memory_gb": total_memory_gb,
        "available_memory_gb": available_memory_gb,
        "memory_usage_percent": memory_usage_percent,
        "disk_total_gb": total_disk_gb,
        "disk_used_gb": used_disk_gb,
        "disk_free_gb": free_disk_gb,
        "disk_usage_percent": disk_usage_percent,
        "network_sent_mb": bytes_sent_mb,
        "network_received_mb": bytes_recv_mb,
    }

    # 组装当前进程的使用情况
    current_usage = {
        "memory_usage_gb": memory_gb,
        "cpu_usage_percent": cpu_percent,
        "cpu_count": cpu_count,
    }

    return {
        "current_usage": current_usage,
        "system_data": system_data,
        # "system_info": system_info,
    }