import psutil
import torch
import os

def get_system_specs():
    """Mendapatkan spesifikasi sistem untuk konfigurasi dinamis"""
    
    # RAM informasi
    total_ram = psutil.virtual_memory().total / (1024**3)  # Convert ke GB
    available_ram = psutil.virtual_memory().available / (1024**3)
    
    # CPU informasi
    cpu_count = psutil.cpu_count(logical=True)
    cpu_physical = psutil.cpu_count(logical=False)
    
    # GPU informasi
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    
    return {
        "total_ram_gb": total_ram,
        "available_ram_gb": available_ram,
        "cpu_cores": cpu_count,
        "cpu_physical_cores": cpu_physical,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name
    }

# Fungsi untuk menentukan konfigurasi optimal berdasarkan spesifikasi sistem

def get_optimal_config():
    """Menentukan konfigurasi optimal berdasarkan RAM"""
    specs = get_system_specs()
    ram_gb = specs["total_ram_gb"]
    
    if ram_gb >= 32:
        config = {
            "llm_model": "llama3.2:3b",
            "embedding_model": "nomic-embed-text-v2-moe",
            "chunk_size": 800,
            "chunk_overlap": 100,
            "batch_size": 64,
            "k_retrieval": 12,
            "max_pages": None,
            "num_ctx": 4096,
            "num_predict": 2048
        }
    elif ram_gb >= 16:
        config = {
            "llm_model": "gemma4:e4b",
            "embedding_model": "nomic-embed-text-v2-moe",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "batch_size": 32,
            "k_retrieval": 10,
            "max_pages": 50,
            "num_ctx": 2048,
            "num_predict": 1024
        }
    else:
        config = {
            "llm_model": "gemma4:e4b",
            "embedding_model": "nomic-embed-text-v2-moe",
            "chunk_size": 300,
            "chunk_overlap": 30,
            "batch_size": 16,
            "k_retrieval": 8,
            "max_pages": 30,
            "num_ctx": 1024,
            "num_predict": 512
        }
    
    config["system_info"] = specs
    return config

def print_system_info():
    """Print sistem info untuk debugging"""
    specs = get_system_specs()
    config = get_optimal_config()
    
    print("="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    print(f"RAM Total: {specs['total_ram_gb']:.1f} GB")
    print(f"RAM Available: {specs['available_ram_gb']:.1f} GB")
    print(f"CPU Cores: {specs['cpu_cores']} (Physical: {specs['cpu_physical_cores']})")
    print(f"GPU Available: {specs['gpu_available']}")
    if specs['gpu_available']:
        print(f"GPU Name: {specs['gpu_name']}")
    
    print("\n" + "="*50)
    print("OPTIMAL CONFIGURATION")
    print("="*50)
    print(f"LLM Model: {config['llm_model']}")
    print(f"Chunk Size: {config['chunk_size']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"K Retrieval: {config['k_retrieval']}")