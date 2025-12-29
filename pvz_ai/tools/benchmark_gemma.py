# -*- coding: utf-8 -*-
"""
Benchmark Gemma Inference Speed
Test tốc độ phản hồi của AI model
"""

import time
import statistics
from pathlib import Path


def benchmark_pytorch(model_path: str, num_runs: int = 20):
    """Benchmark PyTorch model"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading PyTorch model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # Tools definition with full docstrings
    def plant(plant_type: str, row: int, col: int):
        """
        Plant a plant at grid position.
        
        Args:
            plant_type: Type of plant to place
            row: Row index 0-4
            col: Column index 0-8
        
        Returns:
            result: Action result
        """
        return {"result": "planted"}
    
    def wait():
        """
        Wait and do nothing.
        
        Returns:
            result: Action result
        """
        return {"result": "waiting"}
    
    TOOLS = [plant, wait]
    SYSTEM_MSG = "PvZ bot. Choose action based on game state."
    
    test_cases = [
        "PLANTS:[]. ZOMBIES:[]. SEEDS:[(pea_shooter,ready)]",
        "PLANTS:[(pea_shooter,2,0)]. ZOMBIES:[(zombie,2,7)]. SEEDS:[(pea_shooter,cooldown)]",
        "PLANTS:[(pea_shooter,2,0),(pea_shooter,2,1)]. ZOMBIES:[(zombie,2,6)]. SEEDS:[(pea_shooter,ready)]",
    ]
    
    def run_inference(game_state):
        messages = [
            {"role": "developer", "content": SYSTEM_MSG},
            {"role": "user", "content": game_state},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tools=TOOLS, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
        
        start = time.perf_counter()
        out = model.generate(
            **inputs.to(model.device),
            max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        end = time.perf_counter()
        
        return end - start
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        run_inference(test_cases[0])
    
    # Benchmark
    print(f"\nRunning {num_runs} iterations...")
    times = []
    for i in range(num_runs):
        game_state = test_cases[i % len(test_cases)]
        t = run_inference(game_state)
        times.append(t)
        print(f"  [{i+1}/{num_runs}] {t*1000:.1f}ms")
    
    return times


def benchmark_openvino(model_path: str, num_runs: int = 20):
    """Benchmark OpenVINO model"""
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer
    
    print(f"Loading OpenVINO model from {model_path}...")
    model = OVModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # Tools definition with full docstrings
    def plant(plant_type: str, row: int, col: int):
        """
        Plant a plant at grid position.
        
        Args:
            plant_type: Type of plant to place
            row: Row index 0-4
            col: Column index 0-8
        
        Returns:
            result: Action result
        """
        return {"result": "planted"}
    
    def wait():
        """
        Wait and do nothing.
        
        Returns:
            result: Action result
        """
        return {"result": "waiting"}
    
    TOOLS = [plant, wait]
    SYSTEM_MSG = "PvZ bot. Choose action based on game state."
    
    test_cases = [
        "PLANTS:[]. ZOMBIES:[]. SEEDS:[(pea_shooter,ready)]",
        "PLANTS:[(pea_shooter,2,0)]. ZOMBIES:[(zombie,2,7)]. SEEDS:[(pea_shooter,cooldown)]",
        "PLANTS:[(pea_shooter,2,0),(pea_shooter,2,1)]. ZOMBIES:[(zombie,2,6)]. SEEDS:[(pea_shooter,ready)]",
    ]
    
    def run_inference(game_state):
        messages = [
            {"role": "developer", "content": SYSTEM_MSG},
            {"role": "user", "content": game_state},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tools=TOOLS, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
        
        start = time.perf_counter()
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id,
        )
        end = time.perf_counter()
        
        return end - start
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        run_inference(test_cases[0])
    
    # Benchmark
    print(f"\nRunning {num_runs} iterations...")
    times = []
    for i in range(num_runs):
        game_state = test_cases[i % len(test_cases)]
        t = run_inference(game_state)
        times.append(t)
        print(f"  [{i+1}/{num_runs}] {t*1000:.1f}ms")
    
    return times


def print_stats(times, name):
    """Print benchmark statistics"""
    print(f"\n{'='*50}")
    print(f"BENCHMARK RESULTS: {name}")
    print(f"{'='*50}")
    print(f"  Runs: {len(times)}")
    print(f"  Mean: {statistics.mean(times)*1000:.1f}ms")
    print(f"  Median: {statistics.median(times)*1000:.1f}ms")
    print(f"  Min: {min(times)*1000:.1f}ms")
    print(f"  Max: {max(times)*1000:.1f}ms")
    print(f"  Std: {statistics.stdev(times)*1000:.1f}ms")
    print(f"  FPS: {1/statistics.mean(times):.1f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark Gemma inference')
    parser.add_argument('model_path', help='Path to model')
    parser.add_argument('-n', '--runs', type=int, default=20, help='Number of runs')
    parser.add_argument('-t', '--type', choices=['pytorch', 'openvino', 'auto'], default='auto')
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    
    # Auto detect model type
    if args.type == 'auto':
        if (model_path / 'openvino_model.xml').exists():
            model_type = 'openvino'
        else:
            model_type = 'pytorch'
    else:
        model_type = args.type
    
    print(f"Model type: {model_type}")
    
    if model_type == 'openvino':
        times = benchmark_openvino(str(model_path), args.runs)
        print_stats(times, "OpenVINO")
    else:
        times = benchmark_pytorch(str(model_path), args.runs)
        print_stats(times, "PyTorch")


if __name__ == "__main__":
    main()
