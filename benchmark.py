"""
Performance comparison script to benchmark gRPC vs TCP/MPI backends
"""

import time
import numpy as np
import mlx.core as mx
from transformers import AutoTokenizer
from generate import generate
from utils.utils import load_model, log_debug
from config import config as cfg

def benchmark_generation(backend_name, num_iterations=5, max_length=100):
    """Benchmark text generation with specified backend."""
    print(f"\n{'='*50}")
    print(f"Benchmarking {backend_name.upper()} Backend")
    print(f"{'='*50}")
    
    model_path = cfg.get("model_path")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load first half of the model
    print("Loading model...")
    model = load_model(model_path, 0, 35)
    
    # Test prompts
    test_prompts = [
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
        "How does artificial intelligence work?",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis."
    ]
    
    total_times = []
    total_tokens = []
    network_times = []
    
    for i, prompt in enumerate(test_prompts[:num_iterations]):
        print(f"\nIteration {i+1}/{num_iterations}: '{prompt[:30]}...'")
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
        
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        start_time = time.time()
        
        # Generate response
        response = generate(formatted_prompt, model, tokenizer, max_length=max_length)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Count tokens in response
        response_tokens = len(tokenizer.encode(response))
        
        total_times.append(generation_time)
        total_tokens.append(response_tokens)
        
        tokens_per_second = response_tokens / generation_time if generation_time > 0 else 0
        
        print(f"  Generated {response_tokens} tokens in {generation_time:.2f}s")
        print(f"  Speed: {tokens_per_second:.2f} tokens/second")
        print(f"  Response: {response[:100]}...")
    
    # Calculate statistics
    avg_time = np.mean(total_times)
    avg_tokens = np.mean(total_tokens)
    avg_tps = np.mean([tokens/time for tokens, time in zip(total_tokens, total_times)])
    std_tps = np.std([tokens/time for tokens, time in zip(total_tokens, total_times)])
    
    print(f"\n{backend_name.upper()} Performance Summary:")
    print(f"  Average generation time: {avg_time:.2f}s ± {np.std(total_times):.2f}s")
    print(f"  Average tokens generated: {avg_tokens:.1f}")
    print(f"  Average tokens/second: {avg_tps:.2f} ± {std_tps:.2f}")
    print(f"  Total time: {sum(total_times):.2f}s")
    
    return {
        'backend': backend_name,
        'avg_time': avg_time,
        'avg_tokens': avg_tokens,
        'avg_tps': avg_tps,
        'std_tps': std_tps,
        'total_time': sum(total_times)
    }

def run_performance_comparison():
    """Run performance comparison between different backends."""
    print("🚀 LLM Inference Performance Comparison")
    print("This script compares the performance of different network backends")
    
    results = []
    
    # Test current backend
    current_backend = cfg.get('network_backend', 'unknown')
    result = benchmark_generation(current_backend)
    results.append(result)
    
    # Print comparison summary
    print(f"\n{'='*70}")
    print("PERFORMANCE COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    for result in results:
        print(f"{result['backend'].upper():>10}: {result['avg_tps']:>6.2f} ± {result['std_tps']:>5.2f} tokens/sec")
    
    if len(results) > 1:
        # Calculate improvements
        baseline = results[0]
        for result in results[1:]:
            improvement = ((result['avg_tps'] - baseline['avg_tps']) / baseline['avg_tps']) * 100
            print(f"\n{result['backend'].upper()} vs {baseline['backend'].upper()}:")
            print(f"  Speed improvement: {improvement:+.1f}%")
            
            time_saved = baseline['total_time'] - result['total_time']
            print(f"  Time saved: {time_saved:.2f}s ({time_saved/baseline['total_time']*100:.1f}%)")
    
    print(f"\n{'='*70}")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        iterations = int(sys.argv[1])
    else:
        iterations = 3
    
    if len(sys.argv) > 2:
        max_length = int(sys.argv[2])
    else:
        max_length = 50
    
    print(f"Running benchmark with {iterations} iterations, max_length={max_length}")
    
    try:
        results = run_performance_comparison()
        
        # Save results to file
        import json
        with open('benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to benchmark_results.json")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()