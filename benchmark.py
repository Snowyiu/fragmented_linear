from fragmentedlinear6 import FragmentedLinear
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from dataclasses import dataclass
import gc

@dataclass
class BenchmarkResult:
    forward_time: float
    backward_time: float
    peak_memory: float
    model_params: int

class BenchmarkSuite:
    def __init__(
        self,
        in_features: int = 4096,
        out_features: int = 4096,
        batch_size: int = 16,
        num_warmup: int = 10,
        num_iterations: int = 100,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _reset_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
    
    def _get_peak_memory(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB
        return 0.0
    
    def _count_parameters(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def benchmark_model(self, model: nn.Module) -> BenchmarkResult:
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        
        # Generate random data
        x = torch.randn(self.batch_size, self.in_features, device=self.device)
        y = torch.randn(self.batch_size, self.out_features, device=self.device)
        
        # Warmup
        for _ in range(self.num_warmup):
            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        self._reset_gpu_memory()
        
        # Benchmark forward pass
        forward_times = []
        backward_times = []
        
        for _ in range(self.num_iterations):
            # Forward pass timing
            start_time = time.perf_counter()
            output = model(x)
            torch.cuda.synchronize()
            forward_time = time.perf_counter() - start_time
            forward_times.append(forward_time)
            
            # Backward pass timing
            loss = criterion(output, y)
            start_time = time.perf_counter()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            backward_time = time.perf_counter() - start_time
            backward_times.append(backward_time)
        
        peak_memory = self._get_peak_memory()
        num_params = self._count_parameters(model)
        
        return BenchmarkResult(
            forward_time=np.mean(forward_times) * 1000,  # Convert to ms
            backward_time=np.mean(backward_times) * 1000,  # Convert to ms
            peak_memory=peak_memory,
            model_params=num_params
        )

    def run_comparison(
        self,
        fragment_sizes: List[int],
        selection_thresholds: List[float]
    ) -> Dict[str, List[BenchmarkResult]]:
        results = {
            "linear": [],
            "fragmented_train": [],
            "fragmented_inference": []
        }
        
        # Benchmark standard Linear layer
        linear_model = nn.Linear(self.in_features, self.out_features).to('cuda')
        linear_result = self.benchmark_model(linear_model)
        results["linear"] = [linear_result] * len(fragment_sizes)  # For plotting
        
        # Benchmark FragmentedLinear with different configurations
        for num_fragments in fragment_sizes:
            fragmented_model = FragmentedLinear(
                self.in_features,
                self.out_features,
                num_fragments=num_fragments,
                compressed_features=self.in_features//32
            ).to('cuda')
            fragmented_model.train()
            result = self.benchmark_model(fragmented_model)
            results["fragmented_train"].append(result)
            fragmented_model.eval()
            result = self.benchmark_model(fragmented_model)
            results["fragmented_inference"].append(result)
        
        return results

def plot_benchmark_results(
    fragment_sizes: List[int],
    results: Dict[str, List[BenchmarkResult]],
    title_prefix: str = ""
):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract metrics
    metrics = {
        "forward_time": (
            [r.forward_time for r in results["linear"]],
            [r.forward_time for r in results["fragmented_train"]],
            [r.forward_time for r in results["fragmented_inference"]]
        ),
        "backward_time": (
            [r.backward_time for r in results["linear"]],
            [r.backward_time for r in results["fragmented_train"]],
            [r.backward_time for r in results["fragmented_inference"]]
        ),
        "peak_memory": (
            [r.peak_memory for r in results["linear"]],
            [r.peak_memory for r in results["fragmented_train"]],
            [r.peak_memory for r in results["fragmented_inference"]]
        ),
        "model_params": (
            [r.model_params for r in results["linear"]],
            [r.model_params for r in results["fragmented_train"]],
            [r.model_params for r in results["fragmented_inference"]]
        )
    }
    
    # Plot settings
    titles = ["Forward Time (ms)", "Backward Time (ms)", 
              "Peak Memory Usage (MB)", "Number of Parameters"]
    axes = [ax1, ax2, ax3, ax4]
    colors = ['b-', 'r-', 'g-']
    labels = ['Linear', 'FragmentedLinear (Train)', 'FragmentedLinear (Inference)']
    
    for ax, title, (linear_metric, train_metric, inf_metric) in zip(axes, titles, metrics.values()):
        ax.plot(fragment_sizes, linear_metric, colors[0], label=labels[0])
        ax.plot(fragment_sizes, train_metric, colors[1], label=labels[1])
        ax.plot(fragment_sizes, inf_metric, colors[2], label=labels[2])
        ax.set_xlabel('Number of Fragments')
        ax.set_ylabel(title)
        ax.set_title(f"{title_prefix}{title}")
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    return fig

# Example usage
if __name__ == "__main__":
    
    # Test different model sizes
    sizes = [
        (2048, 2048),
        (4096, 4096),
        (8192, 8192)
    ]

    batch_sizes = [1, 8, 16]
    fragment_sizes = [2, 8, 32, 128]
    selection_thresholds = [0.9]

    for in_feat, out_feat in sizes:
        for batch_size in batch_sizes:
            print(f"\nBenchmarking with input_size={in_feat}, output_size={out_feat}, batch_size={batch_size}")
            
            suite = BenchmarkSuite(
                in_features=in_feat,
                out_features=out_feat,
                batch_size=batch_size,
                num_warmup=10,
                num_iterations=100
            )
            
            results = suite.run_comparison(fragment_sizes, selection_thresholds)
            
            # Plot results
            fig = plot_benchmark_results(
                fragment_sizes,
                results,
                title_prefix=f"Size: {in_feat}→{out_feat}, Batch: {batch_size}, "
            )
            plt.savefig(f'benchmark_i{in_feat}_o{out_feat}_b{batch_size}.png')
            plt.close(fig)
            
            # Print detailed results
            print("\nDetailed Results:")
            print("Linear Layer:")
            linear_result = results["linear"][0]
            print(f"  Forward time: {linear_result.forward_time:.2f}ms")
            print(f"  Backward time: {linear_result.backward_time:.2f}ms")
            print(f"  Peak memory: {linear_result.peak_memory:.1f}MB")
            print(f"  Parameters: {linear_result.model_params}")
            
            print("\nFragmentedLinear (Train) Results:")
            for i, result in enumerate(results["fragmented_train"]):
                print(f"\nFragments: {fragment_sizes[i]}")
                print(f"  Forward time: {result.forward_time:.2f}ms")
                print(f"  Backward time: {result.backward_time:.2f}ms")
                print(f"  Peak memory: {result.peak_memory:.1f}MB")
                print(f"  Parameters: {result.model_params}")
                
            print("\nFragmentedLinear (Inference) Results:")
            for i, result in enumerate(results["fragmented_inference"]):
                print(f"\nFragments: {fragment_sizes[i]}")
                print(f"  Forward time: {result.forward_time:.2f}ms")
                print(f"  Backward time: {result.backward_time:.2f}ms")
                print(f"  Peak memory: {result.peak_memory:.1f}MB")
                print(f"  Parameters: {result.model_params}")