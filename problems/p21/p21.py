import torch
from pathlib import Path
from max.torch import CustomOpLibrary


mojo_kernels = Path(__file__).parent / "op"
ops = CustomOpLibrary(mojo_kernels)


def embedding_mojo_1d(
    indices: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """1D coalesced embedding kernel"""
    batch_size, seq_len = indices.shape
    vocab_size, embed_dim = weights.shape

    output = torch.empty(
        (batch_size, seq_len, embed_dim),
        dtype=weights.dtype,
        device=weights.device,
    )

    if indices.dtype != torch.int32:
        indices = indices.to(torch.int32)

    embedding_op = ops.embedding[
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
        }
    ]
    embedding_op(output, indices, weights)
    return output


def embedding_mojo_2d(
    indices: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """2D non-coalesced embedding kernel"""
    batch_size, seq_len = indices.shape
    vocab_size, embed_dim = weights.shape

    output = torch.empty(
        (batch_size, seq_len, embed_dim),
        dtype=weights.dtype,
        device=weights.device,
    )

    if indices.dtype != torch.int32:
        indices = indices.to(torch.int32)

    embedding_op = ops.embedding_2d[
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
        }
    ]
    embedding_op(output, indices, weights)
    return output

def embedding_mojo_3d(
    indices: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """3D non-coalesced embedding kernel"""
    batch_size, seq_len = indices.shape
    vocab_size, embed_dim = weights.shape

    output = torch.empty(
        (batch_size, seq_len, embed_dim),
        dtype=weights.dtype,
        device=weights.device,
    )

    if indices.dtype != torch.int32:
        indices = indices.to(torch.int32)

    embedding_op = ops.embedding_3d[
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
        }
    ]
    embedding_op(output, indices, weights)
    return output

if __name__ == "__main__":
    print("Puzzle 21: Mojo Embedding Kernel Comparison")
    print("=" * 70)
    print()

    batch_size, seq_len, vocab_size, embed_dim = 8, 512, 10000, 512
    print(
        f"Configuration: B={batch_size}, L={seq_len}, V={vocab_size},"
        f" E={embed_dim}"
    )
    print("-" * 60)

    torch.manual_seed(42)
    indices = torch.randint(
        0, vocab_size, (batch_size, seq_len), device="cuda", dtype=torch.int32
    )

    embed_layer = torch.nn.Embedding(
        vocab_size, embed_dim, device="cuda", dtype=torch.float32
    )
    weights = embed_layer.weight.data

    print("Testing Correctness...")

    # PyTorch reference for correctness
    ref_output = embed_layer(indices)

    try:
        # Test 1D coalesced kernel
        mojo_1d_output = embedding_mojo_1d(indices, weights)
        max_diff_1d = (ref_output - mojo_1d_output).abs().max().item()
        print(f"   1D Coalesced - Max difference: {max_diff_1d:.2e}")

        # Test 2D non-coalesced kernel
        mojo_2d_output = embedding_mojo_2d(indices, weights)
        max_diff_2d = (ref_output - mojo_2d_output).abs().max().item()
        print(f"   2D Non-coalesced - Max difference: {max_diff_2d:.2e}")

        # Test 3D non-coalesced kernel
        mojo_3d_output = embedding_mojo_3d(indices, weights)
        max_diff_3d = (ref_output - mojo_3d_output).abs().max().item()
        print(f"   3D Non-coalesced - Max difference: {max_diff_3d:.2e}")

        if max_diff_1d < 1e-5 and max_diff_2d < 1e-5 and max_diff_3d < 1e-5:
            print("   ✅ All implementations CORRECT")
        else:
            print("   ❌ One or more implementations INCORRECT")
            exit(1)

    except Exception as e:
        print(f"   ❌ Implementation failed: {e}")
        exit(1)

    print()
    print("Running Mojo Built-in Benchmarks...")
    print("-" * 60)

    import subprocess
    
    benchmark_file = Path(__file__).parent / "bench_embedding.mojo"
    
    try:
        # Use 'mojo' from the environment, assuming it is available via pixi
        subprocess.run(["mojo", str(benchmark_file)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Mojo benchmark failed with exit code {e.returncode}")
        exit(1)
    except Exception as e:
        print(f"   ❌ Could not run Mojo benchmark: {e}")
        exit(1)

    print()
    print("Key Learning Points:")
    print("• Compare different GPU kernel implementations")
    print("• 1D vs 2D vs 3D grid patterns have different memory access")
    print("• Coalesced memory access should be faster")
    print("• Grid configuration affects GPU utilization")
