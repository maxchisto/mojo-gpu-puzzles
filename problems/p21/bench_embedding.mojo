from benchmark import Bench, BenchConfig, Bencher, BenchId
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from math import ceildiv
from sys import argv
from op.embedding import (
    embedding_kernel_coalesced,
    embedding_kernel_2d,
    embedding_kernel_3d,
)

comptime BATCH_SIZE = 8
comptime SEQ_LEN = 512
comptime VOCAB_SIZE = 10000
comptime EMBED_DIM = 512
comptime DTYPE = DType.float32


@parameter
fn benchmark_coalesced(mut b: Bencher) raises:
    var ctx = DeviceContext()

    comptime indices_layout = Layout.row_major(BATCH_SIZE, SEQ_LEN)
    comptime weights_layout = Layout.row_major(VOCAB_SIZE, EMBED_DIM)
    comptime out_layout = Layout.row_major(BATCH_SIZE, SEQ_LEN, EMBED_DIM)

    var out_buf = ctx.enqueue_create_buffer[DTYPE](
        BATCH_SIZE * SEQ_LEN * EMBED_DIM
    )
    var indices_buf = ctx.enqueue_create_buffer[DType.int32](
        BATCH_SIZE * SEQ_LEN
    )
    var weights_buf = ctx.enqueue_create_buffer[DTYPE](VOCAB_SIZE * EMBED_DIM)

    # Initialize buffers to avoid memory access faults with random indices
    ctx.enqueue_memset(out_buf, 0)
    ctx.enqueue_memset(indices_buf, 0)
    ctx.enqueue_memset(weights_buf, 0)

    var out_tensor = LayoutTensor[DTYPE, out_layout, MutAnyOrigin](
        out_buf.unsafe_ptr()
    )
    var indices_tensor = LayoutTensor[
        DType.int32, indices_layout, MutAnyOrigin
    ](indices_buf.unsafe_ptr())
    var weights_tensor = LayoutTensor[DTYPE, weights_layout, MutAnyOrigin](
        weights_buf.unsafe_ptr()
    )

    comptime THREADS_PER_BLOCK = 256
    var total_elements = BATCH_SIZE * SEQ_LEN * EMBED_DIM
    var blocks = max(1, ceildiv(total_elements, THREADS_PER_BLOCK))

    comptime kernel = embedding_kernel_coalesced[
        indices_layout,
        weights_layout,
        out_layout,
        BATCH_SIZE,
        SEQ_LEN,
        VOCAB_SIZE,
        EMBED_DIM,
        DTYPE,
    ]
    var compiled_kernel = ctx.compile_function[kernel, kernel]()

    @parameter
    @always_inline
    fn workflow(ctx: DeviceContext) raises:
        ctx.enqueue_function(
            compiled_kernel,
            out_tensor,
            indices_tensor,
            weights_tensor,
            grid_dim=(blocks,),
            block_dim=(THREADS_PER_BLOCK,),
        )

    b.iter_custom[workflow](ctx)
    ctx.synchronize()


@parameter
fn benchmark_2d(mut b: Bencher) raises:
    var ctx = DeviceContext()

    comptime indices_layout = Layout.row_major(BATCH_SIZE, SEQ_LEN)
    comptime weights_layout = Layout.row_major(VOCAB_SIZE, EMBED_DIM)
    comptime out_layout = Layout.row_major(BATCH_SIZE, SEQ_LEN, EMBED_DIM)

    var out_buf = ctx.enqueue_create_buffer[DTYPE](
        BATCH_SIZE * SEQ_LEN * EMBED_DIM
    )
    var indices_buf = ctx.enqueue_create_buffer[DType.int32](
        BATCH_SIZE * SEQ_LEN
    )
    var weights_buf = ctx.enqueue_create_buffer[DTYPE](VOCAB_SIZE * EMBED_DIM)

    # Initialize buffers to avoid memory access faults with random indices
    ctx.enqueue_memset(out_buf, 0)
    ctx.enqueue_memset(indices_buf, 0)
    ctx.enqueue_memset(weights_buf, 0)

    var out_tensor = LayoutTensor[DTYPE, out_layout, MutAnyOrigin](
        out_buf.unsafe_ptr()
    )
    var indices_tensor = LayoutTensor[
        DType.int32, indices_layout, MutAnyOrigin
    ](indices_buf.unsafe_ptr())
    var weights_tensor = LayoutTensor[DTYPE, weights_layout, MutAnyOrigin](
        weights_buf.unsafe_ptr()
    )

    comptime BLOCK_X = 32
    comptime BLOCK_Y = 32
    var total_positions = BATCH_SIZE * SEQ_LEN
    var blocks_x = max(1, ceildiv(total_positions, BLOCK_X))
    var blocks_y = max(1, ceildiv(EMBED_DIM, BLOCK_Y))

    comptime kernel = embedding_kernel_2d[
        indices_layout,
        weights_layout,
        out_layout,
        BATCH_SIZE,
        SEQ_LEN,
        VOCAB_SIZE,
        EMBED_DIM,
        DTYPE,
    ]
    var compiled_kernel = ctx.compile_function[kernel, kernel]()

    @parameter
    @always_inline
    fn workflow(ctx: DeviceContext) raises:
        ctx.enqueue_function(
            compiled_kernel,
            out_tensor,
            indices_tensor,
            weights_tensor,
            grid_dim=(blocks_x, blocks_y),
            block_dim=(BLOCK_X, BLOCK_Y),
        )

    b.iter_custom[workflow](ctx)
    ctx.synchronize()


@parameter
fn benchmark_3d(mut b: Bencher) raises:
    var ctx = DeviceContext()

    comptime indices_layout = Layout.row_major(BATCH_SIZE, SEQ_LEN)
    comptime weights_layout = Layout.row_major(VOCAB_SIZE, EMBED_DIM)
    comptime out_layout = Layout.row_major(BATCH_SIZE, SEQ_LEN, EMBED_DIM)

    var out_buf = ctx.enqueue_create_buffer[DTYPE](
        BATCH_SIZE * SEQ_LEN * EMBED_DIM
    )
    var indices_buf = ctx.enqueue_create_buffer[DType.int32](
        BATCH_SIZE * SEQ_LEN
    )
    var weights_buf = ctx.enqueue_create_buffer[DTYPE](VOCAB_SIZE * EMBED_DIM)

    # Initialize buffers to avoid memory access faults with random indices
    ctx.enqueue_memset(out_buf, 0)
    ctx.enqueue_memset(indices_buf, 0)
    ctx.enqueue_memset(weights_buf, 0)

    var out_tensor = LayoutTensor[DTYPE, out_layout, MutAnyOrigin](
        out_buf.unsafe_ptr()
    )
    var indices_tensor = LayoutTensor[
        DType.int32, indices_layout, MutAnyOrigin
    ](indices_buf.unsafe_ptr())
    var weights_tensor = LayoutTensor[DTYPE, weights_layout, MutAnyOrigin](
        weights_buf.unsafe_ptr()
    )

    # Using 8x8x8 to stay within 1024 threads limit
    comptime BLOCK_X = 1
    comptime BLOCK_Y = 1
    comptime BLOCK_Z = 256
    var blocks_x = max(1, ceildiv(BATCH_SIZE, BLOCK_X))
    var blocks_y = max(1, ceildiv(SEQ_LEN, BLOCK_Y))
    var blocks_z = max(1, ceildiv(EMBED_DIM, BLOCK_Z))

    comptime kernel = embedding_kernel_3d[
        indices_layout,
        weights_layout,
        out_layout,
        BATCH_SIZE,
        SEQ_LEN,
        VOCAB_SIZE,
        EMBED_DIM,
        DTYPE,
    ]
    var compiled_kernel = ctx.compile_function[kernel, kernel]()

    @parameter
    @always_inline
    fn workflow(ctx: DeviceContext) raises:
        ctx.enqueue_function(
            compiled_kernel,
            out_tensor,
            indices_tensor,
            weights_tensor,
            grid_dim=(blocks_x, blocks_y, blocks_z),
            block_dim=(BLOCK_X, BLOCK_Y, BLOCK_Z),
        )

    b.iter_custom[workflow](ctx)
    ctx.synchronize()


fn main() raises:
    print("Puzzle 21: Mojo Embedding Kernel Benchmarks")
    print("=" * 80)
    print("Configuration: B=8, L=512, V=10000, E=512")
    print("-" * 80)

    var bench = Bench(BenchConfig(max_iters=100))

    bench.bench_function[benchmark_coalesced](BenchId("1D-Coalesced"))
    bench.bench_function[benchmark_2d](BenchId("2D-NonCoalesced"))
    bench.bench_function[benchmark_3d](BenchId("3D-NonCoalesced"))

    print(bench)
