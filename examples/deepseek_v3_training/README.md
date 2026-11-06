# DeepSeek-V3 training across nodes

![Forward and backward routes over InfiniBand and NVLink](pipeline.svg)

The overview follows all 61 decoder blocks through next-token loss and backward. The first three blocks use dense FFNs. The other 58 use MLA, one shared expert and eight selected routed experts per token. The expanded rows follow a remote expert assignment. A source sends its token row across InfiniBand to a gateway on the destination node, then across NVLink to the expert GPU. Its output comes home in reverse order. The token owner combines contributions using the selected scores. Backward uses the saved route to send weighted output gradients and accumulate input gradients at the source. [DeepSeek-V3 training infrastructure](https://arxiv.org/html/2412.19437v2#S3.SS2.SSS2)

## Generate the figure

From this directory in an installed TileFoundry environment:

```sh
python visualize.py --out pipeline.svg
python visualize.py --layer 60 --tiles 8 --slots 3 --nodes 128 --sequences 240 --out /tmp/layer60.svg
python reference.py
```

A bar's number identifies the same tile across rows. Gold marks intervals when all three resources work concurrently. The bulk column materializes each stage. The tile-ready column passes a tile onward after its producer finishes, and retains a staging slot until its consumer finishes. Both columns fence the gate/up and down kernel phases. Forward applies SwiGLU in the down phase. Backward separates input and weight gradients. The diagram assumes one serial resource for each of IB, NVLink and compute. This models a candidate schedule for the routed branch. Routes within one node bypass IB. Route discovery, shared-expert work, top-k fan-in, CP attention and parameter-gradient collectives occur outside the expanded timing window.

The default study uses 32 nodes of 16 GPUs and 60 full 1,048,576-token sequences per update. Node and batch counts specify workload scale independently. Each routed layer has eight assignments per token. The figure shows the service time of one tile stream. Full-step time requires the other layers, placements, route histogram and available memory. The FP32 CPU reference evaluates source → gateway → expert and both return hops as separate `Reshard` operations. It checks both execution orders, an empty expert node, the sum of owner contributions, selected-score gradients, and expert input/weight gradients against dense autograd.

## Measure the B300 stages

The benchmark takes its target identity from CUDA compute capability **10.3**. It records the driver-reported name and UUID separately. Use the same row count, BF16 wire format, hidden width 7,168 and expert width 2,048 for all probes. Compute samples time warm PyTorch BF16 projections and their explicit backward operations. Link samples time one-way NCCL transfers in each direction on the receiver's CUDA stream. A two-GPU local run verifies an NVLink topology and NCCL P2P channel. An IB run requires different physical nodes, an active InfiniBand port on each, and a confirmed NCCL IB channel. [NVIDIA device capabilities](https://developer.nvidia.com/cuda/gpus)

```sh
python benchmark.py compute --rows 128 --out measurements/compute.json
torchrun --standalone --nproc-per-node 2 benchmark.py nvlink --rows 128 --out measurements/nvlink.json
```

Run the following on **both** IB-connected nodes with the same rendezvous address. Set `NODE_RANK` to 0 or 1 and `HCA` to the local device and port. Rank 0 writes the result.

```sh
torchrun --nnodes 2 --node-rank "$NODE_RANK" --nproc-per-node 1 \
  --master-addr "$MASTER_ADDR" --master-port 29500 \
  benchmark.py ib --hca "$HCA" --rows 128 --out measurements/ib.json
python visualize.py --samples measurements/*.json --out /tmp/calibrated.svg
```

The plot uses dependency ticks until all three probes are available. A complete set switches both columns to **modeled microseconds** from isolated medians. The BF16 reference charges one payload per illustrated hop. Compact dispatch, FP8 wire formats and fused distributed GEMMs need measurements of their own. [Dist-MoE training kernel stages](https://github.com/meta-pytorch/dist_moe/blob/285763b5a0e01059a62c96b0990754a60a6ed94a/docs/mxfp8_kernel_design.md#L97-L128)

To draw **observed** overlap, use `--trace capture.json`. Set `schema` to 1, `cc` to `[10,3]`, `tiles` and `slots` to the selected geometry, and record the raw capture `source`, common `clock_domain` and worst-case timestamp error `clock_error_us`. Under `passes.Forward` and `passes.Backward`, provide `bulk` and `tile_ready` arrays of `[action_key, zero_based_tile, start_us, end_us]`. The action keys are in `workload.py`. Use per-role device activity/completion intervals on a common calibrated time base. A kernel launch envelope cannot identify which roles ran together. The renderer checks tile readiness and trims the gold intervals by the clock uncertainty at each end.
