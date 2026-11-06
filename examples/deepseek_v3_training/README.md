# DeepSeek-V3 training across nodes

![Forward and backward routes over InfiniBand and NVLink](pipeline.svg)

Each training sequence contains **1,048,576 causal tokens**. Global batch size is a separate input. For `B` full sequences, an optimizer update consumes `B × 1,048,576` tokens. The overview uses an illustrative 32 nodes × 16 GPUs and follows the 61 decoder blocks through ordinary next-token loss and backward. It shows three dense FFNs and 58 MoE blocks, each with MLA, one shared expert and eight of 256 routed experts per token. One sequence creates 8,388,608 expert assignments in each routed block. [DeepSeek-V3 model dimensions](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) · [Training infrastructure](https://arxiv.org/html/2412.19437v2#S3.SS2.SSS2)

## Compare training layouts

![MLA token and head ownership, ring visits, and per-rank send volume](context-parallel.svg)

**Ulysses** projects K/V on token owners, sends packed Q/K/V to head owners in one all-to-all, and gathers the head-shared positional key. Head owners run full causal attention and transpose Y back to token owners. Backward reverses both transposes and reduce-scatters the shared-key gradient over heads. **Ring Attention** keeps queries on their token owners. Each traveling block carries normalized KV latents and the shared positional key. The receiver expands K/V and updates the row maximum, normalizer and value sum. Backward replays KV, accumulates dQ locally and returns dLatent and dSharedKey to the original block owner. This is a compact-MLA adaptation of Ring Attention. [Ulysses paper](https://arxiv.org/abs/2309.14509) · [TorchTitan MLA layouts](https://github.com/pytorch/torchtitan/blob/f782e30c1aaf1965c851abeaa6b73d29d35270b6/torchtitan/models/common/attention/cp_mla.py#L77-L155) · [Ring Attention paper](https://arxiv.org/abs/2310.01889)

The figure counts **logical BF16 data sent by one CP rank**, for one sequence and one MLA layer. Let `P` be the CP degree and `L = 1,048,576 / P`. For Ulysses, each transpose sends a fraction `(P−1)/P` of its input away. The packed input has `128 × (192+128+128)` channels, the returning output has `128 × 128`, and each local block of the 64-channel shared key reaches `P−1` peers. The chosen backward sends the same total with the adjoint collectives. The ring sends a `2 × L × (512+64)` byte block for each of `P−1` forward hops. Backward sends those KV blocks again and the gradient accumulator takes `P` hops, including the last hop home. Both directions use BF16 in this accounting. Future-only blocks skip arithmetic and still circulate. At four ranks this trades 27.09 GiB per direction for 0.84 GiB forward and 1.97 GiB backward, with ten forward block projections across the group versus four. These totals cover CP tensor payloads. Weight-gradient collectives, link protocols and elapsed times require separate accounting.

To separate the effect of compact KV from the choice of collective, the fixture also executes TorchTitan’s **compact all-gather** pattern. At four ranks it sends the same 0.84 GiB forward as the ring. With BF16 gradient reduction and the gathered inputs retained, its backward reduce-scatter also sends 0.84 GiB. Each query rank materializes all four compact blocks. The ring consumes blocks successively and replays them in backward. [TorchTitan compact MLA all-gather](https://github.com/pytorch/torchtitan/blob/f782e30c1aaf1965c851abeaa6b73d29d35270b6/torchtitan/models/common/attention/cp_mla.py#L44-L76)

![Expert placement, activation dispatch, and gradient ownership for the same selected experts](expert-parallel.svg)

| EP path | Dispatch and combine | Backward |
|---|---|---|
| Naive variable-split all-to-all | Send one row per assignment. Return each expert result to the token owner for weighting. | The token owner sends a weighted dY per slot, computes selected-score gradients and adds returned dX. |
| Compact DeepEP | Send one row per (token, destination rank) with selected IDs and scores. Fan out and reduce within the destination before returning a row. | Reuse the route for dY. Sum dX by token at the destination and return the selected-score gradients. |
| Dist-MoE-style peer access | Publish each source row once. W13 gathers by peer address, and W2 writes its result to the origin slot. | The source publishes weighted route gradients. Experts read them for input and weight gradients and scatter dX back to slots. |

The illustrated token selects four experts on each of two other ranks. Naive dispatch sends eight 14 KiB activation rows, compact dispatch sends two, and both execute eight expert rows. Peer reads and writes are counted by address. The physical traffic depends on the implementation and fabric. The peer-address example follows Dist-MoE’s symmetric-memory ownership. The two-hop IB/NVLink timeline above models a separate hierarchical schedule. [TorchTitan all-to-all dispatcher](https://github.com/pytorch/torchtitan/blob/f782e30c1aaf1965c851abeaa6b73d29d35270b6/torchtitan/models/common/moe/token_dispatcher.py) · [DeepEP compact dispatch and combine](https://github.com/pytorch/torchtitan/blob/f782e30c1aaf1965c851abeaa6b73d29d35270b6/torchtitan/distributed/deepep/deepep.py#L405-L465) · [Dist-MoE BF16 forward and backward](https://github.com/meta-pytorch/dist_moe/blob/285763b5a0e01059a62c96b0990754a60a6ed94a/docs/bf16_execution.md#L63-L92)

### Run the comparisons

From this directory in an installed TileFoundry environment:

```sh
python context_parallel.py
python expert_parallel.py
python compare.py
python visualize_baselines.py --out-dir .
```

- `context_parallel.py` checks causal Y, dQ, dLatent, dSharedKey and both KV up-projection weight gradients against dense FP32 autograd. The evaluator runs HIR projection and attention blocks, split-to-split exchanges, the shared-key gather and its backward reduction. The rank driver rotates the ring packets. Inputs begin at the normalized latent and post-RoPE boundary.
- `expert_parallel.py` runs all three plans on identical routes with co-located experts, spread experts and empty owners. It checks Y, dX, selected-score gradients and every expert weight gradient. HIR evaluates the bounded exchange mailboxes and expert MLP. The driver supplies the fixed top-eight selections, variable record counts and peer addresses.
- `compare.py` composes **all six CP × EP pairs** at an MLA output projection feeding a routed FFN. The same four logical ranks switch from CP to EP after attention returns token ownership. Across four independent data replicas, FSDP gathers shards of the MLA KV projection, keeps the weights available through backward and reduce-scatters its accumulated gradient. Each final gradient shard and each sequence’s output, input and score gradients are checked against one dense loss. The executable uses 12-token sequences and 32 toy experts. The figures use the full-model dimensions for payload arithmetic. The [FSDP/HSDP training example](../fsdp_training/README.md) covers the wider parameter and optimizer lifecycle.

## Generate the pipeline

From this directory in an installed TileFoundry environment:

```sh
python visualize.py --out pipeline.svg
python visualize.py --layer 60 --tiles 8 --slots 3 --nodes 128 --sequences 8 --out /tmp/layer60.svg
python reference.py
```

A bar's number identifies the same tile across rows. Gold marks intervals when all three resources work concurrently. The bulk column materializes each stage. The tile-ready column passes a tile onward after its producer finishes, and retains a staging slot until its consumer finishes. Both columns fence the gate/up and down kernel phases. Forward applies SwiGLU in the down phase. Backward separates input and weight gradients. The diagram assumes one serial resource for each of IB, NVLink and compute. This models a candidate schedule for the routed branch. Routes within one node bypass IB. Route discovery, shared-expert work, top-k fan-in, CP attention and parameter-gradient collectives occur outside the expanded timing window.

With the default options the report prints tokens and assignments per sequence. Supplying `--sequences B` also prints tokens per optimizer update. Node count and batch size can vary independently. The FP32 reference checks both service orders through the source, gateway and expert `Reshard` operations, including an empty expert node and the sum of owner contributions to dX.

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

Each row displays its available isolated median. The bars use dependency ticks until all three probes are available. A complete set switches both columns to **modeled microseconds** from isolated medians. The BF16 reference charges one payload per illustrated hop. Compact dispatch, FP8 wire formats and fused distributed GEMMs need measurements of their own. [Dist-MoE training kernel stages](https://github.com/meta-pytorch/dist_moe/blob/285763b5a0e01059a62c96b0990754a60a6ed94a/docs/mxfp8_kernel_design.md#L97-L128)

To draw **observed** overlap, use `--trace capture.json`. Set `schema` to 1, `cc` to `[10,3]`, `tiles` and `slots` to the selected geometry, and record the raw capture `source`, common `clock_domain` and worst-case timestamp error `clock_error_us`. Under `passes.Forward` and `passes.Backward`, provide `bulk` and `tile_ready` arrays of `[action_key, zero_based_tile, start_us, end_us]`. The action keys are in `workload.py`. Use per-role device activity/completion intervals on a common calibrated time base. A kernel launch envelope cannot identify which roles ran together. The renderer checks tile readiness and trims the gold intervals by the clock uncertainty at each end.
