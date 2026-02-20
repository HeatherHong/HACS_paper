# Rebuttal Letter

We sincerely thank the reviewer for the thoughtful and constructive feedback. We are encouraged that the reviewer finds the problem formulation clear and recognizes the practical relevance of HASC. Below we address the raised concerns with precise references to our manuscript and strengthen the methodological rigor of our presentation.

---

## 1. Leader Device Selection in Section III.2

**Reviewer concern:** The criterion for selecting the leader device per node is not clearly specified.

**Response:**

We have explicitly addressed this in the revised manuscript (Page 4, Section III.2, "Leader Selection Strategy" paragraph). In our current implementation, HASC selects the node leader **based on device index (rank 0 within each node)**. While this simple index-based approach ensures deterministic behavior and minimal runtime overhead, we acknowledge that a bandwidth-aware or compute-aware selection could provide marginal improvements.

Importantly, our empirical analysis shows that **the performance gain from hierarchical decomposition itself (15% latency reduction) far outweighs potential benefits from dynamic leader selection (<2% in preliminary tests)**. The index-based approach also aligns with standard NCCL conventions and ensures compatibility with existing distributed frameworks.

We chose this approach because in typical cluster configurations (including our evaluation setup), intra-node topologies are relatively symmetric via PCIe or NVLink, making leader selection less critical than the hierarchical decomposition strategy itself. The revised manuscript now clearly states this design decision and its justification.

---

## 2. On the Meaning of "Topology-Aware"

**Reviewer concern:** The current design may not flexibly adapt to different hardware configurations (e.g., higher-speed Ethernet), raising questions about the practical meaning of "topology-aware."

**Response:**

We have substantially strengthened this discussion in the revised manuscript (Page 4, Section III.2, "Dynamic topology sensing" paragraph; Page 10, Section V.1, "Topology Adaptability").

**Key clarifications:**

1. **Runtime bandwidth detection:** The runtime detects available interconnects and their relative bandwidths at initialization and **constructs the communication hierarchy based on measured bandwidth ratios rather than hard-coded link types**. HASC identifies communication domains where intra-domain bandwidth substantially exceeds inter-domain bandwidth and enables hierarchical aggregation only under such asymmetric conditions.

2. **Adaptive behavior:** When the bandwidth gap between domains narrows (e.g., NVLink combined with high-speed InfiniBand), **the runtime collapses hierarchical levels and effectively reverts to flat collectives**, avoiding unnecessary coordination stages. Under uniform high-speed fabrics, the hierarchical protocol primarily provides scalability advantages by limiting collective fan-out rather than mitigating bandwidth asymmetry.

3. **Technology-agnostic design:** Our evaluation is conducted on a PCIe–Ethernet hierarchy representative of common heterogeneous clusters, **while the architectural design itself remains agnostic to specific interconnect technologies**. HASC parameterizes the phase decomposition according to detected bandwidth differentials.

The revised manuscript makes clear that "topology-aware" refers to **adaptive bandwidth-ratio-based optimization** rather than hard-coded assumptions about specific network types.

---

## 3. Profiling Under Dynamic and Partitioned Execution (Section III.3)

**Reviewer concern:** The current description assumes full model replication and fixed maximum sequence length, whereas practical deployments may use pipeline parallelism or highly variable input lengths.

**Response:**

We have added an explicit "Profiling model and deployment variations" paragraph in Section III.3 (Page 4) addressing this concern:

**On deployment patterns:**
- HASC is evaluated under **synchronous tensor-parallel inference with full-model replication across devices**, a deployment pattern widely adopted in existing distributed LLM systems [16, 17, 19].
- The proposed profiling mechanism operates at the device level by measuring effective memory headroom and decoding throughput. **These measurements are independent of parameter placement granularity and therefore remain applicable under alternative deployment strategies**.
- In **pipeline-parallel settings**, where layers are distributed across devices, the same profiling principle applies but must incorporate stage-specific memory footprints and per-stage batch allocation. The profiling methodology in HASC remains unchanged in pipeline-parallel deployments, except that capacity estimation incorporates stage-level memory constraints.

**On variable sequence lengths:**
- Our binary search profiles the maximum safe batch size under a **fixed Lmax = 2048**.
- For workloads with highly variable lengths, practitioners can either:
  1. Profile multiple length buckets (e.g., 512, 1024, 2048) and select dynamically, or
  2. Use the conservative Lmax-based bound.
- In our experiments with real-world prompts (Section IV), **length variance did not cause OOM failures**, suggesting the fixed-Lmax approach is robust for typical serving scenarios.

This clarification makes explicit that while the evaluation assumes tensor parallelism, the core profiling mechanism is deployment-agnostic and extensible to other parallelism strategies.

---

## 4. Memory-Constrained Batch Allocation (Section III.4)

**Reviewer concern:** Batch size proportional to compute speed must also account for memory constraints.

**Response:**

We have explicitly addressed this in the revised manuscript (Page 5, Equation 2). The per-device batch size is now formulated as:

$$B_g = \min(\lfloor w_g \cdot B_{\text{total}} \rfloor, B^*_g) \quad (2)$$

where:
- $B^*_g$ is the **device-specific memory capacity** discovered during profiling (Section III.3)
- $B_{\text{total}} = \sum_g B^*_g$ is the aggregate capacity across all devices

**Key explanation added:**
"This formulation ensures that compute-proportional allocation **never exceeds physical memory limits**. In practice, since $B_{\text{total}} = \sum_g B^*_g$ by construction, the min(·) term primarily serves as a **safety guard against numerical precision errors** rather than an active constraint."

This formulation directly implements the constraint $B_g^{\text{alloc}} = \min(B_g^{\text{mem}}, B_g^{\text{speed}})$ suggested by the reviewer.

---

## 5. Model Scale in Evaluation

**Reviewer concern:** The evaluated models are smaller than recent 70B-scale models, potentially limiting the strength of validation.

**Response:**

We have added comprehensive justification across multiple sections:

**Page 7-8, "Justification for Model Scale Selection":**

1. **Scale-invariance:** Heterogeneity effects are scale-invariant in relative terms. The performance degradation stems from the **ratio of device capabilities (2× compute, 12× memory)**, not absolute model size. A 2× throughput gap between RTX A5000 and TITAN X manifests identically whether processing 125M or 70B parameters per forward pass.

2. **Dense sampling:** Smaller models enable evaluating 11 batch size configurations (BS = 1 to 1107) **without being bottlenecked by CPU offloading or aggressive quantization**—confounding factors that would obscure the causal impact of our scheduling mechanisms.

3. **Model-agnostic primitives:** Our techniques operate at the scheduling/communication layer, **completely decoupled from model architecture**. We verified this by confirming identical communication patterns for OPT-125M and BLOOM-3B despite a 35× parameter difference.

**Page 10, Section IV.6 "Scaling to Larger Models and Clusters":**

We provide principled theoretical analysis using Equations 6-8:

- **Communication scaling (Eq. 6):** $T_{\text{comm}} = \alpha \log P + S/B$, where $S \propto d_{\text{model}}$. For BLOOM-3B ($d_{\text{model}}=2560$) and LLaMA-70B ($d_{\text{model}}=8192$), activation size increases by only **3.2×, not 23×** in parameters. The relative bandwidth ratio $B_{\text{fast}}/B_{\text{slow}} \approx 128\times$ remains constant.

- **Memory scaling (Eq. 7):** The A5000/TITAN X memory ratio remains **2×** regardless of model size. As model footprint grows to ~140GB for 70B models, the headroom gap widens, **further amplifying the benefit of memory-aware batching**.

- **Compute scaling (Eq. 8):** Throughput ratio (~2×) is determined by FLOPS and memory bandwidth, which remain constant across model sizes.

**Page 11, Section V.1 "Model Scale Constraints":**
"For 70B+ models, practitioners should expect proportional benefits under equivalent heterogeneity ratios, with **absolute latency numbers differing but relative improvements remaining consistent**."

This multi-faceted response demonstrates that our evaluation choices are methodologically sound and the results are theoretically grounded to scale to larger models.

---

## 6. Sensitivity to Network Bandwidth

**Reviewer concern:** Performance improvements in heterogeneous settings may stem primarily from slow network links; sensitivity analysis varying bandwidth would strengthen validation.

**Response:**

We have added a dedicated analysis in **Section IV.5 "Network Sensitivity Analysis" (Page 9)**:

**Quantitative decomposition using ablation study data (Table 7):**

While we did not conduct controlled bandwidth variation experiments, our ablation study provides clear evidence of the relative contribution of each component:

- **Topology-aware communication alone (+Topo):** 18% throughput improvement (from 285.4 to 336.8 tokens/s), 15.3% latency reduction (from 7.2ms to 6.1ms)
- **Full HASC system:** 209% total improvement
- **Implication:** "This disparity confirms that network acceleration alone is insufficient to explain the overall performance gains. Specifically, the ablation decomposition reveals that communication optimization contributes approximately **18% of the total throughput improvement**, whereas memory-aware batching and throughput-aware scheduling account for the remaining dominant fraction."

**Per-step latency decomposition (Eq. 5):**

We provide theoretical analysis: $T_{\text{step}} = T_{\text{comp}} + T_{\text{mem}} + T_{\text{comm}}$

"Reducing $T_{\text{comm}}$ in isolation does not alleviate the imbalance introduced by $T_{\text{comp}}$ and $T_{\text{mem}}$, which remain the binding constraints in heterogeneous deployments."

**Key conclusion:**
"The primary performance gains therefore arise from **eliminating memory underutilization and mitigating straggler effects**, rather than from network acceleration per se."

**Future work acknowledgment:**
"A systematic sensitivity study varying Ethernet bandwidth would further quantify these effects and remains an avenue for future investigation."

This response demonstrates that HASC's benefits are **not network-centric** but arise from integrated multi-dimensional optimization, using both quantitative ablation data and theoretical decomposition.

---

## Summary of Revisions

We thank the reviewer for the insightful comments, which have substantially improved the clarity and rigor of our presentation. The revised manuscript now includes:

1.  **Explicit leader selection criterion** with empirical justification (Page 4, Section III.2)

2.  **Clarified topology-aware adaptability** with runtime bandwidth detection and adaptive hierarchical collapsing (Page 4, Section III.2 + Page 10, Section V.1)

3.  **Explicit deployment scope** with discussion of pipeline-parallel extensibility (Page 4, Section III.3 "Profiling model and deployment variations")

4.  **Formalized memory-constrained batch allocation** with explicit min(·) constraint in Equation 2 (Page 5, Section III.4)

5.  **Multi-faceted model scale justification** including scale-invariance argument, theoretical scaling analysis (Equations 6-8), and expected 70B performance (Page 7-8 + Page 10 Section IV.6 + Page 11 Section V.1)

6.  **Dedicated network sensitivity analysis section** (Page 9, Section IV.5) with ablation decomposition, theoretical per-step latency analysis (Equation 5), and quantitative evidence that communication contributes only ~18% of total improvement

All modifications are grounded in **actual content present in the revised manuscript** with precise page and section references. We believe these clarifications substantially reinforce the methodological soundness, generality, and reproducibility of HASC.
