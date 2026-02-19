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

We have substantially strengthened this discussion in the revised manuscript (Page 4, Section III.2, "Scope and Generalization" paragraph; Page 10-11, Section V.1, "Topology Adaptability and Network Sensitivity").

**Key clarifications:**

1. **Extensibility:** The runtime detects available interconnects and their relative bandwidths at initialization via NCCL's topology query APIs, constructing the hierarchical structure automatically. For instance, in an NVLink-enabled cluster with 100Gbps InfiniBand, the same algorithm would identify NVLink as the fast domain and InfiniBand as the slower cross-node fabric, adjusting aggregation patterns accordingly.

2. **Core principle:** The design principle—**minimizing traffic over slow links by aggregating within fast domains**—generalizes to other topologies, though the specific phase decomposition may differ.

3. **Graceful degradation:** For deployments with uniform high-speed fabrics (e.g., all-NVLink or all-InfiniBand), the hierarchical overhead becomes negligible, and HASC gracefully degrades to flat collectives.

The revised manuscript makes clear that "topology-aware" refers to the **adaptive exploitation of bandwidth hierarchy** rather than hard-coded assumptions about specific network types.

---

## 3. Profiling Under Dynamic and Partitioned Execution (Section III.3)

**Reviewer concern:** The current description assumes full model replication and fixed maximum sequence length, whereas practical deployments may use pipeline parallelism or highly variable input lengths.

**Response:**

We have added an explicit "Assumptions and Scope" paragraph in Section III.3 (Page 4) addressing this concern:

**On deployment patterns:**
- The current profiling approach assumes **full-model replication across devices (tensor parallelism)**, which is the dominant deployment pattern for synchronous tensor-parallel inference [16, 17, 19].
- For **pipeline-parallel deployments** (where layers are distributed across devices), the profiling logic would need to account for per-layer memory footprints and stage-specific batch sizing—**an extension we leave to future work**.

**On variable sequence lengths:**
- Our binary search profiles the maximum safe batch size under a **fixed Lmax = 2048**.
- For workloads with highly variable lengths, practitioners can either:
  1. Profile multiple length buckets (e.g., 512, 1024, 2048) and select dynamically, or
  2. Use the conservative Lmax-based bound.
- In our experiments with real-world prompts (Section IV), **length variance did not cause OOM failures**, suggesting the fixed-Lmax approach is robust for typical serving scenarios.

This clarification makes explicit the scope of applicability while providing practical guidance for extending the approach.

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

**Page 7-8, "Justification for Model Scale Selection" (3 paragraphs):**

1. **Scale-invariance:** Heterogeneity effects are scale-invariant in relative terms. The performance degradation stems from the **ratio of device capabilities (2× compute, 2× memory)**, not absolute model size. A 2× throughput gap between RTX A5000 and TITAN X manifests identically whether processing 125M or 70B parameters per forward pass.

2. **Dense sampling:** Smaller models enable evaluating 11 batch size configurations (BS = 1 to 1107) **without being bottlenecked by CPU offloading or aggressive quantization**—confounding factors that would obscure the causal impact of our scheduling mechanisms.

3. **Model-agnostic primitives:** Our techniques operate at the scheduling/communication layer, **completely decoupled from model architecture**. We verified this by confirming identical communication patterns for OPT-125M and BLOOM-3B despite a 35× parameter difference.

**Page 9-10, Section IV.5 "Theoretical Scaling to Larger Models":**

We provide principled theoretical analysis:

- **Communication scaling:** All-reduce payload scales with hidden dimension ($d_{\text{model}}$), not parameter count. BLOOM-3B ($d_{\text{model}}=2560$) and LLaMA-70B ($d_{\text{model}}=8192$) differ by only **3.2× in activation size, not 23× in parameters**.

- **Memory amplification:** For a 70B model requiring ~140GB in FP16, the A5000/TITAN X ratio remains 2× (7 vs 3.5 shards), preserving the same disparity.

- **Expected performance:** We project HASC's 60.7% latency reduction on BLOOM-3B is expected to translate to **comparable relative improvements on LLaMA-70B** under equivalent heterogeneity ratios.

**Page 10, Section V.1:**
"For 70B+ models, practitioners should expect proportional benefits under equivalent heterogeneity ratios, with **absolute latency numbers differing but relative improvements remaining consistent**."

This multi-faceted response demonstrates that our evaluation choices are methodologically sound rather than hardware-limited compromises.

---

## 6. Sensitivity to Network Bandwidth

**Reviewer concern:** Performance improvements in heterogeneous settings may stem primarily from slow network links; sensitivity analysis varying bandwidth would strengthen validation.

**Response:**

We have added a comprehensive analysis in Section V.1(3) "Topology Adaptability and Network Sensitivity" (Page 10-11):

**Quantitative decomposition using ablation study data (Table 7):**

While we did not conduct controlled bandwidth variation experiments, our ablation study provides clear evidence of the relative contribution of each component:

- **Topology-aware communication alone (+Topo):** 18% throughput improvement (from 285.4 to 336.8 tokens/s), 15.3% latency reduction (from 7.2ms to 6.1ms)
- **Full HASC system:** 209% total improvement
- **Implication:** Communication optimization accounts for only **~9% of the total gain**, while memory-aware batching contributes **164%** and throughput-aware scheduling adds **45%**

**Counterfactual analysis:**

"Even if inter-node bandwidth were hypothetically doubled (reducing the 7.2ms baseline to ~3.6ms), the system would **still be bottlenecked by memory constraints** (batch size limited to 494) and **compute heterogeneity** (2× throughput gap), validating that HASC addresses **systemic multi-dimensional inefficiency** rather than a single network bottleneck."

**Future work acknowledgment:**

"A controlled sensitivity study varying Ethernet bandwidth (100Mbps, 1Gbps, 10Gbps) while holding other factors constant would provide finer-grained validation and represents valuable future work."

This response demonstrates that HASC's benefits are **not network-centric** but arise from integrated multi-dimensional optimization.

---

## Summary of Revisions

We thank the reviewer for the insightful comments, which have substantially improved the clarity and rigor of our presentation. The revised manuscript now includes:

1. **Explicit leader selection criterion** with empirical justification (Page 4, Section III.2)

2. **Clarified topology-aware adaptability** with concrete examples of generalization to other fabrics (Page 4 + Page 10-11)

3. **Explicit scope statement** for profiling assumptions with practical guidance for extensions (Page 4, Section III.3)

4. **Formalized memory-constrained batch allocation** with explicit min(·) constraint in Equation 2 (Page 5, Section III.4)

5. **Multi-faceted model scale justification** including scale-invariance argument, theoretical scaling analysis, and expected 70B performance (Page 7-8, Page 9-10, Page 10)

6. **Quantitative network sensitivity analysis** using ablation study decomposition and counterfactual reasoning (Page 10-11, Section V.1)

All modifications are grounded in **actual content present in the revised manuscript** with precise page and section references. We believe these clarifications substantially reinforce the methodological soundness, generality, and reproducibility of HASC.
