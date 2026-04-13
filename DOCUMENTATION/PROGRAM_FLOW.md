# Program Flow Documentation

This document explains how the Julia voter-model code executes, from user input to output, for both the all-to-all and complex-network topologies.

---

## All-to-All Voter Model with Resetting

### Entry Point
```julia
result = simulate_pdf_all_to_all(params; reset, times, bins, nsamples, value_range)
```

### Flow Chart

```
1. INPUT VALIDATION & PREPARATION
   ├─ Convert times to Float64 array
   └─ Check all times ≥ 0

2. MONTE CARLO SAMPLING LOOP (repeat nsamples times)
   │
   └─> For each sample (independent trajectory):
       │
       ├─ INITIALIZE TRAJECTORY
       │  ├─ Convert initial magnetization m0 → count n using positive_count_from_magnetization()
       │  ├─ Set current_time = 0
       │  └─ Create empty trajectory vector (length = # observation times)
       │
       ├─ GILLESPIE LOOP (advance through observation times)
       │  │
       │  └─> For each requested observation time:
       │      │
       │      ├─ While current_time < target_time:
       │      │
       │      │  ├─ Compute total event rate:
      │      │  │  ├─ voter_rate = 2·n·(N-n)/(N-1)
      │      │  │  └─ total_rate = voter_rate + r
       │      │  │
       │      │  ├─ Sample waiting time: dt ~ Exp(total_rate)
       │      │  │
       │      │  ├─ If dt would overshoot target_time → BREAK (record m at current state)
       │      │  │
       │      │  ├─ Else advance time: current_time += dt
       │      │  │
       │      │  └─ Decide event type (random choice, weighted by rates):
       │      │     │
       │      │     ├─ VOTER EVENT (prob = voter_rate / total_rate):
       │      │     │  └─ Flip one agent: n ← n ± 1 (equal probability)
       │      │     │
         │      │     └─ RESET EVENT (prob = r / total_rate):
       │      │        └─ Call apply_all_to_all_reset(protocol, params, ...)
       │      │           └─ Dispatcher returns NEW n based on protocol:
       │      │              ├─ DeltaReset          → n = positive_count_from_magnetization(N, target_m)
       │      │              ├─ StateVectorReset    → n = count of +1 in state vector
       │      │              ├─ UniformMagnetReset  → n ~ Uniform(0, N)
       │      │              ├─ RandomNodeReset     → each node flipped independently
       │      │              └─ FunctionalReset     → user-defined function
       │      │
       │      └─ Record magnetization at checkpoint: m = 2n/N - 1
       │
       └─ Return trajectory vector (length = # observation times)

3. COLLECT ALL SAMPLES
   └─ sample_matrix = (nsamples × ntimes) matrix

4. HISTOGRAM & PDF ESTIMATION
   │
   ├─ resolve_bin_edges(bins; value_range)
   │  └─ Create (nbins+1) edge positions across magnetization range
   │
   ├─ For each observation time t:
   │  │
   │  ├─ Extract all nsamples magnetization values at time t
   │  │
   │  ├─ histogram_counts()
   │  │  └─ Count how many samples fall in each bin
   │  │
   │  └─ counts_to_density()
   │     └─ Normalize: density = counts / (total_count × bin_width)
   │        (integral over all bins = 1)
   │
   └─ Create (ntimes × nbins) density and counts matrices

5. CONSTRUCT RESULT
   └─ Return PDFSimulationResult(
        times,           # original observation times
        bin_edges,       # (nbins+1) edge positions
        bin_centers,     # (nbins) midpoints for plotting
        densities,       # (ntimes × nbins) normalized probability densities
        counts,          # (ntimes × nbins) raw sample counts
        samples          # (nsamples × ntimes) all raw magnetization values
      )
```

### Key Data Transformations
- **m0 → n** : `n = trunc(Int, N*(m0+1)/2)` (initialization)
- **n → m** : `m = 2n/N - 1` (at each checkpoint)
- **samples → density** : histogram counts normalized by (ntimes × bin_width)

---

## Complex Network Voter Model with Resetting

### Entry Point
```julia
result = simulate_pdf_complex(graph, params; reset, times, bins, nsamples, value_range)
```

### Flow Chart

```
1. INPUT VALIDATION & PREPARATION
   ├─ Convert times to Float64 array
   └─ Check all times ≥ 0

2. PRECOMPUTE GRAPH STRUCTURE (once, shared by all samples)
   │
   └─ build_graph_cache(graph)
      ├─ Extract neighbor lists for each node
      ├─ Assign stable edge IDs (numbered in edges() order)
      ├─ Create incident_edge_ids[v] = which edges touch node v
      ├─ Precompute node degrees
      ├─ Sort nodes by degree (highest-first and lowest-first)
      └─ Return ComplexGraphCache (read-only, reused for all trajectories)

3. MONTE CARLO SAMPLING LOOP (repeat nsamples times)
   │
   └─> For each sample (independent trajectory):
       │
       ├─ INITIALIZE STATE & ACTIVE EDGE STRUCTURES
       │  ├─ random_spin_state(N, m0)
       │  │  ├─ Set each node to +1 with probability p = (m0+1)/2
       │  │  └─ Get random state vector (fluctuates around m0)
       │  │
       │  ├─ active_edge_ids_from_state()
       │  │  └─ Scan all edges, find those with endpoints in different states
       │  │
       │  ├─ Initialize pos_map vector
       │  │  └─ pos_map[edge_id] = position in active_list (0 if inactive)
       │  │
       │  ├─ rebuild_active_structures!()
       │  │  └─ Populate active_list and pos_map from initial active edges
       │  │
       │  └─ fixed_reset_plan()
       │     └─ If protocol has fixed reset target (HubReset, StateVectorReset):
       │        ├─ Precompute the fixed reset state
       │        ├─ Precompute active edges in that state
       │        └─ Store in FixedResetPlan (cache for O(1) resets)
       │        Else: return nothing
       │
       ├─ GILLESPIE LOOP (advance through observation times)
       │  │
       │  └─> For each requested observation time:
       │      │
       │      ├─ While current_time < target_time:
       │      │
       │      │  ├─ Compute total event rate:
       │      │  │  ├─ voter_rate = 2 × (# active edges)
      │      │  │  └─ total_rate = voter_rate + r/N
       │      │  │
       │      │  ├─ Sample waiting time: dt ~ Exp(total_rate)
       │      │  │
       │      │  ├─ If dt would overshoot target_time → BREAK (record m at current state)
       │      │  │
       │      │  ├─ Else advance time: current_time += dt
       │      │  │
       │      │  └─ Decide event type (random choice, weighted by rates):
       │      │     │
       │      │     ├─ VOTER EVENT (prob = voter_rate / total_rate):
       │      │     │  ├─ Pick random active edge from active_list
       │      │     │  ├─ Get its two endpoints (u, v)
       │      │     │  ├─ Pick one at random and flip it: state[v] *= -1
       │      │     │  │
       │      │     │  └─ UPDATE ACTIVE EDGES (only incident to flipped node)
       │      │     │     └─ update_incident_edges!(v)
       │      │     │        ├─ For each neighbor of v:
       │      │     │        │  ├─ If state[v] ≠ state[neighbor]: ADD edge to active_list
       │      │     │        │  └─ Else: REMOVE edge from active_list
       │      │     │        │
       │      │     │        └─ Use swap-and-pop for O(1) removal:
       │      │     │           (move last element, update pos_map, pop)
       │      │     │
      │      │     └─ RESET EVENT (prob = (r/N) / total_rate):
       │      │        └─ apply_reset!(state, active_list, pos_map, ...)
       │      │           │
       │      │           ├─ If FixedResetPlan cached:
       │      │           │  ├─ Copy precomputed reset state
       │      │           │  └─ Copy precomputed active edges (O(1) operation)
       │      │           │
       │      │           └─ Else (stochastic reset):
       │      │              ├─ Call apply_dynamic_complex_reset(protocol, ...)
       │      │              │  └─ Dispatcher returns NEW state based on protocol:
       │      │              │     ├─ DeltaReset       → random_exact_state() with fixed count
       │      │              │     ├─ HubReset         → [not for dynamic path, precomputed]
      │      │              │     ├─ RandomNodeReset  → Bernoulli node-wise draw with p=(m+1)/2
       │      │              │     ├─ StateVectorReset → [not for dynamic path, precomputed]
       │      │              │     └─ FunctionalReset  → user-defined function
       │      │              │
       │      │              ├─ Copy new state into current state vector
       │      │              └─ rebuild_active_structures!()
       │      │                 └─ Full rescan to find active edges in new state
       │      │
       │      └─ Record magnetization at checkpoint: m = sum(state) / N
       │
       └─ Return trajectory vector (length = # observation times)

4. COLLECT ALL SAMPLES
   └─ sample_matrix = (nsamples × ntimes) matrix

5. HISTOGRAM & PDF ESTIMATION
   │
   ├─ resolve_bin_edges(bins; value_range)
   │  └─ Create (nbins+1) edge positions across magnetization range
   │
   ├─ For each observation time t:
   │  │
   │  ├─ Extract all nsamples magnetization values at time t
   │  │
   │  ├─ histogram_counts()
   │  │  └─ Count how many samples fall in each bin
   │  │
   │  └─ counts_to_density()
   │     └─ Normalize: density = counts / (total_count × bin_width)
   │        (integral over all bins = 1)
   │
   └─ Create (ntimes × nbins) density and counts matrices

6. CONSTRUCT RESULT
   └─ Return PDFSimulationResult(
        times,           # original observation times
        bin_edges,       # (nbins+1) edge positions
        bin_centers,     # (nbins) midpoints for plotting
        densities,       # (ntimes × nbins) normalized probability densities
        counts,          # (ntimes × nbins) raw sample counts
        samples          # (nsamples × ntimes) all raw magnetization values
      )
```

### Key Data Transformations
- **m0 → state** : `state[i] = rand() < (m0+1)/2 ? +1 : -1` (independent per node, initialization)
- **state → m** : `m = sum(state) / N` (at each checkpoint)
- **samples → density** : histogram counts normalized by (ntimes × bin_width)

### Key Optimization
The **active edge list** prevents rescanning all M edges after every flip:
- After node flip, only check its degree neighbors (not all N nodes)
- Total complexity per voter event: O(degree) instead of O(M)
- Swap-and-pop removal is O(1) instead of O(M)

---

## Output Structure: PDFSimulationResult

Both topologies return the same struct:

```julia
struct PDFSimulationResult
    times::Vector{Float64}              # (ntimes,) observation times
    bin_edges::Vector{Float64}          # (nbins+1,) histogram bin edges
    bin_centers::Vector{Float64}        # (nbins,) bin midpoints
    densities::Matrix{Float64}          # (ntimes, nbins) normalized probability density
    counts::Matrix{Int}                 # (ntimes, nbins) raw sample counts
    samples::Matrix{Float64}            # (nsamples, ntimes) all raw samples
end
```

**Usage:**
- Plot: `plot(result.bin_centers, result.densities[t, :]')` for time index t
- Raw statistics: compute variance, moments, etc. from `result.samples`
- Verify normalization: `sum(result.densities[t, :] .* diff(result.bin_edges)) ≈ 1`
