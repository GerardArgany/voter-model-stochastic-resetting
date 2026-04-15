# =============================================================================
# VoterResetting.jl  —  top-level Julia module
# =============================================================================
#
# This is the single entry-point for the whole Julia implementation.
# Load it from a notebook with:
#
#   include(joinpath(dirname(pwd()), "src", "VoterResetting.jl"))
#   using .VoterResetting
#
# The module is split across three source files that are included in order:
#
#   common/simulation_core.jl
#       Shared types (parameter structs, reset-protocol types, result container),
#       shared maths helpers (magnetization ↔ count conversions), and the shared
#       Monte Carlo sampling + PDF estimation machinery used by both topologies.
#
#   all_to_all/pdf_simulation.jl
#       Gillespie simulation of the voter model on the complete graph (all-to-all
#       coupling) under any resetting protocol.  Produces PDFs of the global
#       magnetization m(t) via repeated independent trajectories.
#
#   complex/pdf_simulation.jl
#       Gillespie simulation on an arbitrary sparse graph (Graphs.jl
#       AbstractGraph).  Uses an active-edge list for O(degree) updates per voter
#       flip instead of rescanning all edges.  Produces PDFs of m(t) in the same
#       format as the all-to-all case.
#
# =============================================================================
module VoterResetting

using Graphs      # AbstractGraph, nv, ne, edges, src, dst, degree, etc.
using Random      # rand, randperm
using Statistics  # mean, std

# ---- public API --------------------------------------------------------------
# Types the caller needs for constructing inputs and reading outputs
export AbstractResetProtocol
export AllToAllParams, ComplexParams, PDFSimulationResult, FPTSimulationResult
export DegreePairEvolutionResult, DegreeGridEvolutionResult
export AMESteadyStateResult, AMEEvolutionResult

# AME (Approximate Master Equation) numerical solver
export solve_ame_evolution, solve_ame_steady_state

# Constructors for the supported resetting protocols
export delta_reset, uniform_reset, random_node_reset, hub_reset, custom_reset

# Utility: compute a PDF from a plain vector of magnetization samples
export magnetization_pdf

# Main simulation functions, one per topology
# PDF (probability density at fixed times)
export simulate_pdf_all_to_all, simulate_pdf_complex
export simulate_degree_evolution_complex, simulate_sikm_pair_complex

# FPT (first passage time to consensus)
export first_passage_time_all_to_all, first_passage_time_complex

# Discrete-time simulations (alternative to Gillespie)
export simulate_pdf_discrete_complex, first_passage_time_discrete_complex
export discrete_pdf_from_samples

# Analytical all-to-all theory (analogous to Python voter_model package)
export dist_laplace, sol_fpt, mean_fpt, variance_fpt
export pip1, pim1, fk, sol
export exact_mfpt_discrete_spectral
# ------------------------------------------------------------------------------

include("common/simulation_core.jl")
include("all_to_all/pdf_simulation.jl")
include("all_to_all/fpt_simulation.jl")
include("all_to_all/theory_solution.jl")
include("complex/pdf_simulation.jl")
include("complex/sikm_simulation.jl")
include("complex/fpt_simulation.jl")
include("complex/ame_solution.jl")
include("discrete_time_voter.jl")

end