---
title: "C++ JIT"
document: P3333R0
date: today
audience: WG21
author:
  - name: Bryce Adelstein Lelbach
    email: brycelelbach@gmail.com
  - name: Basit Ayantunde
    email: basitayde@gmail.com
toc: true
---

# Authors

* Bryce Adelstein Lelbach (he/him/his), NVIDIA, `brycelelbach@gmail.com`

* Basit Ayantunde (he/him/his), NVIDIA, `basitayde@gmail.com`

# Abstract

C++ needs first-class support for just-in-time (JIT) compilation. Machine
learning workloads increasingly require JIT compilation of compute kernels to
specialize code for complex and heterogeneous compute resources like GPUs and
LPUs. This paper surveys the landscape, examines existing practice, and proposes
a direction for standardizing JIT compilation facilities in C++.

# Introduction

<!-- TODO: Write a compelling introduction that sets the stage:
  - C++ has historically been an ahead-of-time (AOT) compiled language.
  - The rise of ML/AI workloads has created a pressing need for runtime
    code generation and specialization.
  - JIT compilation allows code to be specialized for the exact hardware
    it will run on, which is increasingly important as compute resources
    become more heterogeneous (CPUs, GPUs, TPUs, LPUs, custom accelerators).
  - Other languages (Julia, Python via Numba/JAX, etc.) have embraced JIT;
    C++ should too.
-->

# Motivation

<!-- TODO: Flesh out the motivation in detail:
  - Performance-critical ML inference and training pipelines need to
    generate and compile kernels at runtime.
  - Hardware diversity: GPUs (NVIDIA, AMD, Intel), LPUs (Groq), TPUs
    (Google), custom ASICs — each with different ISAs, memory hierarchies,
    and optimal code patterns.
  - AOT compilation cannot anticipate all deployment targets or runtime
    conditions (batch sizes, tensor shapes, sparsity patterns, etc.).
  - JIT enables fusion of operations, elimination of intermediate
    materialization, and specialization on runtime-known constants.
-->

# Existing Practice

<!-- TODO: Survey existing JIT and runtime code generation approaches
  used with or alongside C++:
  - LLVM/Clang JIT (ORC JIT, clang-repl)
  - NVRTC (NVIDIA Runtime Compilation for CUDA)
  - OpenCL runtime compilation
  - SYCL and oneAPI runtime compilation
  - Halide and TVM — ML compiler frameworks that JIT
  - JAX/XLA — JIT compilation for ML in the Python ecosystem
  - Julia — language-level JIT for scientific computing
  - Cling / clang-repl — C++ interpreters/JIT engines
  - libgccjit — GCC's JIT compilation library
  - Domain-specific approaches: expression templates, embedded DSLs
  - MLIR and its role in ML compiler infrastructure
-->

# Problem Statement

<!-- TODO: Clearly articulate the problem:
  - There is no standard way to perform JIT compilation in C++.
  - Existing solutions are compiler-specific, vendor-specific, or
    require leaving the C++ type system entirely (e.g., passing around
    source strings).
  - Lack of standardization leads to fragmentation, portability issues,
    and duplicated effort across the ecosystem.
  - Users are forced to choose between C++ performance and the runtime
    flexibility available in other language ecosystems.
-->

# Design Space

<!-- TODO: Explore the design space for C++ JIT support:
  - What level of abstraction? (source strings, AST, IR, or language
    integrated?)
  - Compilation model: full C++ recompilation vs. a restricted subset
    vs. a new embedded language.
  - Integration with the C++ type system and object model.
  - Interaction with modules, reflection, and other modern C++ features.
  - Ahead-of-time vs. just-in-time vs. adaptive optimization.
  - Fat binaries and deferred compilation strategies.
  - Error handling for compilation failures at runtime.
  - Security implications of runtime code generation.
-->

# Proposed Direction

<!-- TODO: Sketch the proposed direction:
  - What should a C++ JIT facility look like?
  - High-level API sketch or concepts.
  - How it interacts with the rest of the language.
  - Phasing: what could be in C++29, what is longer-term.
-->

# Impact on the Standard

<!-- TODO: Discuss the impact:
  - Core language changes vs. library-only solution.
  - Freestanding considerations.
  - ABI implications.
  - Implementability across major compilers (GCC, Clang, MSVC).
-->

# Acknowledgements

<!-- TODO: Acknowledge contributors and reviewers. -->
