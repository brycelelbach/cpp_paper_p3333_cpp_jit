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

<!-- hardware is now getting more specialized; there used to be a compute GPU,
but there is now compute CPU and compute GPU -->
<!-- provide references for claims -->
<!-- Make the paper smaller and shorter -->

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

C++ is historically an ahead-of-time (AOT) language, but many modern workloads
need runtime specialization: target hardware, tensor shapes, and optimization
choices are often known only at execution time. In these cases, applications do
not just choose among precompiled variants; they synthesize and compile new
functions at runtime.

This pattern now appears across ML, scientific computing, query engines, and
graphics pipelines. JIT compilation is therefore not an edge feature but a core
execution model for high-performance heterogeneous systems.

Other ecosystems expose JIT as a first-class capability (for example, Julia and
Python stacks such as Numba, JAX, and `torch.compile`). Although these systems
often rely on C++/LLVM backends, C++ itself has no standard JIT facility, so
developers rely on non-portable, hard-to-compose mechanisms outside the C++
type system and tooling.

# Motivation

## Hardware Diversity

The compute landscape has fragmented dramatically. A single ML inference
deployment today may target any combination of NVIDIA GPUs, AMD GPUs, Intel
GPUs, Google TPUs, and an ever-growing ecosystem of custom ASICs. Each of
these devices has a distinct ISA, a distinct memory hierarchy, and distinct
performance characteristics.

AOT compilation cannot produce a single binary that is simultaneously optimal
for all of these targets. The set of hardware targets is not closed; new
accelerators with vastly different configurations are released continuously.
More fundamentally, the *optimal code* for a given operation is not a fixed
function of the hardware alone. It also depends on runtime parameters that the
AOT compiler cannot observe.

## Binary Size and Deployment Constraints

Consider a ML framework that wishes to ship optimized kernels for matrix
multiplication across NVIDIA, AMD, and Intel GPUs. For each architecture,
there may be dozens of variants specialized for different tile sizes, data
types (`float8`, `bfloat8`, `nvfp8`, `int8`, `uint8`, `float16`, `bfloat16`,
`nvfp16`, `int16`, `uint16`), and memory layouts. The number of kernels grows
multiplicatively. Compiled GPU kernels are not small; a single highly optimized
GEMM kernel may be hundreds of kilobytes of device code. Shipping thousands of
such variants adds up to gigabytes of device code in the binary, most of which
will never execute on any given deployment.

Instead of shipping all possible specializations ahead of time, the binary
ships a compact, portable JIT-compilable representation of the computation
(e.g., IR, parameterized source, DSL) and generates only the variant that is
actually needed, on the hardware that is actually present, with the parameters
that are actually in use. The deployed binary grows with the number of
*algorithms*, not the number of *specializations*.

## Runtime-Dependent Optimization

The shape of a computation is typically unknown at compile time. In deep
learning, tensor shapes (batch size, sequence length, channel count) are
determined by user input, model configuration, or middleware decisions made
long after the binary is built.

These runtime conditions directly affect the optimal generated code:

- **Kernel fusion**: Fusing two adjacent operations into a
  single kernel avoids writing intermediates to global memory, which can be
  orders of magnitude faster.

- **Constant specialization**: Loops with bounds that are known at JIT
  compile time can be fully unrolled or vectorized in ways the AOT
  compiler could not do.

- **Memory access pattern specialization**: Stride patterns, padding, and
  alignment may not be known until JIT time, enabling the compiler to emit
  vectorized loads and eliminate bounds checks that an AOT compiler must
  conservatively retain.

## Code Generation as a First-Class Operation

In many modern systems, code generation is not an edge case — it *is* the
work. A query engine does not ship a pre-compiled function for every possible
SQL query; it compiles a new function for every query it receives. A shader
compiler does not ship every shader a game might need; it compiles shaders
from descriptors provided by the game at runtime. A ML serving runtime does
not ship a kernel for every possible model graph and hardware combination;
it generates and compiles kernels on demand when a model is first loaded or
when input shapes change.

In these systems, the latency and throughput of the JIT compilation pipeline
is itself a performance-critical concern. The overhead of invoking an external
compiler process, the lack of integration with the application's memory model,
and the loss of type information at the C++ boundary all impose real costs.

## The Limits of Templates and `constexpr`

C++ already has powerful compile-time computation through templates,
`constexpr`/`consteval`/`constinit`, and reflection. These mechanisms are
valuable, but they specialize at *compile time*
of the C++ program, not at runtime. They cannot specialize code for a GPU
architecture discovered when a device driver is queried, for a tensor shape
read from a file, or for a sparsity pattern computed from runtime data.
JIT compilation is a qualitatively different capability that C++ currently
cannot express.

# Existing Practice

Runtime compilation is already widely deployed across the C++ ecosystem, but
it appears through a collection of incompatible tools and programming models.
This section surveys the most relevant existing practice and the tradeoffs each
approach implies.

## LLVM and Clang-Based JIT

LLVM provides JIT infrastructure through
[ORC JIT](https://llvm.org/docs/ORCv2.html), which
supports in-process code generation, symbol resolution, lazy compilation, and
target-specific code emission. In practice, ORC is often used either directly
on LLVM IR or indirectly through higher-level systems that lower to LLVM.

Clang-based interactive tooling
([`clang-repl`](https://clang.llvm.org/docs/ClangRepl.html),
[Cling](https://cling.readthedocs.io/en/latest/)) demonstrates that C++ itself
can be parsed and incrementally compiled at runtime. These systems are valuable
proof points, but they are not standardized, and their embedding interfaces are
implementation-specific.

ClangJIT
[P1609](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1609r1.html)
is relevant prior art that explores integrating JIT compilation directly into
C++ workflows, including syntax and semantic issues specific to C++.

Whilst Clang provides high-quality and low-latency optimization and code
generation for many targets, it is tightly coupled to LLVM/Clang
infrastructure, uses no C++-level abstractions portable to non-LLVM compilers,
and operates on string source code and LLVM IR.

## GPU and Accelerator Runtime Compilation

Accelerator programming ecosystems have made runtime compilation a practical
necessity.

- [**CUDA / NVRTC**](https://docs.nvidia.com/cuda/nvrtc/index.html):
  Applications compile CUDA code at runtime to target the
  exact installed NVIDIA architecture.
- [**OpenCL**](https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html):
  Runtime kernel compilation from source is a core model in many
  implementations.
- [**SYCL / oneAPI**](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html):
  Implementations commonly use deferred compilation and
  device-specific lowering, including combinations of AOT and JIT.

These models are proven for hardware-specific specialization and enabled
single-source deployments across many accelerator targets.

Limitations:

- Vendor or ecosystem lock-in at API and toolchain boundaries.
- Runtime source compilation can impose substantial latency.
- Weak integration with portable C++ type-level abstractions.
- Graphics APIs are predominantly shader-language or IR-string driven (GLSL,
  HLSL, SPIR-V toolchains).
- Using full C++ as the frontend for these pipelines is uncommon in practice
  because repeated runtime specialization would require repeatedly paying
  frontend costs (template instantiation, semantic analysis, AST construction).

## ML Compiler Stacks and Domain-Specific Systems

Modern ML systems rely heavily on JIT techniques, usually via IRs and staged
lowering pipelines.

- [**Halide**](https://halide-lang.org/) and [**TVM**](https://tvm.apache.org/)
  generate specialized kernels from scheduling and IR
  representations, then JIT or AOT compile target code.
- [**JAX/XLA**](https://jax.readthedocs.io/en/latest/) traces host-language
  programs into graph-level IR, performs
  target-aware optimization, and lowers to native code.
- [**MLIR-based pipelines**](https://mlir.llvm.org/) provide multi-level IRs
  to separate frontend
  semantics from backend lowering and runtime specialization.
- [**CUDA TileIR**](https://docs.nvidia.com/cuda/tile-ir/latest/): NVIDIA's
  TileIR exposes a higher-level abstraction for composing and specializing GPU
  kernels at the level of tiles and tensor operations. It is explicitly
  designed to be generated and lowered at runtime, enabling Python-level
  authoring with kernel specialization deferred to JIT time, without
  re-incurring the full cost of high-level semantic analysis per
  specialization.

Most of these systems are DSL- or framework-centric rather than C++-centric,
and crossing the boundary between C++ and framework IRs can lose type and
semantic information.

## Language-Level JIT Ecosystems Adjacent to C++

[Julia](https://julialang.org/) and Python ecosystems
([Numba](https://numba.readthedocs.io/en/stable/), JAX,
[`torch.compile`](https://pytorch.org/get-started/pytorch-2.0/)) expose JIT as
a first-class user model. Although these systems are not C++, they are
relevant because they frequently lower into LLVM or C++-adjacent backend
toolchains.

ML-centric applications are increasingly written in Python or Python-JIT-first
frameworks rather than C++, not for reasons of raw numerical performance (the
backends are C++ and LLVM) but because Python ecosystems make it natural to
express runtime specialization. Model code, kernel fusion strategy, and
operator dispatch logic that would previously have been written in C++ are now
written in Python to take advantage of the extensive JIT infrastructure. The
C++ layer is increasingly relegated to a fixed performance substrate rather
than a productive authoring surface.

This represents an ongoing and accelerating erosion of C++'s position in
numerical and ML computing. First-class JIT support in C++ is not a luxury
feature; it is a prerequisite for C++ remaining a relevant authoring language
in these domains.

## Domain-Specific C++ Techniques

C++ libraries often emulate JIT-like behavior with expression templates,
embedded DSLs, and runtime dispatch tables. These techniques can defer
evaluation and improve specialization, but they typically execute through
precompiled engines and cannot generally emit new machine code at runtime.

They are effective engineering patterns, yet they do not replace a true JIT
facility when runtime code generation is required.

## Synthesis

Existing practice establishes four facts:

- Runtime compilation is already essential in production C++-adjacent systems.
- Current solutions are fragmented across compiler, vendor, and framework
  boundaries.
- The most successful systems separate AOT preparation from low-latency JIT
  specialization, usually through a portable intermediate representation —
  CUDA TileIR being a recent and concrete illustration of this pattern.
- ML-centric authoring is actively migrating away from C++ toward Python-based
  JIT ecosystems precisely because C++ lacks a cohesive runtime compilation
  model — the performance substrate remains C++/LLVM, but the productive
  authoring layer does not.

The standardization opportunity is therefore not to invent JIT from scratch,
but to provide portable C++ abstractions for capabilities that the ecosystem
already depends on — and to do so before C++ loses its relevance as an
authoring language in these domains.

# Problem Statement

C++ has no standard facility for just-in-time compilation. This is not a minor
gap. It is a structural deficiency that forces every application requiring
runtime code generation to build or adopt a private, non-portable solution, and
it is one of the primary drivers behind the migration of ML application
authoring away from C++.

## No Standard Abstraction

The C++ standard library provides no types, no interfaces, and no language
features for requesting compilation at runtime, representing portable
intermediate code, or linking and invoking dynamically generated functions.
Applications that need JIT compilation must step entirely outside of standard
C++ and depend on one of the following:

- Compiler-specific C APIs (`libgccjit`, LLVM ORC via LLVM headers).
- Vendor-specific GPU runtime compilation APIs (NVRTC, OpenCL, SYCL device
  compilers).
- OS-specific dynamic loading primitives (`dlopen`/`LoadLibrary`) combined
  with shelling out to an external compiler process.
- Embedding a language runtime (LLVM, a scripting language, a DSL engine)
  and communicating across the type-system boundary through void pointers or
  string serialization.

None of these is portable. Each ties the application to a specific compiler,
OS, vendor, or framework. Migrating workloads across hardware or compiler
toolchains requires rewriting or re-wrapping the JIT layer, a cost that is
often prohibitive.

## Loss of C++ Type and Semantic Information

All current approaches require exiting the C++ type system at the JIT
boundary. Whether the interface is a source string, an IR module, or a
vendor API, the host program and the JIT-compiled code communicate through
raw function pointers, `void*` buffers, or serialized descriptors. The
compiler cannot verify that caller and callee agree on the type of data being
passed. Template instantiations, type aliases, concepts, and overload
resolution do not cross this boundary. Each specialization is an opaque blob
from the host program's perspective.

This creates a persistent class of bugs — type mismatches between host and
JIT code — that a standard solution with integrated type representation could
eliminate entirely.

## Two-Phase Cost

Source-string JIT approaches (NVRTC, clang-repl invoked programmatically,
`system()` + `dlopen`) impose the full cost of the C++ frontend on every
compilation request: source text must be parsed, template metaprograms must be
re-instantiated, semantic analysis must be re-performed, and target-independent
optimization passes must re-run. For code that is specialized only in its
final code-generation phase — differing only in a tile size, a data type, or
a target architecture — this repeated frontend work is pure waste.

Without a staged representation that separates AOT-precomputable work
from runtime specialization, any JIT facility built on top of standard C++ will
either be too slow for latency-sensitive call sites or will require
non-standard, implementation-specific pre-compilation pipelines.

The model must support **fast incremental specialization**: after an initial
compilation, producing a nearby variant (e.g., different tile size, datatype,
or launch geometry) should reuse prior analysis and cached artifacts rather
than replaying full frontend and midend work.

## Fragmentation and Compounding Costs

Because there is no standard solution, every framework, every vendor library,
and every application solves the problem independently. The costs compound:

- **Integration friction**: composing two libraries that each embed a
  different JIT backend requires managing two sets of incompatible handles,
  contexts, and memory models.
- **Tooling blindness**: debuggers, sanitizers, profilers, and static analysis
  tools have no standard way to understand JIT-compiled code, symbol
  resolution, or the provenance of dynamically generated machine code.
- **Duplicated effort**: NVRTC, ORC JIT wrappers, SYCL compilation pipelines,
  and in-house JIT layers all solve the same underlying problem — runtime code
  specialization — without interoperability or shared abstractions.

## The Ecosystem Consequence

The cumulative effect is that C++ has become unsuitable as an authoring
language for applications that require first-class JIT. The standard does not
provide the primitives; every workaround is costly, non-portable, and
invisible to standard tooling. Developers working in domains where JIT is
essential — ML serving, query compilation, graphics shaders, scientific
computing — increasingly choose language ecosystems (Python, Julia) that
provide cohesive JIT models, even when those ecosystems ultimately rely on
C++-backed backends for performance.

The problem C++ needs to solve is not implementing a JIT engine. It is
providing standard abstractions that allow C++ programs to express, control,
and interoperate with JIT compilation in a portable, type-safe, and tooling-
aware manner.



## Authoring Note

This paper was written by the authors with AI-assistance.

# Acknowledgements

<!-- TODO: Acknowledge contributors and reviewers. -->
