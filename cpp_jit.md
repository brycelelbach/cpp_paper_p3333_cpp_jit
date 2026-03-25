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

C++ has historically been an ahead-of-time (AOT) compiled language: programs
are compiled once, linked into a binary, and executed. This model has served
the language well for decades, delivering predictable performance and tight
control over the generated code. However, the rise of machine learning and
heterogeneous computing has exposed a fundamental limitation of the AOT model:
the hardware a program will run on, and the exact shape of the data it will
process, are often not known until runtime.

Modern ML workloads routinely require runtime code generation. A neural network
inference engine may need to compile a fused kernel for a specific sequence of
operations, a specific batch size, or a specific GPU architecture discovered
only at deployment time. In many cases, the code to be executed does not exist
at all when the binary is shipped — it is synthesized entirely at runtime:
new functions are constructed, specialized, and compiled on the fly in response
to conditions only observable during execution. This is qualitatively different
from template instantiation or link-time optimization; the program is not merely
being specialized from a fixed set of possibilities known to the compiler, but
is generating wholly new code as part of its normal operation.

This need is not unique to ML. Scientific computing, database query engines,
shader compilation in graphics runtimes, and policy-driven dispatch systems all
share the same fundamental requirement: generate code at runtime, compile it
efficiently, and execute it with minimal overhead.

Other language ecosystems have embraced JIT compilation as a first-class
citizen. Julia is built around a JIT model that specializes code for every
concrete argument type. Python's ML ecosystem layers JIT facilities such as
Numba, JAX, and `torch.compile` on top of the interpreter. These approaches
reach C++-adjacent performance precisely because they invoke C++ or LLVM-based
compilation pipelines under the hood — yet C++ itself provides no standard
mechanism for an application to do the same.

The result is fragmentation. Each framework, each hardware vendor, and each
application that needs JIT compilation in C++ invents its own solution: NVRTC
for CUDA, OpenCL runtime compilation, libgccjit, ORC JIT via LLVM, or simply
shelling out to a compiler process and `dlopen`-ing the result. These approaches
are non-portable, difficult to compose, and invisible to the C++ type system and
tooling.

This paper surveys existing practice, articulates the problem clearly, and
proposes a direction for standardizing JIT compilation facilities in C++.

# Motivation

## The Hardware Diversity Problem

The compute landscape has fragmented dramatically. A single ML inference
deployment today may target any combination of NVIDIA GPUs (with
architecture-specific instruction sets spanning Volta, Turing, Ampere,
Hopper, and Blackwell), AMD GPUs (CDNA, RDNA), Intel GPUs (Xe), Google TPUs,
Groq LPUs, and an ever-growing ecosystem of custom ASICs. Each of these
devices has a distinct ISA, a distinct memory hierarchy, distinct warp or
wavefront dimensions, and distinct performance characteristics.

AOT compilation cannot produce a single binary that is simultaneously optimal
for all of these targets. Fat binaries — shipping a pre-compiled variant for
every known target — only partially address the problem. The set of hardware
targets is not closed; new accelerators are released continuously. More
fundamentally, the *optimal code* for a given operation is not a fixed
function of the hardware alone. It also depends on runtime parameters that the
AOT compiler cannot observe.

## Binary Size and Deployment Constraints

The canonical AOT answer to hardware diversity is the fat binary: ship a
pre-compiled variant for every known target and select the right one at load
time. This works at small scale, but it does not scale to the full
combinatorial space of modern deployment targets.

Consider a ML framework that wishes to ship optimized kernels for matrix
multiplication across NVIDIA Hopper, Ampere, and Volta; AMD CDNA3 and CDNA2;
and Intel Xe. For each architecture, there may be dozens of variants
specialized for different tile sizes, data types (float32_t, float16_t, bfloat16_t, float8_t, int8_t),
and memory layouts. The number of kernels grows multiplicatively. Compiled GPU
kernels are not small — a single highly-optimized GEMM kernel may be hundreds
of kilobytes of device code. Shipping thousands of such variants adds up to
gigabytes of device code in the binary, most of which will never execute on any
given deployment.

JIT compilation inverts this tradeoff. Instead of shipping all possible
specializations ahead of time, the binary ships a compact, portable
representation of the computation — whether that is a high-level IR, a
kernel description, or parameterized source — and generates only the variant
that is actually needed, on the hardware that is actually present, with the
parameters that are actually in use. The deployed binary grows with the number
of *algorithms*, not the number of *specializations*.

This matters beyond ML. Embedded and resource-constrained environments where
binary size is strictly bounded can benefit from shipping compact representations
and deferring native code generation to first use. Edge deployments where the
target device is not known at package build time have no other option.

## Runtime-Dependent Optimization

The shape of a computation is typically unknown at compile time. In deep
learning, tensor shapes (batch size, sequence length, channel count) are
determined by user input, model configuration, or middleware decisions made
long after the binary is built. Sparsity patterns in weight matrices may only
be known after a quantization or pruning step that happens at load time.
The memory layout of buffers may depend on which other operations happen to be
co-scheduled on a device.

These runtime conditions directly affect the optimal generated code:

- **Kernel fusion**: Whether two adjacent operations should be fused into a
  single kernel depends on their relative costs and memory pressure — both
  runtime quantities. A fused kernel that avoids writing an intermediate tensor
  to global memory can be orders of magnitude faster, but the fusion must be
  planned and compiled after those costs are known.

- **Constant specialization**: Loops with bounds that are known at JIT
  compile time can be fully unrolled or vectorized in ways the ahead-of-time
  compiler could not do. A matrix multiplication kernel specialized for a
  fixed tile size is fundamentally different from a general one.

- **Memory access pattern specialization**: Stride patterns, padding, and
  alignment may all be known at JIT time, enabling the compiler to emit
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
`constexpr`, and (with C++23 and beyond) `static constexpr` evaluation and
reflection. These mechanisms are valuable, but they operate at *compile time*
of the C++ program, not at runtime. They cannot specialize code for a GPU
architecture discovered when a device driver is queried, for a tensor shape
read from a file, or for a sparsity pattern computed from runtime data.

Expression templates and embedded DSLs can delay the *description* of a
computation until runtime, but they still execute through a fixed,
pre-compiled evaluation engine. They cannot generate and compile new machine
code. JIT compilation is not a generalization of these techniques; it is a
qualitatively different capability that C++ currently cannot express.

# Existing Practice

Runtime compilation is already widely deployed across the C++ ecosystem, but
it appears through a collection of incompatible tools and programming models.
This section surveys the most relevant existing practice and the tradeoffs each
approach implies.

## LLVM and Clang-Based JIT

LLVM provides industrial-strength JIT infrastructure through ORC JIT, which
supports in-process code generation, symbol resolution, lazy compilation, and
target-specific code emission. In practice, ORC is often used either directly
on LLVM IR or indirectly through higher-level systems that lower to LLVM.

Clang-based interactive tooling (`clang-repl`, Cling) demonstrates that C++
itself can be parsed and incrementally compiled at runtime. These systems are
valuable proof points, but they are not standardized, and their embedding
interfaces are implementation-specific.

Strengths:

- High-quality optimization and code generation on many targets.
- Mature ecosystem and production use.
- Supports both low-latency and high-throughput compilation pipelines.

Limitations:

- Tightly coupled to LLVM/Clang internals and versioning.
- No common C++-level abstraction portable to non-LLVM compilers.
- Applications often operate on strings or LLVM IR, not C++ language entities.

## GCC-Based JIT

`libgccjit` exposes GCC as an embeddable JIT compiler through a C API. It is
used in several language runtimes and dynamic instrumentation tools.

Strengths:

- Reuses GCC optimization and backend infrastructure.
- Provides an in-process compilation model.

Limitations:

- Ecosystem is smaller than LLVM-based JIT deployment.
- API surface is compiler-specific and not interoperable with Clang/MSVC-based
  toolchains.

## GPU and Accelerator Runtime Compilation

Accelerator programming ecosystems have made runtime compilation a practical
necessity.

- **CUDA / NVRTC**: Applications compile CUDA code at runtime to target the
  exact installed NVIDIA architecture.
- **OpenCL**: Runtime kernel compilation from source is a core model in many
  implementations.
- **SYCL / oneAPI**: Implementations commonly use deferred compilation and
  device-specific lowering, including combinations of AOT and JIT.

Strengths:

- Proven model for hardware-specific specialization.
- Enables single-source deployments across many accelerator targets.

Limitations:

- Vendor or ecosystem lock-in at API and toolchain boundaries.
- Runtime source compilation can impose substantial latency.
- Weak integration with portable C++ type-level abstractions.
- Graphics APIs are predominantly shader-language or IR-string driven (GLSL,
  HLSL, SPIR-V toolchains), so host C++ semantics are typically erased at the
  boundary and reconstructed through textual interfaces.
- Using full C++ as the frontend for these pipelines is uncommon in practice
  because repeated runtime specialization would require repeatedly paying
  frontend costs (template instantiation, semantic analysis, AST construction)
  unless a staged/cacheable representation is available.

## ML Compiler Stacks and Domain-Specific Systems

Modern ML systems rely heavily on JIT techniques, usually via intermediate
representations and staged lowering pipelines.

- **Halide** and **TVM** generate specialized kernels from scheduling and IR
  representations, then JIT or AOT compile target code.
- **JAX/XLA** traces host-language programs into graph-level IR, performs
  target-aware optimization, and lowers to native code.
- **MLIR-based pipelines** provide multi-level IRs to separate frontend
  semantics from backend lowering and runtime specialization.
- **CUDA TileIR**: NVIDIA's recently announced tile-level IR exposes a
  higher-level abstraction for composing and specializing GPU kernels at the
  level of tiles and tensor operations. It is explicitly designed to be
  generated and lowered at runtime, enabling Python-level authoring with
  kernel specialization deferred to JIT time. TileIR is a concrete instance
  of the staged-representation model: a compact, hardware-independent
  description of tiled computation that a backend can lower to PTX or a
  device-specific ISA on demand, without re-incurring the full cost of
  high-level semantic analysis per specialization.

Strengths:

- Explicit staging boundaries reduce repeated frontend work.
- Rich optimization opportunities at multiple IR levels.
- Demonstrated performance portability for complex workloads.

Limitations:

- Most systems are DSL- or framework-centric rather than C++-centric.
- Crossing the boundary between C++ and framework IRs can lose type and
  semantic information.
- Toolchains are powerful but complex and difficult to standardize wholesale.

## Language-Level JIT Ecosystems Adjacent to C++

Julia and Python ecosystems (Numba, JAX, `torch.compile`) expose JIT as a
first-class user model. Although these systems are not C++, they are relevant
because they frequently lower into LLVM or C++-adjacent backend toolchains.

Key observation: developers routinely choose non-C++ frontends not because C++
cannot generate efficient machine code, but because those ecosystems provide a
cohesive runtime compilation model.

This observation has a measurable consequence for C++. ML-centric applications
are increasingly written in Python or Python-JIT-first frameworks rather than
C++, not for reasons of raw numerical performance — the backends are C++ and
LLVM — but specifically because Python ecosystems make it natural to express
runtime specialization. Model code, kernel fusion strategy, and operator
dispatch logic that would previously have been written in C++ are now written
in Python precisely to take advantage of `torch.compile` or JAX's tracing JIT.
The C++ layer is increasingly relegated to a fixed performance substrate
rather than a productive authoring surface.

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
boundary. Whether the interface is a source string, an IR module handle, or a
vendor API, the host program and the JIT-compiled code communicate through
raw function pointers, `void*` buffers, or serialized descriptors. The
compiler cannot verify that caller and callee agree on the type of data being
passed. Template instantiations, type aliases, concepts, and overload
resolution do not cross this boundary. Each specialization is an opaque blob
from the host program's perspective.

This creates a persistent class of bugs — type mismatches between host and
JIT code — that a standard solution with integrated type representation could
eliminate entirely.

## The Two-Phase Cost Problem

Source-string JIT approaches (NVRTC, clang-repl invoked programmatically,
`system()` + `dlopen`) impose the full cost of the C++ frontend on every
compilation request: source text must be parsed, template metaprograms must be
re-instantiated, semantic analysis must be re-performed, and target-independent
optimization passes must re-run. For code that is specialized only in its
final code-generation phase — differing only in a tile size, a data type, or
a target architecture — this repeated frontend work is pure waste.

Without a standard staged representation that separates AOT-precomputable work
from runtime specialization, any JIT facility built on top of standard C++ will
either be too slow for latency-sensitive call sites or will require
non-standard, implementation-specific pre-compilation pipelines. The standard
must address this directly: a portable, cacheable intermediate form must be
part of the model.

Equally important, the model must support **fast incremental specialization**:
after an initial compilation, producing a nearby variant (e.g., different tile
size, datatype, or launch geometry) should reuse prior analysis and cached
artifacts rather than replaying full frontend and midend work.

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
  - Two-phase compilation model: the standard must accommodate (or mandate)
    a separation between work that can be done AOT (parsing, type-checking,
    target-independent optimization) and work that must be done at JIT time
    (target-specific code generation, runtime-value specialization). This
    rules out "source string at runtime" as the sole primitive and implies
    the need for a portable intermediate representation that can be produced
    AOT, cached/serialized, and cheaply consumed at JIT time. This is a hard
    design constraint: any facility where JIT latency is dominated by
    frontend work that could have been done earlier is not suitable for use
    in hot paths such as query engines or shader compilers.
  - Graphics/shader pipeline integration: how does the facility interoperate
    with existing string/IR-centric APIs while preserving C++ semantics and
    avoiding repeated template-instantiation/AST-front-end work per variant?
  - Fast incremental specialization as a first-class requirement: once a
    kernel/function family is prepared, producing nearby specializations must
    be cheap, cache-friendly, and avoid redoing whole-program frontend work.
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
