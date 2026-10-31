# Pinned HotSpot model reference

Source repository: <https://github.com/yuhc/gpu-rodinia>, commit
`9c10d3ea16ddba2ba057cc3951a9efc4c2cc18a4`.
[provenance.json](provenance.json) records the exact source URLs and byte hashes.
The reference sources and [upstream license](LICENSE) are retained without edits;
they are not compiled into ATLAS.

ATLAS follows the 2D CUDA source's physical constants, five-point thermal model,
and nominal timestep. The OpenMP source at this commit divides its timestep by a
further 1000, so the two programs are not interchangeable numerical references.
ATLAS uses FP64, a conservative explicit-timestep stability limit, rectangular and
degenerate grids, single-timestep kernels, and its own CPU/GPU ownership protocol.
These are documented differences, not historical reproduction claims.

Relevant upstream citations:

- S. Che et al., *Rodinia: A Benchmark Suite for Heterogeneous Computing*,
  IEEE International Symposium on Workload Characterization, 2009.
- J. Meng and K. Skadron, *Performance Modeling and Automatic Ghost Zone
  Optimization for Iterative Stencil Loops on GPUs*, ICS, 2009.

The suite is described by the [University of Virginia Rodinia project](https://rodinia.cs.virginia.edu/).
