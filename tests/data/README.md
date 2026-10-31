# Native kmeans fixture

`kmeans_ties.kmdat` is a project-authored binary fixture in the ATLSKM01 format:
three one-dimensional points `[-2, 0, 2]`, with initial centroids `[-2, 2]`.
The first assignment must be `[0, 0, 1]` (lowest-index exact-tie rule), with
counts `[2, 1]`, updated centroids `[-1, 2]`, and objective decreasing from 4 to 2.

The file consists of an eight-byte ASCII magic, three little-endian uint32 values
N/D/K, then row-major little-endian float64 points and centroids. Its bytes can be
regenerated with `greengpu.kmeans_reference.write_dataset` using the `ties` fixture.
No external dataset or license is involved.

`hotspot_edge.hsdat` is a locally generated 3×5 thermal grid in ATLSHS01 format:
eight-byte magic, little-endian uint32 H/W, then FP64 temperature and power arrays.
Regenerate with `hotspot_reference.generated(3,5,661,'edge')` and `write_input`.
The native CTests move the split repeatedly through endpoints and interior rows,
compare scripted and feedback execution, and check the final field independently
in both per-step verification and deferred-validation modes.
