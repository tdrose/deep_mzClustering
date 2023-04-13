# Deep mzClustering

Implementation of the mzClustering Algorithm from the publication ["A noise-robust deep clustering of biomolecular ions improves interpretability of mass spectrometric images" 2023 _Bioinformatics_](https://doi.org/10.1093/bioinformatics/btad067)


The original implementation from the authors was not functional at commit [d63a469](https://github.com/DanGuo1223/mzClustering/commit/d63a46979ddfb94c0d60ac338463a5e827210a5a).
The code was missing functions (e.g. `pseudo_labeling`) and
many dimensions were hardcoded (only working with 40x40 images).

I forked of the original [code](https://github.com/DanGuo1223/mzClustering) and implemented support for varying image sizes, 
wrote all missing functions, and fixed other minor issues.
Furthermore, I improved compatibility with [METASPACE](https://metaspace2020.eu/) datasets.

