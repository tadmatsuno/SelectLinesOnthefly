Kordopatis et al. (2023) with on-the-fly spectral synthesis with Korg. See examples in SelectOntheFly.py.

A user would need to install 
- Korg (see https://ajwheeler.github.io/Korg.jl/stable/install/ for the installation instruction.)
- Glob, DataFrames in julia
  - in julia, one can install them using
    - using Pkg
    - Pkg.add("Glob")
    - Pkg.add("DataFrames")
  - or from python
    - jl.seval("Pkg")
    - jl.Pkg.add("Glob")
    - jl.Pkg.add("DataFrames")
- matplotlib, numpy, pandas
