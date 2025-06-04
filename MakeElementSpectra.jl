using Korg
using Glob
using DataFrames

"""
    ElementalSpectra(
        marcs_model,
        feh,
        element,
        wvl_min,
        wvl_max,
        resolution;
        A_X_dict = Dict{String,Float64}(),
        vt = 1.0,
        linelist = nothing,
        ion = [1, 2]
    )

Calculate synthetic spectra for a star with non‑standard abundances over a narrow wavelength range,
ready for use with Georges Kordopatis’s line selection code.

# Arguments
- `marcs_model` : model atmosphere (e.g. MARCS)
- `feh::Real`  : metallicity
- `element::String` : element of interest, e.g. `"Th"`
- `wvl_min::Real`, `wvl_max::Real` : wavelength bounds in Å (air)
- `resolution::Real` : spectral resolution
- `A_X_dict::Dict{String,Float64}` : abundances A(X), e.g. `Dict("Th"=>0.5)`
- `vt::Real` : microturbulence in km/s (default = 1.0)
- `linelist` : Korg linelist object or `nothing` to use default VALD
- `ion::Vector{Int}` : ionization stages to include (default = [1,2])
"""
function ElementalSpectra(
    marcs_model :: Union{String,Korg.ModelAtmosphere},
    feh,
    element::String,
    wvl_min,
    wvl_max,
    resolution;
    A_X_dict=Dict{String,Float64}(),
    vt=1.0,
    linelist::Union{Nothing, String, PyDict, Dict } = nothing,
    ion = 0
)

wvl_min = Korg.air_to_vacuum(wvl_min)
wvl_max = Korg.air_to_vacuum(wvl_max)

if linelist === nothing
    linelist = "./linelist/valdall.h5"
    lines = Korg.read_linelist(linelist)
elseif linelist isa String
    lines = Korg.read_linelist(linelist)
else
    lines = Korg.Line.(Korg.air_to_vacuum.(linelist["wl"]), linelist["log_gf"], Korg.Species.(linelist["Species"]), linelist["E_lower"])
end

if marcs_model isa String
    model_atm = Korg.read_model_atmosphere(marcs_model)
else
    model_atm = marcs_model
end
input_abundances = Korg.format_A_X(feh, A_X_dict, solar_relative=false)

full_sp = Korg.synthesize(model_atm, lines, input_abundances, wvl_min, wvl_max, vmic=vt)
flux_unnorm = Korg.apply_LSF(full_sp.flux, full_sp.wavelengths, resolution)
flux_cntm = Korg.apply_LSF(full_sp.cntm, full_sp.wavelengths, resolution)
flx_full = flux_unnorm ./ flux_cntm

wvl_full = Korg.vacuum_to_air.(full_sp.wavelengths)

formula_element = Korg.Formula(element)
istheelement = broadcast(line->line.species.formula == formula_element, lines)
istheion = broadcast(line->line.species.charge == ion, lines)
lines_species = lines[istheelement .&& istheion]

elem_sp = Korg.synthesize(model_atm, lines_species, input_abundances, wvl_min, wvl_max, vmic=vt, hydrogen_lines=false)
flux_unnorm = Korg.apply_LSF(elem_sp.flux, elem_sp.wavelengths, resolution)
flux_cntm = Korg.apply_LSF(elem_sp.cntm, elem_sp.wavelengths, resolution)
flx_elem = flux_unnorm ./ flux_cntm

df_lines = DataFrame([
    (ll = Korg.vacuum_to_air.(line.wl)*1.0e8, Echi=line.E_lower, loggf=line.log_gf) 
    for line in lines_species])


wvl_full, flx_full, flx_elem, df_lines
end


"""
Create a single h5 linelist file from multiple VALD linelist

Parameters
----------
    linelist_files : list of str
        List of paths to the VALD linelist files.
    
    output : str, optional
        Path to t
        he output h5 file.
        Default is "./linelist/valdall.h5".

"""
function create_h5linelist(
    linelist_files::Vector{String}=glob("./linelist/[0-9]000_[0-9]000*");
    output::String="./linelist/valdall.h5"
)
    linelists = Korg.read_linelist.(linelist_files)
    lines_all = vcat(linelists...)
    Korg.save_linelist(output,lines_all)
end

