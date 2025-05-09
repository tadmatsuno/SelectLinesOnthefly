import matplotlib.pyplot as plt
import numpy as np
import select_lines
import pandas

from juliacall import Main as jl
jl.include("./MakeElementSpectra.jl")
#jl.seval("using Korg")
#Korg = jl.Korg

available_models = {\
    (4750, 1.5, -1.0): "s4750_g+1.5_m1.0_t02_an_z-1.00_a-0.40_c+0.00_n+0.00_o-0.40_r+0.00_s+0.00.mod",
    }

def ElementalSpectra(marcs_model, feh, element, wvl_min, wvl_max, resolution, 
    A_X_dict={}, vt=1.0, linelist = None, ion=0):
    """
    Calculate synthetic spectra needed to using Georges Kordopatis's line selection code.
    This function is meant to be used for stars with non-standard abundances for a narrow wavelength range.

    Parameters
    ----------
    teff : float
        Effective temperature.

    logg : float
        Surface gravity.

    feh : float
        Metallicity.

    element : str
        Element of interest. e.g., Th
    
    wvl_min, wvl_max : float
        Wavelength range in Angstroms.

    resolution : float
        Spectral resolution to be simulated.
    
    A_X_dict : dict, optional
        Dictionary of abundances in the A(X) scale.
        e.g., A_X_dict = {'Th': 0.5} to assume A(Th) = 0.5.

    vt : float, optional
        Microturbulence velocity in km/s.
        Default is 1.0 km/s.
    
    linelist : optional
        This has to be a Korg's linelist object.
        If None, it will use the default VALD solar linelist.

    """
    wvl, flx_full, flx_elem, lines_species = jl.ElementalSpectra(\
        marcs_model, feh, element, wvl_min, wvl_max, resolution, 
        A_X_dict=A_X_dict, vt=vt, linelist=linelist, ion=ion)

    return wvl.to_numpy(), flx_full.to_numpy(), flx_elem.to_numpy(),lines_species

def run_select_lines(marcs_model, feh, element, wvl_min, wvl_max, resolution, 
    A_X_dict={}, vt=1.0, linelist = None, ion=0):
    wvl, flx_full, flx_elem, lines_species = ElementalSpectra(marcs_model, \
        feh, \
        element, 
        wvl_min, wvl_max, resolution,
        A_X_dict=A_X_dict, vt=vt, linelist=linelist, ion=ion)
    sp_full = pandas.DataFrame({"ll":wvl,"flux":flx_full})
    sp_elem = pandas.DataFrame({"ll":wvl,"flux":flx_elem})
    linelist_pd_sp = pandas.DataFrame({\
        key: getattr(lines_species,key) for key in ["ll","Echi","loggf"]})
#    return sp_full, sp_elem, linelist_pd
#    print(linelist_pd)
    return select_lines.select_lines(sp_full, sp_elem, 4750, linelist_pd_sp, 0.0, Resolution=resolution, SNR=200, sampling=0.01)

marcs_model = "./model_atms/s4750_g+1.5_m1.0_t02_an_z-1.00_a-0.40_c+0.00_n+0.00_o-0.40_r+0.00_s+0.00.mod"

run_select_lines(marcs_model, -1.0, "Th", 4000, 4500, 50000, A_X_dict={"Th":-0.98}, vt=1.0,ion=1)

sp1,sp2 = run_select_lines(marcs_model, -1.0, "Th", 4000, 4500, 50000, A_X_dict={"Th":-0.98}, vt=1.0,ion=1)
plt.plot(sp1["ll"],sp1["flux"])
plt.plot(sp2["ll"],sp2["flux"])
#plt.xlim(4015,4025)

run_select_lines(marcs_model, -1.0, "Th", 4000, 4500, 50000, A_X_dict={"Th":0.02}, vt=1.0,ion=1)
sp1,sp2 = run_select_lines(marcs_model, -1.0, "Th", 4000, 4500, 50000, A_X_dict={"Th":0.02}, vt=1.0,ion=1)
plt.plot(sp1["ll"],sp1["flux"])
plt.plot(sp2["ll"],sp2["flux"])
#plt.xlim(4015,4025)
