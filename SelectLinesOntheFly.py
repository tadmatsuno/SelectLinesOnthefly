import matplotlib.pyplot as plt
import numpy as np
import select_lines
import pandas

from juliacall import Main as jl
jl.include("./MakeElementSpectra.jl")
#jl.seval("using Korg")
#Korg = jl.Korg


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

def run_select_lines(marcs_model, feh, teff, element, wvl_min, wvl_max, resolution, 
    A_X_dict={}, vt=1.0, linelist = None, ion=0, snr=200, min_purity = 0.5):
    """ 
    Linelist can either be a .h5 file that contains a Korg linelist or 
    a dictionary with (Species, wl, Echi, log_gf).
    The Species format has to be supported by Korg, 
    e.g., "Fe I", "Fe 1", "Fe_1", "Fe.I" or MOOG format BUT WITHOUT ISOTOPES.
    """
    
    wvl, flx_full, flx_elem, lines_species = ElementalSpectra(marcs_model, \
        feh, \
        element, 
        wvl_min, wvl_max, resolution,
        A_X_dict=A_X_dict, vt=vt, linelist=linelist, ion=ion)
    
    sp_full = pandas.DataFrame({"ll":wvl,"flux":flx_full})
    sp_elem = pandas.DataFrame({"ll":wvl,"flux":flx_elem})
    linelist_pd_sp = pandas.DataFrame({\
        key: getattr(lines_species,key) for key in ["ll","Echi","loggf"]})

    select_lines_results = select_lines.select_lines(\
        sp_full, sp_elem, teff, linelist_pd_sp, min_purity, Resolution=resolution, 
        SNR=snr, sampling=0.01)
    return sp_full,sp_elem,select_lines_results


def example():
    """
    Example of how to use the function.
    """
    
    model = "./model_atms/feh-2.00/teff4750_logg1.5.mod"
    sp_full, sp_elem, results = run_select_lines(
        model, 
        feh = -2.0, 
        teff = 4750, # There should be a better way to get Teff and feh from the model atmosphere.
        element="Th", # Create a result for Th
        wvl_min=4000, 
        wvl_max=4200, 
        resolution=50000,
        A_X_dict={"Eu":-0.48,"Th":-0.98},# A(Eu)=-0.48, A(Th) = -0.98, [Eu/Fe]=[Th/Fe] = 1.
        vt=1.0,
        linelist=None, # None means use the default VALD linelist
        ion = 1 # This limits to Th II
        )
    print(results)
    
    fig, axs = plt.subplots(len(results),1, figsize=(10,5*len(results)))
    for idx,ax in zip(results.index,axs):
        plt.sca(ax)            
        plt.plot(sp_full["ll"],sp_full["flux"],label="Full")
        plt.plot(sp_elem["ll"],sp_elem["flux"],label="Element")
        plt.xlim(results.loc[idx,"Bluewidth"]-1.0,results.loc[idx,"Redwidth"]+1.0)
        plt.fill_betweenx([0.0,1.5],results.loc[idx,"Bluewidth"],results.loc[idx,"Redwidth"],alpha=0.2,color="C7")
        plt.ylim(results.loc[idx,"fmin_sp"]*0.9,1.1)
        plt.legend()
        
def example2():
    """
    Example2 with a custom linelist.
    """
    linelist = pandas.read_csv("./linelist/linelist5989_published.csv")
    linelist = linelist.to_dict(orient="list")
    print(linelist)
    
    model = "./model_atms/feh-2.00/teff4750_logg1.5.mod"
    sp_full, sp_elem, results = run_select_lines(
        model, 
        feh = -2.0, 
        teff = 4750, # There should be a better way to get Teff and feh from the model atmosphere.
        element="Th", # Create a result for Th
        wvl_min=5980, 
        wvl_max=6000, 
        resolution=50000,
        A_X_dict={"Eu":-0.48,"Th":-0.98},# A(Eu)=-0.48, A(Th) = -0.98, [Eu/Fe]=[Th/Fe] = 1.
        vt=1.0,
        linelist=linelist,
        ion = 1 # This limits to Th II
        )
    print(results)
    
    fig, axs = plt.subplots(len(results),1, figsize=(10,5*len(results)))
    axs = np.atleast_1d(axs)
    for idx,ax in zip(results.index,axs):
        plt.sca(ax)            
        plt.plot(sp_full["ll"],sp_full["flux"],label="Full")
        plt.plot(sp_elem["ll"],sp_elem["flux"],label="Element")
        plt.xlim(results.loc[idx,"Bluewidth"]-1.0,results.loc[idx,"Redwidth"]+1.0)
        plt.fill_betweenx([0.0,1.5],results.loc[idx,"Bluewidth"],results.loc[idx,"Redwidth"],alpha=0.2,color="C7")
        plt.ylim(results.loc[idx,"fmin_sp"]*0.9,1.1)
        plt.legend()

if __name__ == "__main__":
    example()    

    example2()
