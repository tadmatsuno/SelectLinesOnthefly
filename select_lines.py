# a script from Georges Kordopatis

def select_lines(full_sp,el_sp, Teff, vald, purity_crit, SNR, sampling, Resolution=None, fwhm=None, Element=None, ion=None, verbose=None):
    '''
    input:
    ** full_sp: DataFrame containing a column 'll' with the wavelengths and 'flux' with the full spectrum of a star (all the elements, all the molecules, blends etc)
    ** el_sp: DataFrame containing a column 'll' with the wavelengths and 'flux' with the spectrum of a given element only, computed in a similar way as full_sp (in order to have the same continuum opacities)
    ** Teff: Effective Teff, used in Boltzmann equation. 
    ** vald: pandas Dataframe containing the vald line-parameters (ll, Echi, loggf, Elem, ion) -- (Echi in eV).
    ** purity_crit: minimum required purity to keep the line
    ** Resolution: lambda/fwhm. Replaced previous keyword which was just fwhm
    ** fwhm: in A. Resolution element of the spectrograph
    ** SNR : minimum SNR per resolution element (used for line detection)
    ** sampling: spectrum's sampling (in A)
    ** verbose (optional): print information while running
    
    returns: 
    one panda sdata frame, with the following columns:
    ** ll : central wavelength where either side of the line has a putiry higher than purity_crit
    ** Blueratio: Purity of the line, defined as the ratio of the element spectrum and the full spectrum at lambda0-1.5xFWHM. 
    ** Redratio : Purity of the line, defined as the ratio of the element spectrum and the full spectrum at lambda0+1.5xFWHM. 
    ** Fullratio: Purity of the line, defined as the ratio of the element spectrum and the full spectrum at lambda0+/-1.5xFWHM. 
    ** Maxratio:  max between the right and the left blend of the line.
    ** fmin: depth of the core of line (as identified by the algorithm) for the element spectrum
    ** fmin_sp: depth of the full spectrum at the position of the core of the line. 
    ** width: width in which the ratio has been computed
    ** BlueFlag: Number of pixels that have a flux>0.9 within 1.5 times the FWHM (resolution element) 
    ** RedFlag: Number of pixels that have a flux>0.9 within 1.5 times the FWHM (resolution element)
    
    
    
    Steps: 
    1) Identifies the centers of the lines by computing the first two derivatives of the element spectrum.  
    
    2) Does a x-match with the vald linelist. 
    When several VALD lines are within +/- 1 pixel from the derived line core, 
    the line that has the highest value of the boltzmann equation is selected.
    ==>log(Boltzmann): log(N)=log(A) -(E_chi/kT)+log(gf)
        log(A) is a constant we can neglect
        loggf is in vald
        T is the temperature of the star
        E_chi is the excitation potential 
    
    ==> Caution: By using Boltzmann equation to select the lines,we assume that for a given element, 
        all of the lines correspond to the same ionisation level. If this is not the case, 
        we need to involve Saha's equation too. This is not implemented yet. 
    ==> Additional Caution: when there is hyperfine structure, then the lambda of Vald that we will 
        find is not necessarily the center of the line we will be seeing

    3) Estimates the depth of the line and compares it to Careyl's formula. sigma_fmin = 1.5/SNR_resol 
    If the depth of the line is large enough to be seen at a given SNR, then the line is selected. 
    
    
    4) We estimate the width of the line as the pixel in which the flux of the element itself is close enough to the continuum. 
    
    Once the line is selected, we compute the ratio between the element spectrum and the full spectrum. 
    Note: we require that if ratio<0.8 then we must have at least two pixels of the total spectrum with flux>0.9 within 1.5 FWHM,
    
    History: 
    23 Mar. 2025: Optimised the script with one loop less. Should be faster. I assume that all of the lines I find are in VALD. I don't see when this should not be true
    25 Dec. 2024: replacing the double derivative by the find_peaks routine of scipy. When looking in VALD, looking only for VALD entrie of that element if the element is written in the input.
    23 Dec. 2024: changing the keywords. Now there is the choice between Resolution or fwhm. If resolution is put, then fwhm is recomputed as a fnction of wavelength. 
    28 Nov. 2024: discovered that if the line is saturated and does not have a clear core, then my code does not identify this line. This has not been fixed yet
    20 Apr. 2023: replaced np.argmin (deprecated) with idxmin, that caused code to crash for machines with updated numpy
    10 Feb. 2023: Curated the Code
    04 Feb. 2023: Cleaned the readme. 
    
    Contact: Georges Kordopatis - georges.kordopatis -at- oca.eu
    '''

    import numpy as np
    import pandas as pd
    from scipy.signal import convolve
    from scipy.signal import find_peaks
    import sys
    
    #print(datetime.datetime.now())
    
    #verifying that the wavelength arrays of the two spectra are the same
    test=np.where(el_sp['ll'] != full_sp['ll'])[0]
    ll=full_sp['ll'].values
    if len(test) !=0 :
        print ('NOT SAME WAVELENGTHS')
        
        
    # Program needs either fwhm or R. 
    if ((Resolution is None) & (fwhm is None)): sys.exit('Both Resolution and fwhm are set to None')
    if ((Resolution is not None) & (fwhm is not None)): 
        print('Both Resolution and fwhm provided. Setting fwhm to zero and using Resolution.')
        fwhm=None

    depth=1.-3.*(1.5/SNR) # for a 3sigma detection. Based on Careyl's 1988 formula
    peak=3.*(1.5/SNR)
    #print('DEPTH:',depth)

    #### Using find_peaks to find the lines
    zz, _ = find_peaks(1-el_sp['flux'], height=peak, prominence=peak)
    
    
    # I continue doing stuff if I find lines. Otherwise I skip
    myresult=pd.DataFrame(
        columns=['ll', 'Bluewidth', 'Redwidth', 'Maxratio',
                 'fmin', 'fmin_sp', 'Blueratio', 'Redratio',
                 'Fullratio','Blueflag','Redflag'])
    #print('lines found:',len(zz))
    if len(zz)>0:
        # I have found lines. I don't have to look all VALD. I only keep the one of my element
        if Element is not None:
            if ion is None: vsel=np.where(vald['Elem']==Element)[0]
            else: vsel=np.where( (vald['Elem']==Element) & (vald['ion']==ion) )[0]
            vald=vald.iloc[vsel].reset_index(drop=True)
            
        ####################################################
        ############ BOLTZMANN METHOD *#####################
        ####################################################
        kboltzmann=8.61733034e-5 # in eV/K
        vald_centers_preliminary=np.zeros(len(zz)) 
        # contains the wavelengths (at the pixels) of the peaks
        #print('ALL VALD:',len(vald))
        for j in range(0,len(zz)):
            search=np.abs(ll[zz[j]]-vald['ll'].values)
            if Resolution is not None: fwhm=ll[zz[j]]/Resolution
            myvald=np.where((vald['ll'].values>=ll[zz[j]]-0.5*fwhm) & (vald['ll'].values<=ll[zz[j]]+0.5*fwhm) ) [0]

            if len(myvald)>1:
                myBoltzmann=-1.*vald['Echi'][myvald].values/(kboltzmann*Teff)+vald['loggf'][myvald].values
                mysel=np.where(myBoltzmann==np.max(myBoltzmann))[0]
                vald_centers_preliminary[j]=vald['ll'].values[myvald[mysel[0]]]
                if verbose: print(ll[zz[j]],'len(myvald)>1', vald['ll'].values[myvald[mysel[0]]])

            else:
                if verbose: print(ll[zz[j]],'-->',len(myvald))
                myvald=np.where(search==np.min(search))[0] # Note that this allows the center of the line to be out of the sampling. 
                vald_centers_preliminary[j]=vald['ll'].values[myvald[0]]
                if verbose: print(len(myvald),vald['ll'].values[myvald[0]])
        vald_unique, vald_unique_index=np.unique(vald_centers_preliminary, return_index=True)

        centers_index=zz[vald_unique_index]
        centers_ll=np.array(vald_unique)
        #print('FOUND/xmatched',len(zz),len(centers_ll))
        #####################################


        #Integration of the fluxes in the element spectrum and the full spectrum
        n_lines=len(centers_ll)
        Fratio=np.zeros(n_lines)*np.nan
        Fratio_all=np.zeros(n_lines)*np.nan
        Fratio_blue=np.zeros(n_lines)*np.nan
        Fratio_red=np.zeros(n_lines)*np.nan

        width_blue=np.zeros(n_lines)*np.nan
        width_red=np.zeros(n_lines)*np.nan

        flag_blue=np.empty(n_lines, dtype=int)*0
        flag_red=np.empty(n_lines, dtype=int)*0


        for j in range(0,n_lines):
            #myfwhm=centers_ll[j]/Resolution
            #print(fwhm, myfwhm)
            if Resolution is not None: fwhm=centers_ll[j]/Resolution
            nfwhm=1.5
            half_window_width=nfwhm*fwhm # the total window is 3 fwhm # This needs to be modified in order to take into account the actual wavelength
        

            # two selections: blue (left) part of the line, red (right) part of the line
            window_sel_blue=np.where((ll>=centers_ll[j]-half_window_width) & (ll<=centers_ll[j]))[0]
            window_sel_red=np.where((ll<=centers_ll[j]+half_window_width) & (ll>=centers_ll[j]) )[0]

            width_blue[j]=ll[window_sel_blue[0]] # this will be overwritten if criteria below are fulfilled. 
            width_red[j]=ll[window_sel_red[-1]] # this will be overwritten if criteria below are fulfilled.

            
            for ww in range(0,2): # loop on blue and red wing of the line
                if ww==0: mywindow=window_sel_blue #blue window
                if ww==1: mywindow=window_sel_red # red window

                cont_crit= 1.-(np.min(el_sp['flux'][mywindow])*0.02) #(We are back to the continuum levels more or less 2% of the depth of the line)
                cont_search=np.where(el_sp['flux'][mywindow]>=cont_crit)[0]

                full_continumm_search=np.where(full_sp['flux'][mywindow]>=0.9)[0] # in order to establish the flags. We want the full spectrum to have a flux >0.9. And we search in a range of +/-1.5FWHM and not the width of the line. 

                if len(cont_search)>=1:
                    if ww==0:
                        width_blue[j]=max(ll[mywindow[cont_search]])
                        window_sel_blue=np.where((ll>=width_blue[j]) & (ll<=centers_ll[j]))[0]
                        mywindow=window_sel_blue
                    if ww==1: 
                        width_red[j]=min(ll[mywindow[cont_search]])
                        window_sel_red=np.where((ll<=width_red[j]) & (ll>=centers_ll[j]))[0]
                        mywindow=window_sel_red
                
                if len(cont_search)<1:
                    if ww==0: 
                        #check if the flux decreases. 
                        test=np.where(el_sp['flux'].iloc[mywindow]<el_sp['flux'].iloc[centers_index[j]])[0]                        
                        if len(test)>0: width_blue[j]=max(ll[mywindow[test]])
                            
                    if ww==1: 
                        #check if the flux decreases. 
                        test=np.where(el_sp['flux'].iloc[mywindow]<el_sp['flux'].iloc[centers_index[j]])[0]
                        if len(test)>0: width_red[j]=min(ll[mywindow[test]])
                            
                    

                myflux_element=np.sum(1-el_sp['flux'][mywindow])
                myflux_full_spectrum=np.sum(1-full_sp['flux'][mywindow])
                myline_flux_ratio=myflux_element/myflux_full_spectrum

                if ww==0: 
                    Fratio_blue[j]=np.round(myline_flux_ratio,3)
                    flag_blue[j]=len(full_continumm_search)
                if ww==1: 
                    Fratio_red[j]=np.round(myline_flux_ratio,3)
                    flag_red[j]=len(full_continumm_search)


            full_window_sel=np.append(window_sel_blue,window_sel_red) # this now contains the full width of the line
            flux_element=np.sum(1-el_sp['flux'][full_window_sel])
            flux_full_spectrum=np.sum(1-full_sp['flux'][full_window_sel])
            line_flux_ratio=flux_element/flux_full_spectrum

            Fratio_all[j]=np.round(line_flux_ratio,3)
            Fratio[j]=max([Fratio_blue[j],Fratio_red[j]])


            #print(line_flux_ratio, line_flux_ratio1,line_flux_ratio2,Fratio[j])

        keep=np.where(Fratio>purity_crit)[0]

        
        myresult['ll']=np.round(centers_ll[keep],5)
        myresult['Bluewidth']=np.round(width_blue[keep],5)
        myresult['Redwidth']=np.round(width_red[keep],5)
        myresult['Maxratio']=Fratio[keep]
        myresult['fmin']=np.round(el_sp['flux'][centers_index[keep]].values,3)
        myresult['fmin_sp']=np.round(full_sp['flux'][centers_index[keep]].values,3)
        myresult['Blueratio']=Fratio_blue[keep]
        myresult['Redratio']=Fratio_red[keep]
        myresult['Fullratio']=Fratio_all[keep]
        myresult['Blueflag']=flag_blue[keep]
        myresult['Redflag']=flag_red[keep]

        if verbose: 
            print(centers_ll)
            print('N lines found:',len(vald_unique), ', N lines kept:', len(keep) )

    return(myresult)