# BaroSphere
Barotropic Vorticity Equation on a sphere, using the PySpharm package. The model itself is contained in the BaroSphere module. Using conda the requirements can be installed using 
    conda install requirements.txt

The barotropic vorticity equation can be thought of as a thin, rotating, spherical shell of fluid. The model uses the pyshparm module, which provides access to spherical harmonic transforms between data in physical grid space and data in spectral wavenumber space. The model itself is a pseudospectral model, following the ideas in the GFDL barotropic core description. 

wave_instability.ipynb contains a notebook that runs the model with a jet with a barotropically unstable wave superimposed on top, following Held and Phillips (1987).     

Stochastic_Jet.ipynb contains a notebook that "stirs" the atmosphere in a small jet, meant to mimic the effects of baroclinic instability in the midlatitudes, follows an exmaple in Vallis et. al. (2004) with some small mofidications. 

## References

[pyspharm](https://github.com/jswhit/pyspharm), see especially the Galewsky et. al. Test Case Example.

[GFDL Barotropic Core description](https://www.gfdl.noaa.gov/wp-content/uploads/files/user_files/pjp/barotropic.pdf)

Vallis, G, Gerber, E.P., Kushner, P.J., and B. Cash (2004): A Mechanism and Simple Dynamical Model of the North Atlantic Oscillation and Annular Modes. JAS, Vol. 61, 264-280 available [here](https://edwinpgerber.github.io/files/vallis_etal-JAS-2004.pdf)

Held, I.M., and P.J. Phillips (1987): Linear and Nonliear Barotropic Decay on the Spher. JAS, Vol. 44 200-207, available [here](https://journals.ametsoc.org/view/journals/atsc/44/1/1520-0469_1987_044_0200_lanbdo_2_0_co_2.xml)
