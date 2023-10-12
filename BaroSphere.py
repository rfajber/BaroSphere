import numpy as np
import spharm 

'''
spharm reference  - this is here because I can never remember what is in spharm

Class methods
=============
    - grdtospec: grid to spectral transform (spherical harmonic analysis).
    - spectogrd: spectral to grid transform (spherical harmonic synthesis).
    - getuv:  compute u and v winds from spectral coefficients of vorticity
    and divergence.
    - getvrtdivspec: get spectral coefficients of vorticity and divergence
    from u and v winds.
    - getgrad: compute the vector gradient given spectral coefficients.
    - getpsichi: compute streamfunction and velocity potential from winds.
    - specsmooth:  isotropic spectral smoothing.

Functions
=========
    - regrid:  spectral re-gridding, with optional spectral smoothing and/or
    truncation.
    - gaussian_lats_wts: compute gaussian latitudes and weights.
    - getspecindx: compute indices of zonal wavenumber and degree
    for complex spherical harmonic coefficients.
    - legendre: compute associated legendre functions.
    - getgeodesicpts: computes the points on the surface of the sphere
    corresponding to a twenty-sided (icosahedral) geodesic.
    - specintrp: spectral interpolation to an arbitrary point on the sphere.

'''

class BaroSphere:

    def __init__(
        self,
        omega=7.291e-5,
        rsphere=6.3e6, 
        r=0.1,
        ntrunc=85,
        dt=1800., 
        efold=3*3600,
        damping_order=4,
        nlat=None 
    ):

        # input params 
        self.ntrunc = ntrunc
        self.damping_order = damping_order
        self.r = r
        self.rsphere = rsphere
        self.omega = omega
        self.dt = dt

        # set up the grid 
        # assume that we are using a triangular truncation on gaussian de-alising grid 
        if nlat is None:
            self.nlon = 3*self.ntrunc+1
            self.nlat = int(self.nlon/2)
        else:
            self.nlat=nlat
            self.nlon=2*self.nlat

        delta = 360./self.nlon
        self.lons1 = delta*np.arange(self.nlon)

        self.lats1, self.wt = spharm.gaussian_lats_wts(self.nlat)

        #sometimes we need the 2d fields 
        self.lons, self.lats = np.meshgrid(self.lons1, self.lats1)
        
        #handy conversion factor 
        self.a = np.pi / 180. 
        self.f = 2 * self.omega * np.sin(self.a * self.lats)

        #Spharmt object that does all the work doing transforms
        self.x = spharm.Spharmt(nlon=self.nlon,
                                nlat=self.nlat,
                                rsphere=self.rsphere,
                                gridtype='gaussian')

        #spectral indices
        self.indxm, self.indxn = spharm.getspecindx(self.ntrunc)
        #laplcian operator
        self.lap = -(self.indxn*(self.indxn+1.0)/self.rsphere**2)


        self.efold=efold
        self.hyperdiff_fact = np.exp((-self.dt/self.efold)*(self.lap/self.lap[-1])**(self.damping_order))

        return None 

    def get_vorticity_grid(self,u,v):
        vrtspec,_=self.x.getvrtdivspec(u,v,self.ntrunc)
        return self.x.spectogrd(vrtspec)

    def get_uv_vrtg(self,vrtg):
        vrts = self.x.grdtospec(vrtg)
        return self.x.getuv(vrts,0*vrts)

    def vrt_time_tendency(self,vrtg):

        # get vrtspec, u, and v on the grid 
        vrtspec = self.x.grdtospec(vrtg,self.ntrunc)
        u, v = self.x.getuv(vrtspec, 0 * vrtspec)

        # cheat to get the tendency 
        __,tend_spec=self.x.getvrtdivspec(
                                        (self.f + vrtg) * u,
                                        (self.f + vrtg) * v,
                                        self.ntrunc)

        # do the biharmonic filtering implicitly
        #tend_spec = (tend_spec - self.dfac1 * vrtspec) / self.dfac2

        # get back to grid space
        tend_grid = self.x.spectogrd(tend_spec)

        # return vorticity and tendency on gridspace
        return tend_grid

    def phys_tend(self):
        return 0 * self.vrtg

    def spectral_damping(self,vrtg):
        vrts=self.x.grdtospec(vrtg,self.ntrunc) 
        vrts_damp=vrts*self.hyperdiff_fact
        return self.x.spectogrd(vrts)

    #for time stepping
    def RA_leapfrog(self):

        #m1 n-1 time step
        #0 n time stpe 
        #p1 n+1 time step 

        # get tendencies 
        tend_grid = self.vrt_time_tendency(self.vrtg)

        # leap frog step
        vrtg_p1 = self.vrtg_m1 + 2 * self.dt * tend_grid 
        vrts_p1 = self.x.grdtospec(vrtg_p1,self.ntrunc)*self.hyperdiff_fact

        vrtg_p1 = self.x.spectogrd(vrts_p1)

        #apply spectral damping
#        vrtg_p1=self.spectral_damping(vrtg_p1)

        # Robert Filtering step
        vrtg_f = (1 - 2 *self.r) * self.vrtg + self.r * (vrtg_p1 + self.vrtg_m1) 

        #this is where we do the "physics"
        vrtg_f += self.dt * self.phys_tend()


        # update the variables before next time step             
        self.vrtg_m1 = vrtg_f
        self.vrtg = vrtg_p1

        return None

    def vrtg_unstable_jet(self,
                        U1=25,
                        U3=30,
                        U6=300,
                        A=8e-5,
                        yw=15,
                        y0=45,
                        m=6):

        # set up the jet
        C=np.cos(self.lats * self.a)
        S=np.sin(self.lats * self.a)
        U = U1 * C + U3 * C**3 + U6 * S**2 * C**6 

        # switch to vorticity and add perturbation
        vrtg0 = self.get_vorticity_grid(U,0*U)
        vrtg0 += A / 2 * C * np.exp( -np.power((self.lats - y0) / yw , 2))\
                 * np.cos( self.lons * self.a * m )
        
        return vrtg0

    #ensure that we can get and set attrs from outside.
    def get_attr(self,key):
        return vars(self)[key]

    def set_attr(self,key,val):
        vars(self)[key]=val
        return None