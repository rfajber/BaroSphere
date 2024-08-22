import numpy as np
import scipy.fft as fft

'''
This class makes a barotropic vorticity model on a f plane

'''

class BarotropicPlane:

    def __init__(
        self,
        omega=7.291e-5,
        rsphere=6.3e6, 
        r=0.1,
        dt=1800., 
        efold=3*3600,
        damping_order=4,
        nlat=41,
        nlon=40,
        dlat=10,
        lat0=45
    ):

        # input params 
        self.damping_order = damping_order
        self.r = r
        self.rsphere = rsphere
        self.omega = omega
        self.dt = dt
        self.lat0 = lat0
        self.dlat = dlat
        self.nalt = nlat
        self.nlon = nlon

        # set up the grid 
        # assume that we are using a triangular truncation on gaussian de-alising grid 
        self.nlat=nlat
        self.nlon=nlon

        self.lons1 = 360./self.nlon*np.arange(self.nlon)
        self.lats1 = self.dlat/(self.nlat-1)*np.arange(-self.dlat/2,self.dlat/2+1e-3,self.dlat/(self.nlat-1))

        #sometimes we need the 2d fields 
        self.lons, self.lats = np.meshgrid(self.lons1, self.lats1)

        #handy conversion factor 
        self.a = np.pi / 180. 
        self.s = np.sin(self.a*self.lats)
        self.f = 2 * self.omega * np.sin(lat0*self.a)  

        sh = np.hstack([1,.5*(np.sin(self.lats1*self.a)[1:]+np.sin(self.lats1*self.a)[:-1]),-1])
        ds = sh[1:] - sh[:-1]
        self.ds = (np.tile(ds,[self.nlon,1])).T

        #spectral indices
        self.indxm = fft.fftfreq(self.nlon)
        self.indxn = fft.fftfreq(self.nlat)
        self.indxm2,self.indxn2 = np.meshgrid(self.indxm,self.indxn)
        self.indxk2 = self.indxn2**2+self.indxm2**2

        #laplcian operator
        self.lap = -(self.indxn2*(self.indxn2+1.0)/self.rsphere**2)

        self.efold=efold
        self.hyperdiff_fact = np.exp((-self.dt/self.efold)*(self.lap/self.lap[-1])**(self.damping_order))

        self.vrtg = np.zeros([self.nlat, self.nlon])
        self.vrtg_m1 = np.copy(self.vrtg)

        return None 

    def get_uv(self,vrtg):
        vrts = fft.fft2(vrtg)

        psis = vrts/self.indxk2
        psis[0,0]=0.0
        
        ug = fft.ifft2(self.indxn2*1j*psis)
        vg = -fft.ifft2(self.indxm2*1j*psis)
        return ug,vg

    def jac_term(self,vrtg):

        ug,vg = self.get_uv(vrtg)

        pvs = fft.fft2(vrtg+self.f)

        pvg_x = fft.ifft2(pvs*self.indxm2*1j)
        pvg_y = fft.ifft2(pvs*self.indxn2*1j)

        return ug*pvg_x + vg*pvg_y

    def phys_tend(self):
        return 0 #* self.vrtg

    def spectralDamping(self,f):
        return fft.ifft2(fft.fft2(f)*self.hyperdiff_fact)
        
    #for time stepping
    def RA_leapfrog(self):

        #m1 n-1 time step
        #   n time stpe 
        #p1 n+1 time step 

        tend_grid = -self.jac_term(self.vrtg)

        vrtg_p1 = self.vrtg_m1 + 2 * self.dt * tend_grid 
        vrtg_p1 = self.spectralDamping(vrtg_p1)

        # Robert Filtering step
        vrtg_f = (1 - 2 *self.r) * self.vrtg + self.r * (vrtg_p1 + self.vrtg_m1) 
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
        vrtg0 = 0.001*U1*np.cos(self.lats*self.a)/self.r
        vrtg0 += A / 2 * np.sin(self.lons*self.a*m)*np.exp(-(self.lats-self.lat0)**2/(self.dlat/5)**2) \
                 * np.sin( self.lons * self.a * m )
        
        return vrtg0