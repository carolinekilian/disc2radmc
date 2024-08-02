import numpy as np
import disc2radmc 
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits

#The following two lines aren't applicable for coding done in command line/running a vim file so use as applicable
#%matplotlib inline
#%config InlineBackend.figure_format='retina'

def Sigma_all(r, phi, rc, sigr):
    
    return sigma_0*(r/rc)**-gamma * np.exp(-1.0*(r/rc)**(2.0-gamma))# check units!

########################
###### PARAMETERS ######
########################

## STAR
## STAR
dpc=129.9  # [pc] distance to source 
target='HD110058hales'# name
Rstar=1.6 # [Solar radii]
Tstar= -8000 # [K] If this is negative, the code will consider the star as a blackbody. 
g=4.0
# For a realistic stellar model, you can download bt-settl models 
# from http://svo2.cab.inta-csic.es/theory/newov2/index.php and indicate their directory below
# dir_stellar_templates='/Users/Sebamarino/Astronomy/Stellar_templates/BT-Settl/bt-settl/'
# The code will then search for files named as 
# dir_stellar_templates+'lte%03i-%1.1f-0.0a+0.0.BT-NextGen.7.dat.txt'%(Tstar//100,g) if Tstar>=2600
# dir_stellar_templates+'lte%03i-%1.1f-0.0.BT-Settl.7.dat.txt'%(T//100,g)            else
# and interpolate models with neighbouring temperatures if necessary

## DUST
Mdust=0.080      # [Mearth] Total dust mass
rc=31.        # [au]
#sigr=50.       # [au]
gamma=-0.47    
amin=1.0       # [mu]  minimum grain size (1 by default)
amax=1.0e4     # [mu]  maximum grain size (1e4 by default)
N_species=1    #  (1 by default) Number of dust size bins to use for the radiative transfer calculations. 
slope = -3.5   #  (-3.5 by default) Slope of the size distribution. This is used for computing opacities and for the mass distribution of each size bin
h=0.17         # vertical aspect ratio =H/r, where H is the vertical standard deviation. This is a constant, but other parametrizations are possible and will be shown below. 
par_sigma=(rc, sigr) # list containing the parameters that define the dust surface density. They must be in the same order as in the definition of Sigma_dust

## GAS
gas_species=['12c16o', 'catom'] # species. Each one of these must have a file named molecule_*.inp containing its cross sections. These can be downloaded from https://home.strw.leidenuniv.nl/~moldata/
Masses=np.array([1.0e-2, 1.0e-3 ]) # [Mearth] Total mass of each gas species
masses=np.array([28.*disc2radmc.mp, 12*disc2radmc.mp ]) # [g] molecular weight of each species
mu=14. # mean molecular weight. This could be = np.sum(masses*Masses)/np.sum(Msses)/np.disc2radmc.mp
turbulence=True
alpha_turb=1.0e-2

## MODEL SPATIAL GRID
rmin=10. # [au] make sure it is small enough to sample the surface density
rmax=300.# [au] 
Nr=50    # radial cells (linearly or logspaced)
Nphi=50 # azimuthal cells
Nth=50   # polar angle cells (per emisphere)
thmax=np.arctan(h)*10 # maximum polar angle to sample as measured from the midplane.
axisym=False # Consider the disc to be axisymmetric to speed up calculations? it can overwrite Nphi if True and set it to 1
mirror=False  # Mirror the upper half to speed up calculations. This is incompatible with anisotropic scattering. When including radial velocities in the gas, if the model is mirrored the channel maps appear wrong (not sure why).

logr=True # Sample r logarithmically or linearly

# WAVELENGTH GRID (grid to sample the stellar flux in temperature calculations, see radmc3d manual)
lammin=0.09  # [mu]  minimum wavelength to consider in our model (important for temperature calculation)
lammax=1.0e5 # [mu] minimum wavelength to consider in our model (important for temperature calculation)
Nlam=150     # number of cells logarithmically spaced to sample the wavelength range.

# IMAGE PARAMETERS
Npix=512  # number of pixels
dpix=0.03 # pixel size in arcsec
inc=78.    # inclination
PA=157.    # position angle

wavelength=880. # [um] image wavelength
scattering_mode=1 # scattering mode (0=no scattering, 1=isotropic, 2=anisotropic using H&G function)

### PHYSICAL GRID
gridmodel=disc2radmc.physical_grid(rmin=rmin, rmax=rmax, Nr=Nr, Nphi=Nphi, Nth=Nth, thmax=thmax, mirror=mirror, logr=logr, axisym=axisym)
gridmodel.save()

### WAVELENGTH GRID
lammodel=disc2radmc.wavelength_grid(lammin=lammin, lammax=lammax, Nlam=Nlam)
lammodel.save()

### STAR
starmodel=disc2radmc.star(lammodel, Tstar=Tstar, Rstar=Rstar, g=g,
                               #dir_stellar_templates=dir_stellar_templates # necessary line if Tstar>0
                               )
starmodel.save()

### DUST SIZE DISTRIBUTION AND OPACITY
# path to optical constants that can be found at
# https://github.com/SebaMarino/disc2radmc/tree/main/opacities/dust_optical_constants
path_opct='/Volumes/disks/carolinek/disc2radmc/HD110058/opacities/dust_optical_constants' 
lnk_files=[path_opct+'astrosilicate_ext.lnk',
           path_opct+'ac_opct.lnk',
           path_opct+'ice_opct.lnk']
densities=[4., 3., 1.] # densities in g/cm3
mass_weights=[70.0, 15., 15.] # mixing ratios by mass
dust=disc2radmc.dust(lammodel,
                           Mdust=Mdust,
                           lnk_file=lnk_files,
                           densities=densities,
                           N_species=N_species,
                           slope=slope,
                           N_per_bin=100, # number of species per size bin to have a good representation 
                           mass_weights=mass_weights,
                           tag='mix', # name to give to this new mixed species
                           compute_opct=True) 

#dust.compute_opacities()

### DUST DENSITY DISTRIBUTION
dust.dust_densities(grid=gridmodel,function_sigma=Sigma_all, par_sigma=par_sigma, h=h)
dust.write_density()

sim=disc2radmc.simulation(nphot=10000000, # number of photon packages for thermal monte carlo
                            nphot_scat=1000000, # number of photon packages for image
                            nphot_spec=10000,   # number of photon packages for spectrum
                            nphot_mono=10000,   # number of photon packages for the monochromatic monte carlo
                            scattering_mode=scattering_mode, 
                            incl_lines=1, # whether to include gas lines (1) or not (0)
                            modified_random_walk=0, # for very optically thick medium, this is a useful approximation
                            istar_sphere=0, # consider the star a sphere or a point.
                            tgas_eq_tdust=1, # gas temperature equal to the temperature of the first dust species
                            setthreads=4,
                            verbose=True, 
                               )

### RUN MCTHERM to compute temperature
sim.mctherm()

# check temperature (northern emisphere)
Ts=np.fromfile('./dust_temperature.bdat', count=gridmodel.Nr*gridmodel.Nphi*gridmodel.Nth+4, dtype=float)[4:].reshape( (gridmodel.Nphi, gridmodel.Nth, gridmodel.Nr))
plt.pcolormesh(gridmodel.redge, gridmodel.thedge[::-1], Ts[0,:gridmodel.Nth,:])
plt.xlabel('Radius [Rsun]')
plt.xscale('log')
plt.ylabel(r'Polar angle [rad]')
plt.colorbar(label='Temperature [K]')
plt.show()

#def rhoz_exotic(z, H):
#    return 0.5*( np.exp(-(z-2*H)**2.0/(2.0*(H/2)**2.0))/(np.sqrt(2.0*np.pi)*H/2)+np.exp(-(z+2*H)**2.0/(2.0*(H/2)**2.0))/(np.sqrt(2.0*np.pi)*H/2) )

# vertical parameters
h=0.17
r0=rc # reference radius
gamma=1.

gas=disc2radmc.gas(gas_species=gas_species,
                         star=starmodel,
                         grid=gridmodel,
                         functions_sigma=[Sigma_all, Sigma_all], # surface density function for each gas species
                         pars_sigma=[par_sigma, par_sigma], # parameters for each gas species
                         Masses=Masses, # Total gas mass of each species
                         masses=masses, # molecular weights
                         h=h, # vertical aspect ratio
                         r0=r0,
                         gamma=gamma,
                         mu=mu, # mean molecular weight
                         turbulence=turbulence,
                         alpha_turb=alpha_turb,
                         functions_rhoz=[rhoz_exotic,rhoz_exotic], # one for each gas species
                        )
gas.write_density()
gas.write_velocity()
gas.write_turbulence()
sim.simcube(dpc=dpc,
            imagename='12c16o_3', # name that the image will have
            mol=1,  # indicate which gas species
            line=2, # indicate which transition
            vmax=8, # maximum velocity in km/s
            Nnu=9,# number of channels
            Npix=Npix,
            dpix=dpix,
            inc=90.,
            PA=PA,
            tag=target,
            continuum_subtraction=True # whether you want to subtract the continuum estimated using the first and last channel.
           )

fit1=pyfits.open('./images/cvel_frhb1_data.fits')
cube=fit1[0].data[0,:,:,:]
header1	= fit1[0].header

Nchan=np.shape(cube)[0]

# let's calculate the velocities of each channel
ckms=299792.458 # km/s
f_line=230.538000 # GHz
df=float(header1['CDELT3'])/1.0e9 # GHz
k0=float(header1['CRPIX3'])
f0=float(header1['CRVAL3'])/1.0e9 - (k0-1)*df # GHz
fs=np.linspace(f0,f0+df*(Nchan-1),Nchan) #GHz
vs=-(fs-f_line)*ckms/f_line  # km/s

channels_plot=np.arange(9)
Nchan=len(channels_plot)
fig=plt.figure(figsize=(12,12))
for ichan in range(Nchan):
    axi=fig.add_subplot(331+ichan)
    axi.pcolormesh(cube[channels_plot[ichan],:,:], vmin=0., vmax=np.max(cube), cmap='Blues')
    axi.set_aspect('equal')
    axi.text(50,450, '%1.2f km/s'%vs[channels_plot[ichan]], color='black' )
