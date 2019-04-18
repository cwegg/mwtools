import abc
import copy
from abc import abstractmethod
from enum import Enum

import numpy as np
import scipy
import scipy.optimize
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.special import lambertw

import mwtools.nemo as nemo
from mwtools.nmagic import NMagicParticles

from galpy.util import bovy_conversion
from galpy.potential import evaluateRforces, evaluatezforces


_BIG_G = 4.302E-6  # Gravitational constant in astronomical units kpc/Msun (km/s)^2


class PotentialType(Enum):
    Stellar = 1
    Dark = 2
    ISM = 3
    Baryonic = 4
    Total = 5

    def isa(self, type):
        if self is type:
            return True
        elif type is PotentialType.Total:
            return True
        elif type is PotentialType.Baryonic and \
                ((self is PotentialType.Stellar) or \
                 (self is PotentialType.ISM)):
            return True
        else:
            return False


class AxiSymmetricPotential(object):
    __metaclass__ = abc.ABCMeta

    # We implement addition by as a linked list of potentials
    # which we descend whenever we need a force/label
    def __add__(self, sibling):
        ret = copy.deepcopy(self)
        if ret._sibling is not None:
            ret._sibling = ret._sibling + sibling  # descend list
            return ret
        ret._sibling = copy.deepcopy(sibling)  # add to end of list
        return ret

    def __init__(self, pottype=PotentialType.Dark):
        self._sibling = None
        self.pottype = pottype

    def str_list(self, level=0):
        if self._sibling is None:
            return 'level {:d} {} \n'.format(level, str(self))
        else:
            return 'level {:d} {} \n'.format(level, str(self)) + self._sibling.str_list(
                level + 1)

    @property
    @abstractmethod
    def my_nparm(self):
        """Return the number of parameters of this potential."""

    @property
    def my_param_reasonable(self):
        """Return the number of parameters of this potential."""
        return []

    @property
    @abstractmethod
    def my_labels(self):
        """Return a vector of the parameter names for plot labelling"""

    @property
    @abstractmethod
    def my_bounds(self):
        """Return the parameters bounds vector."""

    @abstractmethod
    def my_f_cylr(self, r, ang, parms=None, rel_tol=None):
        """Return the force in cylindical R direction at (r, theta)"""

    @abstractmethod
    def my_f_z(self, r, ang=90, parms=None, rel_tol=None):
        """Return the force in the z-direction at (r, theta)"""

    # ---------------------------------------------------
    # The following methods all descend the list
    @property
    def dm(self):
        """Return just the first dark matter we find"""
        if self.pottype.isa(PotentialType.Dark):
            ret = copy.deepcopy(self)
            ret._sibling = None
            return ret
        else:
            if self._sibling is None:
                return None
            else:
                return self._sibling.dm

    @property
    def param_reasonable(self):
        """Return the number of parameters of the total potential."""
        if self._sibling is None:
            return self.my_param_reasonable
        else:
            return self.my_param_reasonable + self._sibling.param_reasonable

    @property
    def nparm(self):
        """Return the number of parameters of the total potential."""
        if self._sibling is None:
            return self.my_nparm
        else:
            return self.my_nparm + self._sibling.nparm

    @property
    def labels(self):
        """Return the number of parameters of the total potential."""
        if self._sibling is None:
            return self.my_labels
        else:
            return self.my_labels + self._sibling.labels

    @property
    def bounds(self):
        """Return the number of parameters of the total potential."""
        if self._sibling is None:
            return self.my_bounds
        else:
            my_boundsl, my_boundsu = self.my_bounds
            sibling_boundsl, sibling_boundsu = self._sibling.bounds
            return my_boundsl + sibling_boundsl, my_boundsu + sibling_boundsu

    def f_z(self, r, ang=90, parms=(), pottype=PotentialType.Total, rel_tol=None):
        """Return the force in the z-direction at (r, theta)"""
        if self._sibling is None:
            if self.pottype.isa(pottype):
                my_f_z = self.my_f_z(r, ang, parms, rel_tol)
            else:
                my_f_z = np.zeros_like(r * ang)
            return my_f_z
        else:
            if self.pottype.isa(pottype):
                my_f_z = self.my_f_z(r, ang, parms[0:self.my_nparm], rel_tol)
            else:
                my_f_z = np.zeros_like(r * ang)
            sibling_f_z = self._sibling.f_z(r, ang, parms[self.my_nparm:], pottype, rel_tol)
            return my_f_z + sibling_f_z

    def f_cylr(self, r, ang=90, parms=(), pottype=PotentialType.Total, rel_tol=None):
        """Return the force in the z-direction at (r, theta)"""
        if self._sibling is None:
            if self.pottype.isa(pottype):
                my_f_cylr = self.my_f_cylr(r, ang, parms, rel_tol)
            else:
                my_f_cylr = np.zeros_like(r * ang)
            return my_f_cylr
        else:
            if self.pottype.isa(pottype):
                my_f_cylr = self.my_f_cylr(r, ang, parms[0:self.my_nparm], rel_tol)
            else:
                my_f_cylr = np.zeros_like(r * ang)
            sibling_f_cylr = self._sibling.f_cylr(r, ang, parms[self.my_nparm:], pottype, rel_tol)
            return my_f_cylr + sibling_f_cylr

    # ---------------------------------------------------

    def f_r(self, r, ang=0, *args, **kwargs):
        """Return the radial force at (r, theta)"""
        z, r_cyl = self.spherical_to_cylindrical(r, ang)
        return (r_cyl * self.f_cylr(r, ang, *args, **kwargs) +
                z * self.f_z(r, ang, *args, **kwargs)) / r

    def f_theta(self, r, ang, *args, **kwargs):
        """Return the force in the theta direction at (r, theta)"""
        z, r_cyl = self.spherical_to_cylindrical(r, ang)
        return (z * self.f_cylr(r, ang, *args, **kwargs) -
                r_cyl * self.f_z(r, ang, *args, **kwargs)) / r

    def vc2(self, r, ang=0, *args, **kwargs):
        """Return the circular velocity at (r, theta)"""
        return -self.f_r(r, ang, *args, **kwargs) * r

    def pot_ellip(self, r, ang, *args, **kwargs):
        """Return the ellipticity of the potential"""
        z, r_cyl = self.spherical_to_cylindrical(r, ang)
        return np.sqrt(z * self.f_cylr(r, ang, *args, **kwargs) /
                       (r_cyl * self.f_z(r, ang, *args, **kwargs)))

    @classmethod
    def spherical_to_cylindrical(cls, r, ang):
        z = r * np.sin(np.radians(ang))
        r_cyl = np.sqrt(r ** 2 - z ** 2)
        return z, r_cyl


class AxiSymmetricPotentialFixedParms(AxiSymmetricPotential):
    """Stupid class to fix the parameters of an AxiSymmetricPotential"""

    def __init__(self, pot, parms):
        self.parms = copy.copy(parms)
        self.pot = copy.deepcopy(pot)
        super().__init__(pottype=self.pot.type)

    def __str__(self):
        return 'AxiSymmetricPotentialFixedParms Holding:' + str(self.pot) + ' with ' + str(self.parms)

    @property
    def my_nparm(self):
        """Return the number of parameters of the profile - We have none since we are fixed"""
        return 0

    @property
    def my_labels(self):
        return []

    @property
    def my_bounds(self):
        return [], []

    def my_f_cylr(self, r, ang, parms=None, rel_tol=None):
        return self.pot.my_f_cylr(r, ang, self.parms, rel_tol)

    def my_f_z(self, r, ang, parms=None, rel_tol=None):
        return self.pot.my_f_z(r, ang, self.parms, rel_tol)


class EllipsoidalProfile(AxiSymmetricPotential):
    __metaclass__ = abc.ABCMeta

    def __init__(self, r0=8.2, pottype=PotentialType.Baryonic):
        self.r0 = r0
        super().__init__(pottype=pottype)

    @abstractmethod
    def rho(self, m, parms):
        """Return the density at ellipsoidal coordinate m"""
        return

    def _f_compute(self, r, ang, parms, rel_tol, direction='cyl'):
        if rel_tol is None:
            rel_tol = 1e-6
        if parms is None:
            raise ValueError("We were expecting a parameter array")
        z0, r0 = self.spherical_to_cylindrical(r, ang)
        q = parms[-1]

        if direction == 'cyl':
            # Change variables of the integral from BT's tau over 0->inf, to x = (1/tau-1)**3 over 0->1.
            # Tests suggested 3rd power generally provided better convergence than 1,2,4...
            def integrand(x):
                tau = (1 / x - 1) ** 3
                x_mat = np.broadcast_to(x, r0.shape + x.shape)
                tau_mat = np.broadcast_to(tau, r0.shape + x.shape)
                m = np.sqrt(r0[..., np.newaxis] ** 2 / (tau_mat + 1) + z0[..., np.newaxis] ** 2 / (tau_mat + q ** 2))
                return self.rho(m, parms) / (tau_mat + 1) ** 2 / np.sqrt(tau_mat + q ** 2) * 3 * tau / x_mat / (
                        1 - x_mat)

            integral = r0 * self._fixedquad(integrand, rel_tol=rel_tol)

        elif direction == 'z':

            def integrand(x):
                tau = (1 / x - 1) ** 3
                x_mat = np.broadcast_to(x, r0.shape + x.shape)
                tau_mat = np.broadcast_to(tau, r0.shape + x.shape)
                m = np.sqrt(r0[..., np.newaxis] ** 2 / (tau_mat + 1) + z0[..., np.newaxis] ** 2 / (tau_mat + q ** 2))
                return self.rho(m, parms) / (tau_mat + 1) / (tau_mat + q ** 2) ** 1.5 * 3 * tau / x_mat / (1 - x_mat)

            integral = z0 * self._fixedquad(integrand, rel_tol=rel_tol)

        else:
            raise ValueError("We were expecting a parameter array")

        return -2 * np.pi * _BIG_G * q * integral

    def my_f_cylr(self, r, ang, parms=None, rel_tol=None):
        """Return the force in cylindical R direction at (r, theta)
        with rho in Msun/kpc^3 , R in kpc then returns in (km/s)^2/kpc"""
        return self._f_compute(r, ang, parms, rel_tol, direction='cyl')

    def my_f_z(self, r, ang, parms=None, rel_tol=None):
        """Return the force in the z-direction at (r, theta)
        with rho in Msun/kpc^3 , R in kpc then returns in (km/s)^2/kpc"""
        return self._f_compute(r, ang, parms, rel_tol, direction='z')

    @staticmethod
    def _fixedquad(func, n=None, n_max=100, n_min=20, rel_tol=1e-6):
        """Integrate func from 0->1 using Gaussian quadrature of order n if set.
        Else provide answer with estimated relative error less than rel_tol (up to a
        maximum order of n_max"""
        if n is None:
            val = old_val = integrate.fixed_quad(func, 0, 1, n=n_min)[0]
            for n in range(n_min + 5, n_max, 5):
                val = integrate.fixed_quad(func, 0, 1, n=n)[0]
                rel_err = np.max(np.abs((val - old_val) / val))
                if rel_err < rel_tol:
                    break
                old_val = val
        else:
            val = integrate.fixed_quad(integrand_closed, 0, 1, n=n)[0]
        return val


class EinastoProfile(EllipsoidalProfile):
    # reasonable values to start fits
    @property
    def my_param_reasonable(self):
        """Return reasonable parameters for this potential."""
        return [1.70e7, 8.0, 0.7, 0.8]

    @property
    def my_labels(self):
        return [r'$\rho_{0,dm}$', r'$m_0$', r'$\alpha_{dm}$', r'$q$']

    @property
    def my_nparm(self):
        """Return the number of parameters of the dm profile."""
        return 4

    @property
    def my_bounds(self):
        return [0, 0, 0, 0], [np.inf, np.inf, 8., np.inf]

    def rho(self, m, theta):
        """Einasto profile: rho0,alpha,m0=theta
        rho0 is the density,
        alpha detirmines how quickly dlog rho/dlog changes from -2 to core
        m0 is the radius where dlog rho/dlog r =-2"""
        rhor0, m0, alpha = (theta[0], theta[1], theta[2])

        # first get normalisation from rho(R0)
        rho0 = rhor0 / (np.exp(-(2 / alpha) * ((self.r0 / m0) ** alpha - 1)))

        return rho0 * np.exp(-(2 / alpha) * ((m / m0) ** alpha - 1))


class NFWProfile(EllipsoidalProfile):
    @property
    def my_labels(self):
        return [r'$\rho_{0,dm}$', r'$R_s$', r'$q$']

    @property
    def my_param_reasonable(self):
        """Return reasonable parameters for this potential."""
        return [3e7, 10., 1.0]

    @property
    def my_nparm(self):
        """Return the number of parameters of the dm profile."""
        return 3

    @property
    def my_bounds(self):
        return [0, 0, 0], [np.inf, 250., np.inf]

    def rho(self, m, theta):
        """NFW profile: rho0,Rs=theta
        rho0 is the density at R0,
        Rs is the characteristic radius of switch between
        dlog rho/dlog -1 and -3 profiles"""
        rhor0, rs = theta[0], theta[1]

        # first get normalisation from rho(R0)
        r0 = np.abs(self.r0 / rs)
        rho0 = rhor0 * r0 * (1 + r0) ** 2

        r = np.abs(m / rs)
        return rho0 / r / (1 + r) ** 2


class HubbleProfile(EllipsoidalProfile):
    @property
    def my_labels(self):
        return [r'$\rho_{0,dm}$', r'$a_0$', r'$q$']

    @property
    def my_param_reasonable(self):
        """Return reasonable parameters for this potential."""
        return [1.0, 1., 1.0]

    @property
    def my_nparm(self):
        """Return the number of parameters of the dm profile."""
        return 3

    @property
    def my_bounds(self):
        return [0, 0, 0], [np.inf, 1e3, np.inf]

    def rho(self, m, theta):
        """Test hubble profile. See BT87 Eq 2-92"""
        rho0, a0 = theta[0], theta[1]
        return rho0 * (1 + (m / a0) ** 2) ** (-1.5)


class GeneralizedNFWProfile(EllipsoidalProfile):
    @property
    def my_labels(self):
        return [r'$\rho_{0,dm}$', r'$R_s$', r'$\gamma$', r'$q$']

    @property
    def my_param_reasonable(self):
        """Return reasonable parameters for this potential."""
        return [2e9, 2.2, -2.0, 1.0]

    @property
    def my_nparm(self):
        """Return the number of parameters of the dm profile."""
        return 4

    @property
    def my_bounds(self):
        return [0, 0.01, -5., 0.1], [np.inf, 250., 5., 10.0]

    def rho(self, m, theta):
        """generalized nfw profile: rho0,Rs,gamma=theta
        rho0 is the density at Rs,
        Rs is the characteristic radius of switch between
        dlog rho/dlog -gamma and -3 profiles"""
        rhor0, rs, gamma = theta[0], theta[1], theta[2]

        # first get normalisation from rho(R0)
        r0 = np.abs(self.r0 / rs)
        rho0 = rhor0 * r0 ** gamma * (1 + r0) ** (3 - gamma)

        r = np.abs(m / rs)
        return rho0 / r ** gamma / (1 + r) ** (3 - gamma)


class PseudoIsothermalProfile(EllipsoidalProfile):
    @property
    def my_labels(self):
        return [r'$\rho_{0,dm}$', r'$R_s$', r'$q$']

    @property
    def my_param_reasonable(self):
        """Return reasonable parameters for this potential."""
        return [3e7, 10., 1.0]

    @property
    def my_nparm(self):
        """Return the number of parameters of the dm profile."""
        return 3

    @property
    def my_bounds(self):
        return [0, 0, 0], [np.inf, np.inf, np.inf]

    def rho(self, m, theta):
        """NFW profile: rho0,Rs=theta
        rho0 is the density at Rs,
        Rs is the characteristic radius of switch between
        dlog rho/dlog -1 and -3 profiles"""
        rhor0, rs = theta[0], theta[1]
        r = np.abs(m / rs)

        # first get normalisation from rho(R0)
        r0 = np.abs(self.r0 / rs)
        rho0 = rhor0 * (1 + r0) ** 2

        return rho0 / (1 + r) ** 2

class BurkertProfile(EllipsoidalProfile):
    @property
    def my_labels(self):
        return [r'$\rho_{0,dm}$', r'$R_0$', r'$q$']

    @property
    def my_param_reasonable(self):
        """Return reasonable parameters for this potential."""
        return [3e7, 10., 1.0]

    @property
    def my_nparm(self):
        """Return the number of parameters of the dm profile."""
        return 3

    @property
    def my_bounds(self):
        return [0, 0, 0], [np.inf, np.inf, np.inf]

    def rho(self, m, theta):
        """NFW profile: rho0,Rs=theta
        rho0 is the density at Rs,
        Rs is the characteristic radius of switch between
        dlog rho/dlog -1 and -3 profiles"""
        rhor0, rs = theta[0], theta[1]
        r = np.abs(m / rs)

        # first get normalisation from rho(R0)
        r0 = np.abs(self.r0 / rs)
        rho0 = rhor0 * (1 + r0) * (1 + r0**2)

        return rho0 / (1 + r) / (1 + r**2)


class AzimuthallyAveragedParticles(AxiSymmetricPotential):
    def __init__(self, particles=None, nr=401, maxr=20, nphi=37, ntheta=20,
                 pottype=PotentialType.Baryonic):
        # We compute in internal units and but return in physical so need to store diu and viu
        if particles is not None:
            self.diu = particles.Diu
            self.viu = particles.Viu
            self._f_z_interp, self._f_cyl_interp = self._constuct_ang_interp(particles, nr, maxr, nphi, ntheta)
        super().__init__(pottype=pottype)

    @property
    def my_labels(self):
        return []

    @property
    def my_nparm(self):
        """Return the number of parameters of the profile - We have none since """
        return 0

    @property
    def my_bounds(self):
        return [], []

    def my_f_cylr(self, r, ang, *args, **kwargs):
        """Return the force in cylindical R direction at (r, theta)"""
        theta_rad = np.pi / 2 - np.radians(ang)
        return self._f_cyl_interp(r / self.diu, theta_rad) * self.viu * self.viu / self.diu

    def my_f_z(self, r, ang, *args, **kwargs):
        """Return the force in the z-direction at (r, theta)"""
        theta_rad = np.pi / 2 - np.radians(ang)
        return self._f_z_interp(r / self.diu, theta_rad) * self.viu * self.viu / self.diu

    @staticmethod
    def _constuct_ang_interp(particles, nr, maxr, nphi, ntheta):
        # first use nemo to compute forces on spherical grid
        particle_array = particles[particles.m > 0].toiu.asarray()
        phi = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        theta = np.linspace(0, np.pi, nphi)
        r = np.linspace(0, maxr, nr)
        grav = nemo.gravity_spherical_grid(particle_array, r, theta, phi)

        # compute Fz and Fcyl on this grid
        phi_v, theta_v, r_v = np.meshgrid(phi, theta, r, indexing='ij')
        x = np.sin(theta_v) * np.cos(phi_v)
        y = np.sin(theta_v) * np.sin(phi_v)
        r_cyl = np.sqrt(x ** 2 + y ** 2)

        # We ignore errors related to the f_cyl force being undefined on the z-axis
        old_settings = np.seterr(divide='ignore', invalid='ignore')
        f_cyl = (x * grav['F'][:, :, :, 0] + y * grav['F'][:, :, :, 1]) / r_cyl
        # azimuthally average forces
        f_z_avg = np.mean(grav['F'][:, :, :, 2], axis=0)
        f_cyl_avg = np.mean(f_cyl, axis=0)
        f_cyl_avg[np.isnan(f_cyl_avg)] = 0  # force is zero on axis
        f_cyl_avg[np.isnan(f_cyl_avg)] = 0  # force is zero at center
        np.seterr(**old_settings)

        # Construct interpolation objects
        r_v_avg, theta_v_avg = np.meshgrid(r, theta)
        _f_z_interp = scipy.interpolate.CloughTocher2DInterpolator((r_v_avg.ravel(), theta_v_avg.ravel()),
                                                                   f_z_avg.ravel())
        _f_cyl_interp = scipy.interpolate.CloughTocher2DInterpolator((r_v_avg.ravel(), theta_v_avg.ravel()),
                                                                     f_cyl_avg.ravel())
        return _f_z_interp, _f_cyl_interp


class ExponentialDisk(AzimuthallyAveragedParticles):
    """
    We could be clever when we set up our exponential disk and compute analytically with Hankel transforms etc....
    But instead we just put particles in an exponential disk and use nemo...
    """

    def __init__(self, n_particles=10000, sigma_sun=13.0, Rd=2.4 * 2, h=0.13, Rinner=0., Router=np.inf,
                 r0=8.2, nr=401, maxr=20, nphi=37, ntheta=40, pottype=PotentialType.Baryonic):
        # We compute in internal units and but return in physical so need to store diu and viu
        G = 4.302E-3  # Gravitational constant in astronomical units
        disk_mass = np.exp(r0 / Rd) * sigma_sun * 2 * np.pi * Rd ** 2 * 1e6
        # correct for finite size disk
        if Router != np.inf:
            disk_mass_corr = np.exp(-Rinner / Rd) * (Rinner / Rd + 1) - np.exp(-Router / Rd) * (Router / Rd + 1)
        else:
            disk_mass_corr = np.exp(-Rinner / Rd) * (Rinner / Rd + 1)
        disk_mass *= disk_mass_corr
        self.diu = Rd
        self.viu = np.sqrt(G * disk_mass / (self.diu * 1e3))
        z = h * np.log(1 - np.random.random(n_particles)) / self.diu  # z is easy. 1-rand  to handle half open interval
        z[np.random.randint(2, size=n_particles) == 0] *= -1
        cdf = np.random.random(n_particles)
        if Router != np.inf:
            x = cdf * disk_mass_corr + np.exp(-Router / Rd) * (Router / Rd + 1)
        else:
            x = cdf * disk_mass_corr
        # radial is harder. Need to invert xe^-x using lambert see wolfram alpha...
        r_cyl = -np.real(lambertw(-x / np.exp(1), k=-1)) - 1
        phi = 2 * np.pi * np.random.random(n_particles)  # finally scatter in azimuth
        x = r_cyl * np.cos(phi)
        y = r_cyl * np.sin(phi)

        v = np.zeros((n_particles, 3))
        m = np.full(n_particles, 1. / n_particles)  # with the selected Viu we want total disk mass to be unity
        ptype = np.full(n_particles, 3)
        particles = NMagicParticles(x=np.column_stack((x, y, z)), v=v, m=m, Viu=self.viu, Diu=self.diu, ptype=ptype)
        super().__init__(particles, nr=nr, maxr=maxr, nphi=nphi, ntheta=ntheta, pottype=pottype)


class TransformedPotential(AxiSymmetricPotential):
    """Class to transform the parameters of a potential more physical parameters"""
    """We replace the first two parameters by Vc(R0) and rho_dm(R0)"""

    def __init__(self, pot, r0=8.2):
        self.pot = copy.deepcopy(pot)
        self.r0 = r0
        self._lastp = None
        super().__init__(pottype=self.pot.pottype)

    def __str__(self):
        return 'TransformedPotential Holding:' + str(self.pot)

    @property
    def my_nparm(self):
        """Return the number of parameters of the stored potential"""
        return self.pot.nparm

    @property
    def my_param_reasonable(self):
        """Return the number of parameters of this potential."""
        param_reasonable = self.pot.param_reasonable
        param_reasonable[0:2] = [238., 0.008]
        return param_reasonable

    @property
    def my_labels(self):
        labels = self.pot.labels
        labels[0:2] = [r'$V_c(R_\odot)~[{\rm km/s}]$', r'$\rho_{\rm dm}(R_\odot)~[M_\odot/{\rm kpc}^3]$']
        return labels

    # Never called since we override bounds
    @property
    def my_bounds(self):
        boundsl, boundsu = self.pot.bounds
        boundsl[0:2] = [0, 0]
        boundsu[0:2] = [350, 0.1]
        return boundsl, boundsu

    # Should never be called since we override bounds
    def my_f_cylr(self, r, ang, parms=None, rel_tol=None):
        converged, xformed_parms = self.xform_parms(parms)
        if converged:
            return self.pot.f_cylr(r, ang, parms=xformed_parms, rel_tol=rel_tol)
        else:
            return np.full_like(r * ang, np.nan)

    def my_f_z(self, r, ang, parms=None, rel_tol=None):
        converged, xformed_parms = self.xform_parms(parms)
        if converged:
            return self.pot.f_z(r, ang, parms=xformed_parms, rel_tol=rel_tol)
        else:
            return np.full_like(r * ang, np.nan)

    def xform_parms(self, parms):
        if self._lastp is None:
            p0 = self.pot.param_reasonable
        else:
            p0 = self._lastp

        def F(p):
            # return residual in (100km/s)**2 and 0.01Msun/kpc
            testparms = np.copy(parms)
            testparms[0:2] = p
            vc2_sun = self.pot.vc2(self.r0, 0, parms=testparms)  # in (km/s)^2
            dm_sun = self.pot.dm.rho(self.r0, testparms) / 1e9  # in Msun/kpc3
            return [(vc2_sun - parms[0] ** 2) / 100 ** 2,
                    (dm_sun - parms[1]) / 0.01]

        p = scipy.optimize.root(F, np.array(p0[0:2]), tol=1e-4)
        parms = np.copy(parms)
        parms[0:2] = p['x']
        if p['success']:
            self._lastp = parms
        return p['success'], parms


class GalpyPotentialWrapper(AxiSymmetricPotential):
    """
    Wraps a galpy potential so we can use it
    """
    def __init__(self, func, Vc=220, R0=8.0, pottype=PotentialType.Baryonic):
        self.force_to_kpckms2 = bovy_conversion.force_in_kmsMyr(Vc, R0) / 1.023e-3
        self.func = func
        self.Vc = Vc
        self.R0 = R0
        super().__init__(pottype=pottype)

    @property
    def my_labels(self):
        return []

    @property
    def my_nparm(self):
        """Return the number of parameters of the profile - We have none since """
        return 0

    @property
    def my_bounds(self):
        return [], []

    def my_f_cylr(self, r, ang, *args, **kwargs):
        """Return the force in cylindical R direction at (r, theta)"""
        z, r_cyl = self.spherical_to_cylindrical(r, ang)
        return evaluateRforces(self.func, r_cyl / self.R0, z / self.R0) * self.force_to_kpckms2

    def my_f_z(self, r, ang, *args, **kwargs):
        """Return the force in the z-direction at (r, theta)"""
        z, r_cyl = self.spherical_to_cylindrical(r, ang)
        return evaluatezforces(self.func, r_cyl / self.R0, z / self.R0) * self.force_to_kpckms2
