

import copy
import os
import numpy as np

class NMagicParticles(object):
    def __init__(self, x=None, v=None, m=None, ptype=None, file=None,
                 Viu=None, Diu=None, phys=False, parameterfile=None):
        """
        Initialize either from nmagic snapshot with file= keyword or positions
        and velocities with (x,v,m,ptype) keywords, of which only x is required.
        If you would like to use physical units either:
        -supply diu and viu to scale,
        -provide the parameter file (this has diu and viu inside)
        -of if your particles are already in physical units set phys keyword to True.

        To alter the galactic parameters set the following attributes after initialisation:
        alpha - bar angle - default 27 deg
        R0 - galactic center distance - default 8.2 kpc
        z0 - solar height above galactic plane - default 0.025 kpc
        Vsun - velocity of sun wrt circular orbit at sun - default (U,V,W)=[11.1, 12.24, 7.25]km/s
        Vc - circular velocity - default 238 km/s
        """
        self.alpha = 27.
        self.R0 = 8.2
        self.z0 = 0.025
        self.Vsun = [11.1, 12.24, 7.25]
        self.Vc = 238.0

        self._readparameterfile(parameterfile)
        if file is None and 'Model' in  self.parameters:
            file = os.path.join(os.path.dirname(parameterfile), self.parameters['Model'])
        if Viu is None and 'iu2kms' in self.parameters:
            Viu = self.parameters['iu2kms']
        if Diu is None and 'kpc2iu' in self.parameters:
            Diu = 1 / self.parameters['kpc2iu']
        if x is None and file is None:
            raise ValueError("x or file must be set")
        if file is not None:
            x = np.loadtxt(file)
        self.pos = x[:, 0:3]
        if x.shape[1] >= 6:
            self.v = x[:, 3:6]
        if v is not None:
            self.v = v
        if x.shape[1] >= 7:
            self.m = x[:, 6]
        if m is not None:
            self.m = m
        if x.shape[1] >= 8:
            self.ptype = x[:, 7]
        if ptype is not None:
            self.ptype = ptype
        if phys == True:
            self.phys = True
        else:
            self.phys = False
            if Diu is not None:
                self.pos *= Diu
                self.Diu = Diu
                if self.v is not None:
                    if Viu is None:
                        raise ValueError("diu and viu are both required to use physical units")
                    self.v *= Viu
                    self.Viu = Viu
                    if self.m is not None:
                        G = 4.302E-3  # Gravitational constant in astronomical units
                        Miu = Diu * 1e3 * Viu ** 2 / G
                        self.m *= Miu
                self.phys = True

    def _readparameterfile(self,parameterfile=None):
        self.parameters = {}
        if parameterfile is not None:
            parameters = {}
            with open(parameterfile) as f:
                for line in f:
                    (key, val) = line.split()
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                    parameters[key] = val
            self.parameters = parameters

    def writenmagicsnap(self,fname):
        if self.phys:
            self.toiu
            wasphys = True
        else:
            wasphys = False
        np.savetxt(fname, np.c_[self.pos[:, 0], self.pos[:, 1], self.pos[:, 2],
                                self.v[:, 0], self.v[:, 1], self.v[:, 2], self.m, self.ptype],
                   fmt=['%.6e', '%.6e', '%.6e', '%.6e', '%.6e', '%.6e', '%.6e', '%i'])
        if wasphys:
            self.tophys

    def asarray(self):
        """Returns the raw array data"""
        width = 3
        if self.v is not None:
            width += 3
        if self.m is not None:
            width += 1
        if self.ptype is not None:
            width += 1
        ret = np.zeros((self.n, width))
        ret[:, 0:3] = self.pos
        if self.v is not None:
            ret[:, 3:6] = self.v
        if self.m is not None:
            ret[:, 6] = self.m
        if self.ptype is not None:
            ret[:, 7] = self.ptype
        return ret

    @property
    def tophys(self, Diu=None, Viu=None):
        """Convert to physical units"""
        if Diu is not None:
            self.Diu = Diu
        if Viu is not None:
            self.Viu = Viu
        if self.phys == True:
            return self
        else:
            self.pos *= self.Diu
            if self.v is not None:
                self.v *= self.Viu
                if self.m is not None:
                    G = 4.302E-3  # Gravitational constant in astronomical units
                    Miu = self.Diu * 1e3 * self.Viu ** 2 / G
                    self.m *= Miu
        self.phys = True
        return self

    @property
    def toiu(self):
        """Convert to internal units"""
        if self.phys == False:
            return self
        self.pos /= self.Diu
        if self.v is not None:
            self.v /= self.Viu
            if self.m is not None:
                G = 4.302E-3  # Gravitational constant in astronomical units
                Miu = self.Diu * 1e3 * self.Viu ** 2 / G
                self.m /= Miu
        self.phys = False
        return self

    @property
    def n(self):
        """Return the number of particles."""
        return (self.pos.shape)[0]

    @property
    def x(self):
        """Return the x position of particles."""
        return self.pos[:, 0]

    @property
    def y(self):
        """Return the y position of particles."""
        return self.pos[:, 1]

    @property
    def z(self):
        """Return the z position of particles."""
        return self.pos[:, 2]

    @property
    def vx(self):
        """Return the vx position of particles."""
        return self.v[:, 0]

    @property
    def vy(self):
        """Return the vy position of particles."""
        return self.v[:, 1]

    @property
    def vz(self):
        """Return the vz position of particles."""
        return self.v[:, 2]

    @property
    def dm(self):
        """Return the dm particles."""
        return self[self.ptype == 0]

    @property
    def stellar(self):
        """Return the stellar particles."""
        return self[self.ptype > 0]

    def mask(self, mask):
        """Return a masked copy of the particles."""
        masked = copy.deepcopy(self)
        masked.pos = self.pos[mask, :].reshape(-1, 3)  # retain leading dimension with reshape
        if self.v is not None:
            masked.v = self.v[mask, :].reshape(-1, 3)
        if self.m is not None:
            masked.m = self.m[mask]
        if self.ptype is not None:
            masked.ptype = self.ptype[mask]
        try:
            if self.frac_d_change is not None:
                masked.frac_d_change = self.frac_d_change[mask]
        except AttributeError:
            pass
        return masked

    def __getitem__(self, index):
        return self.mask(index)

    def __add__(self, other):
        if self.phys != other.phys:
            raise ValueError('Both sets of particles should be in the same units')
        ret = copy.deepcopy(self)
        ret.pos = np.vstack((self.pos, other.pos))
        if self.v is not None:
            ret.v = np.vstack((self.v, other.v))
        if self.m is not None:
            ret.m = np.vstack((self.m, other.m))
        if self.ptype is not None:
            ret.ptype = np.vstack((self.ptype, other.ptype))
        return ret

    @property
    def xyzmw(self):
        """Return position of particles in galactic cartesian coordinates.
        """
        if self.phys == False:
            raise ValueError("Model should be in physical units")
        ddtor = np.pi / 180.
        ang = self.alpha * ddtor
        xyz = np.zeros_like(self.pos)
        xyz[:, 0] = (self.pos[:, 0] * np.cos(-ang) - self.pos[:, 1] * np.sin(-ang)) + self.R0
        xyz[:, 1] = (self.pos[:, 0] * np.sin(-ang) + self.pos[:, 1] * np.cos(-ang))
        xyz[:, 2] = self.pos[:, 2] - self.z0
        return xyz

    @property
    def uvwmw(self):
        """Return UVW velocities.
        """
        if self.phys == False:
            raise ValueError("Model should be in physical units")
        if self.v is None:
            raise ValueError("Velocities Required")
        ddtor = np.pi / 180.
        ang = self.alpha * ddtor
        vxyz = np.zeros_like(self.pos)
        vxyz[:, 0] = (self.v[:, 0] * np.cos(-ang) - self.v[:, 1] * np.sin(-ang)) + self.Vsun[
            0]  # sun moves at Vsun[0] towards galactic center i.e. other stars are moving away towards larger x
        vxyz[:, 1] = (self.v[:, 0] * np.sin(-ang) + self.v[:, 1] * np.cos(-ang)) - self.Vsun[
            1] - self.Vc  # sun moves at Vsun[1] in direction of rotation, other stars are going slower than (0,-Vc,0)
        vxyz[:, 2] = self.v[:, 2] - self.Vsun[
            2]  # sun is moving towards ngp i.e. other stars on average move at negative v
        return vxyz

    @property
    def r_cyl(self):
        """Return distance to galactic center in cylindrical coordianates"""
        return np.sqrt(self.pos[:, 0] ** 2 + self.pos[:, 1] ** 2)

    @property
    def rgc(self):
        """Return galactocentric distances. """
        return np.sqrt(self.pos[:, 0] ** 2 + self.pos[:, 1] ** 2 + self.pos[:, 2] ** 2)

    @property
    def r(self):
        """Return distance from sun. """
        xyz = self.xyzmw
        return np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2)

    @property
    def rlb(self):
        """Return array of particles in galactic (r,l,b) corrdinates angles in degrees. """
        rlb = np.zeros_like(self.pos)
        xyz = self.xyzmw
        rlb[:, 0] = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2)
        rlb[:, 1] = np.arctan2(xyz[:, 1], xyz[:, 0]) * 180 / np.pi
        rlb[:, 2] = np.arcsin(xyz[:, 2] / rlb[:, 0]) * 180 / np.pi
        return rlb

    @property
    def vrmulmub(self):
        """Return array of particles velocities in galactic (r,l,b) corrdinates i.e.
        (radial velocity [km/s], mul [mas/yr], mub [mas/yr])  """
        vrmulmub = np.zeros_like(self.pos)
        xyz = self.xyzmw
        vxyz = self.uvwmw
        r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2)
        vrmulmub[:, 0] = (vxyz[:, 0] * xyz[:, 0] + vxyz[:, 1] * xyz[:, 1] + vxyz[:, 2] * xyz[:, 2]) / r
        rxy = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
        vrmulmub[:, 1] = (-vxyz[:, 0] * xyz[:, 1] / rxy + vxyz[:, 1] * xyz[:, 0] / rxy) / r / 4.74057
        vrmulmub[:, 2] = (-vxyz[:, 0] * xyz[:, 2] * xyz[:, 0] / rxy -
                          vxyz[:, 1] * xyz[:, 2] * xyz[:, 1] / rxy + vxyz[:, 2] * rxy) / (r ** 2) / 4.74057
        return vrmulmub
