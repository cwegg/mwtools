"""
Routines for calling nemo from python.

To use you need to have installed nemo from https://teuben.github.io/nemo/

All routines that take input particles accept either be a array with rows
containing (x,y,z,vx,vy,vz,mass) as the first rows. Most routines also accept
rows of (x,y,z,mass) when velocities aren't needed.
They also accept an NMagicParticles instance. If you input NMagicParticles in
physical units then you outputs are in physical units, otherwise internal units
are used (i.e. G=1).

Calling Nemo can be a bit tempramental - the executables are dyanmically linked,
and the libraries and executables are all in their own envirnoment. We try to
handle this, but if it fails then first add verbose=True to the failing command,
and see if there's a useful message being printed.
"""
import os
import subprocess
import tempfile
import numpy as np

try:
    from .nmagic import NMagicParticles
except ImportError:
    NMagicParticles = type(None)

try:
    from dotenv import load_dotenv, find_dotenv

    _dotenv_path = find_dotenv(usecwd=True)
    load_dotenv(_dotenv_path, override=True)
except (ModuleNotFoundError, ImportError) as e:
    pass

_NEMO_LOCATION = os.environ.get("NEMO_LOCATION")
_NEMO_BIN_LOCATION = os.path.join(_NEMO_LOCATION, 'bin')


def _make_nemo_env():
    """
    Nemo doesnt make static executables for GyrFalcon/getgravity. normally
    nemo_start.csh would add the paths to the libraries. here we call
    nemo_start.csh and use the environment variables it sets to create our
    environment
    """
    command = ['csh', '-c', 'source ' +
               os.path.join(_NEMO_LOCATION, 'nemo_start.csh') + '  && env']
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)

    for line in proc.stdout:
        (key, equals, value) = line.decode('utf-8').rstrip().partition('=')
        if equals == '=':
            os.environ[key] = value

    proc.communicate()

    env = dict(os.environ)
    return env


_NEMO_ENV = _make_nemo_env()


def call_nemo_executable(executable, arguments):
    """Call nemo executable (should be a string) with arguments (should be a list).
    Returns Popen object with .stdin, .stdout and .stderr attributes."""
    p = subprocess.Popen([os.path.join(_NEMO_BIN_LOCATION, executable)] + arguments,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_NEMO_ENV)
    return p


def writesnap(particles, filename, time=0.0, verbose=False, positions_only=False, filter_mass=False):
    """
    Write a nemo snapshot of particles to filename, uses snapshot time=0 by deafult
    This writes a snapshot with postitions, velocities and masses. Use positions_only to just write only postitions.
    filter_mass=True removes zero (or negative) mass particles that can cause problems for various nemo routines

    particles can either be:
      -If positions_only=True, an array whose rows contain (x,y,z,...)
      -Otherwise we either need an array (i) whose rows contain (x,y,z,mass)
       or (ii) whose rows con (x,y,z,vx,vy,vz,mass)
      -An NMagicParticles object
    """

    if isinstance(particles, NMagicParticles):
        x = particles.pos
        n_particles = particles.n
        if not positions_only:
            m = particles.m
            v = particles.v
    else:
        x = particles[:, 0:3]
        n_particles = len(particles[:, 0])
        if not positions_only:
            if particles.shape[1] >= 7:
                m = particles[:, 6]
                v = particles[:, 3:6]
            else:
                # Seems like we don't have positions
                m = particles[:, 3]
                v = np.zeros_like(x)

    if positions_only:
        p = call_nemo_executable('atos', ["in=-", "out=" + filename, "options=pos"])
        p.stdin.write('{} 3 {} 0'.format(n_particles, time).encode('ascii'))
        np.savetxt(p.stdin, x, fmt='%e', delimiter=' ')
    else:
        if filter_mass:
            good = (m > 0)
            n_particles = np.sum(good)
            x, m, v = x[good], m[good], v[good]

        p = call_nemo_executable('atos', ["in=-", "out=" + filename, "options=mass,pos,vel"])
        p.stdin.write('{} 3 {} 0'.format(n_particles, time).encode('ascii'))
        np.savetxt(p.stdin, m, fmt='%e', delimiter=' ')
        np.savetxt(p.stdin, x, fmt='%e', delimiter=' ')
        np.savetxt(p.stdin, v, fmt='%e', delimiter=' ')
    p.stdin.flush()
    (stdoutdata, stderrdata) = p.communicate()
    if verbose:
        print(stdoutdata.decode('utf-8'), stderrdata.decode('utf-8'))


def readsnap(filename, times='all', verbose=False, give='t,x,y,z,vx,vy,vz,m'):
    """
    Read a nemo snapshot of particles
    This reads a snapshot with postitions, velocities and masses.
    """
    p = call_nemo_executable("snapprint", ["in=" + filename, "options=" + give, f"times={times}"])

    snaps = np.genfromtxt(p.stdout)
    if verbose:
        print(p.stderr.read().decode('utf-8'))

    columns = give.count(',') + 1
    tlist, ti = np.unique(snaps[:, 0], return_inverse=True)
    snaps = (np.reshape(snaps, (len(tlist), np.size(snaps) // (columns * len(tlist)), columns)))[:, :, 1:]
    return tlist, snaps


def rotationcurve(particles, nr=100, ntheta=100, rrange=(0, 10), verbose=False,
                  units=None):
    """
    Computes the rotation curve of particles as a function of r in the X,Y plane using gyrfalcON

    Inputs:
        particles : n-body model of which to compute rotation curve.
          particles can either be:
            -An array whose rows contain (i) (x,y,z,mass) or (ii) (x,y,z,vx,vy,vz,mass)
            -An NMagicParticles object
        nr : number of radial locations to compute curve (default:100)
        ntheta : number of azimuthal locations to average over in computation (default:100)
        rrange : range of radii to compute rotation curve, nr points are spaced linearly over this range
        (default:[0,10])
        units : either None or 'physical'.
          None assumes G=1, unless particles are a NMagicParticles instance
          physical assumes positions in kpc, velocities in km/s, masses in Msun
    Outputs:
        (r,vcirc2) : Where vcirc2 is the square of circular velocity and is computed at each point in r
                    If particles is a NMagicParticles instance in physical units then return in kpc,(km/s)**2
                    else return in internal units i.e. assuming G=1.
    """
    theta = np.linspace(0, 2 * np.pi, num=ntheta, endpoint=False)
    r = np.linspace(rrange[1], rrange[0], num=nr, endpoint=False)[::-1]
    thetamat, rmat = np.meshgrid(theta, r)
    x = rmat * np.sin(thetamat)
    y = rmat * np.cos(thetamat)
    positions = np.zeros((ntheta * nr, 3))
    positions[:, 0] = x.flatten()
    positions[:, 1] = y.flatten()
    ax, ay, _, _ = getgravity(particles, positions, verbose, units).T
    ax = ax.reshape(x.shape)
    ay = ay.reshape(y.shape)
    ar = - (x * ax + y * ay) / np.sqrt(x * x + y * y)
    vcirc2 = np.mean(ar, 1) * r
    return r, vcirc2


def getpartpot(particles, verbose=False, units=None):
    """
    Computes the potential at each particle using gyrfalcON.

    Inputs:
        particles : n-body model of which to compute potential
          particles can either be:
            -An array whose rows contain (i) (x,y,z,mass) or (ii) (x,y,z,vx,vy,vz,mass)
            -An NMagicParticles object
        units : either None (default) or 'physical'
          None assumes G=1, unless particles are a NMagicParticles instance
          physical assumes positions in kpc, velocities in km/s, masses in Msun

    Outputs:
        potential at each particle
        If units=='physical' or particles is a NMagicParticles object in physical units
        then return in kpc,(km/s)**2 else return assuming G=1.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        filein = tmpdirname + 'partpot.in'
        writesnap(particles, filename=filein, time=0.0)
        fileout = tmpdirname + 'partpot.falc'
        p = call_nemo_executable("gyrfalcON", ["in=" + filein, "out=" + fileout, "tstop=0", "theta=0.5",
                                               "eps=0.05", "kmax=6", "give=phi", "Ncrit=20"])
        (stdout, stderr) = p.communicate()
        if verbose:
            print(stdout.decode('utf-8'), stderr.decode('utf-8'))
        p = call_nemo_executable("snapprint", ["in=" + fileout, "options=phi"])
        from_falcon = np.genfromtxt(p.stdout)
        if verbose:
            print(p.stderr.read().decode('utf-8'))

        if units == 'physical' or (isinstance(particles, NMagicParticles) and particles.phys):
            from_falcon *= 4.301e-6

    return from_falcon


def getgravity(particles, positions, verbose=False, units=None):
    """
    Computes the forces and potential at each position due to paticles using gyrfalcON
    Inputs:
        particles : n-body model of which to compute rotation curve
          particles can either be:
            -An array whose rows contain (i) (x,y,z,mass) or (ii) (x,y,z,vx,vy,vz,mass)
            -An NMagicParticles object
        postions : positions in z,y,z to compute gravity. Format (N,3) array
        units : either None (default) or 'physical'
          -None assumes G=1, unless particles are a NMagicParticles instance
          -physical assumes positions in kpc, masses in Msun

    Outputs:
        (Fx,Fy,Fz,potential) at each postion. If particles is a NMagicParticles instance in
        physical units then return in (km/s)**2/kpc, (km/s)**2 else return assuming G=1.
    """

    with tempfile.TemporaryDirectory() as tmpdirname:
        filein = os.path.join(tmpdirname, 'part.in')
        writesnap(particles, filename=filein, time=0.0, filter_mass=True, verbose=verbose)

        posfilein = os.path.join(tmpdirname, 'pos.in')
        writesnap(positions, filename=posfilein, time=0.0, positions_only=True, verbose=verbose)

        fileout = os.path.join(tmpdirname, 'gravity.out')
        p = call_nemo_executable("getgravity", ["srce=" + filein, "sink=" + posfilein, "out=" + fileout, "Ncrit=20"])
        stdout, stderr = p.communicate()
        if verbose:
            print(stdout.decode('utf-8'), stderr.decode('utf-8'))
        p = call_nemo_executable("snapprint", ["in=" + fileout, "options=ax,ay,az,phi"])
        fromfalc = np.genfromtxt(p.stdout)

        if verbose:
            print(p.stderr.read().decode('utf-8'))

        if units == 'physical' or (isinstance(particles, NMagicParticles) and particles.phys):
            fromfalc *= 4.301e-6

        return fromfalc


def integrate(particles, t=1., step=None, verbose=False):
    """
    Integrates particles using gyrfalcON
    Inputs:
        particles : n-body model to integrate
          particles can either be:
            -An array whose rows contain (x,y,z,vx,vy,vz,mass)
            -An NMagicParticles object
          NOTE - We assume G=1 when integrating. BE CAREFULL ABOUT UNITS!
        t : time to integrate for
        step : return a snapshot each step units of time
    Outputs:
        a tuple (t,snaps) with snaps containing a snapshot at each t
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        filein = os.path.join(tmpdirname, 'integrate.in')
        writesnap(particles, filename=filein, time=0.0)
        fileout = os.path.join(tmpdirname, 'integrate.falc')
        p = call_nemo_executable("gyrfalcON", ["in=" + filein, "out=" + fileout,
                                               "tstop=" + str(t), "step=" + str(step), "theta=0.5", "eps=0.05",
                                               "kmax=6", "give=mxvp"])
        (stdout, stderr) = p.communicate()
        if verbose:
            print(stdout.decode('utf-8'), stderr.decode('utf-8'))
        p = call_nemo_executable("snapprint", ["in=" + fileout, "options=t,x,y,z,vx,vy,vz,phi"])

        snaps = np.genfromtxt(p.stdout)
        if verbose:
            print(p.stderr.read().decode('utf-8'))

        tlist, ti = np.unique(snaps[:, 0], return_inverse=True)
        snaps = (np.reshape(snaps, (len(tlist), np.size(snaps) // (9 * len(tlist)), 9)))[:, :, 1:]
    return tlist, snaps


def _cartesian_to_polar_forces(force_dict, xmat, ymat, zmat):
    # Helper to convert forces
    Fx = force_dict['F'][..., 0]
    Fy = force_dict['F'][..., 1]
    Fz = force_dict['F'][..., 2]
    rmat = np.sqrt(xmat ** 2 + ymat ** 2 + zmat ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        Fr = (Fx * xmat + Fy * ymat + Fz * zmat) / rmat
        Rcyl = np.sqrt(xmat ** 2 + ymat ** 2)
        Ftheta = ((xmat * Fx + ymat * Fy) * zmat - Rcyl ** 2 * Fz) / rmat / Rcyl
        Fphi = (-ymat * Fx + xmat * Fy) / Rcyl
    Fr[rmat == 0] = 0
    Ftheta[Rcyl == 0] = 0
    Fphi[Rcyl == 0] = 0
    force_dict['F'] = np.stack((Fr, Ftheta, Fphi), axis=-1)


def gravity_cartesian_grid(particles, x, y, z, polar_forces=False,
                           verbose=False, units=None):
    """
    Takes vectors of r theta and phi and then on the resultant 3d spherical
    polar grid computes the gravity using gyrfalcON.

    Inputs:
        particles : n-body model to integrate
          particles can either be:
            -An array whose rows contain (x,y,z,vx,vy,vz,mass)
            -An NMagicParticles object
        x, y, z :  the cartesian grid points
        polar_forces : return the forces in polar coordinates
        units : either None (default) or 'physical'
          None assumes G=1, unless particles are a NMagicParticles instance
          physical assumes positions in kpc, velocities in km/s, masses in Msun

    Outputs:
        (Fx,Fy,Fz,potential) at each postion. If particles is a NMagicParticles instance in
        physical units then return in (km/s)**2/kpc, (km/s)**2 else return assuming G=1.

    Output:
      A dict with entries for (r,theta,phi,F,pot).
        -F is an [Nx,Ny,Nz,3] array: [Fr,Ftheta,Fphi] if polar_forces=True, [Fx,Fy,Fz] otherwise

      If units=='physical' or  particles is a NMagicParticles instance in physical units
      then return forces in (km/s)**2/kpc, potential in (km/s)**2 else return everything
      assuming G=1."""

    xmat, ymat, zmat = np.meshgrid(x, y, z, indexing='ij')
    pos = np.zeros((len(xmat.flatten()), 3))
    pos[:, 0] = xmat.flatten()
    pos[:, 1] = ymat.flatten()
    pos[:, 2] = zmat.flatten()
    # gyrfalcon crashes if there are 2 particles in the same location, remove duplicates
    # (eg along z axis), but keep track of indexes to add them back at the end
    unique_pos, inds = np.unique(pos, axis=0, return_inverse=True)
    grav = getgravity(particles, unique_pos, verbose, units)
    ret = {}
    ret['x'] = x
    ret['y'] = y
    ret['z'] = z
    ret['pot'] = np.reshape(grav[inds, 3], xmat.shape)
    ret['F'] = np.reshape(grav[inds, 0:3], xmat.shape + (3,))
    if polar_forces:
        _cartesian_to_polar_forces(ret, xmat, ymat, zmat)
    ret['pot'] = ret['pot'].squeeze()
    ret['F'] = ret['F'].squeeze()
    return ret


def gravity_spherical_grid(particles, r, theta, phi, polar_forces=False,
                           verbose=False, units=None):
    """
    Takes vectors of x,y,z and then on the resultant 3d spherical
    polar grid computes the gravity using gyrfalcON.

    Inputs:
        particles : n-body model to integrate
          particles can either be:
            -An array whose rows contain (x,y,z,vx,vy,vz,mass)
            -An NMagicParticles object
        x, y, z :  the cartesian grid points
        polar_forces : return the forces in polar coordinates
        units : either None (default) or 'physical'
          None assumes G=1, unless particles are a NMagicParticles instance
          physical assumes positions in kpc, velocities in km/s, masses in Msun

    Outputs:
        (Fx,Fy,Fz,potential) at each postion. If particles is a NMagicParticles instance in
        physical units then return in (km/s)**2/kpc, (km/s)**2 else return assuming G=1.

    Output:
      A dict with entries for (r,theta,phi,F,pot).
        -F is an [Nx,Ny,Nz,3] array: [Fr,Ftheta,Fphi] if polar_forces=True, [Fx,Fy,Fz] otherwise

      If units=='physical' or  particles is a NMagicParticles instance in physical units
      then return forces in (km/s)**2/kpc, potential in (km/s)**2 else return everything
      assuming G=1."""

    r_v, theta_v, phi_v = np.meshgrid(r, theta, phi, indexing='ij')
    x = r_v * np.sin(theta_v) * np.cos(phi_v)
    y = r_v * np.sin(theta_v) * np.sin(phi_v)
    z = r_v * np.cos(theta_v)
    pos = np.zeros((len(x.flatten()), 3))
    pos[:, 0] = x.flatten()
    pos[:, 1] = y.flatten()
    pos[:, 2] = z.flatten()
    # gyrfalcon crashes if there are 2 particles in the same location, remove duplicates
    # (eg along z axis), but keep track of indexes to add them back at the end
    unique_pos, inds = np.unique(pos, axis=0, return_inverse=True)
    grav = getgravity(particles, unique_pos, verbose, units)
    ret = {}
    ret['r'] = r
    ret['theta'] = theta
    ret['phi'] = phi
    ret['pot'] = np.reshape(grav[inds, 3], x.shape)
    ret['F'] = np.reshape(grav[inds, 0:3], x.shape + (3,))
    if polar_forces:
        _cartesian_to_polar_forces(ret, x, y, z)
    ret['pot'] = ret['pot'].squeeze()
    ret['F'] = ret['F'].squeeze()
    return ret
