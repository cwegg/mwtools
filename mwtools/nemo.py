"""
Routines for The input particles can either be a [n,8] array ordered like the NMagic file format
        or NMagicParticles instance


"""
import os
import subprocess
import tempfile
from .nmagic import NMagicParticles
import numpy as np
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)

_NEMO_LOCATION = os.environ.get("NEMO_LOCATION")
_NEMO_DEHNEN_LOCATION = os.environ.get("NEMO_DEHNEN_LOCATION")


def writesnap(particles, filename, time=0.0, verb=False, positions_only=False):
    """
    Write a nemo snapshot of particles to filename, uses snapshot time=0 by deafult
    This writes a snapshot including postitions, velocities and masses
    c.f. writepossnap which only writes the positions and masses

    The input particles can either be a [n,8] array ordered like the NMagic file format
    or NMagicParticles instance
    """

    if isinstance(particles, NMagicParticles):
        m = particles.m
        x = particles.pos
        v = particles.v
        npart = particles.n
    else:
        m = particles[:, 6]
        x = particles[:, 0:3]
        v = particles[:, 3:6]
        npart = len(particles[:, 0])

    if os.path.isfile(filename):
        os.remove(filename)

    p = subprocess.Popen([_NEMO_LOCATION + "atos", "in=-", "out=" + filename, "options=mass,pos,vel"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.stdin.write('{} 3 {} 0'.format(npart, time).encode('ascii'))
    np.savetxt(p.stdin, m, fmt='%e', delimiter=' ')
    np.savetxt(p.stdin, x, fmt='%e', delimiter=' ')
    np.savetxt(p.stdin, v, fmt='%e', delimiter=' ')
    p.stdin.flush()
    (stdoutdata, stderrdata) = p.communicate()
    if verb:
        print(stdoutdata, stderrdata)


def writepossnap(positions, filename, time=0.0, verb=False):
    """
    Write a nemo snapshot of particles to filename, uses snapshot time=0 by deafult
    This writes a snapshot of postitionsand masses
    c.f. writesnap which also writes the velocities
    """

    if isinstance(positions, NMagicParticles):
        x = positions.pos
        npart = positions.n
    else:
        x = positions[:, 0:3]
        npart = len(positions[:, 0])

    if os.path.isfile(filename):
        os.remove(filename)


    p = subprocess.Popen([_NEMO_LOCATION + "atos", "in=-", "out=" + filename, "options=pos"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.stdin.write('{} 3 {} 0'.format(npart, time).encode('ascii'))
    np.savetxt(p.stdin, x, fmt='%e', delimiter=' ')
    p.stdin.flush()
    (stdoutdata, stderrdata) = p.communicate()
    if verb:
        print(stdoutdata, stderrdata)


def rotationcurve(particles, nr=100, ntheta=100, rrange=(0, 10)):
    """
    Computes the rotation curve of particles as a function of r in the X,Y plane using gyrfalcON
    Inputs:
        particles : n-body model of which to compute rotation curve
        nr : number of radial locations to compute curve (default:100)
        ntheta : number of azimuthal locations to average over in computation (default:100)
        rrange : range of radii to compute rotation curve, nr points are spaced linearly over this range
        (default:[0,10])
    Outputs:
        (vcirc2,r) : where vcirc2 is the square of circular velocity and is computed at each point in r
    """
    theta = np.linspace(0, 2 * np.pi, num=ntheta, endpoint=False)
    r = np.linspace(rrange[1], rrange[0], num=nr, endpoint=False)
    thetamat, rmat = np.meshgrid(theta, r)
    x = rmat * np.sin(thetamat)
    y = rmat * np.cos(thetamat)
    positions = np.zeros((ntheta * nr, 3))
    positions[:, 0] = x.flatten()
    positions[:, 1] = y.flatten()
    ax, ay, _, _ = getgravity(particles, positions)
    ar = (x * ax + y * ay) / np.sqrt(x * x + y * y)
    vcirc2 = np.mean(ar, 1) * r
    return vcirc2, r


def _make_dehnen_env():
    """
    Nemo doesnt make static executables for GyrFalcon/getgravity. normally nemo_start.sh would add the paths to the
    libraries. Here we do it manually.
    """
    env = dict(os.environ)
    from sys import platform
    if platform == "linux" or platform == "linux2":
        env['LD_LIBRARY_PATH'] = _NEMO_DEHNEN_LOCATION + 'falcON/lib/:' \
                                 + _NEMO_DEHNEN_LOCATION + 'utils/lib/'
    elif platform == "darwin":
        env['DYLD_LIBRARY_PATH'] = _NEMO_DEHNEN_LOCATION + 'falcON/lib/:' \
                                   + _NEMO_DEHNEN_LOCATION + 'utils/lib/'
    elif platform == "win32":
        raise WindowsError('Windows not supported yet')
    return env


def getpartpot(particles):
    """
    Computes the potential at each particle using gyrfalcON
    Inputs:
        particles : n-body model of which to compute rotation curve
    Outputs:
        potential at each particle
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        filein = tmpdirname + 'partpot.in'
        writesnap(particles, filename=filein, time=0.0)
        fileout = tmpdirname + 'partpot.falc'
        env = dict(os.environ)
        p = subprocess.Popen([_NEMO_DEHNEN_LOCATION + "falcON/bin/gyrfalcON", "in=" + filein,
                              "out=" + fileout, "tstop=0", "theta=0.5", "eps=0.05", "kmax=6", "give=phi"], env=env)
        (_, _) = p.communicate()
        p = subprocess.Popen([_NEMO_LOCATION + "snapprint in=" + fileout + " options=phi"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        fromfalc = np.genfromtxt(p.stdout)
    return fromfalc


def getgravity(particles, positions):
    """
    Computes the forces and potential at each position due to paticles using gyrfalcON
    Inputs:
        particles : n-body model of which to compute rotation curve
        postions : positions in z,y,z to compute gravity. Format (N,3) array
    Outputs:
        (Fx,Fy,Fz,potential) at each postion
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        filein = tmpdirname + 'part.in'
        writesnap(particles, filename=filein, time=0.0)
        posfilein = tmpdirname + 'pos.in'
        writepossnap(positions, filename=posfilein, time=0.0)
        fileout = tmpdirname + 'gravity.out'

        env = _make_dehnen_env()

        p = subprocess.Popen([_NEMO_DEHNEN_LOCATION + "falcON/bin/getgravity",
                              "srce=" + filein, "sink=" + posfilein, "out=" + fileout, "Ncrit=20"], env=env)
        (_, _) = p.communicate()
        p = subprocess.Popen([_NEMO_LOCATION + "snapprint in=" + fileout + " options=ax,ay,az,phi"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=env)
        fromfalc = np.genfromtxt(p.stdout)
        return fromfalc


def integrate(particles, t=1., step=None):
    """
    Integartes paticles using gyrfalcON
    Inputs:
        particles : n-body model of which to compute rotation curve
        t : time to integrate for
        step : return a snapshot each step units of time
    Outputs:
        a tuple (t,snaps) with snaps containing a snapshot at each t
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        filein = tmpdirname + 'integrate.in'
        writesnap(particles, filename=filein, time=0.0)
        fileout = tmpdirname + 'integrate.falc'
        if os.path.isfile(fileout):
            os.remove(fileout)
        env = dict(os.environ)
        p = subprocess.Popen([_NEMO_DEHNEN_LOCATION + "falcON/bin/gyrfalcON", "in=" + filein,
                              "out=" + fileout, "tstop=" + str(t), "step=" + str(step), "theta=0.5", "eps=0.05",
                              "kmax=6",
                              "give=mxvp"], env=env)
        (_, _) = p.communicate()
        p = subprocess.Popen([_NEMO_LOCATION + "snapprint in=" + fileout + " options=t,x,y,z,vx,vy,vz,phi"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=env)
        snaps = np.genfromtxt(p.stdout)
        tlist, ti = np.unique(snaps[:, 0], return_inverse=True)
        snaps = (np.reshape(snaps, (len(tlist), np.size(snaps) / (9 * len(tlist)), 9)))[:, :, 1:]
    return tlist, snaps


def gravity_spherical_grid(particles, r, theta, phi, polar_forces=False):
    """
    Takes vectors of r theta and phi and then on the 3d sphical polar grid and 
    computes the gravity using gyrfalcON.

    Returns a dict with entries for (r,theta,phi,F(cartesian or polar),pot)
    """
    phi_v, theta_v, r_v = np.meshgrid(phi, theta, r)
    x = r_v * np.sin(phi_v) * np.cos(theta_v)
    y = r_v * np.sin(phi_v) * np.sin(theta_v)
    z = r_v * np.cos(phi_v)
    pos = np.zeros((len(x.flatten()), 3))
    pos[:, 0] = x.flatten()
    pos[:, 1] = y.flatten()
    pos[:, 2] = z.flatten()
    # gyrfalcon crashes if there are 2 particles in the same location, remove duplicates
    # (eg along z axis), but keep track of indexes to add them back at the end
    unique_pos, inds = np.unique(pos, axis=0, return_inverse=True)
    grav = getgravity(particles, unique_pos)
    ret = {}
    ret['r'] = r
    ret['theta'] = theta
    ret['phi'] = phi
    ret['pot'] = np.reshape(grav[inds, 3], x.shape)
    ret['F'] = np.reshape(grav[inds, 0:3], x.shape + (3,))
    if polar_forces:
        Fx, Fy, Fz = ret['F']
        Fr = (Fx * x + Fy * y + Fz * z) / r_v
        Rcyl = np.sqrt(x ** 2 + y ** 2)
        Ftheta = ((x * Fx + y * Fy) * z - Rcyl ** 2 * Fz) / r_v / Rcyl
        Fphi = (-y * Fx + x * Fy) / Rcyl
        ret['F'] = np.vstack((Fr, Ftheta, Fphi))
    return ret
