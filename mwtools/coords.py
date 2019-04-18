import numpy as np

# A couple of lightweight coordinate transformations that aren't easy to access in astropy

def galcencart_to_dlb_rot(d, l, b, r0=8.2, z0=0.014):
    """Compute the rotation matrix that transforms velocities from UVW
    to (d,l,b) - matrix S in Ratnatunga+89 notation"""
    cl, sl = np.cos(l), np.sin(l)
    cb, sb = np.cos(b), np.sin(b)

    z = d * sb - z0
    cylr = np.sqrt(r0 ** 2 + (d * cb) ** 2 - 2 * r0 * d * cb * cl)  # A1 of RBC89

    smat = np.array([[(r0 * cl - d * cb) * cb, r0 * sl * cb, cylr * sb],
                     [-r0 * sl, (r0 * cl - d * cb), np.zeros_like(d)],
                     [-(r0 * cl - d * cb) * sb, -r0 * sl * sb, cylr * cb]])  # A4 of RBC89
    smat = np.transpose(smat / cylr, [2, 0, 1])

    # We follow Ratnatunga+89 (RBC89) in conventions:
    # isigma is in (1,2,3) coordinate system which is aligned to (U,V,W) near the sun
    # where U is positive *towards* GC, V is in direction of rotation, W towards NGP.
    # to rotate we use  (U,V,W) = tmat @ (1,2,3)
    # we rotate by velocities by an angle t anti-clockwise from the (1,3) plane to get (U,W)
    # Near the sun then spherical alignment this means tan t = Z/cylR (see RBC89 theta defn. between
    # A3 and A4).
    # Near the minor axis then spherical alignment means tan t = -cylR/Z if we want 3 to align
    # with the z axis on the minor axis

    # We need to be careful when z<0. We could reverse the sign of vrvtheta but instead we fold z < 0 back to z> 0
    # by using abs(z) and flipping W
    smat[z < 0, :, 2] *= -1  # Reverse the sign of W when we are below the galactic plane
    return smat

def sph_to_dlb_rot(d, l, b, r0=8.2, z0=0.014):
    """Compute the rotation matrix that transforms velocities from galactocentric spherical coordinates (r,phi,theta)
    to (d,l,b)"""

    smat = galcencart_to_dlb_rot(d, l, b, r0=r0, z0=z0)
    cl, sl = np.cos(l), np.sin(l)
    cb, sb = np.cos(b), np.sin(b)
    z = d * sb - z0
    cylr = np.sqrt(r0 ** 2 + (d * cb) ** 2 - 2 * r0 * d * cb * cl)  # A1 of RBC89
    r = np.sqrt(cylr ** 2 + z ** 2)  # A2 of RBC89

    t = np.arcsin(np.abs(z) / r)

    ct, st = np.cos(t), np.sin(t)
    cp, sp = np.full(t.shape, 1.), np.full(t.shape, 0.)

    tmat = np.array([[ct * cp, -ct * sp, st],
                     [sp, cp, np.zeros_like(cp)],
                     [-st * cp, st * sp, ct]])  # A3 of RBC89
    tmat = np.transpose(tmat, [2, 0, 1])

    return smat @ tmat

def subtract_vsun(l, b, vl=None, vb=None, vr=None, vsun=(11.1, 12.24 + 238., 7.25),add=False):
    cl, sl = np.cos(l), np.sin(l)
    cb, sb = np.cos(b), np.sin(b)
    vlcorr = (vsun[0] * sl - vsun[1] * cl) # can be found in eg. Schönrich+11 eq 4
    vbcorr = (vsun[0] * cl * sb + vsun[1] * sl * sb - vsun[2] * cb)
    vrcorr = -vsun[0] * cb * cl - cb * sl * vsun[1] - vsun[2] * sb
    #print(vlcorr,vbcorr,vrcorr)
    if add:
        if vl is not None:
            vlcorr = vl + vlcorr
        if vb is not None:
            vbcorr = vb + vbcorr
        if vr is not None:
            vrcorr = vr + vrcorr
    else:
        if vl is not None:
            vlcorr = vl - vlcorr
        if vb is not None:
            vbcorr = vb - vbcorr
        if vr is not None:
            vrcorr = vr - vrcorr

    ret=[]
    if vl is not None:
        ret.append(vlcorr)
    if vb is not None:
        ret.append(vbcorr)
    if vr is not None:
        ret.append(vrcorr)

    return np.array(ret)
