import galpy.util.bovy_coords as bovy_coords
import numpy as np
import astropy.units as u


def add_gaia_galactic_pms(df, errors=True):
    mul, mub = bovy_coords.pmrapmdec_to_pmllpmbb(np.array(df['pmra']),
                                                 np.array(df['pmdec']),
                                                 np.array(df['ra_gaia']),
                                                 np.array(df['dec_gaia']), degree=True).T
    df['pml'] = mul
    df['pmb'] = mub
    if errors:
        add_gaia_galactic_pm_errors(df)


def add_gaia_galactic_pm_errors(df):
    off_diag = df['pmra_error'] * df['pmdec_error'] * df['pmra_pmdec_corr']
    covpmrapmdec = np.array([[df['pmra_error'] ** 2, off_diag], [off_diag, df['pmdec_error'] ** 2]])
    ra_deg = np.array(df['ra_gaia'] / u.deg)
    dec_deg = np.array(df['dec_gaia'] / u.deg)

    cov = bovy_coords.cov_pmrapmdec_to_pmllpmbb(np.transpose(covpmrapmdec, [2, 0, 1]), ra_deg, dec_deg, degree=True)

    df['pml_error'] = np.sqrt(cov[:, 0, 0])
    df['pmb_error'] = np.sqrt(cov[:, 1, 1])
    df['pml_pmb_corr'] = cov[:, 0, 1] / (np.sqrt(cov[:, 0, 0]) * np.sqrt(cov[:, 1, 1]))