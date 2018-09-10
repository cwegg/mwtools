import galpy.util.bovy_coords as bovy_coords
import numpy as np
import astropy.units as u
import tempfile
import os
import sys
import warnings
from astropy.table import Table
from contextlib import contextmanager

# Below is to supress the output produced when importing Gaia from astroquery
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

with suppress_stdout():
    from astroquery.gaia import Gaia


def add_gaia_galactic_pms(df, errors=True):
    old_settings = np.seterr(invalid='ignore')
    try:
        ra,dec = np.array(['ra_gaia']),np.array(df['dec_gaia'])
    except KeyError:
        ra,dec = np.array(['ra']),np.array(df['dec'])
    pmra, pmdec = np.array(['pmra']),np.array(df['pmdec'])

    mul, mub = bovy_coords.pmrapmdec_to_pmllpmbb(pmra,pmdec,ra,dec, degree=True).T
    df['pml'] = mul
    df['pmb'] = mub
    if errors:
        add_gaia_galactic_pm_errors(df)
    np.seterr(**old_settings)


def add_gaia_galactic_pm_errors(df):
        off_diag = df['pmra_error'] * df['pmdec_error'] * df['pmra_pmdec_corr']
    covpmrapmdec = np.array([[df['pmra_error'] ** 2, off_diag], [off_diag, df['pmdec_error'] ** 2]])
    try:
        ra_deg = np.array(df['ra_gaia'] / u.deg)
        dec_deg = np.array(df['dec_gaia'] / u.deg)
    except KeyError:
        ra_deg = np.array(df['ra'] / u.deg)
        dec_deg = np.array(df['dec'] / u.deg)

    cov = bovy_coords.cov_pmrapmdec_to_pmllpmbb(np.transpose(covpmrapmdec, [2, 0, 1]), ra_deg, dec_deg, degree=True)

    df['pml_error'] = np.sqrt(cov[:, 0, 0])
    df['pmb_error'] = np.sqrt(cov[:, 1, 1])
    df['pml_pmb_corr'] = cov[:, 0, 1] / (np.sqrt(cov[:, 0, 0]) * np.sqrt(cov[:, 1, 1]))

def Gaia_adql(query,upload=None):
    """Run query on gaia archive and return results as a pandas dataframe.
    If upload is a pandas dataframe then this is uploaded to the archive
    and available as the table tap_upload.uploadedtable"""

    # We take and return pandas DataFrames but use astropy tables to make a votable of coordinates to upload

    with tempfile.NamedTemporaryFile(suffix='.xml', mode='w') as file_to_upload:
        if upload is not None:
            table = Table.from_pandas(upload)
            table.write(file_to_upload, format='votable')

        # It seems Gaia DR2 source table produces many votable warnings that aren't important
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            with suppress_stdout():
                if upload is not None:
                    job = Gaia.launch_job_async(query=query, upload_resource=file_to_upload.name,
                                                upload_table_name="uploadedtable")
                else:
                    job = Gaia.launch_job_async(query=query)

        result = job.get_results()
        result_df = result.to_pandas()

    return result_df