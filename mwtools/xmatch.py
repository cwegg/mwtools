import tempfile
import warnings

from astropy.table import Table
from astroquery.gaia import Gaia
import numpy as np

def _get_column_from_keylist(df, keylist):
    for key in keylist:
        if key in df.columns:
            return df[key]
    raise ValueError("None of {} found in DataFrame".format(keylist))


def _make_coordinate_table(df):
    ra_column = _get_column_from_keylist(df, ['ra', 'RA'])
    dec_column = _get_column_from_keylist(df, ['dec', 'DEC', 'de', 'DE'])
    coordinate_table = Table(data=[ra_column, dec_column], names=('RA', 'DEC'))
    return coordinate_table


def Gaia_DR2_Xmatch(df, dist=1):
    """Cross match a pandas dataframe to Gaia DR2. The dataframe should have columns named ra and de(c). Returns
    the nearest cross match"""

    # We take and return pandas DataFrames but use astropy tables to make a votable of coordinates to upload
    coordinate_table = _make_coordinate_table(df)
    xmatch_id = np.arange(len(df.index))
    coordinate_table['xmatch_id'] = xmatch_id  # add a column to keep cross of cross matches on
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xml', mode='w') as file_to_upload:
        coordinate_table.write(file_to_upload, format='votable')

    # Construct cross-match query. Taken from the Gaia archive examples
    cross_match_query = """SELECT distance(POINT('ICRS', mystars.ra, mystars.dec), POINT('ICRS', gaia.ra, gaia.dec))
        AS dist, mystars. *, gaia. * FROM tap_upload.table_test
        AS mystars, gaiadr2.gaia_source
        AS gaia WHERE 
        1 = CONTAINS(POINT('ICRS', mystars.ra, mystars.dec), CIRCLE('ICRS', gaia.ra, gaia.dec, {}))""".format(
        dist / 3600.)

    # It seems Gaia DR2 source table produces many votable warnings that aren't important
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        job = Gaia.launch_job_async(query=cross_match_query, upload_resource=file_to_upload.name,
                                    upload_table_name="table_test")
    xmatched_table = job.get_results()
    xmatched_table.remove_columns(['ra', 'dec'])
    xmatched_table.rename_column('ra_2', 'ra_gaia')
    xmatched_table.rename_column('dec_2', 'dec_gaia')
    xmatched_df = xmatched_table.to_pandas()
    if xmatched_df.empty:
        raise ValueError('No crossmatches found')
    # joined_table = astropy.table.join(table, xmatched_table, join_type='left', keys='xmatch_id')
    # joined_table.remove_column('xmatch_id')
    joined_df = df.merge(xmatched_df,how='left',left_on=xmatch_id,right_on='xmatch_id')
    joined_df.drop(['xmatch_id'],1,inplace=True)
    return joined_df
