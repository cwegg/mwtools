import tempfile
import warnings

import astropy
from astropy.table import Table
from astroquery.gaia import Gaia


def _get_column_from_keylist(table, keylist):
    for key in keylist:
        if key in table.colnames:
            return table[key]
    raise ValueError("None of {} found in table".format(keylist))


def _make_coordinate_table(table):
    ra_column = _get_column_from_keylist(table, ['ra', 'RA'])
    dec_column = _get_column_from_keylist(table, ['dec', 'DEC', 'de', 'DE'])
    coordinate_table = Table(data=[ra_column, dec_column], names=('RA', 'DEC'))
    return coordinate_table


def Gaia_DR2_Xmatch(table, dist=1):
    coordinate_table = _make_coordinate_table(table)
    coordinate_table['xmatch_id'] = range(len(table))  # add a column to keep cross of cross matches on
    table['xmatch_id'] = range(len(table))
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xml', mode='w') as file_to_upload:
        coordinate_table.write(file_to_upload, format='votable')

    # Construct cross-match query. Taken from the Gaia archive examples
    cross_match_query = """SELECT distance(POINT('ICRS', mystars.ra, mystars.dec), POINT('ICRS', gaia.ra, gaia.dec))
        AS dist, mystars. *, gaia. * FROM tap_upload.table_test
        AS mystars, gaiadr2.gaia_source
        AS gaia WHERE 
        1 = CONTAINS(POINT('ICRS', mystars.ra, mystars.dec), CIRCLE('ICRS', gaia.ra, gaia.dec, {}))""".format(
        dist / 3600.)

    # Gaia DR2 source table produces many votable warnings that aren't important it seems
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        job = Gaia.launch_job_async(query=cross_match_query, upload_resource=file_to_upload.name,
                                    upload_table_name="table_test")
    xmatched_table = job.get_results()
    xmatched_table.remove_columns(['ra', 'dec'])
    xmatched_table.rename_column('ra_2', 'ra_gaia')
    xmatched_table.rename_column('dec_2', 'dec_gaia')
    joined_table = astropy.table.join(table, xmatched_table, join_type='left', keys='xmatch_id')
    joined_table.remove_column('xmatch_id')
    return joined_table
