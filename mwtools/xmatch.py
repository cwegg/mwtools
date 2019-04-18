import tempfile
import warnings
from astropy.table import Table
import numpy as np
from contextlib import contextmanager
import os
import sys

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


def _get_column_from_keylist(df, keylist):
    for key in keylist:
        if key in df.columns:
            return df[key]
    raise ValueError("None of {} found in DataFrame".format(keylist))


def _make_coordinate_table(df):
    ra_column = _get_column_from_keylist(df, ['ra', 'RA','radeg','RAdeg','RA_J2000'])
    dec_column = _get_column_from_keylist(df, ['dec', 'DEC', 'de', 'DE', 'dedeg', 'DEdeg','DEC_J2000'])
    coordinate_table = Table(data=[ra_column, dec_column], names=('RA', 'DEC'))
    return coordinate_table


def Gaia_DR2_Xmatch(df, dist=1, nearest=True):
    """Cross match a pandas dataframe to Gaia DR2. The dataframe should have columns named ra and de(c). Returns
    the nearest cross match if nearest is True (and the returned dataframe is the same size as the input),
    otherwise return all the cross matches (and the returned dataframe will be larger than the input)"""

    # We take and return pandas DataFrames but use astropy tables to make a votable of coordinates to upload
    coordinate_table = _make_coordinate_table(df)
    xmatch_id = np.arange(len(df.index))
    coordinate_table['xmatch_id'] = xmatch_id  # add a column to keep cross of cross matches on
    with tempfile.NamedTemporaryFile(suffix='.xml', mode='w') as file_to_upload:
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
            with suppress_stdout():
                job = Gaia.launch_job_async(query=cross_match_query, upload_resource=file_to_upload.name,
                                            upload_table_name="table_test")
        xmatched_table = job.get_results()

        # Rename columns so we have ra and dec as _gaia 
        xmatched_table.remove_columns(['ra', 'dec'])
        xmatched_table.rename_column('ra_2', 'ra_gaia')
        xmatched_table.rename_column('dec_2', 'dec_gaia')
        xmatched_df = xmatched_table.filled().to_pandas()

        if xmatched_df.empty:
            raise ValueError('No crossmatches found')

        # We have to hack to get the join to preserve our int64 numbers. Because we do an outer join the unmatched
        # stars have NaNs and to facilitate this pandas converts int64 to float64. But this precision loss is
        # unacceptable. We store the upper and lower 32bits seperately and convert back to int64 afterwards
        int64_columns = xmatched_df.dtypes[xmatched_df.dtypes == 'int64']
        for column in int64_columns.index:
            xmatched_df[f'{column}_low'] = xmatched_df[column].astype(np.int32)
            xmatched_df[f'{column}_high'] = (xmatched_df[column].values >> 32).astype(np.int32)

        joined_df = df.merge(xmatched_df,how='left',left_on=xmatch_id,right_on='xmatch_id',suffixes=('_orig',''))
        if nearest:
            # To select the nearest we group by the xmatch_id and select the nearest cross-match
            joined_df = joined_df.sort_values(['xmatch_id', 'dist'], ascending=True).groupby('xmatch_id').first().reset_index()
        joined_df.drop(['xmatch_id'],1,inplace=True)

        # Some DR2 columns end up messed up... fix them. The problem maybe due to NaNs in otherwise bool/string columns
        columns=['designation','datalink_url']
        joined_df.loc[:,columns]=joined_df[columns].applymap(str)
        columns=['astrometric_primary_flag','duplicated_source','phot_variable_flag']
        joined_df.loc[:,columns]=joined_df[columns].applymap(bool)

        # reconstruct the int64 columns from the high and low 32bit parts
        for column in int64_columns.index:
            joined_df[column] = (joined_df[f'{column}_high'].values.astype(np.int64) << 32) + \
                                  joined_df[f'{column}_low'].values.astype(np.int64)
            joined_df.drop([f'{column}_low'],1,inplace=True)
            joined_df.drop([f'{column}_high'],1,inplace=True)


    return joined_df
