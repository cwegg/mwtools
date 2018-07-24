import warnings

import astropy.units as u
import numpy as np
import numpy.testing as nptest
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
import pandas as pd
from astroquery.gaia import Gaia

import mwtools


def test_getcolumn_lower():
    fake_ras = np.arange(10., 20.0)
    fake_coordinate_df = pd.DataFrame({'ra':fake_ras})
    nptest.assert_equal(mwtools.xmatch._get_column_from_keylist(fake_coordinate_df, ['ra']), fake_ras)


def test_getcolumn_upper():
    fake_ras = np.arange(10., 20.0)
    fake_coordinate_df = pd.DataFrame({'RA':fake_ras})
    nptest.assert_equal(mwtools.xmatch._get_column_from_keylist(fake_coordinate_df, ['RA']), fake_ras)


def test_getcolumn_raises_error():
    fake_ras = np.arange(10., 20.0)
    fake_coordinate_df = pd.DataFrame({'ra':fake_ras})
    with pytest.raises(ValueError):
        mwtools.xmatch._get_column_from_keylist(fake_coordinate_df, ['RA', 'DEC'])


def test_make_coordinate_table():
    fake_ras = np.arange(10., 20.0)
    fake_decs = np.arange(20., 30.0)
    fake_coordinate_df = pd.DataFrame({'ra':fake_ras,'de':fake_decs})
    coordinate_table = mwtools.xmatch._make_coordinate_table(fake_coordinate_df)
    nptest.assert_equal(coordinate_table['RA'], fake_coordinate_df['ra'])
    nptest.assert_equal(coordinate_table['DEC'], fake_coordinate_df['de'])


def test_make_coordinate_table_raises_error():
    fake_ras = np.arange(10., 20.0)
    fake_decs = np.arange(20., 30.0)
    fake_coordinate_df = pd.DataFrame({'RANDOM':fake_ras, 'de':fake_decs})
    with pytest.raises(ValueError):
        _ = mwtools.xmatch._make_coordinate_table(fake_coordinate_df)

def test_gaiadr2_xmatch():
    # get a sample of stars to cross match
    center_coord = SkyCoord(ra=280, dec=-60, unit=(u.degree, u.degree), frame='icrs')
    width = u.Quantity(0.1, u.deg)
    height = width
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        sample_of_stars = Gaia.query_object_async(coordinate=center_coord, width=width, height=height)

    # cross match the coordinates and check we get back the same thing
    coordinates_of_stars = pd.DataFrame({'ra':sample_of_stars['ra'],'de':sample_of_stars['dec']})
    xmatch_results = mwtools.xmatch.Gaia_DR2_Xmatch(coordinates_of_stars)
    nptest.assert_equal(xmatch_results['source_id'], sample_of_stars['source_id'])
