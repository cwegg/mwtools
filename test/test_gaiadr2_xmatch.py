import warnings

import astropy.units as u
import numpy as np
import numpy.testing as nptest
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.gaia import Gaia

import gaiabulge.xmatch as xmatch


def test_getcolumn_lower():
    fake_ras = np.arange(10., 20.0)
    fake_coordinate_table = Table(data=(fake_ras,), names=('ra',))
    nptest.assert_equal(xmatch._get_column_from_keylist(fake_coordinate_table, ['ra']), fake_ras)


def test_getcolumn_upper():
    fake_ras = np.arange(10., 20.0)
    fake_coordinate_table = Table(data=(fake_ras,), names=('RA',))
    nptest.assert_equal(xmatch._get_column_from_keylist(fake_coordinate_table, ['RA']), fake_ras)


def test_getcolumn_raises_error():
    fake_ras = np.arange(10., 20.0)
    fake_coordinate_table = Table(data=(fake_ras,), names=('ra',))
    with pytest.raises(ValueError):
        xmatch._get_column_from_keylist(fake_coordinate_table, ['RA', 'DEC'])


def test_make_coordinate_table():
    fake_ras = np.arange(10., 20.0)
    fake_decs = np.arange(20., 30.0)
    fake_coordinate_table = Table(data=(fake_ras, fake_decs), names=('ra', 'de'))
    coordinate_table = xmatch._make_coordinate_table(fake_coordinate_table)
    nptest.assert_equal(coordinate_table['RA'], fake_coordinate_table['ra'])
    nptest.assert_equal(coordinate_table['DEC'], fake_coordinate_table['de'])


def test_make_coordinate_table_raises_error():
    fake_ras = np.arange(10., 20.0)
    fake_decs = np.arange(20., 30.0)
    fake_coordinate_table = Table(data=(fake_ras, fake_decs), names=('RANDOM', 'de'))
    with pytest.raises(ValueError):
        coordinate_table = xmatch._make_coordinate_table(fake_coordinate_table)

def test_gaiadr2_xmatch():
    # get a sample of stars to cross match
    center_coord = SkyCoord(ra=280, dec=-60, unit=(u.degree, u.degree), frame='icrs')
    width = u.Quantity(0.1, u.deg)
    height = width
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        sample_of_stars = Gaia.query_object_async(coordinate=center_coord, width=width, height=height)

    # cross match the coordinates and check we get back the same thing
    coordinates_of_stars = Table(sample_of_stars)
    coordinates_of_stars.keep_columns(['ra', 'dec'])
    xmatch_results = xmatch.Gaia_DR2_Xmatch(coordinates_of_stars)
    nptest.assert_equal(xmatch_results['source_id'], sample_of_stars['source_id'])
