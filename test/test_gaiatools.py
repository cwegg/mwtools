import pytest
import numpy as np
import mwtools
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia
import warnings

@pytest.fixture
def gaia_xmatch_results():
    # get a sample of stars to cross match
    center_coord = SkyCoord(ra=280, dec=-60, unit=(u.degree, u.degree), frame='icrs')
    width = u.Quantity(0.05, u.deg)
    height = width
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        sample_of_stars = Gaia.query_object_async(coordinate=center_coord, width=width, height=height)

    # cross match the coordinates and check we get back the same thing
    coordinates_of_stars = pd.DataFrame({'ra': sample_of_stars['ra'], 'de': sample_of_stars['dec']})
    xmatch_results = mwtools.xmatch.Gaia_DR2_Xmatch(coordinates_of_stars)
    return xmatch_results


def test_gaiadr2_pms(gaia_xmatch_results):
    mwtools.add_gaia_galactic_pms(gaia_xmatch_results)
    assert gaia_xmatch_results.pml[0] == -6.331141220895921


def test_gaiadr2_pmerrors(gaia_xmatch_results):
    mwtools.add_gaia_galactic_pms(gaia_xmatch_results)
    assert gaia_xmatch_results.pml_pmb_corr[0] == -0.030277463861580655
