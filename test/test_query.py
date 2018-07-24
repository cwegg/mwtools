import pytest
import mwtools
import os
import mwtools.xmatch as xmatch
import numpy as np
import pandas as pd
import tempfile


def test_query_wsa():
    df = mwtools.query_wsa('select top 1 ra from gcsSource', database='UKIDSSDR10PLUS')
    assert df.RA[0] == 3.2520437684847763


def test_query_wsa_with_filename():
    testfile = 'testfile.fits'
    df = mwtools.query_wsa('select top 1 ra from gcsSource', filename=testfile)
    assert df.RA[0] == 3.2520437684847763
    os.remove(testfile)


def test_query_wsa_raises_error():
    with pytest.raises(RuntimeError):
        _ = mwtools.query_wsa('select top 1 r from gcsSource')


def test_query_wsa_uploadvot():
    fake_ras = np.arange(10., 20.0)
    fake_decs = np.arange(20., 30.0)
    fake_coordinate_df = pd.DataFrame({'ra': fake_ras, 'de': fake_decs})
    coordinate_table = xmatch._make_coordinate_table(fake_coordinate_df)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.xml', mode='w') as file_to_upload:
        coordinate_table.write(file_to_upload, format='votable')

    df = mwtools.query_wsa('select top 1 ra from gcsSource', file_to_upload=file_to_upload.name)
    assert df.RA[0] == 3.2520437684847763
