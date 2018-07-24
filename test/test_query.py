import pytest
import mwtools
import os

def test_QueryWSA():
    df = mwtools.QueryWSA('select top 1 ra from gcsSource',database='UKIDSSDR10PLUS')
    assert df.RA[0] == 3.2520437684847763

def test_QueryWSA_with_filename():
    testfile='testfile.fits'
    df = mwtools.QueryWSA('select top 1 ra from gcsSource',filename=testfile)
    assert df.RA[0] == 3.2520437684847763
    os.remove(testfile)

def test_QueryWSA_raises_error():
    with pytest.raises(RuntimeError):
        _ = mwtools.QueryWSA('select top 1 r from gcsSource')
