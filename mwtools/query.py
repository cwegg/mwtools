from astropy.io import fits
import re
import urllib
from http.cookiejar import CookieJar
import urllib.request as request
import requests
import os
import tempfile
import pandas as pd
import numpy as np

from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)


def query_wsa(sql, database='UKIDSSDR10PLUS', filename=None, file_to_upload=None, lowercase=True):
    """Submits a SQL query to the (WSA - WFCAM Science Archive) and returns the result as a pandas dataframe.
    Query is submitted to http://wsa.roe.ac.uk:8080/wsa/SQL_form.jsp. If a file_to_upload is a votable then
    this can be accessed as #userTable. See the WSA website for full information. It filename is set then we write
    the results to a fits file here (useful for caching results of long running queries).

    For acknowledgement see http://wsa.roe.ac.uk/pubs.html"""

    login_details = {'username': '', 'password': '', 'community': ''}
    login_details['username'] = os.environ.get("WSAUSERNAME")
    login_details['password'] = os.environ.get("WSAPASSWORD")
    login_details['community'] = os.environ.get("WSACOMMUNITY")

    loginurl = "http://surveys.roe.ac.uk:8080/wsa/DBLogin?user=%s&passwd=%s&community=+&community2=+%s&submit=Login"
    sqlurl = "http://wsa.roe.ac.uk:8080/wsa/WSASQL"

    return _query_w_or_v_sa(sql, database, filename, file_to_upload, loginurl, sqlurl, login_details,
                            lowercase=lowercase)


def query_vsa(sql, programmeID='VVV', database='VVVDR4', filename=None, file_to_upload=None, lowercase=True):
    """Submits a SQL query to the VSA (VISTA Science Archive) and returns the result as a pandas dataframe.
    Query is submitted to http://horus.roe.ac.uk:8080/vdfs/VSQL_form.jsp If a file_to_upload is a votable then
    this can be accessed as #userTable. See the WSA website for full information. It filename is set then we write
    the results to a fits file here (useful for caching results of long running queries).

    For acknowledgement see http://horus.roe.ac.uk/vsa/pubs.html
    Adapted from http://casu.ast.cam.ac.uk/surveys-projects/wfcam/data-access/wsa-freeform.py"""

    login_details = {'username': '', 'password': '', 'community': ''}
    login_details['username'] = os.environ.get("VSAUSERNAME")
    login_details['password'] = os.environ.get("VSAPASSWORD")
    login_details['community'] = os.environ.get("VSACOMMUNITY")

    loginurl = "http://surveys.roe.ac.uk:8080/wsa/DBLogin?user=%s&passwd=%s&community=+&community2=+%s&submit=Login"
    sqlurl = "http://horus.roe.ac.uk:8080/vdfs/WSASQL"

    programs = {'VHS': "110", 'VVV': "120", 'VMC': "130", 'VIKING': "140", 'VIDEO': "150", 'SHARKS': "175",
                'VVVX': "170", 'GCAV': "180", 'VISIONS': "185", 'VEILS': "190", 'VINROUGE':"195",
               'UltraVISTA':"160", 'Calibration': "200"}
    if programmeID in programs:
        programmeIDnumber = programs[programmeID]
    else:
        raise ValueError("programmeID {} not recognised".format(programmeID))

    return _query_w_or_v_sa(sql, database, filename, file_to_upload, loginurl, sqlurl,
                            login_details, programme_id=programmeIDnumber, lowercase=lowercase)


def _query_w_or_v_sa(sql, database, filename, file_to_upload, loginurl, sqlurl, login_details, programme_id=None,
                     lowercase=True):
    """Adapted from http://casu.ast.cam.ac.uk/surveys-projects/wfcam/data-access/wsa-freeform.py"""
    # Send request to login to the archive

    cj = CookieJar()
    if login_details['username'] is not None and login_details['username'].strip():
        # There are non-empty login details
        q = loginurl % (login_details['username'], login_details['password'], login_details['community'])
        response = urllib.request.urlopen(q)
        request = urllib.request.Request(q)

        # Extract the cookies from the response header and use them for future connections
        cj.extract_cookies(response, request)
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    # Construct and post the request
    postdata = {'formaction': 'freeform', 'sqlstmt': sql, 'emailAddress': '',
                'database': database, 'timeout': 1800,
                'format': 'FITS', 'compress': 'GZIP', 'rows': 30, 'iFmt': 'VOTable'}
    if programme_id is not None:
        postdata['programmeID'] = programme_id

    if file_to_upload is not None:
        files = {'uploadSQLFile': (file_to_upload, open(file_to_upload, 'rb'))}
    else:
        files = None

    response = requests.post(sqlurl, data=postdata, files=files)
    res = response.text
    # Find where our output file is
    try:
        fitsfile = re.compile("<a href=\"(\S+%s).+" % 'fits.gz').search(str(res)).group(1)
    except AttributeError:
        raise RuntimeError('Query Failed, Reponse: {}'.format(res))

    # Request the fitsfile
    fitsres = opener.open(fitsfile).read()
    # Save file to local disk
    if filename:
        with open(filename, 'wb') as f:
            f.write(fitsres)

    else:
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix='.fits.gz', mode='wb') as f:
            f.write(fitsres)

    hdulist = fits.open(f.name)

    # FITs files are big endian, while pandas assumes native byte order i.e. little endian on x86
    # calling .byteswap().newbyteorder() on a numpy array switches to native order
    df = pd.DataFrame(np.array(hdulist[1].data).byteswap().newbyteorder())

    if lowercase:
        df.columns = map(str.lower, df.columns)

    return df
