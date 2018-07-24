from astropy.io import fits
import re
import urllib
from http.cookiejar import CookieJar
import urllib.request as request
import requests
import os
import tempfile
import pandas as pd


from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


def QueryWSA(sql, database='UKIDSSDR10PLUS', filename=None):
    USERNAME  = os.environ.get("WSAUSERNAME")
    PASSWORD = os.environ.get("WSAPASSWORD")
    COMMUNITY = os.environ.get("WSACOMMUNITY")

    # Send request to login to the archive
    URLDBLOGIN = "http://surveys.roe.ac.uk:8080/wsa/DBLogin?user=%s&passwd=%s&community=+&community2=+%s&submit=Login"
    q = URLDBLOGIN % (USERNAME, PASSWORD, COMMUNITY)
    response = urllib.request.urlopen(q)
    request = urllib.request.Request(q)

    # Extract the cookies from the response header and use them for future connections
    cj = CookieJar()
    cj.extract_cookies(response, request)
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    # Construct and post the request
    dd = {'formaction': 'freeform', 'sqlstmt': sql, 'emailAddress': '',
          'database': database, 'timeout': 1800,
          'format': 'FITS', 'compress': 'GZIP', 'rows': 30}
    response = requests.post('http://wsa.roe.ac.uk:8080/wsa/WSASQL', data=dd)
    res = response.text
    # Find where our output file is
    try:
        fitsfile = re.compile("<a href=\"(\S+%s).+" % 'fits.gz').search(str(res)).group(1)
    except:
        raise RuntimeError('WSA Query Failed, Reponse: {}'.format(res))

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
    return pd.DataFrame(hdulist[1].data)

