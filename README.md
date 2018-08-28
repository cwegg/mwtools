# mwtools

Some Common tools for dealing with MW data and e.g. cross matching, querying WSA/VSA

For example usage look at examples/examples.ipynb

To install run either

    pip install .
    
or

    python setup.py install
    

After installation, if you'd like to query the private databases on WSA or VSA 
place your usename and password in the .env file like:

    WSAUSERNAME = MaxMustermann
    WSAPASSWORD = password1234
    WSACOMMUNITY = myinstitute.edu

Or similar for the VSA.

To use nemo interface at mwtools.nemo then:
1. Install Nemo
2. Give the locations of the nemo executables as NEMO_LOCATION in the .env file
If there are problems with the routines calling nemo from python then add
verbose=True and see what error messages are produced.