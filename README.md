# mw-data-tools

Some Common tools for dealing with MW data and e.g. cross matching, querying WSA/VSA

To query private WSA or VSA place your usename and password in the .env file like: 
WSAUSERNAME = MaxMustermann
WSAPASSWORD = password1234
WSACOMMUNITY = myinstitute.edu
Or similar for VVV.

To use nemo interface at mwtools.nemo then:
-Install Nemo
-Give the locations of the nemo executables as NEMO_LOCATION in the .env file
-Give the location of Walter Dehnens falcOn as NEMO_DEHNEN_LOCATION in .env (this should have the directories 'falcON' and 'utils' inside)
