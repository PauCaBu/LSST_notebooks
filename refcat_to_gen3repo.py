import glob
import astropy.table
import os 

table = astropy.table.Table(names=("filename", "htm7"), dtype=('str', 'int'))
for file in glob.glob("~/ps1_pv3_3pi_20170110/*.fits"):
    u = os.path.splitext(file)[0]
    a = u.split('/')
    table.add_row((file, a[1]))
table.write("~/filename_to_htm.ecsv", overwrite=True)
