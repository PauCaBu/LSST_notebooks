import glob
import astropy.table
import os 

table = astropy.table.Table(names=("filename", "htm7"), dtype=('str', 'int'))
for file in glob.glob("~/gaia_dr2_20200414/*.fits"):
    u = os.path.splitext(file)[0]
    a = u.split('/')
    table.add_row((file, a[1]))
table.write("~/filename_to_htm_gaia.ecsv", overwrite=True)

#ps1_pv3_3pi_20170110