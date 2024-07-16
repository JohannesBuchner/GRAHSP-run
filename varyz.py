"""
Create a modified photometry input file, with new rows with slightly altered redshifts.

The new rows are for example <origid>-deltaz0.3 where <origid> is
the original id with the redshift changed by +0.3.

Synopsis: python3 varyz.py input.fits input-new.fits
"""

import numpy as np
from astropy.table import Table, vstack
import sys

infile = sys.argv[1]
outfile = sys.argv[2]

f = Table.read(infile)
flux_columns = [col for col in f.colnames if col + '_err' in f.colnames]

tables = [f]
for deltaz in np.arange(-0.7, +0.8, 0.1):
	if deltaz == 0: continue
	# add a identical row, with redshift changed
	modf = f.copy()
	modf['id'] = [str(i) + '-deltaz%.1f' % deltaz for i in modf['id']]
	modf['redshift'] = f['redshift'] + deltaz
	tables.append(modf[modf['redshift'] > 0])

vstack(tables).write(outfile, overwrite=True)
