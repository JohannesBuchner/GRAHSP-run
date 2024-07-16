"""
Create a modified photometry input file, with new rows that leave out one filter at a time.

The new ids are <origid>-no-<filtername> where <origid> is
the original id, and filtername is the name of the filter left out.

Synopsis: python3 loofilter.py input.fits input-new.fits
"""

from astropy.table import Table, vstack
import sys

infile = sys.argv[1]
outfile = sys.argv[2]

f = Table.read(infile)
flux_columns = [col for col in f.colnames if col + '_err' in f.colnames]

tables = [f]
for col in flux_columns:
	# add a identical row, with original_row[col] set to a negative number (-9999)
	modf = f.copy()
	modf[col] = -9999
	modf['id'] = [str(i) + '-no-' + str(col) for i in modf['id']]
	tables.append(modf)
	print(modf)

vstack(tables).write(outfile, overwrite=True)
