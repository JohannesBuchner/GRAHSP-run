import numpy as np
import os
import sys
from pcigale.data import Database, _Filter
from pcigale.data.filters import Filter
import argparse
from pathlib import Path


class HelpfulParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


parser = HelpfulParser(
    description=__doc__,
    epilog="""Johannes Buchner (C) 2013-2023 <johannes.buchner.acad@gmx.com>""",
    formatter_class=argparse.RawDescriptionHelpFormatter
)

parser.add_argument(
    '-l', '--loc', type=str, required=True,
    help='Localisation of the filter transmission.')

parser.add_argument(
    '-n', '--name', type=str,
    help='Name of the filter to register.')

parser.add_argument(
    '-d', '--description', type=str, default='',
    help='Description of the filter to register.')

parser.add_argument(
    '-ttp', '--trans_type', type=str, choices=['energy', 'photon'], default='photon',
    help="Type of transmission table ('energy' or 'photon', default 'photon')")

parser.add_argument(
    '-eff','--eff_lambda', type=float, 
    help='Effective_wavelength of the filter.')

parser.add_argument(
    '-u', '--units', type=str, choices=['nm', 'A'], default='nm',
    help="Units of the wavelength ('nm' or 'A'). Default: nm")

parser.add_argument(
    '-del','--delete', action='store_true', default=False,
    help='Delete the filter from the database.')


args = parser.parse_args()
print(args)


Loc = Path(args.loc)
assert Loc is not None, 'The localisation of the transmission file is needed.'
assert Loc.exists(), 'No file found at the given location.'


if args.name is None:
    name = Loc.stem
else:
    name = args.name
print(name)


trans_table = np.loadtxt(Loc)
shape = trans_table.shape

assert len(shape)==2 and (shape[0]==2 or shape[1]==2), 'The transmission file does not have the correct dimension, it should be a 2-dimensional array.'

if shape[1] == 2:
    trans_table = trans_table.T


if args.units == 'A':
    trans_table[0] /= 10


new_filter = Filter(name, trans_table=trans_table, 
                    trans_type=args.trans_type, description=args.description,
                   effective_wavelength=args.eff_lambda)

if args.eff_lambda is None:
    new_filter.normalise()

print(new_filter)


with Database(writable=True) as base:
    if args.delete:
        filter_to_delete = base.session.query(_Filter).filter(_Filter.name==name).first()
        assert filter_to_delete is not None, 'The filter to delete does not exists.'
        base.session.delete(filter_to_delete)
        base.session.commit()
        filter_to_delete = base.session.query(_Filter).filter(_Filter.name==name).first()
        print(f'Filter {name} deleted.')
    else:
        base.add_filter(new_filter)
    









