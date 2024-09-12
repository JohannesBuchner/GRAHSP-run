import numpy as np
from pcigale.data import Database
from astropy.table import Table
import argparse
import os
import sys
from collections import Counter
import importlib
from pathlib import Path
import astropy.units as u


def Prepare_Data(path, SAVE=True, savepath=None, delete_missing=False):
    if path.suffix in ['.fits', '.csv']:
        T = Table.read(path)
    else:
        T = Table.read(path, format='ascii')
        
    with Database() as base:
        k, dico = base.get_filter_list()
    keys = [key for key in dico.keys()]
    vals = [val for val in dico.values()]
    
    list_old_names = []
    list_new_names = []
    list_del_names = []
    
    id_name = T.colnames[0]
    redshift_name = T.colnames[1]
    assert 'id' in id_name.lower(), "The 'id' column is missing."
    assert 'z' in redshift_name.lower() or 'redshift' in redshift_name.lower(), "the 'redshift' column is missing."
    T.rename_columns(T.colnames[:2], ['id', 'redshift'])
    
    for name in [colname for colname in T.colnames 
                 if colname not in ['id', 'redshift', 'redshift_err', 'zphot_distrib', 'flag_zphot', 'PZ', 'ZGRID']]:
        if name[-3:] != 'err':
            if name not in keys:
                if delete_missing:
                    print(f"\nFilter '{name}' not registered. Data deleted.")
                    list_del_names += [name, name + '_err']
                else:
                    L_possibilities = []
                    for part in name.replace('.', '_').split('_'):
                        if len(part) == 1:
                            L_possibilities += [name for name in keys if part.lower() in name.lower().replace('.', '_').split('_')]
                        else:
                            L_possibilities += [name for name in keys if part.lower() in name.lower()]

                    print(f"\nFilter '{name}' not registered!")
                    if len(L_possibilities) > 0:
                        C = Counter(L_possibilities)
                        
                        L_possibilities = np.unique(L_possibilities)
                        print('Available filters that might correspond:')
                        print(L_possibilities)
                        
                        best_option = None
                        if Counter(C.values())[max(C.values())] == 1: # unique most likely filter
                            print('Best option: ', C.most_common()[0][0])
                            best_option = C.most_common()[0][0]
                            
                        print()
                        question = f'If filter available, please write its name.\nIf not, type "delete" to delete the filter of your data. \n("Q" to exit)\nFilter name (default: {best_option}): '
                        
                        valid_answer = False
                        while not valid_answer:
                            answer = input(question)
                            if answer.lower() == 'delete':
                                list_del_names += [name, name + '_err']
                                valid_answer = True
                            elif answer in L_possibilities:
                                list_old_names += [name, name + '_err']
                                list_new_names += [answer, answer + '_err']
                                valid_answer = True
                            elif answer.lower() == 'q':
                                return
                            elif best_option is not None and answer == '':
                                list_old_names += [name, name + '_err']
                                list_new_names += [best_option, best_option + '_err']
                                valid_answer = True
                            else:
                                print('Not a valid anser. Please retry.')
                        print()
                    else:
                        print(f"No match to the '{name}' filter was found. \nPlease, verify column names and/or register corresponding filter.")
                        list_del_names += [name, name + '_err']

        old_units = T[name].unit
        if old_units is None:
            print(f"\nNo units were given for the {name} column. \nIt is assumed to be flux in mJy.")
            T[name] = T[name] * u.mJy
        elif old_units != u.mJy:
            T[name] = T[name].to(u.mJy)
            print(f"\nThe flux units of the {name} column have been changed from {old_units} to mJy.")
                
    
    T.remove_columns(list_del_names)
    print('The following columns have been removed:')
    print(list_del_names)
    print()

    if len(list_new_names) == 0:
        print('All the column names were correct.')
    elif not delete_missing:
        T.rename_columns(list_old_names, list_new_names)
        print('The columns are renamed.')
    else:
        print('The unknown columns were deleted.')
    
    if SAVE:
        if savepath is None:
            question = "No path to save indicated, please indicate it: "
            savepath = input(question)
        T.write(savepath, overwrite=True)
        print('New file saved.')
        
    txt = 'column_list = '
    for filt in [name for name in T.colnames[2:] if name not in ['redshift_err', 'zphot_distrib', 'flag_zphot', 'PZ', 'ZGRID']]:
        txt += filt + ', '
    txt = txt[:-2]    
        
    return T, txt

def get_stellar_pop():
    question = "What stellar population do you want to use?\n\
Type 'bc03' for Bruzual&Charlot (2003) or 'm2005' for Maraston (2005)\n\
Stellar pop (default: m2005): "
    valid_answer = False
    while not valid_answer:
        answer = input(question)
        if answer.lower() == 'bc03':
            valid_answer = True
            stellar_pop = 'bc03'
        elif answer.lower() == 'm2005' or answer == '':
            valid_answer = True
            stellar_pop = 'm2005'
    return stellar_pop



def create_pcigale(path_pcigale_empty, path_new_pcigale, filename, txt):
    with open(path_pcigale_empty, 'r') as f:
        pcigale_empty = f.readlines()
    with open(path_new_pcigale, 'w') as fout:
        for line in pcigale_empty:
            if line[:9] == 'data_file':
                line = f'data_file = {filename}\n'
            elif line[:16] == 'creation_modules':
                stellar_pop = get_stellar_pop()
                list_modules = f'sfhdelayed, {stellar_pop}, nebular, activate, activatelines, activategtorus, activatepl, activatebol, biattenuation, galdale2014, redshifting'
                line = f'creation_modules = {list_modules}\n'
            elif line[:11] == 'column_list':
                line = txt + '\n'
            elif line[:15] == 'analysis_method':
                line = "analysis_method = pdf_analysis\n"
#            elif '[sed_creation_modules]' in line:
#                for module_name in list_modules.split():
#                     module_name = module_name.replace(',', '')
#                     module = importlib.import_module('pcigale.creation_modules.' + module_name)
#                     klass = module.Module
#                     line += f"\n  [[{module_name}]]\n"
#                     for param, (dtype, comment, default_value) in klass.parameter_list.items():
#                        line += f"      # {comment} (type: {dtype})\n"
#                        line += f"      {param} = {default_value}\n"
#                     line += f"    # the code of this module is in: {module.__file__}\n"
            fout.write(line)



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
    '-p', '--path', type=str, required=True,
    help='Path of the catalogue to prepare (required).')
args = parser.parse_args()
assert os.path.exists(args.path), "The given path does not exist."



def main():
    filename = Path(args.path)
    current_path = Path.cwd()
    new_filename = current_path / ('new_' + filename.name)
    path_pcigale_empty = Path(__file__).parent / 'template_pcigale.ini'
    path_new_pcigale = current_path / 'pcigale.ini'

    data, txt = Prepare_Data(filename, savepath=new_filename)

    need_pcigale_file = True
    if path_new_pcigale.exists():
        question = f"The file {path_new_pcigale} already exists.\nDo you want to replace it? [y]/n "
        valid_answer = False
        while not valid_answer:
            answer = input(question)
            if answer.lower() == 'y' or answer == '':
                valid_answer = True
                print(f"The file {path_new_pcigale} has been replaced.")
            if answer.lower() == 'n':
                valid_answer = True
                need_pcigale_file = False
                
    if need_pcigale_file:
        create_pcigale(path_pcigale_empty, path_new_pcigale, new_filename, txt)
                


if __name__ == '__main__':
    main()




































