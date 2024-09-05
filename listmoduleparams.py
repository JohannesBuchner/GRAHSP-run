#import pcigale.creation_modules
import importlib

modules = 'sfhdelayed bc03 m2005 sfhdelayed nebular dustatt_calzleit dustatt_powerlaw galdale2014 activate activatelines activategtorus activatepl activatebol biattenuation galdale2014 redshifting'.split()

for module_name in modules:
     module = importlib.import_module('pcigale.creation_modules.' + module_name)
     klass = module.Module
     print(f"[{module_name}]")
     for param, (dtype, comment, default_value) in klass.parameter_list.items():
        print(f"    # {comment} (type: {dtype})")
        print(f"    {param} = {default_value}")
     print()
     print(f"  # the code of this module is in: {module.__file__}")
     print()
     

