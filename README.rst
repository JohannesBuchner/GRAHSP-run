GRAHSP Installation
===================

Current use policy
--------------------

GRAHSP is a private code developed with several years of effort.
There will be a public release in the future. 

That you have received GRAHSP means I trust you to keep the code private
and respect the policy:

* You will get access to the GRAHSP code (GRAHSP, GRAHSP-run repositories)
* You will get access to code documentation and example data sets (GRAHSP-example repository)
* You will get help applying the code and interpreting the results:

  * Resources include scripts and notebooks in GRAHSP-example, and user issues for questions and answers
  * If your questions are generic and may help other users, please open an issue (in GRAHSP-run/ for most issues, in GRAHSP-examples/ for the scripts there and for new example data or notebooks, in GRAHSP/ only for model bugs)

* Scientific publications are encouraged to demonstrate the capabilities and benefits of the code
and to obtain scientific insights.

You are expected to

* help make improvements to the GRAHSP collaboration, such as:

  * opening issues pointing out how to improve the documentation for newcomers (for example you do not understand something while reading a file)
  * help resolve open issues, helping other users
  * sharing code or data sets with other users, such as a jupyter notebook or data set

* offer co-authorship to core GRAHSP developers (Johannes, Mara) and anyone else who helps you with the code during the project

SED fitting is a subtle endevour where one can make many 
mistakes. When preparing the photometry, watch out for 

* type of magnitude (petrosian, Hall, total/aperture, )
* that the aperture matches across bands so you look at the same physical region
* flux conversion (units nanomaggies, mJy, uJy, ...)
* AB vs Vega
* Milky way extinction correction
* redshifts (photo-z, spec-z, reliability)
* galaxy and AGN modelling assumptions

Experts co-authors reviewing the final manuscript can improve your work.

You may be interested in RainbowLasso as well: https://github.com/JohannesBuchner/RainbowLasso

Preliminaries
---------------

1. Have a look at the Cigale Documentation. 
   
GRAHSP is built on top of CIGALE, so there are many commonalities,
such as the input data format.

Currently, GRAHSP and CIGALE cannot be installed alongside each other.

2. Install necessary packages with pip or conda:

 * ultranest 
 * getdist
 * tqdm
 * joblib
 * numba
 * matplotlib
 * scipy
 * astropy
 * sqlalchemy
 * configobj

For example::

	conda install -c conda-forge ultranest tqdm joblib numba h5py sqlalchemy matplotlib configobj astropy
	pip3 install getdist

You need about 4 GB of free space in your python site-package directories 
(usually inside ~/.local or conda folder).

Download instructions
---------------------

3. Download GRAHSP components

* Option 1: get a release tarball from https://github.com/JohannesBuchner/GRAHSP/releases/
* Option 2: clone the latest version from the git repositories

  * GRAHSP: contains the SED model engine: "git clone https://github.com/JohannesBuchner/GRAHSP"
  * GRAHSP-run: contains the SED fitting engine, visualisations, typically updates more often than GRAHSP: "git clone https://github.com/JohannesBuchner/GRAHSP-run"
  * GRAHSP-examples: example data and scripts: "git clone https://github.com/JohannesBuchner/GRAHSP-examples"

Installation instructions
--------------------------

4. go into GRAHSP/ folder: "cd GRAHSP"

5. install with conda or pip3::

	$ SPEED=2 pip3 install -v .
	...
	##############################################################################
	1- Importing filters...

	Importing BX_B90... (21 points)
	Importing B_B90... (21 points)
	...

This takes a while as all the filters and models are imported into the
database (data.db). 

The SPEED environment variable controls how many models to include::

    2 -- quick and small GRAHSP install, 700MB
    1 -- full physical AGN models (Fritz,Netzer), 2800MB, or 
    0 -- like 1 but also include non-solar metallicity galaxies and Draine&Li dust models), 3200MB, 

Building the database may fail on NFS-mounted file systems. Use a local file system if this happens.

Verifying the installation
---------------------------

Verify that the installation was successful:

In the GRAHSP-examples/DR16QWX folder, run::

	$ python3 ../../GRAHSP-run/dualsampler.py --help
	$ python3 ../../GRAHSP-run/dualsampler.py list-filters


Plotting the model
------------------

The GRAHSP-examples/ directory contains python scripts that allow plotting the 
model and its components, and playing with parameter settings.

In particular:

 * **plotgagn.py**: plots the full GRAHSP model and its components

 * **plotstellarpop.py**: allows you to play with stellar populations

   * set the age and tau of the star formation history
   * compare Maraston2005 and BC03 templates

 * **plotattenuation.py**: illustrates the impact of different levels of attenuation

Running GRAHSP
---------------

The fitting is performed with **dualsampler.py**. To understand the interface, run::

	$ python3 ../../GRAHSP-run/dualsampler.py --help

* The model setup is described with a pcigale.ini file. This is virtually identical to CIGALE.
* The data is described with a data file (pointed to in the pcigale.ini). This is virtually identical to CIGALE.
* What to do is set by command line options.
* How to do it (performance settings) is set by environment variables (see below).

In a directory with pcigale.ini file, the following command

	$ python3 ../../GRAHSP-run/dualsampler.py analyse --cores=2 --plot

does:

* load data file and filters
* run in parallel with 2 cores
* for each catalog entry

  * run fitting with ultranest+slice sampling 
  * create posterior chains
  * create plots and diagnostics
  * output files for custom plots
  * create summary file (analysis_results.txt)

* output the fit summary file <inputfilename>_analysis_results.txt. You can convert this to a fits file with::

	$ stilts tpipe in=inputfilename_analysis_results.txt ifmt=CSV out=analysis_results.fits

To obtain a file which contains also the input file columns, the following may be useful::

	$ stilts tmatch2 in1=input.fits_analysis_results.txt ifmt1=ASCII suffix1= values1=id \
		in2=input.fits suffix2=_in values2=id \
		out=analysis_results.fits fixcols=all matcher=exact 

There is also a post-processing script which does the same, and makes
a diagnostic plot of the fit residuals::

	$ python3 ../../GRAHSP-run/postprocess.py


Environment flags
-----------------

On large machines, speed-ups are possible with more memory and CPUs. 
This can be enabled by setting the following environment variables:

* OMP_NUM_THREADS: numpy and other libraries also parallelise some of their functions. 
  If you already parallelise with --cores, 
  you should prevent double parallelisation (which causes slowdown)::

	# do not parallelize within each process
	export OMP_NUM_THREADS=1

* MP_METHOD: This controls how parallelisation is performed, see:
  https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
  It is 'forkserver' by default, alternatives are 'spawn' and 'fork.
  If set to 'joblib', joblib is used for parallelisation: https://joblib.readthedocs.io/en/latest/parallel.html::

	# run with many cores
	python3 ~/workspace/Storage/jbuchner/persistent/GRAHSP/sampler/dualsampler.py analyse --cores=40 --plot --randomize

* CACHE_MAX (default: 10000, suitable for laptops): How many SEDs to cache. Try to increase this. 
  If you see crashes (processes killed), it is likely that you have exceeded the 
  available memory. Then, reduce CACHE_MAX and/or the number of cores::

	# the following relaxes cache constraints since we have huge memory
	# number of models to keep in cache
	export CACHE_MAX=200000
	# report when the cache maximum is reached?
	export CACHE_VERBOSE=1

* DB_IN_MEMORY (default: 0): copy database to memory to avoid mutual blocking of processes::

	# copy database to memory to avoid mutual blocking of processes
	export DB_IN_MEMORY=1

* HDF5_USE_FILE_LOCKING: Most shared remote machines use NFS-mounted file systems. 
  HDF5 (used by ultranest for resume files) has issues with this::

	# locking if running on multiple machines is problematic on NFS
	export HDF5_USE_FILE_LOCKING=FALSE

Other scripts
--------------

Here you can also find:
* loofilter.py: Create a modified photometry input file, with new rows that leave out one filter at a time.
  * Usage: python3 loofilter.py input.fits input-new.fits
* varyz.py: Create a modified photometry input file, with new rows with slightly altered redshifts.
  * Usage: python3 varyz.py input.fits input-new.fits
