#!/usr/bin/env python

"""
Calculate ensemble mean of soil moisture content from all ensemble members

Created by Elliott Warren Wed 20th Nov 2019: elliott.warren@metoffice.gov.uk
"""

import os, pdb, datetime, shutil
import numpy as np

import mule
import mule.operators
from mule import FieldsFile

# Environement variables:
# Directories:
ROSE_DATACPT6H = os.getenv('ROSE_DATACPT6H')
ENS_SMC_DIR = os.getenv('ENS_SMC_DIR')
ENS_PERTS_DIR = os.getenv('ENS_PERTS_DIR')
# Files:
# initial conditions of unperturbed ensemble forecast:
ENS_ANAL_FILE = os.getenv('ENS_ANAL_FILE')

# global dump for masking perturbations under snow:
ENS_SNOW_FILE = os.getenv('ENS_SNOW_FILE')

# number of ensemble members to do perturbations for
NUM_PERT_MEMBERS = os.getenv('NUM_PERT_MEMBERS')

# when called from a suite, picking up environment variables, then
# we don't write out diagnostic files of each stage to disk:
WRITE_DIAG_FILES = False

# actually overwrite the main perturbation files at the end of the script
OVERWRITE_PERT_FILES = True

if ROSE_DATACPT6H is None:
    # if not set, then this is being run for development, so have canned variable settings to hand:
    ROSE_DATACPT6H = "/home/d03/freb/cylc-run/u-ai506/share/cycle/20150629T1200Z"
    ENS_SMC_DIR = "/home/d03/freb/cylc-run/u-ai506/share/cycle/20150629T1800Z/engl_etkf/smc"
    ENS_PERTS_DIR = "/home/d03/freb/cylc-run/u-ai506/share/cycle/20150629T1800Z/engl_etkf/perts"
    ENS_ANAL_FILE = "/home/d03/freb/cylc-run/u-ai506/share/cycle/20150629T1800Z/engl_atmanl"
    ENS_SNOW_FILE = "/home/d03/freb/cylc-run/u-ai506/share/cycle/20150629T1800Z/engl_um_000/englaa_da006"
    NUM_PERT_MEMBERS = "44"
    NUM_PERT_MEMBERS = "3"
    WRITE_DIAG_FILES = False
    OVERWRITE_PERT_FILES = False

# conversions to type:
NUM_PERT_MEMBERS = int(NUM_PERT_MEMBERS)
MEMBERS_PERT_INTS = range(1, NUM_PERT_MEMBERS+1)

print 'ROSE_DATACPT6H = "{}"'.format(ROSE_DATACPT6H)
print 'ENS_SMC_DIR = "{}"'.format(ENS_SMC_DIR)
print 'ENS_PERTS_DIR = "{}"'.format(ENS_PERTS_DIR)
print 'NUM_PERT_MEMBERS = "{}"'.format(NUM_PERT_MEMBERS)

## Config:
# STASH codes to use:
STASH_LAND_SEA_MASK = 30
STASH_SMC = 9
STASH_TSOIL = 20
STASH_SNOW_AMNT = 23
# volumetric soil moisture contents at wilting, critical and saturation:
STASH_VSMC_WILT = 40
STASH_VSMC_CRIT = 41
STASH_VSMC_SAT = 43
# land use fractions:
STASH_LANDFRAC = 216
# pseudo level of land ice tile (this should remain unchanged at 9!)
PSEUDO_LEVEL_LANDICE = 9

# STASH codes to load and mean:
STASH_TO_LOAD = [STASH_SMC, STASH_TSOIL, STASH_LAND_SEA_MASK]
# these need to be all multi-level (not pseudo level) stash codes
MULTI_LEVEL_STASH = [STASH_SMC, STASH_TSOIL]
# a list of stash codes we want to actually act on to produce perturbations in this routine:
STASH_TO_MAKE_PERTS = [STASH_SMC, STASH_TSOIL]


# The minimum allowed soil moisture content, after perturbation, uses
# the SMC at wilting point, scaled by this factor:
WILTING_POINT_SCALING_FACTOR = 0.1

# snow amount (STASH_SNOW_AMNT, kg/m2) above which the smc perturbations are set to zero:
SNOW_AMOUNT_THRESH = 0.05

# layer depth of each soil layer, to be set once input files are read in:
DZ_SOIL_LEVELS = {}
# density of pure water (kg/m3)
RHO_WATER = 1000.0



def mean_prev_ens_data(in_data, in_files):
    'produces the mean of the required fields, input as in_data'
    TIMER.start('Meaning fields')

    mean_data = {}

    # define the operators we need to use:
    # add the fields up:
    adder = mule.operators.AddFieldsOperator(preserve_mdi=True)
    # dived them by the number:
    divver = mule.operators.ScaleFactorOperator(1.0 / NUM_PERT_MEMBERS)
    # and save the result in memory rather than recalculating it each time:
    cahceoperator = CachingOperator()

    #for stash in STASH_TO_MAKE_PERTS:
    stash = STASH_SMC
    #if stash in MULTI_LEVEL_STASH:
    mean_data = {}
    for level in in_data[stash]:
        # add them up:
        sum_field = adder(in_data[stash][level])
        # now divied:
        mean_field = divver(sum_field)
        # and store that in the output strcuture, as cached data:
        # mean_data[stash][level] = cahceoperator(mean_field)
        mean_data[level] = cahceoperator(mean_field)
        # now use get_data to actually do the meaning at this point
        # and cache the result. This means the timer is accurate.
        _ = mean_data[level].get_data() #??
    else:
        # sum the fields:
        sum_field = adder(in_data[stash])
        # now divided:
        mean_field = divver(sum_field)
        # and store that in the output strcuture, as cached data:
        mean_data[stash] = cahceoperator(mean_field)
        # now use get_data to actually do the meaning at this point
        # and cache the result  This means the timer is accurate.
        _ = mean_data[stash].get_data()
    TIMER.end('Meaning fields')

    if WRITE_DIAG_FILES:
        mean_ff = in_files[0].copy(include_fields=False)
        for stash in STASH_TO_MAKE_PERTS:
            if stash in MULTI_LEVEL_STASH:
                for level in in_data[stash]:
                    mean_ff.fields.append(mean_data[stash][level])
            else:
                mean_ff.fields.append(mean_data[stash])

        # now add the land/sea mask:
        mean_ff.fields.append(in_data[STASH_LAND_SEA_MASK][0])

        mean_file = '%s/mean.ff' % ENS_SMC_DIR
        print 'writing out diagnostic mean file "%s"' % mean_file
        TIMER.start('writing diagnostic mean smc file')
        mean_ff.to_file(mean_file)
        TIMER.end('writing diagnostic mean smc file')

    return mean_data
