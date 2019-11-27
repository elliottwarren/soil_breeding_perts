#!/usr/bin/env python2

"""
Calculate ensemble mean of soil moisture content (SMC) and soil temperature (TSOIL) from all ensemble members, and
save the output in a 'bpert' file.

Created by Elliott Warren Wed 20th Nov 2019: elliott.warren@metoffice.gov.uk
Based on engl_ens_smc_pert.py by Malcolm Brooks 18th Sept 2016: Malcolm.E.Brooks@metoffice.gov.uk

Tested versions:
python 2.7.16
mule 2019.01.1
numpy 1.16.5

Testing (including soil temperature) carried out in:
/data/users/ewarren/R2O_projects/soil_moisture_pertubation/
"""

import os, pdb, datetime, shutil
import numpy as np

import mule
import mule.operators
from mule import FieldsFile

# Environement variables:
# number of ensemble members to do perturbations for
NUM_PERT_MEMBERS = os.getenv('NUM_PERT_MEMBERS')

# current cycle (style = 20190831T1200Z)
THIS_CYCLE = os.getenv('THIS_CYCLE')

# directory the ensemble mean file will be saved in
ENS_MEAN_DIR = os.getenv('ENS_MEAN_DIR')

# directory the ensemble pertubation files will be saved in
# named [engl_smc_bpert] to differentiate the pertubations as being produced from the breeding method
ENS_PERT_DIR = os.getenv('ENS_PERT_DIR')


if NUM_PERT_MEMBERS is None:
    # if not set, then this is being run for development, so have canned variable settings to hand:
    NUM_PERT_MEMBERS = '3'
    THIS_CYCLE = '20181201T0600Z'
    SUITE_DIR = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/' + THIS_CYCLE
    ENS_MEAN_DIR = SUITE_DIR + '/engl_smc'
    ENS_PERT_DIR = SUITE_DIR + '/engl_smc/engl_smc_bpert'


# conversions to type:
NUM_PERT_MEMBERS = int(NUM_PERT_MEMBERS)
MEMBERS_PERT_INTS = range(1, NUM_PERT_MEMBERS+1)

## Config:
# STASH codes to use:
STASH_LAND_SEA_MASK = 30
STASH_TSOIL = 20
STASH_SMC = 9

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

class CachingOperator(mule.DataOperator):
    """
    Operator which caches the data returned by another
    field object (it does no calculations of its own.

    In a situation where the data payload of a specific
    field object is to be referenced many times,
    wrapping it in this operator will save the result
    in memory rather than re-calculating it.

    Note that this means any updates to the source
    objects referenced by this operator *after* it has
    been invoked will not have any effect.
    """
    def __init__(self):
        pass
    def new_field(self, src_field):
        """A simple copy of the source field."""
        return src_field.copy()
    def transform(self, src_field, new_field):
        """
        The data returned by the source field is saved
        as the "_data" attribute of the new field. On
        any subsequent calls the data is referenced
        directly from the "_data" attribute.
        """
        if not hasattr(src_field, "_data"):
            src_field._data = src_field.get_data()
        return src_field._data

def mem_to_str(member):
    'converts an ensemble member intput number into the filename string used to locate files'
    fail_str = 'Cannot convert ensemble member input to filename string: {0}'.format(member)
    if isinstance(member, str):
        if member.isdigit():
            mem_str = '{0:03d}'.format(int(member))
        else:
            raise ValueError(fail_str)
    elif isinstance(member, int):
        mem_str = '{0:03d}'.format(member)
    else:
        raise ValueError(fail_str)
    return mem_str

def engl_cycle_tplus6_dump(member):

    """ locates T+6 hour start dump from the current cycle for an ensemble member"""
    # ToDo set the filename (da006) to be an environment variable passed down to python script
    return '{0}/engl_um/engl_um_{1}/englaa_da006'.format(SUITE_DIR, mem_to_str(member))

def engl_cycle_bpert_filename(member):

    """ create filename for the breeding pertubation to be saved under, for an ensemble member"""
    # extra 0 before the member number to be consistent with existing engl_smc pertubation files
    return '{0}/engl_smc/engl_smc_bpert/engl_smc_bpert_0{1}'.format(SUITE_DIR, mem_to_str(member))

def load_engl_soil_member_data():

    """

    Loads in input data from ensemble run (all members). This is from a large number of files, and sorted into a dict
    strucure by field, level to produce a list containing the data for ensemble members.

    :return: ens_data (dictionary, keys = STASH, level; items = fields], e.g. {9: {1: [field1a, field2a, field3a],
        2: [field1b, field2b ...]}} where 9 = stash code, 1 and 2 are levels. The list of fields within are from all the
        different members and have length n where n = maximum number of members): all ensemble field data for STASH
        variables.
    :return: ens_ff_files (list): field files for all members

    """

    # set up the returned data structure...
    ens_data = {}
    for stash in STASH_TO_LOAD:
        if stash in MULTI_LEVEL_STASH:
            ens_data[stash] = {}
        else:
            ens_data[stash] = []

    # and the fieldsfile objects, which we can use as templates later on:
    ens_ff_files = []

    # load in the required data for each file:
    for member in MEMBERS_PERT_INTS:

        # file with SMC and TSOIL data (after t+3)
        soil_src_file = engl_cycle_tplus6_dump(member)
        # data file to open:
        ff_file_in = mule.DumpFile.from_file(soil_src_file)
        ff_file_in.remove_empty_lookups()

        # pull out the fields:
        for field in ff_file_in.fields:
            # is this the SMC?
            if field.lbuser4 in STASH_TO_LOAD:
                if field.lbuser4 in MULTI_LEVEL_STASH:
                    # multi-level fields are a dict, with a list for each level
                    if field.lblev in ens_data[field.lbuser4].keys():
                        ens_data[field.lbuser4][field.lblev].append(field)
                    else:
                        ens_data[field.lbuser4][field.lblev] = [field]
                else:
                    # single level fields are a flat list:
                    ens_data[field.lbuser4].append(field)
        # keep the fieldsfile objects for reference:
        ens_ff_files.append(ff_file_in)

    return ens_data, ens_ff_files

def mean_ens_data(in_data):

    """
    Create ensemble mean of the soil moisture content field

    :param in_data :(dictionary, keys = STASH, level; items = fields]): all ensemble data in. See [ens_data] in
        load_engl_soil_member_data()
    :return:
    """

    # Assumes soil moisture content is provided on multiple levels
    mean_data = {}

    # Define the operators we need to use:
    # add the fields up:
    adder = mule.operators.AddFieldsOperator(preserve_mdi=True)
    # Dived them by the number:
    divver = mule.operators.ScaleFactorOperator(1.0 / NUM_PERT_MEMBERS)
    # And save the result in memory rather than recalculating it each time:
    cahceoperator = CachingOperator()

    for stash in STASH_TO_MAKE_PERTS:
        if stash in MULTI_LEVEL_STASH:
            mean_data[stash] = {}
            for level in in_data[stash]:
                # add them up:
                sum_field = adder(in_data[stash][level])
                # now divide:
                mean_field = divver(sum_field)
                # and store that in the output structure, as cached data:
                mean_data[stash][level] = cahceoperator(mean_field)
                # now use get_data to actually do the meaning at this point
                # and cache the result. This means the timer is accurate.
                _ = mean_data[stash][level].get_data()
        # encase code is expanded to include other vars such as soil surface temperature
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

    return mean_data

def save_ens_mean(ens_data, ens_mean_data, in_files):

    """
    Additional function to save the ensemble mean calculated from mean_ens_data().

    :param ens_data (dictionary): fields from all ensemble members (see [ens_data] in load_engl_smc_member_data)
    :param ens_mean_data:
    :param in_files: 
    :return:
    """

    # Ensure ENS_MEAN_DIR exists
    if os.path.isdir(ENS_MEAN_DIR) == False:
        # use mkdir -p (recursive dir creation) as pythonic equivalent functions variy between Python2 and Python3
        os.system('mkdir -p '+ENS_MEAN_DIR)

    ens_mean_filepath = ENS_MEAN_DIR + '/smc_ens_mean'
    # open a fieldfile object:
    ens_mean_ff = in_files[0].copy(include_fields=False)
    # ens_mean_ff.fixed_length_header.dataset_type = 3
    # add the fields:
    for stash in STASH_TO_MAKE_PERTS:
        if stash in MULTI_LEVEL_STASH:
            for level in ens_mean_data[stash]:
                ens_mean_ff.fields.append(ens_mean_data[stash][level])
        else:
            ens_mean_ff.fields.append(STASH_TO_MAKE_PERTS[stash])

    # add land sea mask - can this be removed somehow?
    ens_mean_ff.fields.append(ens_data[STASH_LAND_SEA_MASK][0])
    # save
    ens_mean_ff.to_file(ens_mean_filepath)

    return

def in_data_minus_mean(in_data, mean_data, in_files):

    """
    Subtracts the mean of a set of fields from the input data, to produce a perturbation

    :param in_data: all ensemble data
    :param mean_data: ensemble mean
    :param in_files: all ensemble files
    :return:
    """

    subber = mule.operators.SubtractFieldsOperator(preserve_mdi=True)

    pert_data = {}
    for stash in STASH_TO_MAKE_PERTS:
        if stash in MULTI_LEVEL_STASH:
            pert_data[stash] = {}
            for level in in_data[stash]:
                # for this field and level, make a new field by looping through the input in_data,
                # and the result is the in data, minus the mean:
                pert_data[stash][level] = []
                for member in in_data[stash][level]:
                    pert_data[stash][level].append(subber([member, mean_data[stash][level]]))
        else:
            pert_data[stash] = []
            for member in in_data[stash]:
                pert_data[stash].append(subber([member, mean_data[stash]]))

    # now add the land sea mask from the first member as well.
    # Set within a list to match the storage style of other STASH fields in the pert_data dictionary
    pert_data[STASH_LAND_SEA_MASK] = [in_data[STASH_LAND_SEA_MASK][0]]

    return pert_data

def save_ens_perts(in_pert_data, in_files):

    """
    Save ensemble perturbations

    :param in_pert_data: perturbation data
    :param in_files:
    :return:
    """

    # create directory in engl_smc to save the pertubations
    # use mkdir -p to avoid errors due to varying python interpreter version
    os.system('mkdir -p ' + ENS_PERT_DIR)

    for member in MEMBERS_PERT_INTS:

        # path to save under
        ens_pert_filepath = engl_cycle_bpert_filename(member)

        # open a fieldfile object:
        ens_bpert_member_ff = in_files[0].copy(include_fields=False)
        # ens_mean_ff.fixed_length_header.dataset_type = 3
        # add the fields:
        for stash in STASH_TO_MAKE_PERTS:
            if stash in MULTI_LEVEL_STASH:
                for level in in_pert_data[stash]:
                    # ens_bpert_member_ff.fields.append(in_pert_data[stash][level])
                    # member - 1 needed for indexing. E.g. member 1 index = 0
                    ens_bpert_member_ff.fields.append(in_pert_data[stash][level][member-1])
            else:
                ens_bpert_member_ff.fields.append(STASH_TO_MAKE_PERTS[stash][member-1])

        # add land sea mask
        ens_bpert_member_ff.fields.append(ens_data[STASH_LAND_SEA_MASK][0])
        # save
        ens_bpert_member_ff.to_file(ens_pert_filepath)

    return



if __name__ == '__main__':

    """
    Top level soil moisture content (SMC) and soil temperature (TSOIL) perturbation routine
    1) Loads in data
    2) Produces the mean of the input fields
    2a (added option) save ensemble mean
    3) Save the ensemble perturbations
    """
    # load the smc and tsoil input data for all members:
    ens_data, ens_ff_files = load_engl_soil_member_data()

    # mean the required fields
    ens_mean = mean_ens_data(ens_data)

    # create pertubations pert_i = (member_i - mean(member_k)),
    # where member_k is a vector and k = 1 ... n maximum members
    ens_perts = in_data_minus_mean(ens_data, ens_mean, ens_ff_files)

    # save the ensemble mean used in making pertubations (mean(all_members_of_same_cycle))
    # save_ens_mean(ens_data, prev_ens_mean, prev_ens_ff_files)

    # save ensemble pertubations (pert_i)
    save_ens_perts(ens_perts, ens_ff_files)





