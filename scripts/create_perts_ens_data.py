#!/usr/bin/env/python

"""
Calculate perturbations of soil moisture content (SMC) and soil temperature (TSOIL) from all ensemble members,
using the breeding method. Data used is for t+3 of the current cycle and saves the output for each member
separately in a 'bpert' file. Output gets used next cycle, then valid for t-3.

Perturbations calculated for each soil level as: each member minus the ensemble mean

Created by Elliott Warren Wed 20th Nov 2019: elliott.warren@metoffice.gov.uk
Based on engl_ens_smc_pert.py by Malcolm Brooks 18th Sept 2016: Malcolm.E.Brooks@metoffice.gov.uk

Tested versions using canned data:
python 2.7.16
python 3.7.5
mule 2019.01.1
numpy 1.16.5 (python2)
numpy 1.17.3 (python3)

Testing carried out in:
/data/users/ewarren/R2O_projects/soil_moisture_pertubation/

Note: Global variables named in all caps e.g. NUM_PERT_MEMBERS
"""

import os
import numpy as np

import mule
import mule.operators

# Environment variables:
# number of ensemble members to do perturbations for
NUM_PERT_MEMBERS = os.getenv('NUM_PERT_MEMBERS')

# cycle directory
ROSE_DATAC = os.getenv('ROSE_DATAC')

# directory the ensemble perturbation files will be saved in
# named [engl_smc_bpert] to differentiate the perturbations as being produced from the breeding method
ENS_PERT_DIR = os.getenv('ENS_PERT_DIR')

# filename of dump with soil variables in that will be used for members (not the full filepath)
ENS_SOIL_DUMP_FILE = os.getenv('ENS_SOIL_DUMP_FILE')

# Diagnostics - True = save ensemble mean for analysis
# Set False for routine runs
DIAGNOSTICS = False

if NUM_PERT_MEMBERS is None:
    os.system('echo script being ran in development mode!')
    # if not set, then this is being run for development, so have canned variable settings to hand:
    NUM_PERT_MEMBERS = '3'
    ROSE_DATAC = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20181201T0600Z'
    ENS_PERT_DIR = ROSE_DATAC + '/engl_smc/engl_smc_bpert'
    ENS_SOIL_DUMP_FILE = 'englaa_da003'  # Breo's change
    # ENS_SOIL_DUMP_FILE = 'englaa_da006'  # before Breo's change
    DIAGNOSTICS = True

# conversions to type:
NUM_PERT_MEMBERS = int(NUM_PERT_MEMBERS)
MEMBERS_PERT_INTS = range(1, NUM_PERT_MEMBERS+1)

# ------------------------------------

# Configuration:
# STASH codes to use:
STASH_LAND_SEA_MASK = 30
STASH_TSOIL = 20
STASH_SMC = 9
STASH_LANDFRAC = 216
# pseudo level for land-ice mask
PSEUDO_LEVEL_LANDICE = 9

# STASH codes to load and mean:
STASH_TO_LOAD = [STASH_SMC, STASH_TSOIL, STASH_LAND_SEA_MASK, STASH_LANDFRAC]

# these need to be all multi-level (not pseudo level) stash codes
MULTI_LEVEL_STASH = [STASH_SMC, STASH_TSOIL, STASH_LANDFRAC]

# a list of stash codes we want to actually act on to produce perturbations in this routine:
STASH_TO_MAKE_PERTS = [STASH_SMC, STASH_TSOIL]

# constraints on which fields to load in for a STASH variable
STASH_LEVEL_CONSTRAINTS = {STASH_LANDFRAC: [PSEUDO_LEVEL_LANDICE]}

# layer depth of each soil layer, to be set once input files are read in:
DZ_SOIL_LEVELS = {}


# ------------------------------------------


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

    """
    Converts an ensemble member input number into the filename string used to locate files
    :param member: (int or char)
    :return: mem_str: (str) 3 char string with leading zeros e.g. 001
    """

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


# filename creation functions


def engl_cycle_tplus3_dump(member):

    """ locates T+6 hour start dump from the current cycle for an ensemble member"""
    return '{0}/engl_um/engl_um_{1}/{2}'.format(ROSE_DATAC, mem_to_str(member), ENS_SOIL_DUMP_FILE)


def engl_cycle_bpert_filename(member):

    """ create filename for the breeding pertubation to be saved under, for an ensemble member"""
    # extra 0 before the member number to be consistent with existing engl_smc pertubation files
    return '{0}/engl_smc/engl_smc_bpert/engl_smc_bpert_{1}'.format(ROSE_DATAC, mem_to_str(member))


# loading functions


def load_engl_member_data():

    """
    Loads in input data from ensemble run (all members). This is from a large number of files, and sorted into a dict
    structure by field, level to produce a list containing the data for ensemble members.

    :return: ens_data (dictionary, keys = STASH, level; items = member fields], e.g.{9: {1: [field1a, field2a, field3a],
        2: [field1b, field2b ...]}} where 9 = stash code, 1 and 2 are levels. The list of fields within are from all the
        different members and have length n where n = maximum number of members): all ensemble field data for STASH
        variables.
    :return: ens_ff_files (list): field files for all members

    Extra help:
    field.lblev = level code (e.g. 1 to 70. 3 special entries for special fields e.g. 7777 or 8888 (see UM F03 docs.)
        soil fields use lblev as their level. Land fraction has special 4 digit value here instead.
    field.lbuser4 = stash code
    field.lbuser5 = pseudo level (land surface tiles will have these)
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

        # file with SMC and TSOIL data (valid for t+3 for this cycle and t-3 next cycle)
        soil_src_file = engl_cycle_tplus3_dump(member)
        # data file to open:
        ff_file_in = mule.DumpFile.from_file(soil_src_file)
        ff_file_in.remove_empty_lookups()

        # pull out the fields:
        for field in ff_file_in.fields:
            # is this the SMC or TSOIL?
            if field.lbuser4 in STASH_TO_LOAD:
                if field.lbuser4 in MULTI_LEVEL_STASH:

                    # 1. special instance for landfrac
                    if field.lbuser4 == STASH_LANDFRAC:
                        if field.lbuser5 in STASH_LEVEL_CONSTRAINTS[STASH_LANDFRAC]:
                            # Read in landfrac pseudo-level...
                            if field.lbuser5 in ens_data[field.lbuser4].keys():
                                ens_data[field.lbuser4][field.lbuser5].append(field)
                            else:
                                ens_data[field.lbuser4][field.lbuser5] = [field]

                    else:
                        # 2. Read in multi-level stash...
                        if field.lblev in ens_data[field.lbuser4].keys():
                            ens_data[field.lbuser4][field.lblev].append(field)
                        else:
                            ens_data[field.lbuser4][field.lblev] = [field]

                else:
                    # 3. single level fields are a flat list:
                    ens_data[field.lbuser4].append(field)

        # keep the fieldsfile objects for reference:
        ens_ff_files.append(ff_file_in)

    # verify that all the data is included:
    for stash in STASH_TO_LOAD:
        if stash in MULTI_LEVEL_STASH:
            for level in ens_data[stash]:
                n_mems = len(ens_data[stash][level])
                if n_mems != NUM_PERT_MEMBERS:
                    raise ValueError('{} fields found in ensemble for STASH code {}, level {}, '
                                     'but NUM_PERT_MEMBERS ({}) is expected'.format(
                                     n_mems, stash, level, NUM_PERT_MEMBERS))
        else:
            if len(ens_data[stash]) != NUM_PERT_MEMBERS:
                raise ValueError('Not all data found!')

    return ens_data, ens_ff_files

# processing


def mean_ens_data(in_data):

    """
    Create ensemble mean of the soil moisture content field

    :param in_data: (dictionary, keys = STASH, level; items = fields]): all ensemble data in. See [ens_data] in
        load_engl_soil_member_data()
    :return: mean_data: (dictionary, keys = STASH, level, item = field): mean of in_data by level
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
            # and store that in the output structure, as cached data:
            mean_data[stash] = cahceoperator(mean_field)
            # now use get_data to actually do the meaning at this point
            # and cache the result  This means the timer is accurate.
            _ = mean_data[stash].get_data()

    return mean_data


def in_data_minus_mean(in_data, mean_data):

    """
    Subtracts the mean of a set of fields from the input data, to produce a perturbation

    :param in_data : (dictionary, keys = STASH, level; items = fields]): all ensemble data in. See [ens_data] in
    :param mean_data: (dictionary): ensemble mean
    :return: pert_data: (dictionary): ensemble perturbations
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


def zero_land_ice_perts(pert_data, ens_data):

    """

    :param ens_perts:
    :return:
    """

    for member in MEMBERS_PERT_INTS:

        # load in the land-ice mask for this member (member-1 because index 0 of ens_data[...][...] relates to member 1)
        landice_data = ens_data[STASH_LANDFRAC][PSEUDO_LEVEL_LANDICE][member-1].get_data()

        # create land-ice mask
        landice_mask = np.logical_and(landice_data == 1.0,
                                      landice_data != ens_data[STASH_LANDFRAC][PSEUDO_LEVEL_LANDICE][member-1].bmdi)

        for stash in STASH_TO_MAKE_PERTS:
            if stash in MULTI_LEVEL_STASH:
                for level in pert_data[stash]:

                    # extract out this member's pert data field
                    pert_data_i = pert_data[stash][level][member-1]


                    # extract data and set values to 0.0 where there is ice
                    tmp_pert_data = pert_data_i.get_data()
                    tmp_pert_data[landice_mask] = 0.0

                    # now put that data back into the pert_data_i field:
                    array_provider = mule.ArrayDataProvider(tmp_pert_data)
                    pert_data_i.set_data_provider(array_provider)

                    # and explicitly put that field back into the ens pert dictionary encase it deep copied the variable
                    pert_data[stash][level][member-1] = pert_data_i

    return pert_data

# saving functions


def save_ens_mean(ens_data, ens_mean_data, in_files):

    """
    Additional function to save the ensemble mean calculated from mean_ens_data().

    :param ens_data: (dictionary): fields from all ensemble members (see [ens_data] in load_engl_smc_member_data)
    :param ens_mean_data: (dictionary): ensemble mean for each variable and level
    :param in_files: (list): list of file files for all member's data
    :return:
    """

    # Ensure smc directory exists to save into
    if os.path.isdir(ROSE_DATAC + '/engl_smc') is False:
        # use mkdir -p (recursive dir creation) as pythonic equivalent functions variy between Python2 and Python3
        os.system('mkdir -p '+ROSE_DATAC + '/engl_smc')

    ens_mean_filepath = ROSE_DATAC + '/engl_smc/smc_ens_mean'
    # open a FieldFile object:
    ens_mean_ff = in_files[0].copy(include_fields=False)

    # add the fields:
    for stash in STASH_TO_MAKE_PERTS:
        if stash in MULTI_LEVEL_STASH:
            for level in ens_mean_data[stash]:
                ens_mean_ff.fields.append(ens_mean_data[stash][level])
        else:
            ens_mean_ff.fields.append(STASH_TO_MAKE_PERTS[stash])

    # add land sea mask
    ens_mean_ff.fields.append(ens_data[STASH_LAND_SEA_MASK][0])

    # save
    ens_mean_ff.to_file(ens_mean_filepath)

    print('saved: '+ens_mean_filepath)

    return


def save_ens_perts(in_pert_data, in_files):

    """
    Save ensemble perturbations for each member in a different file

    :param in_pert_data: (dictionary): perturbation data
    :param in_files: (list): list of field files for all members
    :return:
    """

    # create directory in engl_smc to save the perturbations
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

    # load the SMC, TSOIL and other input data for all members:
    ens_data, ens_ff_files = load_engl_member_data()

    # mean the required fields
    ens_mean = mean_ens_data(ens_data)

    # create perturbations pert_i = (member_i - mean(member_k)),
    # where member_k is a vector and k = 1 ... n maximum members
    ens_perts = in_data_minus_mean(ens_data, ens_mean)

    # set pert values of SMC and TSOIL to 0 where ice is present on land
    ens_perts = zero_land_ice_perts(ens_perts, ens_data)

    # save the ensemble mean used in making perturbations (mean(all_members_of_same_cycle))
    if DIAGNOSTICS is True:
        save_ens_mean(ens_data, ens_mean, ens_ff_files)

    # save ensemble perturbations (pert_i)
    save_ens_perts(ens_perts, ens_ff_files)

    exit(0)