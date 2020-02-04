#!/usr/bin/env python

"""
Calculate the correction (perturbation) needed to recenter the mean of the ensemble, toward the ensemble control member.
I.e. ensemble mean = 28, and control member = 32, then recentering correction to apply on all members on the next
cycle is +4. Corrections applied to soil moisture content (SMC). Data used is for t+3 of
the current cycle. Output then gets used next cycle, and is valid for t-3.

This script is doing the recentering step of the 'breeding method'.

A boolean of the combined snow field is needed for the next cycle,  in order to determine where the EKF IAU increments
 need to be set to 0.

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

# directory the correction file will be saved in
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
    ENS_PERT_DIR = ROSE_DATAC + '/engl_smc'
    ENS_SOIL_DUMP_FILE = 'englaa_da003'
    DIAGNOSTICS = True

# control member number (can be 0 or 000)
CONTROL_MEMBER = 0

# conversions to type:
NUM_PERT_MEMBERS = int(NUM_PERT_MEMBERS)
MEMBERS_PERT_INTS = range(1, NUM_PERT_MEMBERS+1)

# ------------------------------------

# Configuration:
# STASH codes to use:
STASH_LAND_SEA_MASK = 30
STASH_SMC = 9
STASH_LANDFRAC = 216
STASH_SNOW_AMNT = 23  # can be overwritten by other programs therefore not the ideal choice for use as a snow mask
STASH_SNOW_ANY_LAYER = 380  # preferred alternative to STASH_SNOW_AMNT that does not get overwritten
# pseudo level for land-ice mask
PSEUDO_LEVEL_LANDICE = 9


# STASH codes to load and mean:
STASH_TO_LOAD = [STASH_SMC, STASH_LAND_SEA_MASK, STASH_SNOW_ANY_LAYER, STASH_LANDFRAC]

# these need to be all multi-level (not pseudo level) stash codes
MULTI_LEVEL_STASH = [STASH_SMC, STASH_LANDFRAC]

# a list of stash codes we want to actually act on to produce perturbations in this routine:
STASH_TO_MAKE_PERTS = [STASH_SMC]

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


def engl_cycle_dump(member):

    """ locates hour start dump from the current cycle for an ensemble member (e.g. t+3)"""
    return '{0}/engl_um/engl_um_{1}/{2}'.format(ROSE_DATAC, mem_to_str(member), ENS_SOIL_DUMP_FILE)


# loading functions


def load_engl_member_data(member_list):

    """
    Loads in input data from ensemble run members. This can be from a large number of files, and sorted into a dict
    structure by field, level to produce a list containing the data for ensemble members. Can read in data from a single
    member (e.g. control)

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

    # ensure member_list is actually an iterable, if not already.
    if isinstance(member_list, (float, int, str)):
        member_list = [member_list]

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
    for member in member_list:

        # file with SMC and TSOIL data (valid for t+3 for this cycle and t-3 next cycle)
        soil_src_file = engl_cycle_dump(member)
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
                if n_mems != len(member_list):
                    raise ValueError('{} fields found in ensemble for STASH code {}, level {}, '
                                     'but NUM_PERT_MEMBERS ({}) is expected'.format(
                                     n_mems, stash, level, len(member_list)))
        else:
            if len(ens_data[stash]) != len(member_list):
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

    # add the land-sea mask
    mean_data[STASH_LAND_SEA_MASK] = in_data[STASH_LAND_SEA_MASK][0]

    # add the land-use masks
    mean_data[STASH_LANDFRAC] = {key: item[0] for key, item in in_data[STASH_LANDFRAC].items()}

    return mean_data


def mean_minus_control(control_data, mean_data):

    """
    Subtracts the control field from the ensemble mean, for all required fields, to produce a correction. This
    correction can be used in the next cycle to adjust each of the ensemble member's starting points, effectively
    'centering' the ensembles analyses around the control.

    :param control_data : (dictionary, keys = STASH, level; items = field]) same hierarchy as mean_data. Fields will
                be within separate lists with only one entry. Therefore fields need to be indexed from
                the list. i.e. [field], therefore control_data[stash][level]][0] to get the field.
    :param mean_data: (dictionary): ensemble mean
    :return: corrction_data: (dictionary): ensemble perturbations
    """

    subber = mule.operators.SubtractFieldsOperator(preserve_mdi=True)

    corrction_data = {}
    for stash in STASH_TO_MAKE_PERTS:
        if stash in MULTI_LEVEL_STASH:
            corrction_data[stash] = {}
            for level in control_data[stash]:

                # ensemble mean field - control field
                corrction_data[stash][level] = subber([mean_data[stash][level], control_data[stash][level][0]])

        else:
            # ensemble mean field - control field
            corrction_data[stash] = subber([mean_data[stash], control_data[stash][0]])

    # now add the land sea mask from the first member as well.
    # Take the land-sea mask out of the single element lists
    corrction_data[STASH_LAND_SEA_MASK] = control_data[STASH_LAND_SEA_MASK][0]

    #corrction_data[STASH_LANDFRAC] = control_data[STASH_LANDFRAC]
    # add the land use as well (taken the items out of a single element list
    corrction_data[STASH_LANDFRAC] = {key: item[0] for key, item in control_data[STASH_LANDFRAC].items()}

    # add the snow cover from the control as well.
    # Intended only to be used as a field's file template for storing the combined ensemble snow field boolean later

    return corrction_data


def zero_land_ice_snow_perts(corr_data, ens, ctrl):

    """
    Set perturbations to be 0.0 over ice or where snow was present (>0.05 kg m-2) in any of the members, including the
    control. The mule subtraction can introduce small non-zero values over ice where values were very close. This sets
    the perturbations to be 0.0 over ice, regardless.
    :param corr_data: ensemble mean - control, correction data.
    :param ens: ensemble data (contains the snow fields from all members except the control)
    :param ctrl: control ensemble member (contains control member's snow field)
    :return corr_data
    :return comb_snow_field: snow field with a boolean defining whether snow (>0.05 kg m-2), is present on any grid cell
        for any member
    """

    def apply_mask(field, mask):

        """
        apply the mask to the data
        :param field: (field) field to partially mask
        :param mask: (numpy array with boolean values, same shape as field.get_data()) True for where to mask
        :return:
        """

        # extract data and set values to 0.0 where there is ice
        tmp_pert_data = field.get_data()
        tmp_pert_data[mask] = 0.0

        # now put that data back into the corr_data field:
        array_provider = mule.ArrayDataProvider(tmp_pert_data)
        field.set_data_provider(array_provider)

        return field

    # 1. Get land-ice mask
    # load in the land-ice mask
    landice_data = corr_data[STASH_LANDFRAC][PSEUDO_LEVEL_LANDICE].get_data()
    # create land-ice mask
    landice_mask = np.logical_and(landice_data == 1.0,
                                  landice_data != corr_data[STASH_LANDFRAC][PSEUDO_LEVEL_LANDICE].bmdi)

    # 2. Get snow mask (snow from any member is > 0.05 kg m-2)
    # append two lists of snow fields (all members and control member)
    snow_fields = ens[STASH_SNOW_ANY_LAYER] + ctrl[STASH_SNOW_ANY_LAYER]
    # Create a list of booleon arrays (one for each member), where the field has more than 0.05 kg m-2 snow and does
    # not have the 'missing data' number.
    snow_masks = [np.logical_and(snow_field_i.get_data() > 0.05, snow_field_i.get_data() != snow_field_i.bmdi) for snow_field_i in snow_fields]
    # stack the arrays together along a third dimension to enable the next step
    stacked_snow_masks = np.stack(snow_masks, axis=2)
    # Create a 2d booleon array showing whether any of the ensemble members had more than 0.05 snow
    comb_snow_mask = np.any(stacked_snow_masks, axis=2)

    # 3. combine land-ice and snow mask into one (True if permafrost, or enough snow from any member, is present)
    ice_snow_mask = np.logical_or(landice_mask, comb_snow_mask)

    # 4. set perturbations to 0.0 where ice or enough snow is present
    for stash in STASH_TO_MAKE_PERTS:
        if stash in MULTI_LEVEL_STASH:
            for level in corr_data[stash]:
                corr_data[stash][level] = apply_mask(corr_data[stash][level], ice_snow_mask)
        else:
            corr_data[stash] = apply_mask(corr_data[stash], ice_snow_mask)

    # 5. Add the snow and ice field to the corr_data, for saving. Done by taking a copy of the control snow field,
    #     replacing the data, and then adding the field to the correction dictionary
    comb_snow_field = ctrl[STASH_SNOW_ANY_LAYER][0]
    snow_array_provider = mule.ArrayDataProvider(comb_snow_mask)
    comb_snow_field.set_data_provider(snow_array_provider)
    corr_data[STASH_SNOW_ANY_LAYER] = comb_snow_field

    return corr_data

# saving functions


def save_fields_file(data, in_files, filename):

    """
    Additional function to save fields calculated. Intended for saving the ensemble mean, and the correction fields
    separately.

    :param data: (dictionary): fields needing to be saved
    :param in_files: (list): list of file files for all member's used in creating [data]. Used to create a template.
    :param stash_list (list): list of stash numbers to save e.g. [9] for soil moisture
    :param filename (str): name of the file to save (not the full path)
    :return:
    """

    # Ensure smc directory exists to save into
    if os.path.isdir(ENS_PERT_DIR) is False:
        # use mkdir -p (recursive dir creation) as equivalent pythonic functions vary between Python2 and Python3
        os.system('mkdir -p '+ENS_PERT_DIR)

    ens_filepath = ENS_PERT_DIR+'/'+filename
    # open a FieldFile object:
    ens_ff = in_files[0].copy(include_fields=False)

    # add the fields:
    for stash in data.keys():
        if stash in MULTI_LEVEL_STASH:
            for level in data[stash]:
                ens_ff.fields.append(data[stash][level])
        else:
            ens_ff.fields.append(data[stash])

    # # add land sea mask if it wasn't in the stash_list
    # if STASH_LAND_SEA_MASK not in stash_list:
    #     if STASH_LAND_SEA_MASK in data.keys():
    #         ens_ff.fields.append(data[STASH_LAND_SEA_MASK])
    #     else:
    #         raise ValueError('land sea mask not present in data. It needs to be included in saved field files!')

    # save
    ens_ff.to_file(ens_filepath)

    print('saved: '+ens_filepath)

    return

def save_ice_snow_fields(comb_snow_mask, corr_data, ctrl_file):


    # replace data in the control field with the combined snow boolean field (snow > 0.05 = True for any member),
    #   so it can be saved and used for the next cycle
    comb_snow_field = corr_data[STASH_SNOW_ANY_LAYER]
    snow_array_provider = mule.ArrayDataProvider(comb_snow_mask)
    comb_snow_field.set_data_provider(snow_array_provider)

    # snow field in a dictionary and filename, ready for save_fields_file() function
    snow_ice_field_dict = {STASH_SNOW_ANY_LAYER: [comb_snow_field]}
    # add any land-use fields (including ice)
    snow_ice_field_dict.update({STASH_LANDFRAC: corr_data[STASH_LANDFRAC]})
    #{stash: field_i[0] for (stash, field_i) in ctrl_data[STASH_LANDFRAC].items()}
    # add land-sea mask, as it is needed for any saved fields file
    snow_ice_field_dict.update({STASH_LAND_SEA_MASK: corr_data[STASH_LAND_SEA_MASK]})

    filename = 'engl_snow_ice_masks'
    stash_list = snow_ice_field_dict.keys()

    # restructure the dictionary, so the fields are not in single element lists. This is needed for the
    # save_fields_file() to work properly
    snow_ice_field_dict

    # save the snow fields file.
    # It will save the ice fields file too
    save_fields_file(snow_ice_field_dict, ctrl_file, stash_list, filename)

    return


if __name__ == '__main__':

    """
    Routine 1 of 2 for creating the ensemble soil moisture content (SMC) and soil temperature (TSOIL) perturbation 
    correction.
    
    1) Loads in data
    2) Produces the mean of the input fields
    3) Produces the correction fields (ensemble mean - control field)
    4) Checks correction fields
    5) Save ensemble mean
    6) Saves the correction
    """

    ## Read
    # load the SMC and other input data for all members:
    ens_data, ens_ff_files = load_engl_member_data(MEMBERS_PERT_INTS)

    # load the SMC and other input data for the control member only:
    ctrl_data, ctrl_ff_files = load_engl_member_data(CONTROL_MEMBER)

    ## Process
    # create the mean for all required ensemble member fields
    ens_mean = mean_ens_data(ens_data)

    # create the soil recentering correction (ensemble mean - control).
    ens_correction = mean_minus_control(ctrl_data, ens_mean)

    # set pert values of SMC to 0 where ice or snow is present on land, in any member (including control)
    # the snow field used is appended to the ens_correction dictionary for use in the next cycle.
    ens_correction = zero_land_ice_snow_perts(ens_correction, ens_data, ctrl_data)

    ## Save
    # save the ensemble mean used in making perturbations (mean(all_members_of_same_cycle))
    # includes the composite boolean snow field map produced in zero_land_ice_snow_perts()
    save_fields_file(ens_mean, ens_ff_files, 'engl_soil_mean')

    # save the correction
    # use the control ensemble file as a file template for saving
    save_fields_file(ens_correction, ctrl_ff_files, 'engl_soil_correction')

    exit(0)