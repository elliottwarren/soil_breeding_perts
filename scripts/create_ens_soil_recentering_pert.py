#!/usr/bin/env python

"""
Calculate the correction (perturbation) needed to recenter the mean of the ensemble, toward the ensemble control member.
I.e. ensemble mean = 28, and control member = 32, then recentring correction to apply on all members on the next
cycle is +4. Corrections can be applied to soil moisture content (SMC), soil temperature (TSOIL), snow temperature
(TSNOW) and surface temperature (TSKIN). Data used is for t+3 of the current cycle. Output then gets used next cycle,
and is valid for t-3. This script is doing the recentring step of the 'breeding method'. A boolean of the combined snow
field is needed for the next cycle,  in order to determine where the EKF IAU increments need to be set to 0.

Created by Elliott Warren Wed 20th Nov 2019: elliott.warren@metoffice.gov.uk
Based on engl_ens_smc_pert.py by Malcolm Brooks 18th Sept 2016: Malcolm.E.Brooks@metoffice.gov.uk

Tested versions using canned data:
python 2.7.16
python 3.7.5
mule 2019.01.1
numpy 1.16.5 (python2)
numpy 1.17.3 (python3)


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
DIAGNOSTICS = True

if NUM_PERT_MEMBERS is None:
    os.system('echo script being ran in development mode!')
    # if not set, then this is being run for development, so have canned variable settings to hand:
    NUM_PERT_MEMBERS = '3'
    #ROSE_DATAC = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20181201T0600Z'  # 1 aggregate tile
    ROSE_DATAC = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20190615T0600Z'  # 9 pseudo-tile
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
STASH_TSOIL = 20
STASH_TSKIN = 233  # a.k.a. surface temperature
STASH_TSNOW = 384
STASH_LANDFRAC = 216
STASH_SNOW_AMNT = 23  # can be overwritten by other programs therefore not the ideal choice for use as a snow mask
STASH_NUM_SNOW_LAYERS = 380  # preferred alternative to STASH_SNOW_AMNT that does not get overwritten
# pseudo level for land-ice mask:
PSEUDO_LEVEL_LANDICE = 9


# STASH codes to load.
# Includes single and multi-level codes
# STASH codes in here but not in the multi lists are assumed to be single level fields without pseudo-levels.
STASH_TO_LOAD = [STASH_SMC, STASH_TSOIL, STASH_TSNOW, STASH_TSKIN,
                 STASH_LAND_SEA_MASK, STASH_NUM_SNOW_LAYERS, STASH_LANDFRAC]

# A list of STASH codes we want to act on to produce perturbations in this routine:
STASH_TO_MAKE_PERTS = [STASH_SMC, STASH_TSOIL, STASH_TSNOW, STASH_TSKIN]

# These are the multi soil level STASH codes
MULTI_SOIL_LEVEL_STASH = [STASH_SMC, STASH_TSOIL]

# These are multi pseudo-level stash codes
# If a single aggregate tile is used for these stash codes, the field will simply be read in having a single
# pseudo-level.
MULTI_PSEUDO_LEVEL_STASH = [STASH_LANDFRAC, STASH_TSKIN, STASH_NUM_SNOW_LAYERS, STASH_TSNOW]

# Constraint to only read in a specific tile, e.g. just the ice tile for the land fraction
STASH_LEVEL_CONSTRAINTS = {STASH_LANDFRAC: [PSEUDO_LEVEL_LANDICE]}

# Multiple surface layers
MULTI_LAYER_STASH = [STASH_TSNOW]

# Combined list for multiple soil and pseudo level stash codes to aid with dictionary looping
# Currently having both multiple soil and pseudo levels is not supported
MULTI_LEVEL_STASH = MULTI_SOIL_LEVEL_STASH + MULTI_PSEUDO_LEVEL_STASH
for stash in MULTI_SOIL_LEVEL_STASH:
    if stash in MULTI_PSEUDO_LEVEL_STASH:
        raise EnvironmentError('STASH {} present in MULTI_SOIL_LEVEL_STASH and MULTI_PSEUDO_LEVEL_STASH. Currently '\
                               'processing a STASH present in both is unsupported'.format(stash))

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


# def load_engl_member_data(member_list):
#
#     """
#     Loads in input data from ensemble run members. This can be from a large number of files, and sorted into a dict
#     structure by field, level to produce a list containing the data for ensemble members. Can read in data from a single
#     member (e.g. control)
#
#     :return: ens_data (multi level dictionary =
#         {STASH: {level or surface layer: {pseudo level: [items = [n member fields]}}}).
#         For STASH codes that cannot be on pseudo-levels, that dictionary level is not used.
#         e.g.{9: {1: [field1a, field2a, field3a], 2: [field1b, field2b ...]}} where 9 = stash code, 1 and 2 are model
#         levels. The list of fields within are from all the different members and have length n where n = maximum number
#         of members): all ensemble field data for STASH variables.
#     :return: ens_ff_files (list): field files for all members
#
#     Extra help:
#     field.lblev = level code (e.g. 1 to 70. 3 special entries for special fields e.g. 7777 or 8888 (see UM F03 docs.)
#         soil fields use lblev as their level. Land fraction has special 4 digit value here instead.
#     field.lbuser4 = STASH code
#     field.lbuser5 = pseudo level (land surface tiles will have these). For STASH codes with multiple surface layers and
#         on pseudo-levels, this will be = (pseudo-level * 1000) + surface layer number. e.g. 9002 = pseudo-level 9,
#         surface layer = 2.
#     """
#
#     def check_list_length(stash_list, member_list, **kwargs):
#
#         """
#         Check the length of each fields list to ensure it is equal to the number of ensemble members. If not, throw an
#         error
#         :param stash_list: (list) list of fields
#
#         :keyword: level: (int) information on what stash, level or pseudo level we're on. e.g. stash = 9 or level = 2.
#         """
#
#         # Find the number of members that should be read in (if control then 1, if ensemble then the number of ensemble
#         # members
#         if member_list == CONTROL_MEMBER:  # single member, therefore 1
#             n_mem = 1
#         else:
#             n_mem = len(member_list)
#
#         # Length of the lists read in
#         list_length = len(stash_list)
#
#         # Test whether the read in list length is equal to the number of expected members.
#         if n_mem != list_length:
#
#             # level detail string (which stash, level and/or pseudo-level are we on?)
#             level_detail = '; '.join(['{} = {}'.format(key, item) for key, item in kwargs.items()])
#
#             raise ValueError('{} fields found in ensemble for {}, but number of members expected is {} is expected'.
#                 format(list_length, level_detail, n_mem))
#
#         return
#
#     # ensure member_list is actually an iterable list of strings, if not already.
#     if isinstance(member_list, (float, int, str)):
#         member_list = [member_list]
#
#     # set up the returned data structure...
#     ens_data = {}
#     for stash in STASH_TO_LOAD:
#         if stash in MULTI_LEVEL_STASH or stash in STASH_LEVEL_CONSTRAINTS or stash in MULTI_PSEUDO_LEVEL_STASH\
#                 or stash in MULTI_LAYER_STASH:
#             ens_data[stash] = {}
#         else:
#             ens_data[stash] = []
#
#     # and the fieldsfile objects, which we can use as templates later on:
#     ens_ff_files = []
#
#     # load in the required data for each file:
#     for member in member_list:
#
#         # dump file with the data fields (valid for t+3 for this cycle and t-3 next cycle)
#         soil_src_file = engl_cycle_dump(member)
#         # data file to open:
#         ff_file_in = mule.DumpFile.from_file(soil_src_file)
#         ff_file_in.remove_empty_lookups()
#
#         # pull out the fields:
#         for field in ff_file_in.fields:
#             # is this a stash to load?
#             if field.lbuser4 in STASH_TO_LOAD:
#
#                 # 1. Multiple soil level stash
#                 if field.lbuser4 in MULTI_SOIL_LEVEL_STASH:
#
#                     # 1.1 Also on multiple pseudo levels (tiles)
#                     # Dictionary will be 3 levels deep here:
#                     # lbuser4 = stash; lblev = level; lbuser5 = pseudo-level (tile)
#                     if field.lbuser4 in MULTI_PSEUDO_LEVEL_STASH:
#                         # Read in pseudo-level...
#                         if field.lblev in ens_data[field.lbuser4].keys():  # is level already present
#                             if field.lbuser5 in ens_data[field.lbuser4][field.lblev].keys():  # is pseudo-level present
#                                 ens_data[field.lbuser4][field.lblev][field.lbuser5].append(field)
#                             else:
#                                 ens_data[field.lbuser4][field.lblev][field.lbuser5] = [field]
#                         else:
#                             ens_data[field.lbuser4][field.lblev] = {field.lbuser5: [field]}
#
#                     # 1.2 Multi-level stash without pseudo-levels
#                     # Dictionary is 2 levels deep
#                     else:
#                         if field.lblev in ens_data[field.lbuser4].keys():  # is level already present in dict
#                             ens_data[field.lbuser4][field.lblev].append(field)
#                         else:
#                             ens_data[field.lbuser4][field.lblev] = [field]
#
#                 # 2. Multiple surface layers
#                 elif field.lbuser4 in MULTI_LAYER_STASH:
#
#                     # 2.1 Also on multiple pseudo-levels (currently just TSNOW)
#                     if field.lbuser4 in MULTI_PSEUDO_LEVEL_STASH:
#
#                         # Check if lbuser5 represents surface layer and pseudo-level or just surface layer
#                         # if it is both, lbuser5 will be equal to (pseudo-level * 1000) + surface layer
#                         # else, a single aggregate tile is being used and lbuser5 will equal 0 + surface layer
#
#                         surface_layer = field.lbuser5 % 1000  # % finds the remainder and works even if lbuser <= 1000
#                         if field.lbuser5 > 1000:
#                             pseudo_level = field.lbuser5 // 1000  # // finds the quotient or number of times divided
#                         else:
#                             pseudo_level = 1  # state that a single aggregate tile is being used
#
#                         # Dictionary will be 3 levels deep here:
#                         # STASH; surface-layer; pseudo_level (tile)
#                         # Read in pseudo-level...
#                         if surface_layer in ens_data[field.lbuser4].keys():  # is layer already present
#                             if pseudo_level in ens_data[field.lbuser4][surface_layer].keys():  # is pseudo-level present
#                                 ens_data[field.lbuser4][surface_layer][pseudo_level].append(field)
#                             else:
#                                 ens_data[field.lbuser4][surface_layer][pseudo_level] = [field]
#                         else:
#                             ens_data[field.lbuser4][surface_layer] = {pseudo_level: [field]}
#
#                 # 3. Single model level
#                 else:
#                     # 3.1 Multiple pseudo levels on a single model level
#                     if field.lbuser4 in MULTI_PSEUDO_LEVEL_STASH:
#                         if field.lbuser5 in ens_data[field.lbuser4].keys():
#                             ens_data[field.lbuser4][field.lbuser5].append(field)
#                         else:
#                             ens_data[field.lbuser4][field.lbuser5] = [field]
#
#                     # 3.2. Special instance, only want the ice land fraction
#                     elif field.lbuser4 == STASH_LANDFRAC:
#                         if field.lbuser5 in STASH_LEVEL_CONSTRAINTS[STASH_LANDFRAC]:
#                             # Read in landfrac pseudo-level...
#                             if field.lbuser5 in ens_data[field.lbuser4].keys():
#                                 ens_data[field.lbuser4][field.lbuser5].append(field)
#                             else:
#                                 ens_data[field.lbuser4][field.lbuser5] = [field]
#
#                     # 3.3 Single level fields without pseudo-levels
#                     else:
#                         ens_data[field.lbuser4].append(field)
#
#         # keep the fieldsfile objects for reference:
#         ens_ff_files.append(ff_file_in)
#
#     # Verify that all the STASH codes have data from every ensemble member (all lists equal to number of members):
#     # Three nested if statements as <ens_data> can be up to three dictionary levels deep
#     # STASH
#     for stash in STASH_TO_LOAD:
#         if type(ens_data[stash]) is not dict:
#             check_list_length(ens_data[stash], member_list, stash=stash)
#         else:
#             # Model level or surface layer
#             for level in ens_data[stash]:
#                 if type(ens_data[stash][level]) is not dict:
#                     check_list_length(ens_data[stash][level], member_list, stash=stash, level=level)
#                 else:
#                     # Pseudo-level
#                     for pseudo_level in ens_data[stash][level]:
#                         check_list_length(ens_data[stash][level][pseudo_level], member_list,
#                                           stash=stash, level=level, pseudo_level=pseudo_level)
#
#     return ens_data, ens_ff_files

def load_engl_member_data(member_list):

    """
    Loads in input data from ensemble run members. This can be from a large number of files, and sorted into a dict
    structure by field, level to produce a list containing the data for ensemble members. Can read in data from a single
    member (e.g. control)

    :return: ens_data (multi level dictionary =
        {STASH: {level or surface layer: {pseudo level: [items = [n member fields]}}}).
        For STASH codes that cannot be on pseudo-levels, that dictionary level is not used.
        e.g.{9: {1: [field1a, field2a, field3a], 2: [field1b, field2b ...]}} where 9 = stash code, 1 and 2 are model
        levels. The list of fields within are from all the different members and have length n where n = maximum number
        of members): all ensemble field data for STASH variables.
    :return: ens_ff_files (list): field files for all members

    Extra help:
    field.lblev = level code (e.g. 1 to 70. 3 special entries for special fields e.g. 7777 or 8888 (see UM F03 docs.)
        soil fields use lblev as their level. Land fraction has special 4 digit value here instead.
    field.lbuser4 = STASH code
    field.lbuser5 = pseudo level (land surface tiles will have these). For STASH codes with multiple surface layers and
        on pseudo-levels, this will be = (pseudo-level * 1000) + surface layer number. e.g. 9002 = pseudo-level 9,
        surface layer = 2.
    """

    def get_field_level(field):

        """Get the level that this field is on. If the field is not on a specific level, set it = 1."""

        # soil level
        if field.lbuser4 in MULTI_SOIL_LEVEL_STASH:
            level = field.lblev
        # for layer fields, use the layer as the level. If also on pseudo levels, lbuser5 has both layer and
        # pseduo-level, therefore need to isolate the surface layer.
        elif field.lbuser4 in MULTI_LAYER_STASH:
            level = field.lbuser5 % 1000  # % finds the remainder and works even if lbuser <= 1000
        # if field isn't on a specific layer, set it equal to 1.
        else:
            level = 1

        return level

    def get_field_pseudo_level(field):

        """Get the pseudo-level from the field. If on aggregate tiles, pseudo-level extracted will be 1.
        """

        # setup pseudo level
        if field.lbuser4 in MULTI_PSEUDO_LEVEL_STASH:
            # If field is on multiple layers and pseudo-levels, lbuser5 will contain information on
            # both, therefore need to extract the pseudo-level.
            if field.lbuser4 in MULTI_LAYER_STASH:
                if field.lbuser5 > 1000:
                    # // finds the quotient or number of times divided
                    pseudo_level = field.lbuser5 // 1000
                else:
                    # if aggregate tiles used, lbuser5 will be the pseudo-level
                    pseudo_level = field.lbuser5
                    # normal attribute with pseudo-level on will be 1 if on aggregate tiles
            else:
                pseudo_level = field.lbuser5
        # if tile cannot be on pseudo-levels, simply set it = 1.
        else:
            pseudo_level = 1

        return pseudo_level

    def check_list_length(stash_list, member_list, **kwargs):

        """
        Check the length of each fields list to ensure it is equal to the number of ensemble members. If not, throw an
        error
        :param stash_list: (list) list of fields

        :keyword: level: (int) information on what stash, level or pseudo level we're on. e.g. stash = 9 or level = 2.
        """

        # Find the number of members that should be read in (if control then 1, if ensemble then the number of ensemble
        # members
        if member_list == CONTROL_MEMBER:  # single member, therefore 1
            n_mem = 1
        else:
            n_mem = len(member_list)

        # Length of the lists read in
        list_length = len(stash_list)

        # Test whether the read in list length is equal to the number of expected members.
        if n_mem != list_length:

            # level detail string (which stash, level and/or pseudo-level are we on?)
            level_detail = '; '.join(['{} = {}'.format(key, item) for key, item in kwargs.items()])

            raise ValueError('{} fields found in ensemble for {}, but number of members expected is {} is expected'.
                format(list_length, level_detail, n_mem))

        return


    # ensure member_list is actually an iterable list of strings, if not already.
    if isinstance(member_list, (float, int, str)):
        member_list = [member_list]

    # set up the returned data structure...
    ens_data = {}
    for stash in STASH_TO_LOAD:
        ens_data[stash] = {}
        # if stash in MULTI_LEVEL_STASH or stash in STASH_LEVEL_CONSTRAINTS or stash in MULTI_PSEUDO_LEVEL_STASH\
        #         or stash in MULTI_LAYER_STASH:
        #     ens_data[stash] = {}
        # else:
        #     ens_data[stash] = {}

    # and the fieldsfile objects, which we can use as templates later on:
    ens_ff_files = []

    # load in the required data for each file:
    for member in member_list:

        # dump file with the data fields (valid for t+3 for this cycle and t-3 next cycle)
        soil_src_file = engl_cycle_dump(member)
        # data file to open:
        ff_file_in = mule.DumpFile.from_file(soil_src_file)
        ff_file_in.remove_empty_lookups()

        # pull out the fields:
        for field in ff_file_in.fields:
            # is this a stash to load?
            if field.lbuser4 in STASH_TO_LOAD:

                # flag to say "load the data". This is used so if there are STASH constraints saying this
                # particular field shouldn't be loaded, it can be set to False.
                load_flag = True

                # If only certain fields are needed for this STASH, 
                if field.lbuser4 in STASH_LEVEL_CONSTRAINTS:
                    # if not the field we are looking for...
                    if field.lbuser5 not in STASH_LEVEL_CONSTRAINTS[field.lbuser4]:
                        load_flag = False

                # If this field meets all the constraints ...
                if load_flag is True:

                    # Get field level and pseudo-level to help store them in a dictionary
                    # if field doesn't have one of them, it shall be set equal to 1 and used to ensure the number of
                    # nested dictionaries and lists is the same across all the loaded variables.
                    level = get_field_level(field)

                    pseudo_level = get_field_pseudo_level(field)

                    # load...
                    # load in the fields
                    if level in ens_data[field.lbuser4].keys():  # is level already present
                        if pseudo_level in ens_data[field.lbuser4][level].keys():  # is pseudo-level present
                            ens_data[field.lbuser4][level][pseudo_level].append(field)
                        else:
                            ens_data[field.lbuser4][level][pseudo_level] = [field]
                    else:
                        ens_data[field.lbuser4][level] = {pseudo_level: [field]}

        # keep the fieldsfile objects for reference:
        ens_ff_files.append(ff_file_in)

    # Verify that all the STASH codes have data from every ensemble member (all lists equal to number of members):
    # Three nested if statements as <ens_data> can be up to three dictionary levels deep
    # STASH

    for stash in STASH_TO_LOAD:
        for level in ens_data[stash]:
            for pseudo_level in ens_data[stash][level]:
                check_list_length(ens_data[stash][level][pseudo_level], member_list,
                                      stash=stash, level=level, pseudo_level=pseudo_level)


    # for stash in STASH_TO_LOAD:
    #     if type(ens_data[stash]) is not dict:
    #         check_list_length(ens_data[stash], member_list, stash=stash)
    #     else:
    #         # Model level or surface layer
    #         for level in ens_data[stash]:
    #             if type(ens_data[stash][level]) is not dict:
    #                 check_list_length(ens_data[stash][level], member_list, stash=stash, level=level)
    #             else:
    #                 # Pseudo-level
    #                 for pseudo_level in ens_data[stash][level]:
    #                     check_list_length(ens_data[stash][level][pseudo_level], member_list,
    #                                       stash=stash, level=level, pseudo_level=pseudo_level)

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
        mean_data[stash] = {}
        for level in in_data[stash]:
            mean_data[stash][level] = {}
            for pseudo_level in in_data[stash][level]:
                # add them up:
                sum_field = adder(in_data[stash][level][pseudo_level])
                # now divide:
                mean_field = divver(sum_field)
                # and store that in the output structure, as cached data:
                mean_data[stash][level][pseudo_level] = cahceoperator(mean_field)
                # now use get_data to actually do the meaning at this point
                # and cache the result. This means the timer is accurate.
                _ = mean_data[stash][level][pseudo_level].get_data()

    # add the land-sea mask (assumes common for all members, hence take the first member's mask)
    mean_data[STASH_LAND_SEA_MASK] = in_data[STASH_LAND_SEA_MASK][1][1][0]

    # add each of the land-use mask (assumes common for all members, hence take the first member's mask)
    mean_data[STASH_LANDFRAC] = {key: item[0] for key, item in in_data[STASH_LANDFRAC][1].items()}

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
        corrction_data[stash] = {}
        for level in control_data[stash]:
            corrction_data[stash][level] = {}
            for pseudo_level in control_data[stash][level]:

                # ensemble mean field - control field
                corrction_data[stash][level][pseudo_level] = \
                    subber([mean_data[stash][level][pseudo_level], control_data[stash][level][pseudo_level][0]])


    # now add the land sea mask from the first member as well.
    # Take the land-sea mask out of the single element list
    corrction_data[STASH_LAND_SEA_MASK] = control_data[STASH_LAND_SEA_MASK][1][1][0]

    # add the land use as well (taken the items out of a single element list
    corrction_data[STASH_LANDFRAC] = {key: item[0] for key, item in control_data[STASH_LANDFRAC][1].items()}

    return corrction_data


def pert_check_correction(corr_data, ens, ctrl):

    """
    Set perturbations to be 0.0:
    1. Over ice or where snow was present on any layer, in any of the members, including the
    control.
    2. Where soil temperature is below -10 degC
    3. Where the absolute perturbation values are more than 1 standard deviation of the full field.
    :param corr_data: ensemble mean - control, correction data.
    :param ens: ensemble data (contains the snow fields from all members except the control)
    :param ctrl: control ensemble member (contains control member's snow field)
    :return corr_data
    """

    def apply_mask(field, mask):

        """
        apply the mask to the data
        :param field: (field) field to partially mask
        :param mask: (numpy array with boolean values, same shape as field.get_data()) True for where to mask
        :return:
        """

        # extract data and set the elements to <value>
        tmp_pert_data = field.get_data()

        # Print how many values will be masked that have not been already.
        if DIAGNOSTICS:
            legit_perts = np.logical_and(tmp_pert_data != field.bmdi, tmp_pert_data != 0.0)
            masked = np.sum(np.logical_and(legit_perts, mask))
            print('STASH: {}; lblev: {}; lbuser5 {}; Additional number of perturbation values masked: {}'.format(field.lbuser4, field.lblev, field.lbuser5, masked))

        # mask the data
        tmp_pert_data[mask] = 0.0

        # now put that data back into the field:
        array_provider = mule.ArrayDataProvider(tmp_pert_data)
        field.set_data_provider(array_provider)

        return field

    def zero_land_ice_snow_perts(corr_data, ens, ctrl):

        """
        Zero perturbations where:
            1) Land-ice is present
            2) Number of snow layers differs
            3) There is any snow on the tile in any member (mask for all except TSNOW)
        :param corr_data:
        :param ctrl:
        :return ice_snow_mask: ice and snow mask with True being present on any grid cell for any member
        :return:
        """

        def num_snow_layers_differ_mask(snow_fields):

            """
            Determine whether any ensemble member had different number of snow layers on any pseudo level.
            Loop through each pseudo-level first and determine whether the number of snow levels is the same across
            all the ensemble members. If not, then the mask is set to True at those points.
            :param snow_fileds: (dict): All snow fields across all the ensemble members
            :return: snow_masks: (dict): Snow masks for each pseudo-level
            :return: comb_snow_mask: (bool array): Whether any ensemble member had any snow on any pseudo level

            Pseudo-level is a land-surface tile type e.g. ice, urban, soil
            """

            # Define list to fill of 2D bool arrays
            snow_masks = {}

            for (level, snow_fields_level) in snow_fields.items():


                # Extract out all snow data for this pseudo-level
                snow_data = [snow_field_i.get_data() for snow_field_i in snow_fields_level]

                # Setup mask array (initialised as False everywhere, so no masking)
                snow_masks[level] = np.zeros(snow_data[0].shape, dtype=bool)

                # loop through all idx positions. If not a missing data point, check whether all the snow fields data
                # are the same at this point. Done by comparing all field's data to the first field. If any member
                # has a different value, then set the mask point to True for later masking.
                for (idx_x, idx_y), snow_data_0_value in np.ndenumerate(snow_data[0]):
                    if snow_data_0_value != snow_fields_level[0].bmdi:
                        # if any fields have different values, then this resolves to True
                        snow_masks[level][idx_x, idx_y] = \
                            np.any([snow_data_i[idx_x, idx_y] != snow_data_0_value for snow_data_i in snow_data])

            # create a combined snow mask, where for each point, it is True if True on any pseudo-level
            stack = np.stack([i for i in snow_masks.values()], axis=2)
            comb_snow_mask = np.any(stack, axis=2)

            return snow_masks, comb_snow_mask

        def num_snow_layers_gt_0_mask(snow_fields):

            """
            Create a single 2D bool array to show whether any ensemble member had any snow on any pseudo level.
            Loop through each pseudo-level first and determine whether any snow was present across the ensemble members,
            then combine together to see if any snow was present on any pseudo level.
            :param snow_fileds: (dict): All snow fields across all the ensemble members
            :return: comb_snow_mask: (bool array): Whether any ensemble member had any snow on any pseudo level
            Pseudo-level is a land-surface tile type e.g. ice, urban, soil
            """

            # Define list to fill of 2D bool arrays
            snow_masks = {}

            # Iterate over each pseudo-level to combine snow data across all ensemble members
            for (level, snow_fields_level) in snow_fields.items():
                # Turn snow field arrays (values: number of snow layers) into boolean arrays (values: >=one layers of
                # snow = 1)
                snow_masks_level = [np.logical_and(snow_field_i.get_data() >= 1,
                                                   snow_field_i.get_data() != snow_field_i.bmdi)
                                    for snow_field_i in snow_fields_level]

                # Stack the arrays together along a third dimension to enable the next step
                stacked_snow_masks = np.stack(snow_masks_level, axis=2)

                # Create a list of  2d boolean arrays showing whether any of the ensemble members had snow on any layer, for
                # each pseudo-level
                snow_masks[level] = np.any(stacked_snow_masks, axis=2)

            # Combine the snow masks across the pseudo-levels to show whether there was snow, on any pseudo level, for any
            # ensemble member
            snow_mask_stack = np.stack([i for i in snow_masks.values()], axis=2)
            comb_snow_mask = np.any(snow_mask_stack, axis=2)

            return snow_masks, comb_snow_mask

        # 1. Get land-ice mask
        # load in the land-ice data
        landice_data = corr_data[STASH_LANDFRAC][PSEUDO_LEVEL_LANDICE].get_data()
        # create land-ice mask
        landice_mask = np.logical_and(landice_data == 1.0,
                                      landice_data != corr_data[STASH_LANDFRAC][PSEUDO_LEVEL_LANDICE].bmdi)

        # 2. Combine ens and ctrl snow field dict into a single dict (order is irrelevant)
        # snow_fields = {pseudo_level: [list of snow fields for each level equal to number of ensemble plus ctrl member]}
        snow_fields = {pseudo_level: ens[STASH_NUM_SNOW_LAYERS][1][pseudo_level] + ctrl[STASH_NUM_SNOW_LAYERS][1][pseudo_level]
                       for pseudo_level in ens[STASH_NUM_SNOW_LAYERS][1].keys()}

        # 3. Masks for whether the number of snow layers differed across the members, on each pseudo-level
        snow_diff_masks, comb_snow_diff_mask = num_snow_layers_differ_mask(snow_fields)

        # 4. Masks for whether there was any snow on any member, on each pseduo-level
        snow_any_masks, comb_snow_any_mask = num_snow_layers_gt_0_mask(snow_fields)

        # 5. Apply the masks
        for stash in STASH_TO_MAKE_PERTS:
            for level in corr_data[stash].keys():
                for pseudo_level in corr_data[stash][level]:

#                   # 5.1 Apply the ice mask (set = 0.0)
                    corr_data[stash][level][pseudo_level] = \
                        apply_mask(corr_data[stash][level][pseudo_level], landice_mask)

                    # 5.2 If number of snow layers differed (set = 0.0)
                    corr_data[stash][level][pseudo_level] = \
                        apply_mask(corr_data[stash][level][pseudo_level], snow_diff_masks[pseudo_level])

                    # 5.3 If there was any snow on the tile (set = 0.0)
                    # For all pert variables except snow temperature, where having snow on a tile is fine.
                    if stash != STASH_TSNOW:
                        corr_data[stash][level][pseudo_level] = \
                            apply_mask(corr_data[stash][level][pseudo_level], snow_any_masks[pseudo_level])


        return corr_data, landice_mask, snow_diff_masks, snow_any_masks

    def zero_perts_lt_m10degc(corr_data, ens, ctrl):

        """
        Mask perturbation data on each level where the same level tsoil temperature is below -10degC. If perts are
        on a single level, each tsoil temperature mask will be applied in turn.
        :param corr_data:
        :param ens:
        :param ctrl:
        :return:

        Only applied to the SMC perturbations as the other variables are fine to vary below freezing.
        """

        # Combine TSOIL dicts from ensemble and ctrl
        tsoil_fields = {level: ens[STASH_TSOIL][level][1] + ctrl[STASH_TSOIL][level][1] for level in ens[STASH_TSOIL].keys()}

        tsoil_masks = {}

        for (level, tsoil_fields_level) in tsoil_fields.items():

            # Create tsoil mask where True is data below -10 degC (263.15 K)
            tsoil_masks_level = [np.logical_and(tsoil_field_i.get_data() < 263.15,
                                                tsoil_field_i.get_data() != tsoil_field_i.bmdi)
                                 for tsoil_field_i in tsoil_fields_level]

            # Stack the arrays together along a third dimension to enable the next step
            stacked_tsoil_masks = np.stack(tsoil_masks_level, axis=2)

            # Create a list of 2d boolean arrays showing whether any of the ensemble members had tsoil < -10 degC for
            #  this level. Append mask to list for function export
            tsoil_comb_mask = np.any(stacked_tsoil_masks, axis=2)
            # Store using the same [level][pseuo-level] format as other dictionaries in this code. Pseudo-level is
            #   always 1 as it's not applicable for TSOIL.
            tsoil_masks[level] = {1: tsoil_comb_mask}

            # apply the mask to the SMC perts ONLY
            corr_data[STASH_SMC][level][1] = apply_mask(corr_data[STASH_SMC][level][1], tsoil_comb_mask)

        return corr_data, tsoil_masks

    def store_masks_in_fields(ctrl, snow_diff_masks, snow_any_masks, tsoil_masks):

        """
        Store the masks in fields and all within a dicitonary for saving
        :param ctrl:
        :param snow_diff_masks:
        :param snow_any_masks:
        :param tsoil_masks:
        :return: mask_fields:
        """

        def mask_to_field(field, mask, lbuser6=None):

            """
            Store the masks in fields and output in a dictionary
            :param field:
            :param mask:
            :param lbuser6:
            :return:
            """

            mask_field = mule.Field.copy(field)
            snow_array_provider = mule.ArrayDataProvider(mask)
            mask_field.set_data_provider(snow_array_provider)

            # lbuser6 is free space for users. Use it to define this mask is for differing number of snow layers
            if lbuser6 is not None:
                mask_field.lbuser6 = lbuser6

            return mask_field

        mask_fields = {}

        # Add the land-ice mask
        mask_fields[STASH_LANDFRAC] = {1: {key: item[0] for key, item in ctrl[STASH_LANDFRAC][1].items()}}

        # Put the two different snow masks into fields, and store them in a dictionary. Use [level] of dictionary
        #   to split the two different masks up.
        mask_fields[STASH_NUM_SNOW_LAYERS] = {1: {}, 2: {}}
        # mask_fields[STASH_NUM_SNOW_LAYERS][level] = {}
        for pseudo_level in ctrl[STASH_NUM_SNOW_LAYERS][1].keys():
            # field to take a copy of to store snow masks
            orig_snow_field = ctrl[STASH_NUM_SNOW_LAYERS][1][pseudo_level][0]

            # Store the snow masks in fields for exporting
            # Change the lbuser6 value in each to later distinguish between the two NUM_SNOW_LAYER masks
            # Use lbuser6 as the [level] in the stored mask field dictionary
            mask_fields[STASH_NUM_SNOW_LAYERS][1][pseudo_level] = \
                mask_to_field(orig_snow_field, snow_diff_masks[pseudo_level], lbuser6=1)

            mask_fields[STASH_NUM_SNOW_LAYERS][2][pseudo_level] = \
                mask_to_field(orig_snow_field, snow_any_masks[pseudo_level], lbuser6=2)

        # Store the <-  10 degC TSOIL masks
        mask_fields[STASH_TSOIL] = {}
        for level in tsoil_masks.keys():
            mask_fields[STASH_TSOIL][level] = {}
            for pseudo_level in tsoil_masks[level].keys():
                # field to take a copy of to store tsoil masks
                orig_tsoil_field = ctrl[STASH_TSOIL][level][pseudo_level][0]

                # Store TSOIL mask in copy of TSOIL field.
                # No need for lbuser6 as there is only one set of TSOIL masks
                mask_fields[STASH_TSOIL][level][pseudo_level] = \
                    mask_to_field(orig_tsoil_field, tsoil_masks[level][pseudo_level])

        # add land-sea mask because it needs it (extract it out of the single element list)
        mask_fields[STASH_LAND_SEA_MASK] = {1: {1: ctrl[STASH_LAND_SEA_MASK][1][1][0]}}

        return mask_fields


    print('Making perturbations:')
    # 1. Zero perturbation where for any ensemble member, on any tile there is
    #    1) Land ice
    #    2) Snow is present (all but TSNOW where having snow is fine)
    #    3) Number of snow layers differs between members.
    print('Ice and snow masking:')
    corr_data, landice_mask, snow_diff_masks, snow_any_masks = zero_land_ice_snow_perts(corr_data, ens, ctrl)

    # 2. Zero SMC perturbations ONLY where TSOIL is below -10 degC (other pert variables are unaffected)
    print('TSOIL < -10 degC masking:')
    corr_data, tsoil_masks = zero_perts_lt_m10degc(corr_data, ens, ctrl)

    # 3. Zero perturbations where absolute pert values are larger than 1 standard deviation of the original field.
    #print('abs(pert) > 1 standard deviation of field masking:')
    #corr_data, stdev_masks = cap_perts_gt_bgerr(corr_data, ens, ctrl)

    # print out additional diagnostics (descriptive statistics of fields)
    if DIAGNOSTICS:
        print('\nPerturbation descriptive statistics:')
        for stash in STASH_TO_MAKE_PERTS:
            for level in corr_data[stash].keys():
                for (pseudo_level, pert_field) in corr_data[stash][level].items():
                    data = pert_field.get_data()
                    data_flat = data[np.where(data != pert_field.bmdi)].flatten()
                    # print min, max , rms
                    print('STASH: {}; lblev: {}; lbuser5: {}:'.format(pert_field.lbuser4, pert_field.lblev, pert_field.lbuser5))
                    print('maximum: {0:.2e}'.format(np.amax(data_flat)))
                    print('minimum: {0:.2e}'.format(np.amin(data_flat)))
                    print('rms    : {0:.2e}'.format(np.sqrt(np.mean(data_flat**2))))
        print('')

    # 4. Put masks into fields for later saving, with a valid land-sea mask
    #   1) Land-ice field (used to make mask)
    #   2) Number of snow layers present differed between the members
    #   3) Snow present on any layer
    #   4) TSOIL < -10 degC
    mask_fields = store_masks_in_fields(ctrl, snow_diff_masks, snow_any_masks, tsoil_masks)

    return corr_data, mask_fields

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
        if type(data[stash]) == dict:
            for level in data[stash]:
                if type(data[stash][level]) == dict:
                    for pseudo_level in data[stash][level]:
                        ens_ff.fields.append(data[stash][level][pseudo_level])
                else:
                    ens_ff.fields.append(data[stash][level])
        else:
            ens_ff.fields.append(data[stash])

    # save
    ens_ff.to_file(ens_filepath)

    print('Saved: '+ens_filepath)

    return


if __name__ == '__main__':

    """
    Routine 1 of 2 for creating the ensemble soil moisture content (SMC) perturbation 
    correction.
    
    1) Loads in data
    2) Produces the mean of the input fields
    3) Produces the correction fields (ensemble mean - control field)
    4) Checks correction fields
    5) Save ensemble mean
    6) Saves the correction
    """

    ## Read
    # load the SMC, TSOIL, and other input data for all members:
    ens_data, ens_ff_files = load_engl_member_data(MEMBERS_PERT_INTS)

    # load the SMC, TSOIL, and other input data for the control member only:
    ctrl_data, ctrl_ff_files = load_engl_member_data(CONTROL_MEMBER)

    ## Process
    # create the mean for all required ensemble member fields
    ens_mean = mean_ens_data(ens_data)

    # create the soil recentering correction (ensemble mean - control).
    ens_correction = mean_minus_control(ctrl_data, ens_mean)

    # Carry out checks and corrections to ensure the perts are phystically sensible:
    # 1. Set pert values to 0 where ice or snow is present on land, in any member (including control)
    # the snow field used is appended to the ens_correction dictionary for use in the next cycle.
    # 2. Set pert values to 0 where TSOIL < -10 degC
    # 3. Set pert values to 0 where absolute perts are more than 1 standard deviations of the original field.
    # Export all masks for saving
    ens_correction, mask_fields = pert_check_correction(ens_correction, ens_data, ctrl_data)

    ## Save
    # save the ensemble mean used in making perturbations (mean(all_members_of_same_cycle))
    # includes the composite boolean snow field map produced in zero_land_ice_snow_perts()
    save_fields_file(ens_mean, ens_ff_files, 'engl_soil_mean')

    # save the correction
    # use the control ensemble file as a file template for saving
    save_fields_file(ens_correction, ctrl_ff_files, 'engl_soil_correction')

    # save masks
    # use the control ensemble file as a file template for saving
    save_fields_file(mask_fields, ctrl_ff_files, 'engl_soil_correction_masks')

    exit(0)