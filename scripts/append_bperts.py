#!/usr/bin/env python2

"""
Load in breeding perturbations, calculated from and valid for t+3 in the previous cycle. Then,
and add them to EKF perturbations of current cycle. The combined output should be valid for t-3 for this cycle.


Created by Elliott Warren Thurs 21th Nov 2019: elliott.warren@metoffice.gov.uk
Based on engl_ens_smc_pert.py by Malcolm Brooks 18th Sept 2016: Malcolm.E.Brooks@metoffice.gov.uk

Tested versions using canned data:
python 2.7.16
mule 2019.01.1
numpy 1.16.5

Testing (including soil temperature) carried out in:
/data/users/ewarren/R2O_projects/soil_moisture_pertubation/
"""

import os
import shutil
import numpy as np

import mule
import mule.operators

# Environment variables:
# last cycle directory
ROSE_DATACPT6H = os.getenv('ROSE_DATACPT6H')

# this cycle directory
ROSE_DATAC = os.getenv('ROSE_DATAC')

# ensemble member
ENS_MEMBER = os.getenv('ENS_MEMBER')

# breeding perturbation data from the previous cycle
ENS_PERT_DIR = os.getenv('ENS_PERT_DIR')

# directory with the existing soil EKF perturbations for this member
ENS_SMC_DIR = os.getenv('ENS_SMC_DIR')

# tuning factor to determine how much of the breeding perturbation to add to the existing EKF perturbation.
# as final, combined perturbation will be EKF_pert + (TUNING_FACTOR*breeding_pert)
# i.e. 0 = no breeding added, 1 = full added, >1 = amplified breeding perturbation added
TUNING_FACTOR = os.getenv('TUNING_FACTOR')

# actually overwrite the main perturbation files at the end of the script
OVERWRITE_PERT_FILES = True


if ROSE_DATACPT6H is None:
    # if not set, then this is being run for development, so have canned variable settings to hand:
    ROSE_DATACPT6H = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20181201T0600Z'
    ROSE_DATAC = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20181201T1200Z'
    ENS_MEMBER = '1'
    ENS_PERT_DIR = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20181201T0600Z/' \
                   'engl_smc/engl_smc_bpert'
    ENS_SMC_DIR = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20181201T1200Z/engl_um/' \
                  'engl_um_{0:03d}'.format(int(ENS_MEMBER))  # made to match the ENS_MEMBER
    TUNING_FACTOR = '1'
    OVERWRITE_PERT_FILES = False

# conversions to type:
ENS_MEMBER_INT = int(ENS_MEMBER)
TUNING_FACTOR_FLOAT = float(TUNING_FACTOR)

# quick check TUNING_FACTOR is not less than 0
if TUNING_FACTOR_FLOAT < 0.0:
    raise ValueError('TUNING_FACTOR value of {0} is invalid. MUst be set >0.0'.format(TUNING_FACTOR_FLOAT))

# Configuration:
# STASH codes to use:
STASH_LAND_SEA_MASK = 30
STASH_SMC = 9
STASH_TSOIL = 20
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

        # return src_field.get_data()

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


def set_time_metadata(data_field, time_template_field):
    """
    Sets the time metadata of an input data field to that of a template field
    :param data_field: (field) data field that needs its time changed
    :param time_template_field: (field) template field that has the time we want to change to
    :return:
    """

    data_field.lbyr = time_template_field.lbyr
    data_field.lbmon = time_template_field.lbmon
    data_field.lbdat = time_template_field.lbdat
    data_field.lbhr = time_template_field.lbhr
    data_field.lbmin = time_template_field.lbmin

    data_field.lbyrd = time_template_field.lbyrd
    data_field.lbmond = time_template_field.lbmond
    data_field.lbdatd = time_template_field.lbdatd
    data_field.lbhrd = time_template_field.lbhrd
    data_field.lbmind = time_template_field.lbmind

    data_field.lbtim = time_template_field.lbtim
    data_field.lbft = time_template_field.lbft

    # field may not have lbsec or lbsecd - so if not, set them to 0
    if not hasattr(time_template_field, 'lbsec'):
        data_field.lbsec = 0
    else:
        data_field.lbsec = time_template_field.lbsec

    if not hasattr(time_template_field, 'lbsecd'):
        data_field.lbsecd = 0
    else:
        data_field.lbsecd = time_template_field.lbsec

    return


# filename creation functions


def engl_cycle_smc_filename():
    """ locates soil moisture content data for this current cycle"""
    return '{0}/engl_surf_inc'.format(ENS_SMC_DIR)


# loading functions

def load_soil_data(stash_and_constraints, cache=False):
    """
    Loads required data from a single file, supplied as a file name to load
    or a ff object, into a simple dictionary.

    Returns the requested FieldsFile data, and the FieldsFile object
    :param stash_and_constraints: (dictionary) stash codes and additional constraints to load
    :param cache: (bool) To use mule cache functionality to assist loading
    :return: ff_data: (dictionary) soil data fields from current cycle
    :return: ff_obj: (object) the UM file read in
    """

    # identify engl_smc file to load, containing soil moisture and temperature with existing EKF perturbation
    ff_file = engl_cycle_smc_filename()

    if isinstance(ff_file, str):
        ff_obj = mule.load_umfile(ff_file)
        ff_obj.remove_empty_lookups()
    else:
        ff_obj = ff_file
        ff_obj.remove_empty_lookups()

    ff_data = {}
    for field in ff_obj.fields:
        # is this stash code we want?
        if field.lbuser4 in stash_and_constraints.keys():
            # passes the first filter, some of these fields are needed:
            if field.lbuser5 == 0:
                # not a psudolevel field, so don't test them
                fld_needed = True
            else:
                # need to test against pseudo levels:
                fld_needed = False
                # now check the individual reqeusts to see if any match:
                if 'lbuser5' in stash_and_constraints[field.lbuser4]:
                    fld_needed = field.lbuser5 in stash_and_constraints[field.lbuser4]['lbuser5']
                else:
                    # not constraining on pseudo level (lbuser5)
                    fld_needed = True

            if fld_needed:
                if cache:
                    # apply the cache, if required:
                    cacheoperator = CachingOperator()
                    field = cacheoperator(field)
                if field.lbuser5 == 0:
                    # multi-level fields are a dict, with a list for each level
                    if field.lbuser4 in MULTI_LEVEL_STASH:
                        if field.lbuser4 not in ff_data.keys():
                            ff_data[field.lbuser4] = {}
                        ff_data[field.lbuser4][field.lblev] = field
                    else:
                        # single level fields are a flat list:
                        ff_data[field.lbuser4] = field
                else:
                    if field.lbuser4 in MULTI_LEVEL_STASH:
                        raise NotImplementedError('Cannot have multi-level fields with '
                                                  'psuedo-levels are not in this script yet')
                    else:
                        if field.lbuser4 not in ff_data.keys():
                            ff_data[field.lbuser4] = {}
                        # now this is like a multi-level field:
                        ff_data[field.lbuser4][field.lbuser5] = field

    return ff_data, ff_obj


def load_prev_ens_bpert_data():

    """
    Loads in previous cycle ensemble breeding perturbation data. Sorted into a dict structure by field, level to
    produce a list containing the data for ensemble members.
    E.g. {9: {1: [field1], 2: [field2], ...}} where 9 = STASH code and 1 and 2 are levels

    Also contains the land_sea mask (STASH = 30), required by the soil moisture content field
    :return: bpert_data_in (dictionary) breeding perturbation data
    :return: ff_file_in (object) bpert file
    """

    def engl_prev_cycle_bpert_filename():
        """ locates breeding pertubations calulated from t+6 of the last cycle"""
        return '{0}/engl_smc_bpert_{1}'.format(ENS_PERT_DIR, mem_to_str(ENS_MEMBER_INT))

    # load in the required data for each file:
    # for member in MEMBERS_PERT_INTS:

    # set up the returned data structure...
    bpert_data_in = {}
    for stash in STASH_TO_LOAD:
        if stash in MULTI_LEVEL_STASH:
            bpert_data_in[stash] = {}
        else:
            bpert_data_in[stash] = []

    # file with pre-calculated breeding perturbations (valid for t-3 this cycle)
    ens_smc_bpert_file = engl_prev_cycle_bpert_filename()
    # data file to open:
    # ff_file_in = mule.load_umfile(smc_src_file)
    ff_file_in = mule.DumpFile.from_file(ens_smc_bpert_file)
    #     raise ValueError("wrong file type")
    ff_file_in.remove_empty_lookups()

    # pull out the fields:
    for field in ff_file_in.fields:
        # is this the SMC?
        if field.lbuser4 in STASH_TO_LOAD:
            if field.lbuser4 in MULTI_LEVEL_STASH:
                # multi-level fields are a dict, with a list for each level
                if field.lblev in bpert_data_in[field.lbuser4].keys():
                    bpert_data_in[field.lbuser4][field.lblev] = field
                else:
                    bpert_data_in[field.lbuser4][field.lblev] = field
            else:
                # single level fields are a flat list:
                bpert_data_in[field.lbuser4] = field

    return bpert_data_in, ff_file_in


def load_combine_ekf_bpert():

    """
    Load in the EKF and breeding perturbations, and combine them. Also load in the land-sea mask
    from the breeding perturbations file.

    :return: soil_comb_pert: (dictionary) combined perturbations of the EKF and breeding scheme
    :return: bpert_landsea_field (field) land sea mask from the breeding perturbations file
    """

    # STASH codes to load in from the analysis dump:
    # stash codes as dict key, then a dict of constraints, if any:
    stash_from_dump = {STASH_SMC: None,
                       STASH_TSOIL: None,
                       STASH_LAND_SEA_MASK: None,
                       STASH_LANDFRAC: {'lbuser5': [PSEUDO_LEVEL_LANDICE]}}

    # load smc data in (current cycle analysis)
    soil_fields, smc_ff = load_soil_data(stash_from_dump, cache=True)

    # load in the breeding perturbation file, for this member, from the last cycle
    # includes the land-sea mask
    bpert_fields, ff_file_in = load_prev_ens_bpert_data()
    landsea_field = bpert_fields[STASH_LAND_SEA_MASK]


    # Combined perturbation from breeding method (bpert) with the EKF pertubation, for each soil level.
    # Use TUNING_FACTOR to scale the amount of bpert to add (scaling applied equally to all levels).
    tuning_factor_scaling = mule.operators.ScaleFactorOperator(TUNING_FACTOR_FLOAT)
    adder = mule.operators.AddFieldsOperator(preserve_mdi=True)

    soil_comb_pert = {stash: {} for stash in STASH_TO_MAKE_PERTS}
    for stash in STASH_TO_MAKE_PERTS:
        for level in bpert_fields[stash].iterkeys():
            bpert_l = bpert_fields[stash][level]
            bpert_l_scaled = tuning_factor_scaling(bpert_l)
            ekf_pert_l = soil_fields[stash][level]
            soil_comb_pert[stash][level] = adder([bpert_l_scaled, ekf_pert_l])

    return soil_comb_pert, landsea_field

# save


def save_comb_pert(soil_pert, landsea_field):

    # now write out the perturbation on top of the fields in the actual perturbation file.
    output_pert_file = engl_cycle_smc_filename()
    tmp_output_pert_file = output_pert_file + '_tmp.ff'

    # pert_ff_in = mule.AncilFile.from_file(output_pert_file, remove_empty_lookups=True)  # the full existing file
    #mule.load_umfile(ff_file)
    pert_ff_in = mule.load_umfile(output_pert_file)  #
    pert_ff_out = pert_ff_in.copy(include_fields=False)  # empty copy

    # Remove the level_dependend_constants fixed header if present, as it is no longer needed in an ancillary file and
    # mule will not save file with it in.
    if hasattr(pert_ff_out, 'level_dependent_constants'):
        pert_ff_out.level_dependent_constants = None

    # now go through the fields in the pert_ff_in and as long as they are not duplicates
    # of the fields in the this_perts object, add them to pert_ff_out.
    #
    # We MUST not output duplicate fields as the forecast will apply all perturbations
    # even if they are duplicated.
    # This can make the forecast go bad in a variety of interesting ways!
    #
    # the pert_ff_out must also have a land sea mask in it.
    out_pert_has_lsm = False
    template_time_field = None
    for field in pert_ff_in.fields:
        if field.lbuser4 in STASH_TO_MAKE_PERTS:
            pass  # don't load in the existing perturbed data. We want to replace this!
        elif field.lbuser4 == STASH_LAND_SEA_MASK:
            out_pert_has_lsm = True
            if not np.all(field.get_data() == soil_pert[STASH_LAND_SEA_MASK].get_data()):
                raise ValueError('Inconsistent land sea masks in perturbation files')
            pert_ff_out.fields.append(field)
        else:
            pert_ff_out.fields.append(field)
            if template_time_field is None:
                template_time_field = field
    # # does the pert_ff_out have a land sea mask already?
    if not out_pert_has_lsm:
        # add it:
        pert_ff_out.fields.append(landsea_field)

    # Now add in the perturbations:
    for stash in STASH_TO_MAKE_PERTS:
        if stash in MULTI_LEVEL_STASH:
            # write out each level (the order now matters!):
            for level in sorted(soil_pert[stash]):
                # # modify the headers for time:
                set_time_metadata(soil_pert[stash][level], template_time_field)
                # and append:
                pert_ff_out.fields.append(soil_pert[stash][level])
        else:
            # set_time_metadata(soil_comb_pert[stash], template_time_field)
            pert_ff_out.fields.append(soil_pert[stash])

    # pert_ff_out.validate
    pert_ff_out.to_file(tmp_output_pert_file)

    # if we are done, and this id being run 'for real'
    # then we want to overwrite the real pert file with the temporary one:
    if OVERWRITE_PERT_FILES:
        shutil.move(tmp_output_pert_file, output_pert_file)

    return


if __name__ == '__main__':

    # load the breeding perturbation data, EFK data and combine them.
    # Also load the land-sea mask for saving the data later.
    soil_comb_pert, bpert_landsea_field = load_combine_ekf_bpert()

    # save combined perturbations in the original perturbation file
    save_comb_pert(soil_comb_pert, bpert_landsea_field)

    exit(0)


