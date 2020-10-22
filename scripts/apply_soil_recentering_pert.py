#!/usr/bin/env python

"""
Load in ensemble correction (ensemble mean - control), calculated from and valid for t+3 in the previous cycle. Then,
take it away from the EKF analysis of current cycle. The combined output should be valid for t-3 for this cycle.
Scientific checks are assumed to be carried out by IAU. This file is then to be used by all members, effectively
centering the ensemble members around the control.

Created by Elliott Warren Thurs 21th Nov 2019: elliott.warren@metoffice.gov.uk
Based on engl_ens_smc_pert.py by Malcolm Brooks 18th Sept 2016: Malcolm.E.Brooks@metoffice.gov.uk

Tested versions using canned data:
python 2.7.16
python 3.7.5
mule 2019.01.1
numpy 1.16.5 (python2)
numpy 1.17.3 (python3)

Testing carried out in:
/data/users/ewarren/R2O_projects/soil_moisture_pertubation/
"""

import os
import numpy as np

import mule
import mule.operators

# Environment variables:
# last cycle directory
ROSE_DATACPT6H = os.getenv('ROSE_DATACPT6H')

# this cycle directory
ROSE_DATAC = os.getenv('ROSE_DATAC')

# ensemble correction data from the previous cycle for this member (full path)
ENS_SOIL_CORR_FILEPATH = os.getenv('ENS_SOIL_CORR_FILEPATH')

# filepath with the existing soil EKF perturbations for this member
ENS_SOIL_EKF_FILEPATH = os.getenv('ENS_SOIL_EKF_FILEPATH')

if ROSE_DATACPT6H is None:
    # if not set, then this is being run for development, so have canned variable settings to hand:
    ROSE_DATACPT6H = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20181201T0600Z'
    ROSE_DATAC = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20181201T1200Z'
    ENS_MEMBER = '1'
    ENS_SOIL_CORR_FILEPATH = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20181201T0600Z/' \
                   'engl_smc/engl_soil_correction'
    ENS_SOIL_EKF_FILEPATH = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20181201T1200Z/engl_smc/' \
                            'engl_surf_inc' # control member

# ------------------------------------

# Configuration:
# STASH codes to use:
STASH_LAND_SEA_MASK = 30
STASH_SMC = 9
STASH_TSOIL = 20
STASH_SNOW_AMNT = 23  # can be overwritten by other programs therefore not the ideal choice for use as a snow mask
STASH_NUM_SNOW_LAYERS = 380  # preferred alternative to STASH_SNOW_AMNT that does not get overwritten
# land use fractions:
STASH_LANDFRAC = 216
# pseudo level of land ice tile (this should remain unchanged at 9!)
PSEUDO_LEVEL_LANDICE = 9

# STASH codes to load and mean:
STASH_TO_LOAD = [STASH_SMC, STASH_LAND_SEA_MASK, STASH_NUM_SNOW_LAYERS]
# these need to be all multi-level (not pseudo level) stash codes
MULTI_LEVEL_STASH = [STASH_SMC]
# a list of stash codes we want to actually act on to produce perturbations in this routine:
STASH_TO_MAKE_PERTS = [STASH_SMC]

# ------------------------------------


def validate_overide(*args, **kwargs):
    pass


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


# load and process functions


def load_prev_ens_correction_data():

    """
    Loads in the correction data (ensemble mean - control field) from the previous cycle. Sorted into a dict structure
    by field E.g. {9: {1: [field1], 2: [field2], ...}} where 9 = STASH code and 1 and 2 are levels

    Also contains the land_sea mask (STASH = 30), required by the soil moisture content and soil temperature field
    :return: corr_data_in (dictionary) correction data
    """

    # set up the returned data structure...
    corr_data_in = {}
    for stash in STASH_TO_LOAD:
        if stash in MULTI_LEVEL_STASH:
            corr_data_in[stash] = {}
        else:
            corr_data_in[stash] = []

    # data file to open:
    ff_file_in = mule.load_umfile(ENS_SOIL_CORR_FILEPATH)
    ff_file_in.remove_empty_lookups()

    # pull out the fields:
    for field in ff_file_in.fields:
        if field.lbuser4 in STASH_TO_LOAD:
            if field.lbuser4 in MULTI_LEVEL_STASH:
                # multi-level fields are a dict, with a list for each level
                if field.lblev in corr_data_in[field.lbuser4].keys():
                    corr_data_in[field.lbuser4][field.lblev] = field
                else:
                    corr_data_in[field.lbuser4][field.lblev] = field
            else:
                # single level fields are a flat list:
                corr_data_in[field.lbuser4] = field

    return corr_data_in

# load and process EKF functions


def load_field_data(stash_and_constraints, ff_file, cache=False):
    """
    Loads EKF soil data

    :param stash_and_constraints: (dictionary) stash codes and additional constraints to load
    :param: ff_file: (filepath str) filepath of the data to load
    :param cache: (bool) To use mule cache functionality to assist loading
    :return: ff_data: (dictionary) soil data fields from current cycle
    :return: ff_obj: (object) the UM file read in
    """

    if isinstance(ff_file, str):
        ff_obj = mule.load_umfile(ff_file)
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


def load_ekf_combine_with_correction(corr_data):

    """
    Load in EFK soil perturbations and take away the correction needed to centre the overall ensemble mean for this
    cycle

    :return: soil_centred_pert: (dictionary) combined perturbations of the EKF and correction
    """

    # STASH codes to load in from the analysis dump:
    # stash codes as dict key, then a dict of constraints, if any:
    stash_from_dump = {STASH_SMC: None,
                       STASH_LAND_SEA_MASK: None,
                       STASH_LANDFRAC: {'lbuser5': [PSEUDO_LEVEL_LANDICE]}}

    # load in the EKF and other
    soil_fields, smc_ff = load_field_data(stash_from_dump, ENS_SOIL_EKF_FILEPATH, cache=True)

    # reduce EKF by the correction (now that EKF is done via IAU).
    subber = mule.operators.SubtractFieldsOperator(preserve_mdi=True)

    soil_centred_pert = {stash: {} for stash in STASH_TO_MAKE_PERTS}
    for stash in STASH_TO_MAKE_PERTS:
        for level, corr_l in corr_data[stash].items():
            ekf_pert_l = soil_fields[stash][level]
            soil_centred_pert[stash][level] = subber([ekf_pert_l, corr_l])

    # put land-sea mask in the soil_centred_pert dictionary
    soil_centred_pert[STASH_LAND_SEA_MASK] = corr_data[STASH_LAND_SEA_MASK]

    return soil_centred_pert

def load_combine_snow_fields():

    stash_from_dump = {STASH_NUM_SNOW_LAYERS: None}

    # load in the EKF and other
    snow_fields, snow_ff = load_field_data(stash_from_dump, ENS_SOIL_EKF_FILEPATH, cache=True)

    return

def zero_snow_cell_perts(corr_data, total_pert):

    """
    Set the perturbations for any grid cells with snow on them (defined from the snow field of the correction file)
     to 0.0
    :param corr_data: (dictionary) contains the snow field
    :param total_pert: (EKF - correction) perturbations that need to be set to 0.0 where snow is present.
    :return:
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

    # extract out snow mask
    snow_field = corr_data[STASH_NUM_SNOW_LAYERS]
    snow_data = snow_field.get_data()
    snow_mask = np.logical_and(snow_data == 1.0, snow_data != snow_field.bmdi)

    # zero any perturbations over tiles that 
    for stash in STASH_TO_MAKE_PERTS:
        if stash in MULTI_LEVEL_STASH:
            for level in total_pert[stash]:
                total_pert[stash][level] = apply_mask(total_pert[stash][level], snow_mask)
        else:
            total_pert[stash] = apply_mask(total_pert[stash], snow_mask)

    return total_pert

# save


def save_total_pert(centred_pert, template_file=ENS_SOIL_EKF_FILEPATH):

    """
    Save the total perturbation ready for the IAU to ingest. Use the EKF file as a template,
    as this is already intended for the IAU and to help future-proof the process as the file may
    include more information in the future.

    :param centred_pert: (dictionary) EKF - correction. The increment to apply, to center the ensemble mean
    """

    # Ideally use the existing EKF file as a template
    pert_ff_in = mule.AncilFile.from_file(template_file)
    pert_ff_out = pert_ff_in.copy(include_fields=False)  # empty copy
    # name the output file for EKF - correction
    output_pert_file = ROSE_DATAC + '/engl_smc/engl_surf_inc_correction'

    os.system('echo file being saved using '+template_file+' as a template')

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
            pert_ff_out.fields.append(field)
        else:
            pert_ff_out.fields.append(field)
            if template_time_field is None:
                template_time_field = field
    # add land-sea mask if not present already
    if not out_pert_has_lsm:
        # add it:
        pert_ff_out.fields.append(centred_pert[STASH_LAND_SEA_MASK])

    # Now add in the perturbations:
    # Correct times to ensure validity time matches
    for stash in STASH_TO_MAKE_PERTS:
        if stash in MULTI_LEVEL_STASH:
            # write out each level (the order now matters!):
            for level in sorted(centred_pert[stash]):
                pert_ff_out.fields.append(centred_pert[stash][level])
        else:
            pert_ff_out.fields.append(centred_pert[stash])

    # overide validate function to keep the level dependent constants in the ancillary file and save
    # pert_ff_out.validate = validate_overide
    pert_ff_out.to_file(output_pert_file)

    print('saved: ' + output_pert_file)

    return


if __name__ == '__main__':

    """
    Routine 2 of 2 for applying the ensemble soil moisture content (SMC) correction, to 
    the EKF perturbation.
    """

    ## Read and process
    # load the ensemble correction data (ensemble mean - control field, both from previous cycle).
    corr_data = load_prev_ens_correction_data()

    # if the EKF file exists, read it in and combine with the ensemble correction.
    if os.path.exists(ENS_SOIL_EKF_FILEPATH):
        total_pert = load_ekf_combine_with_correction(corr_data)
    else:
        raise ValueError(ENS_SOIL_EKF_FILEPATH +' is missing!')

    ## Save
    # save total perturbations in the original perturbation file
    save_total_pert(total_pert)

    exit(0)


