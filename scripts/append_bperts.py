#!/usr/bin/env python2

"""
Load in breeding perturbations, calculated from and valid for t+3 in the previous cycle. Then,
and add them to EKF perturbations of current cycle. The combined output should be valid for t-3 for this cycle.
Scientific checks are assumed to be carried out by IAU.

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

import mule
import mule.operators

# Environment variables:
# last cycle directory
ROSE_DATACPT6H = os.getenv('ROSE_DATACPT6H')

# this cycle directory
ROSE_DATAC = os.getenv('ROSE_DATAC')

# ensemble member
ENS_MEMBER = os.getenv('ENS_MEMBER')

# breeding perturbation data from the previous cycle for this member (full path)
ENS_SOIL_BPERT_FILEPATH = os.getenv('ENS_SOIL_BPERT_FILEPATH')

# filepath with the existing soil EKF perturbations for this member
ENS_SOIL_EKF_FILEPATH = os.getenv('ENS_SOIL_EKF_FILEPATH')

# tuning factor to determine how much of the breeding perturbation to add to the existing EKF perturbation.
# Final, combined perturbation will be EKF_pert + (TUNING_FACTOR*breeding_pert)
# i.e. 0 = no breeding added, 1 = fully added, >1 = amplified breeding perturbation added
TUNING_FACTOR = os.getenv('TUNING_FACTOR')

# actually overwrite the main perturbation files at the end of the script
OVERWRITE_PERT_FILES = True

if ROSE_DATACPT6H is None:
    # if not set, then this is being run for development, so have canned variable settings to hand:
    ROSE_DATACPT6H = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20181201T0600Z'
    ROSE_DATAC = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20181201T1200Z'
    ENS_MEMBER = '1'
    ENS_SOIL_BPERT_FILEPATH = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20181201T0600Z/' \
                   'engl_smc/engl_smc_bpert/engl_smc_bpert_{0:03d}'.format(int(ENS_MEMBER))
    ENS_SOIL_EKF_FILEPATH = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20181201T1200Z/engl_um/'\
                            'engl_um_{0:03d}/engl_surf_inc'.format(int(ENS_MEMBER))  # Breo's change
    # ENS_SOIL_EKF_FILEPATH = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20181201T1200Z/engl_smc/' \
    #                         'engl_smc_p{0:04d}'.format(int(ENS_MEMBER))  # before Breo's change
    TUNING_FACTOR = '1'
    OVERWRITE_PERT_FILES = False

# conversions to type:
ENS_MEMBER_INT = int(ENS_MEMBER)
TUNING_FACTOR_FLOAT = float(TUNING_FACTOR)

# quick check TUNING_FACTOR is not less than 0
if TUNING_FACTOR_FLOAT <= 0.0:
    raise ValueError('TUNING_FACTOR value of {0} is invalid. MUst be set >= 0.0'.format(TUNING_FACTOR_FLOAT))

# ------------------------------------

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

# ------------------------------------


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

# load and process breeding functions


def load_prev_ens_bpert_data():

    """
    Loads in previous cycle ensemble breeding perturbation data. Sorted into a dict structure by field, level to
    produce a list containing the data for ensemble members.
    E.g. {9: {1: [field1], 2: [field2], ...}} where 9 = STASH code and 1 and 2 are levels

    Also contains the land_sea mask (STASH = 30), required by the soil moisture content field
    :return: bpert_data_in (dictionary) breeding perturbation data
    :return: ff_file_in (object) bpert file
    """

    # load in the required data for each file:
    # for member in MEMBERS_PERT_INTS:

    # set up the returned data structure...
    bpert_data_in = {}
    for stash in STASH_TO_LOAD:
        if stash in MULTI_LEVEL_STASH:
            bpert_data_in[stash] = {}
        else:
            bpert_data_in[stash] = []

    # data file to open:
    ff_file_in = mule.load_umfile(ENS_SOIL_BPERT_FILEPATH)
    #ff_file_in = mule.DumpFile.from_file(ENS_SOIL_BPERT_FILEPATH)
    ff_file_in.remove_empty_lookups()

    # pull out the fields:
    for field in ff_file_in.fields:
        # is this SMC or land sea mask?
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


def load_scale_bpert():

    """
    Load and scale the breeding perturbations
    :return:
    """

    # load in the breeding perturbation file, for this member, from the last cycle
    # includes the land-sea mask
    bpert_fields, ff_file_in = load_prev_ens_bpert_data()
    landsea_field = bpert_fields[STASH_LAND_SEA_MASK]

    # Combined perturbation from breeding method (bpert) with the EKF perturbation, for each soil level.
    # Use TUNING_FACTOR to scale the amount of bpert to add (scaling applied equally to all levels).
    tuning_factor_scaling = mule.operators.ScaleFactorOperator(TUNING_FACTOR_FLOAT)

    bpert_scaled = {stash: {} for stash in STASH_TO_MAKE_PERTS}
    for stash in STASH_TO_MAKE_PERTS:
        for level in bpert_fields[stash].iterkeys():
            bpert_l = bpert_fields[stash][level]
            bpert_scaled[stash][level] = tuning_factor_scaling(bpert_l)

    return bpert_scaled, landsea_field


# load and process EKF functions


def load_soil_EFK_data(stash_and_constraints, cache=False):
    """
    Loads EKF soil data

    Returns the requested FieldsFile data, and the FieldsFile object
    :param stash_and_constraints: (dictionary) stash codes and additional constraints to load
    :param cache: (bool) To use mule cache functionality to assist loading
    :return: ff_data: (dictionary) soil data fields from current cycle
    :return: ff_obj: (object) the UM file read in
    """

    # identify file with existing EKF perturbation
    ff_file = ENS_SOIL_EKF_FILEPATH

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


def load_ekf_combine_with_bpert(bpert_scaled):

    """
    Load in EFK soil perturbations and combine with the already scaled breeding perturbations

    :return: soil_comb_pert: (dictionary) combined perturbations of the EKF and breeding scheme
    :return: bpert_landsea_field (field) land sea mask from the breeding perturbations file
    """

    # STASH codes to load in from the analysis dump:
    # stash codes as dict key, then a dict of constraints, if any:
    stash_from_dump = {STASH_SMC: None,
                       STASH_TSOIL: None,
                       STASH_LAND_SEA_MASK: None,
                       STASH_LANDFRAC: {'lbuser5': [PSEUDO_LEVEL_LANDICE]}}

    # load in the EKF and other
    soil_fields, smc_ff = load_soil_EFK_data(stash_from_dump, cache=True)

    # Combined perturbation from breeding method (bpert) with the EKF perturbation, for each soil level.
    adder = mule.operators.AddFieldsOperator(preserve_mdi=True)

    soil_total_pert = {stash: {} for stash in STASH_TO_MAKE_PERTS}
    for stash in STASH_TO_MAKE_PERTS:
        for level, bpert_l in bpert_scaled[stash].iteritems():
            ekf_pert_l = soil_fields[stash][level]
            soil_total_pert[stash][level] = adder([bpert_l, ekf_pert_l])

    return soil_total_pert

# save


def save_total_pert(soil_pert, landsea_field, template_file=ENS_SOIL_EKF_FILEPATH):

    """
    Save the total perturbation ready for the IAU to ingest. Use the EKF file as a template, if present,
    as this is already intended for the IAU and to help future-proof the process as the file may
    include more information in the future.

    :param soil_pert: (dictionary) total soil perturabtion
    :param landsea_field: (field) land sea mask from the breeding perturbation file
    :return:
    """

    def ensure_fields_file():
        # Set the dataset_type to 3 (fields file) encase the file type isn't already
        pert_ff_out.fixed_length_header.dataset_type = 3
        return

    def ensure_ancil_file():
        # If file is an ancillary (dataset_type = 4) and has level_dependent_constants as a header...
        # Remove the level_dependent_constants fixed header, as it should not exist in an ancillary file and
        # mule will not save file with it in.
        if (pert_ff_in.fixed_length_header.dataset_type == 4) & hasattr(pert_ff_out, 'level_dependent_constants'):
            pert_ff_out.level_dependent_constants = None
        return

    # Ideally use the existing EKF file as a template
    if os.path.exists(ENS_SOIL_EKF_FILEPATH):
        output_pert_file = ENS_SOIL_EKF_FILEPATH
        tmp_output_pert_file = output_pert_file + '_tmp.ff'
        pert_ff_in = mule.AncilFile.from_file(output_pert_file, remove_empty_lookups=True)
        pert_ff_out = pert_ff_in.copy(include_fields=False)  # empty copy
        # EKF file is an ancillary that can contain bad headers, so ensure it is suitable for saving as an ancillary.
        ensure_ancil_file()
    else:
        # Use the existing breeding perturbation file as a template
        output_pert_file = ENS_SOIL_BPERT_FILEPATH
        tmp_output_pert_file = output_pert_file + '_tmp.ff'
        pert_ff_in = mule.FieldsFile.from_file(output_pert_file, remove_empty_lookups=True)
        pert_ff_out = pert_ff_in.copy(include_fields=False)  # empty copy
        # breeding perturbation file is a dump file, so convert it to a fields file for saving
        ensure_fields_file()

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
        pert_ff_out.fields.append(landsea_field)

    # Now add in the perturbations:
    # Correct times to ensure validity time matches
    for stash in STASH_TO_MAKE_PERTS:
        if stash in MULTI_LEVEL_STASH:
            # write out each level (the order now matters!):
            for level in sorted(soil_pert[stash]):
                pert_ff_out.fields.append(soil_pert[stash][level])
        else:
            pert_ff_out.fields.append(soil_pert[stash])

    # pert_ff_out.validate to check validity
    pert_ff_out.to_file(tmp_output_pert_file)

    # if we are done, and this id being run 'for real'
    # then we want to overwrite the real pert file with the temporary one:
    if OVERWRITE_PERT_FILES:
        shutil.move(tmp_output_pert_file, output_pert_file)

    return


if __name__ == '__main__':

    # load the breeding perturbation data, EFK data and combine them.
    # Also load the land-sea mask for saving the data later.
    # soil_comb_pert, bpert_landsea_field = load_combine_ekf_bpert()

    # load and scale the breeding perturbation data based on the TUNING_FACTOR_INT value
    bpert_scaled, bpert_landsea_field = load_scale_bpert()

    # if the EKF file exists, read it in and combine with the scaled breeding perturbation, to create a total.
    # Else take the breeding perturbation data alone as the 'total'.
    if os.path.exists(ENS_SOIL_EKF_FILEPATH):
        total_pert = load_ekf_combine_with_bpert(bpert_scaled)
    else:
        total_pert = bpert_scaled

    # save total perturbations in the original perturbation file
    save_total_pert(total_pert, bpert_landsea_field)

    exit(0)


