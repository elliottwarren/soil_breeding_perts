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

# filepath with soil masks to mask out EKF + breeding perts in unsuitable areas (e.g. snow is present)
ENS_SOIL_MASK_FILEPATH = os.getenv('ENS_SOIL_MASK_FILEPATH')

# Diagnostic information and saving
DIAGNOSTICS = True

if ROSE_DATACPT6H is None:
    # if not set, then this is being run for development, so have canned variable settings to hand:
    #ROSE_DATACPT6H = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20181201T0600Z'  # 1-tile scheme
    #ROSE_DATAC = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20181201T1200Z'  # 1-tile scheme
    ROSE_DATACPT6H = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20190615T0600Z'  # 9-tile scheme
    ROSE_DATAC = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20190615T1200Z'  # 9-tile scheme
    ENS_MEMBER = '1'
    ENS_SOIL_CORR_FILEPATH = ROSE_DATACPT6H +'/engl_smc/engl_soil_correction'
    ENS_SOIL_EKF_FILEPATH = ROSE_DATAC+'/engl_smc/engl_surf_inc' # control member ETKF stochastic perts
    ENS_SOIL_MASK_FILEPATH = ROSE_DATACPT6H +'/engl_smc/engl_soil_correction_masks'
    DIAGNOSTICS = True

# ------------------------------------

# Configuration:
# STASH codes to use:
STASH_LAND_SEA_MASK = 30
STASH_SMC = 9
STASH_TSOIL = 20
STASH_TSKIN = 233  # a.k.a. surface temperature
STASH_TSNOW = 384
STASH_SNOW_AMNT = 23  # can be overwritten by other programs therefore not the ideal choice for use as a snow mask
STASH_NUM_SNOW_LAYERS = 380  # preferred alternative to STASH_SNOW_AMNT that does not get overwritten
# land use fractions:
STASH_LANDFRAC = 216
# pseudo level of land ice tile (this should remain unchanged at 9!)
PSEUDO_LEVEL_LANDICE = 9

# STASH codes to load:
STASH_TO_LOAD = [STASH_SMC, STASH_TSOIL, STASH_TSNOW, STASH_TSKIN,
                 STASH_LAND_SEA_MASK, STASH_NUM_SNOW_LAYERS, STASH_LANDFRAC]

# STASH codes to fields that are for masking.
STASH_MASKS_TO_LOAD = [STASH_TSOIL, STASH_NUM_SNOW_LAYERS]

# These are the multi soil level STASH codes
MULTI_SOIL_LEVEL_STASH = [STASH_SMC, STASH_TSOIL]

# These need to be all multi-level (not pseudo level) stash codes
# STASH_NUM_SNOW_LAYERS isn't in MULTI_LEVEL_STASH as the field used is simply a single 2D mask
#MULTI_LEVEL_STASH = [STASH_SMC, STASH_TSOIL]

# These are multi pseudo-level stash codes
# If a single aggregate tile is used for these stash codes, the field will simply be read in having a single
# pseudo-level.
MULTI_PSEUDO_LEVEL_STASH = [STASH_LANDFRAC, STASH_TSKIN, STASH_NUM_SNOW_LAYERS, STASH_TSNOW]

# Constraint to only read in a specific tile, e.g. just the ice tile for the land fraction
STASH_LEVEL_CONSTRAINTS = {STASH_LANDFRAC: [PSEUDO_LEVEL_LANDICE]}

# Multiple surface layers
MULTI_LAYER_STASH = [STASH_TSNOW]

# a list of stash codes we want to actually act on to produce perturbations in this routine:
STASH_TO_MAKE_PERTS = [STASH_SMC, STASH_TSOIL, STASH_TSNOW, STASH_TSKIN]


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


# load and process functions


def load_um_fields(filepath):

    """
    Loads in UM field data. Sorted into a dict structure  by field
    E.g. {9: {1: [field1], 2: [field2], ...}} where 9 = STASH code and 1 and 2 are levels

    :return: data_in (dictionary) correction data
    """

    def get_field_level(field):

        """Get the level that this field is on. If the field is not on a specific level, set it = 1."""

        # soil level
        if field.lbuser4 in MULTI_SOIL_LEVEL_STASH:
            level = field.lblev

        # for layer fields, use the layer as the level. If also on pseudo levels, lbuser5 has both layer and
        # pseduo-level, therefore need to isolate the surface layer.
        elif field.lbuser4 in MULTI_LAYER_STASH:
            # This will split up any multi layer stash level from the pseudo-level, if they have been merged.
            level = field.lbuser5 % 1000  # % finds the remainder and works even if lbuser <= 1000

        # special instance for snow layers. There are two sets of snow layer field masks, and lbuser6 was
        # set differently for each to tell them apart. Use it as the level to keep them separate.
        elif field.lbuser4 == STASH_NUM_SNOW_LAYERS:
            level = field.lbuser6

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

    # set up the returned data structure...
    data_in = {}
    #for stash in STASH_TO_LOAD:
    #if stash in MULTI_LEVEL_STASH:
    #data_in[stash] = {}
    #else:
    #data_in[stash] = []

    # data file to open:
    ff_file_in = mule.load_umfile(filepath)
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

                # Add STASH entry if it doesn't exist yet
                if field.lbuser4 not in data_in.keys():
                    data_in[field.lbuser4] = {}

                # Get field level and pseudo-level to help store them in a dictionary
                # if field doesn't have one of them, it shall be set equal to 1 and used to ensure the number of
                # nested dictionaries and lists is the same across all the loaded variables.
                level = get_field_level(field)

                pseudo_level = get_field_pseudo_level(field)

                # load...
                # load in the fields
                if level in data_in[field.lbuser4].keys():  # is level already present
                    data_in[field.lbuser4][level][pseudo_level] = field
                else:
                    data_in[field.lbuser4][level] = {pseudo_level: field}

    return data_in, ff_file_in


def load_ekf_combine_with_correction(corr_data):

    """
    Load in EFK soil perturbations and take away the correction needed to centre the overall ensemble mean for this
    cycle

    :return: soil_centred_pert: (dictionary) combined perturbations of the EKF and correction
    """

    # def load_field_data(stash_and_constraints, ff_file, cache=False):
    #     """
    #     Loads EKF soil data
    #
    #     :param stash_and_constraints: (dictionary) stash codes and additional constraints to load
    #     :param: ff_file: (filepath str) filepath of the data to load
    #     :param cache: (bool) To use mule cache functionality to assist loading
    #     :return: ff_data: (dictionary) soil data fields from current cycle
    #     :return: ff_obj: (object) the UM file read in
    #     """
    #
    #     if isinstance(ff_file, str):
    #         ff_obj = mule.load_umfile(ff_file)
    #     else:
    #         ff_obj = ff_file
    #     ff_obj.remove_empty_lookups()
    #
    #     ff_data = {}
    #     for field in ff_obj.fields:
    #         # is this stash code we want?
    #         if field.lbuser4 in stash_and_constraints.keys():
    #             # passes the first filter, some of these fields are needed:
    #             if field.lbuser5 == 0:
    #                 # not a psudolevel field, so don't test them
    #                 fld_needed = True
    #             else:
    #                 # need to test against pseudo levels:
    #                 fld_needed = False
    #                 # now check the individual reqeusts to see if any match:
    #                 if 'lbuser5' in stash_and_constraints[field.lbuser4]:
    #                     fld_needed = field.lbuser5 in stash_and_constraints[field.lbuser4]['lbuser5']
    #                 else:
    #                     # not constraining on pseudo level (lbuser5)
    #                     fld_needed = True
    #
    #             if fld_needed:
    #                 if cache:
    #                     # apply the cache, if required:
    #                     cacheoperator = CachingOperator()
    #                     field = cacheoperator(field)
    #                 if field.lbuser5 == 0:
    #                     # multi-level fields are a dict, with a list for each level
    #                     if field.lbuser4 in MULTI_LEVEL_STASH:
    #                         if field.lbuser4 not in ff_data.keys():
    #                             ff_data[field.lbuser4] = {}
    #                         ff_data[field.lbuser4][field.lblev] = field
    #                     else:
    #                         # single level fields are a flat list:
    #                         ff_data[field.lbuser4] = field
    #                 else:
    #                     if field.lbuser4 in MULTI_LEVEL_STASH:
    #                         raise NotImplementedError('Cannot have multi-level fields with '
    #                                                   'psuedo-levels are not in this script yet')
    #                     else:
    #                         if field.lbuser4 not in ff_data.keys():
    #                             ff_data[field.lbuser4] = {}
    #                         # now this is like a multi-level field:
    #                         ff_data[field.lbuser4][field.lbuser5] = field
    #
    #     return ff_data, ff_obj

    # STASH codes to load in from the EKF file
    # include all perts first, then update dict with land_sea_mask stash and land fraction
    stash_from_dump = {stash: None for stash in STASH_TO_MAKE_PERTS}
    stash_from_dump.update({STASH_LAND_SEA_MASK: None,
                       STASH_LANDFRAC: {'lbuser5': [PSEUDO_LEVEL_LANDICE]}})

    # load in the EKF and other
    # soil_fields, smc_ff = load_field_data(stash_from_dump, ENS_SOIL_EKF_FILEPATH, cache=True)
    soil_fields, smc_ff = load_um_fields(ENS_SOIL_EKF_FILEPATH)


    # reduce EKF by the correction (now that EKF is done via IAU).
    subber = mule.operators.SubtractFieldsOperator(preserve_mdi=True)

    soil_centred_pert = {}
    for stash in STASH_TO_MAKE_PERTS:
        soil_centred_pert[stash] = {}
        for level in corr_data[stash].keys():
            soil_centred_pert[stash][level] = {}
            for pseudo_level, corr_l in corr_data[stash][level].items():
                ekf_pert_l = soil_fields[stash][level][pseudo_level]
                soil_centred_pert[stash][level][pseudo_level] = subber([ekf_pert_l, corr_l])

    # put land-sea mask in the soil_centred_pert dictionary
    soil_centred_pert[STASH_LAND_SEA_MASK] = corr_data[STASH_LAND_SEA_MASK]

    return soil_centred_pert


def apply_masks_to_perts(mask_data, total_pert):

    """
    # Apply field masking to the combined perturbations (fields applied to in brackets). This includes:
    # 1) Snow masks:
    #    1a) Where snow is present in any ensemble member (SMC, TSOIL, TSKIN)
    #    1b) Where number of snow tiles is different between any of the members for each cell (all pert variables)
    # 2) TSOIL min threshold mask: For any grid cell where TSOIL < -10 degC and should be frozen
    # 3) Limit the total (perturbation + increment) to not be beyond sensible limits (SURF background error * factor)

    STASH code for the combined mask and the perts will match e.g. combined mask for STASH=9, level=1 will be in
    [mask_data][9][1]. The masks were created in the previous cycle and are reapplied here to the EKF increments.
    :param mask_data: (dictionary) contains the mask fields - fields are [0.0, 1.0, bmdi] and will need changing to bool
    :param total_pert: (EKF - correction) perturbations that need to be set to 0.0 where snow is present.
    :return: total_pert: total_pert but now with its data masked appropriately.

    """

    def extract_mask(mask_field):

        """
        Extract mask from field, and convert it from an array of floats of [0.0, 1.0, bmdi] to a boolean array
        :param mask_field: field with an array of floats as its data
        :return: mask: (array of booleans)
        """

        # Extract out mask data from field and turn it into a boolean array
        mask_data = mask_field.get_data()
        # Convert array of floats [1.0, 0.0, bmdi] into boolean mask array
        mask = np.logical_and(mask_data == 1.0, mask_data != mask_field.bmdi)

        return mask

    def apply_mask(field, mask, replacement_value):

        """
        Apply the mask to the data
        :param field: (field) field to partially mask
        :param mask: (numpy array with boolean values, same shape as field.get_data()) True for where to mask
        :return:
        """

        # Extract data and set values to 0.0 where there is ice
        tmp_pert_data = field.get_data()

        # Print how many values will be masked that have not been already.
        if DIAGNOSTICS:
            legit_perts = np.logical_and(tmp_pert_data != field.bmdi, tmp_pert_data != 0.0)
            masked = np.sum(np.logical_and(legit_perts, mask))
            print('STASH: {}; lblev: {}; lbuser5 {}; Additional number of perturbation values masked: {}'.format(field.lbuser4, field.lblev, field.lbuser5, masked))

        # mask the data
        tmp_pert_data[mask] = replacement_value

        # now put that data back into the field:
        array_provider = mule.ArrayDataProvider(tmp_pert_data)
        field.set_data_provider(array_provider)

        return field

    def apply_ice_snow_masks(mask_data, total_pert):

        """
        Apply the snow mask to the total perturbation data. A single mask that is applied to all levels.
        :param mask_data: (dict)
        :param total_pert: (dict)
        :return: total_pert: (dict)
        """

        # 1. Number of snow layers differed between ensemble members
        # Applied to all perturbation variables on each level and pseudo-level
        print('   Differing number snow layers:')
        for stash in STASH_TO_MAKE_PERTS:
            for level in total_pert[stash]:
                for pseudo_level in total_pert[stash][level]:

                    # level 1 in <mask_data> is the "any layers differ" mask
                    snow_mask_diff = extract_mask(mask_data[STASH_NUM_SNOW_LAYERS][1][pseudo_level])
                    total_pert[stash][level][pseudo_level] = apply_mask(total_pert[stash][level][pseudo_level], snow_mask_diff, 0.0)

        # 2. Number of snow layers differed between ensemble members
        # 2nd mask is applied to all perturbation variables except TSNOW where it's ok to have snow on the tiles.
        print('   Any snow layers present:')
        for stash in STASH_TO_MAKE_PERTS:
            if stash != STASH_TSNOW:
                for level in total_pert[stash]:
                    for pseudo_level in total_pert[stash][level]:

                        # level 2 in <mask_data> is the "one or more layers present"
                        snow_mask_any = extract_mask(mask_data[STASH_NUM_SNOW_LAYERS][2][pseudo_level])
                        total_pert[stash][level][pseudo_level] = apply_mask(total_pert[stash][level][pseudo_level], snow_mask_any, 0.0)

        # 3. Apply the land-ice mask
        # Applied to all perturbation variables on each level and pseudo-level
        print('   Land-ice:')
        for stash in STASH_TO_MAKE_PERTS:
            for level in total_pert[stash]:
                for pseudo_level in total_pert[stash][level]:
                    ice_mask = extract_mask(mask_data[STASH_LANDFRAC][1][PSEUDO_LEVEL_LANDICE])
                    total_pert[stash][level][pseudo_level] = apply_mask(total_pert[stash][level][pseudo_level], ice_mask, 0.0)

        return total_pert

    def zero_perts_lt_m10degc(mask_data, total_pert):

        """
        Extract and apply the combined masks, onto each total perturbation. The combined mask was created in the
        previous cycle and is True where:
        1) Any grid cell where TSOIL < -10 degC

        Applied to SMC only!
        :param mask_data: (dict)
        :param total_pert:
        :return: total_pert:
        """

        for level in total_pert[STASH_SMC]:
            for pseudo_level in total_pert[STASH_SMC][level]:

                # Extract out the combined mask (STASH and level specific)
                tsoil_mask = extract_mask(mask_data[STASH_TSOIL][level][pseudo_level])

                # Apply the mask from each level to the total pert on the same level
                total_pert[STASH_SMC][level][pseudo_level] = apply_mask(total_pert[stash][level][pseudo_level], tsoil_mask, 0.0)

        return total_pert

    def cap_perts_gt_bgerr(mask_data, total_pert):

        """
        Cap perturbations that are greater than the back ground error in SURF * factor, where factor is also taken
        from SURF

        Functions for single or multi-level fields
        """

        # Background errors and factor for pert variables from SURF
        stash_errors = {STASH_TSOIL: 2.0,  # [K]
               STASH_SMC: [0.03] + [0.026]*(len(corr_data[STASH_SMC].keys())-1),  # [m3 m-3] All but first soil level is 0.026
               STASH_TSNOW: 2.0,  # [K]
               STASH_TSKIN: 2.3}  # [K]
        err_factor = 3.0  # values are capped beyond (err * err_factor)

        # store the masks
        mask_max_tol = {}

        # Find where totals are beyond acceptable tolerances (stash_errors * factor)
        for stash in STASH_TO_MAKE_PERTS:
            mask_max_tol[stash] = {}

            for level in total_pert[stash]:
                mask_max_tol[stash][level] = {}

                # accepted tolerance for this stash and level
                if stash != STASH_SMC:
                    tolerance = stash_errors[stash] * err_factor
                else:
                    tolerance = stash_errors[stash][level] * err_factor

                for pseudo_level, field in total_pert[stash][level].items():

                    field_data = field.get_data()

                    # 1. Find where positive totals are greater than the allowed tolerance
                    pos_mask = np.logical_and(field_data > tolerance, field_data != field.bmdi)

                    # Apply the mask from each level to the total pert on the same level.
                    # Cap values to the maximum tolerance instead of zero
                    total_pert[STASH_SMC][level][pseudo_level] = apply_mask(total_pert[stash][level][pseudo_level],
                                                                            pos_mask, tolerance)

                    # 2. Find where negative totals are greater than the total
                    neg_mask = np.logical_and(field_data < -tolerance, field_data != field.bmdi)

                    # Apply the mask from each level to the total pert on the same level.
                    # Cap values to the maximum tolerance instead of zero
                    total_pert[STASH_SMC][level][pseudo_level] = apply_mask(total_pert[stash][level][pseudo_level],
                                                                            neg_mask, -tolerance)

                    # 3. Combine positive and negative masks together for diagnostic output
                    # masks should not overlap, therefore their sum should identify:
                    # a) positive caps (tolerance), b) negative caps (-tolerance), c) no cap (0.0)
                    mask_max_tol[stash][level][pseudo_level] = (pos_mask * tolerance) - (neg_mask * tolerance)


        return total_pert, mask_max_tol

    print('Making perturbations:')

    # 1. Apply the ice and snow masking
    print('Ice and snow masking:')
    total_pert = apply_ice_snow_masks(mask_data, total_pert)

    # 2. Apply the combined mask (True when TSOIL < -10 degC and absolute pert > 1 standard deviation of original field.
    print('Combined masking (TSOIL < -10degC or abs(pert) > 1 standard deviation of field):')
    total_pert = zero_perts_lt_m10degc(mask_data, total_pert)

    # 3. Cap the maximum combined perturbation + increment allowed based on SURF background error tolerance * factor
    # Output the mask (created in this app unlike the other masks) and save for later diagnostics
    total_pert, mask_max_tol = cap_perts_gt_bgerr(mask_data, total_pert)

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

    return total_pert


# Save


def save_total_pert(centred_pert, template_file=ENS_SOIL_EKF_FILEPATH):

    """
    Save the total perturbation ready for the IAU to ingest. Use the EKF file as a template,
    as this is already intended for the IAU and to help future-proof the process as the file may
    include more information in the future.

    :param centred_pert: (dictionary) EKF - correction. The increment to apply, to center the ensemble mean
    :keyword: template_file: (str) template file to cmake a copy and reate a new file from.
    """

    # Ideally use the existing EKF file as a template
    pert_ff_in = mule.AncilFile.from_file(template_file)
    pert_ff_out = pert_ff_in.copy(include_fields=False)  # empty copy
    # name the output file for EKF - correction
    output_pert_file = ROSE_DATAC + '/engl_smc/engl_surf_inc_correction'

    os.system('echo File being saved using '+template_file+' as a template')

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

    # Output to file
    pert_ff_out.to_file(output_pert_file)

    print('Saved: ' + output_pert_file)

    return


if __name__ == '__main__':

    """
    Routine 2 of 2 for applying the ensemble breeding corrections, to the EKF perturbation.
    
    1) Loads correction data from last cycle (correction = ensemble mean - control field)
    2) Loads EKF data and applies correction (total perturbation = EKF - correction)
    3) Loads mask data created last cycle for zeroing unsuitable perturbations
    4) Apply masks to total perturbations
    5) Save the total perturbations
    """

    ## Read and process
    # Load the ensemble correction data (ensemble mean - control field, both from previous cycle).
    corr_data, _ = load_um_fields(ENS_SOIL_CORR_FILEPATH)

    # If the EKF file exists, read it in and combine with the ensemble correction.
    # NOTE: file will not exist during a fast run cycle
    if os.path.exists(ENS_SOIL_EKF_FILEPATH):
        total_pert = load_ekf_combine_with_correction(corr_data)
    else:
        raise ValueError(ENS_SOIL_EKF_FILEPATH +' is missing!')

    # Load in soil masks created from the previous cycle (used to mask out EKF perts created in this cycle).
    # File doesn't include land fraction, but its not necessary
    mask_data, _ = load_um_fields(ENS_SOIL_MASK_FILEPATH)

    # Apply field masking to the combined perturbations (fields applied to in brackets). This includes:
    # 1) Snow masks:
    #    1a) Where snow is present in any ensemble member (SMC, TSOIL, TSKIN)
    #    1b) Where number of snow tiles is different between any of the members for each cell (all pert variables)
    # 2) TSOIL min threshold mask: For any grid cell where TSOIL < -10 degC and should be frozen
    # 3) Limit the total (perturbation + increment) to not be beyond sensible limits (SURF background error * factor)
    total_pert = apply_masks_to_perts(mask_data, total_pert)

    ## Save
    # Save total perturbations in a new perturbation file
    save_total_pert(total_pert)

    exit(0)


