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
    ROSE_DATACPT6H = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20181201T0600Z'  # 1-tile scheme
    ROSE_DATAC = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20181201T1200Z'  # 1-tile scheme
    #ROSE_DATACPT6H = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20190615T0600Z'  # 9-tile scheme
    #ROSE_DATAC = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/data/20190615T1200Z'  # 9-tile scheme
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
STASH_SNOW_AMNT = 23  # can be overwritten by other programs therefore not the ideal choice for use as a snow mask
STASH_NUM_SNOW_LAYERS = 380  # preferred alternative to STASH_SNOW_AMNT that does not get overwritten
# land use fractions:
STASH_LANDFRAC = 216
# pseudo level of land ice tile (this should remain unchanged at 9!)
PSEUDO_LEVEL_LANDICE = 9

# STASH codes to load and mean:
STASH_TO_LOAD = [STASH_SMC, STASH_TSOIL, STASH_LAND_SEA_MASK, STASH_NUM_SNOW_LAYERS, STASH_LANDFRAC]

# STASH codes to fields that are for masking.
STASH_MASKS_TO_LOAD = [STASH_SMC, STASH_TSOIL, STASH_NUM_SNOW_LAYERS]

# These need to be all multi-level (not pseudo level) stash codes
# STASH_NUM_SNOW_LAYERS isn't in MULTI_LEVEL_STASH as the field used is simply a single 2D mask
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


# load and process functions


def load_um_fields(filepath):

    """
    Loads in UM field data. Sorted into a dict structure  by field
    E.g. {9: {1: [field1], 2: [field2], ...}} where 9 = STASH code and 1 and 2 are levels

    :return: data_in (dictionary) correction data
    """

    # set up the returned data structure...
    data_in = {}
    for stash in STASH_TO_LOAD:
        if stash in MULTI_LEVEL_STASH:
            data_in[stash] = {}
        else:
            data_in[stash] = []

    # data file to open:
    ff_file_in = mule.load_umfile(filepath)
    ff_file_in.remove_empty_lookups()

    # pull out the fields:
    for field in ff_file_in.fields:
        if field.lbuser4 in STASH_TO_LOAD:
            if field.lbuser4 in MULTI_LEVEL_STASH:
                # multi-level fields are a dict, with a list for each level
                if field.lblev in data_in[field.lbuser4].keys():
                    data_in[field.lbuser4][field.lblev] = field
                else:
                    data_in[field.lbuser4][field.lblev] = field
            else:
                # single level fields are a flat list:
                data_in[field.lbuser4] = field

    return data_in


def load_ekf_combine_with_correction(corr_data):

    """
    Load in EFK soil perturbations and take away the correction needed to centre the overall ensemble mean for this
    cycle

    :return: soil_centred_pert: (dictionary) combined perturbations of the EKF and correction
    """

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

    # STASH codes to load in from the EKF file
    # include all perts first, then update dict with land_sea_mask stash and land fraction
    stash_from_dump = {stash: None for stash in STASH_TO_MAKE_PERTS}
    stash_from_dump.update({STASH_LAND_SEA_MASK: None,
                       STASH_LANDFRAC: {'lbuser5': [PSEUDO_LEVEL_LANDICE]}})

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


def apply_masks_to_perts(mask_data, total_pert):

    """
    Set the perturbations to zero where the masks apply. This includes:
    1) Snow mask: For any grid cell where snow is present in any ensemble member on any tile
    2) A COMBINED mask created specifically for each STASH where:
        2a) TSOIL min threshold mask: For any grid cell where TSOIL < -10 degC and should be frozen
        2b) standard deviation mask: For any grid cell where the absolute perturbations are greater than 1 standard
        deviation of the original field.
    STASH code for the combined mask and the perts will match e.g. combined mask for STASH=9, level=1 will be in
    [mask_data][9][1]. The masks were created in the previous cycle and are reapplied here to also zero the EKF perts.
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

    def apply_mask(field, mask):

        """
        Apply the mask to the data
        :param field: (field) field to partially mask
        :param mask: (numpy array with boolean values, same shape as field.get_data()) True for where to mask
        :return:
        """

        # check mask is an array of booleans and not of floats by checking the first element of the array
        if mask.flatten()[0].dtype != bool:
            raise ValueError('Mask for STASH {}, lblev {} needs to contain boolean values, not {}'.format(field.lbuser4, field.lblev, mask.flatten()[0].dtype))

        # Extract data and set values to 0.0 where there is ice
        tmp_pert_data = field.get_data()

        # Print how many values will be masked that have not been already.
        if DIAGNOSTICS:
            legit_perts = np.logical_and(tmp_pert_data != field.bmdi, tmp_pert_data != 0.0)
            masked = np.sum(np.logical_and(legit_perts, mask))
            print('STASH: {}; lblev: {}; Additional number of perturbation values masked: {}'.format(field.lbuser4, field.lblev, masked))

        # mask the data
        tmp_pert_data[mask] = 0.0

        # now put that data back into the field:
        array_provider = mule.ArrayDataProvider(tmp_pert_data)
        field.set_data_provider(array_provider)

        return field

    def apply_snow_mask(mask_data, total_pert):

        """
        Apply the snow mask to the total perturbation data. A single mask that is applied to all levels.
        :param mask_data: (dict)
        :param total_pert: (dict)
        :return: total_pert: (dict)
        """

        # Extract out snow mask
        snow_mask = extract_mask(mask_data[STASH_NUM_SNOW_LAYERS])

        # Loop through perts and mask where appropriate
        for stash in STASH_TO_MAKE_PERTS:
            if stash in MULTI_LEVEL_STASH:
                for level in total_pert[stash]:
                    total_pert[stash][level] = apply_mask(total_pert[stash][level], snow_mask)
            else:
                total_pert[stash] = apply_mask(total_pert[stash], snow_mask)

        return total_pert

    def apply_combined_masks(mask_data, total_pert):

        """
        Extract and apply the combined masks, onto each total perturbation. The combined mask was created in the
        previous cycle and is True where:
        1) Any grid cell where TSOIL < -10 degC and should be frozen
        2) Any grid cell where the absolute perturbations are greater than 1 standard deviation of the original field.
        :param mask_data: (dict)
        :param total_pert:
        :return: total_pert:
        """

        # Loop through perts and masks where appropriate
        for stash in STASH_TO_MAKE_PERTS:
            for level in total_pert[stash]:

                # Extract out the combined mask (STASH and level specific)
                comb_mask = extract_mask(mask_data[stash][level])

                if stash in MULTI_LEVEL_STASH:
                    # Apply the mask from each level to the total pert on the same level
                    total_pert[stash][level] = apply_mask(total_pert[stash][level], comb_mask)
                else:
                    # Apply all the masks in turn onto the singular total pert field
                    total_pert[stash] = apply_mask(total_pert[stash], comb_mask)

        return total_pert

    print('Making perturbations:')

    # Apply the snow mask (True where snow was present on any member, on any tile)
    print('Ice and snow masking:')
    total_pert = apply_snow_mask(mask_data, total_pert)

    # Apply the combined mask (True when TSOIL < -10 degC and absolute pert > 1 standard deviation of original field.
    print('Combined masking (TSOIL < -10degC or abs(pert) > 1 standard deviation of field):')
    total_pert = apply_combined_masks(mask_data, total_pert)

    # print out additional diagnostics (descriptive statistics of fields)
    if DIAGNOSTICS:
        print('\nPerturbation descriptive statistics:')
        for stash in STASH_TO_MAKE_PERTS:
            if stash in MULTI_LEVEL_STASH:
                for (level, pert_field) in total_pert[stash].items():
                    data = pert_field.get_data()
                    data_flat = data[np.where(data != corr_data[stash][level].bmdi)].flatten()
                    # print min, max , rms
                    #print('STASH:' + str(pert_field.lbuser4) + '; ' + 'lblev:' + str(pert_field.lblev)+':')
                    print('STASH: {}; lblev: {}:'.format(pert_field.lbuser4, pert_field.lblev))
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
    Routine 2 of 2 for applying the ensemble soil moisture content (SMC) correction, to 
    the EKF perturbation.
    
    1) Loads correction data from last cycle (correction = ensemble mean - control field)
    2) Loads EKF data and applies correction (total perturbation = EKF - correction)
    3) Loads mask data created last cycle for zeroing unsuitable perturbations
    4) Apply masks to total perturbations
    5) Save the total perturbations
    """

    ## Read and process
    # Load the ensemble correction data (ensemble mean - control field, both from previous cycle).
    corr_data = load_um_fields(ENS_SOIL_CORR_FILEPATH)

    # If the EKF file exists, read it in and combine with the ensemble correction.
    # NOTE: file will not exist during a fast run cycle
    if os.path.exists(ENS_SOIL_EKF_FILEPATH):
        total_pert = load_ekf_combine_with_correction(corr_data)
    else:
        raise ValueError(ENS_SOIL_EKF_FILEPATH +' is missing!')

    # Load in soil masks created from the previous cycle (used to mask out EKF perts created in this cycle).
    # File doesn't include land fraction, but its not necessary
    mask_data = load_um_fields(ENS_SOIL_MASK_FILEPATH)

    # Apply field masking to the combined perturbations. This includes:
    # 1) Snow mask: For any grid cell where snow is present in any ensemble member on any tile
    # 2) A COMBINED mask created specifically for each STASH where:
    #     2a) TSOIL min threshold mask: For any grid cell where TSOIL < -10 degC and should be frozen
    #     2b) standard deviation mask: For any grid cell where the absolute perturbations are greater than 1 standard
    #     deviation of the original field.
    total_pert = apply_masks_to_perts(mask_data, total_pert)

    ## Save
    # Save total perturbations in a new perturbation file
    save_total_pert(total_pert)

    exit(0)


