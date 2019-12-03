#!/usr/bin/env python2

"""
Load in breeding perturbations, calculated from the previous cycle and valid for t+3 there,
and add them to EKF perturbations of current cycle. The combined output should be valid for t-3 of this cycle.


Created by Elliott Warren Thurs 21th Nov 2019: elliott.warren@metoffice.gov.uk
Based on engl_ens_smc_pert.py by Malcolm Brooks 18th Sept 2016: Malcolm.E.Brooks@metoffice.gov.uk

Tested versions using canned data:
python 2.7.16
mule 2019.01.1
numpy 1.16.5

Testing (including soil temperature) carried out in:
/data/users/ewarren/R2O_projects/soil_moisture_pertubation/
"""

import os, shutil
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

def engl_prev_cycle_bpert_filename(ENS_MEMBER_INT):
    """ locates breeding pertubations calulated from t+6 of the last cycle"""
    return '{0}/engl_smc_bpert{1}'.format(ENS_PERT_DIR, mem_to_str(ENS_MEMBER_INT))


def engl_cycle_smc_filename(ENS_MEMBER_INT):
    """ locates soil moisture content data for this current cycle"""
    return '{0}/engl_surf_inc'.format(ENS_SMC_DIR, mem_to_str(ENS_MEMBER_INT))


def engl_snow_analysis_filename():
    """ locates snow data for this current cycle"""
    return '{0}/engl_um/umglaa_dz003'.format(ROSE_DATAC)
    # return '{0}/engl_um/engl_um_000/englaa_da006'.format(THIS_CYCLE_DIR)


def engl_cycle_saturation_filename():
    """ locates soil moisture saturation limit data for this current cycle"""
    return '{0}/engl_um/umglaa_dz003'.format(ROSE_DATAC)


# loading functions

def load_soil_data(stash_and_constraints, cache=False):
    """
    Loads required data from a single file, supplied as a file name to load
    or a ff object, into a simple dictionary.

    Returns the requested FieldsFile data, and the FieldsFile object
    :param stash_and_constraints: (dictionary) stash codes and additional constraints to load
    :param cache: (bool) use mule cache functionality to assist loading
    :return: ff_data: (dictionary) soil data fields from current cycle
    :return: ff_obj:
    """

    # identify engl_smc file to load, containing soil moisture and temperature with existing EKF perturbation
    ff_file = engl_cycle_smc_filename(ENS_MEMBER_INT)

    if isinstance(ff_file, str):
        ff_obj = mule.load_umfile(ff_file)
        # if ff_obj.fixed_length_header.dataset_type not in (1, 2, 3):
        #     raise ValueError("wrong file type")
        ff_obj.remove_empty_lookups()
    else:
        ff_obj = ff_file
        ff_obj.remove_empty_lookups()

    if cache:
        cacheoperator = CachingOperator()
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
    '''
    Loads in previous cycle ensemble breeding perturbation data. Sorted into a dict structure by field, level to
    produce a list containing the data for ensemble members.
    E.g. {9: {1: [field1], 2: [field2], ...}} where 9 = STASH code and 1 and 2 are levels

    Also contains the land_sea mask (STASH = 30), required by the soil moisture content field
    '''

    # load in the required data for each file:
    # for member in MEMBERS_PERT_INTS:

    # set up the returned data structure...
    bpert_data_in = {}
    for stash in STASH_TO_LOAD:
        if stash in MULTI_LEVEL_STASH:
            bpert_data_in[stash] = {}
        else:
            bpert_data_in[stash] = []

    # file with SMC data (after t+3)
    ens_smc_bpert_file = engl_prev_cycle_bpert_filename(ENS_MEMBER_INT)
    # print('reading in file: %s' % smc_src_file)
    # data file to open:
    # TIMER.start('opening file')
    # ff_file_in = mule.load_umfile(smc_src_file)
    ff_file_in = mule.DumpFile.from_file(ens_smc_bpert_file)
    #     raise ValueError("wrong file type")
    ff_file_in.remove_empty_lookups()
    # TIMER.end('opening file')

    # pull out the fields:
    # TIMER.start('checking/extracting input data')
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
    # TIMER.end('checking/extracting input data')

    return bpert_data_in, ff_file_in


def load_saturation_limit():
    '''
    Loads in saturation limit using a temporary work around. Load in from dz003 (reconfiguration for the deterministic
    ensemble member)
    '''

    # load in the required data for each file:
    # for member in MEMBERS_PERT_INTS:

    # set up the returned data structure...
    data_in = {}
    data_in[STASH_VSMC_SAT] = []

    # get filename containing saturation limit
    sat_limit_filename = engl_cycle_saturation_filename()

    ff_file_in = mule.DumpFile.from_file(sat_limit_filename)
    #     raise ValueError("wrong file type")
    ff_file_in.remove_empty_lookups()
    # TIMER.end('opening file')

    # pull out the fields:
    # TIMER.start('checking/extracting input data')
    for field in ff_file_in.fields:
        # is this the SMC?
        if field.lbuser4 in [STASH_VSMC_SAT]:
            # single level fields are a flat list:
            data_in[field.lbuser4] = field
    # TIMER.end('checking/extracting input data')

    return data_in, ff_file_in


def load_snow():
    '''
    Loads in previous cycle ensemble breeding perturbation data. Sorted into a dict structure by field, level to
    produce a list containing the data for ensemble members.
    E.g. {9: {1: [field1], 2: [field2], ...}} where 9 = stash code and 1 and 2 are levels
    '''

    # load in the required data for each file:
    # for member in MEMBERS_PERT_INTS:

    # set up the returned data structure...
    data_in = {}
    data_in[STASH_SNOW_AMNT] = []

    # file with SMC data (after t+3)
    snow_filename = engl_snow_analysis_filename()
    # print('reading in file: %s' % smc_src_file)
    # data file to open:
    # TIMER.start('opening file')
    # ff_file_in = mule.load_umfile(smc_src_file)
    ff_file_in = mule.DumpFile.from_file(snow_filename)
    #     raise ValueError("wrong file type")
    ff_file_in.remove_empty_lookups()
    # TIMER.end('opening file')

    # pull out the fields:
    # TIMER.start('checking/extracting input data')
    for field in ff_file_in.fields:
        # is this the SMC?
        if field.lbuser4 in [STASH_SNOW_AMNT, STASH_LAND_SEA_MASK]:
            # single level fields are a flat list:
            data_in[field.lbuser4] = field
    # TIMER.end('checking/extracting input data')

    return data_in, ff_file_in


def load_ekf_bpert():

    '''
    Checks that the differences (first guess perturbations) from the
    previous ensemble run do not cause problems when applied to the initial
    conditions of the next ensemble run.

    Calls smc_check and tsoil_check on each member to ensure that the
    perturbations are valid.

    smc_check uses a snow_or_ice mask to remove soil moisture perturbations
    under snow or ice. This uses the land fraction (ice tile) being 1.0 for the land ice
    (Land ice is currently either 0.0 or 1.0), and snow amounts > SNOW_AMOUNT_THRESH
    which is currently 0.05.

    After checking, the perturbation is written out onto a copy of the current perturbation
    file, which already contains the atmospheric perturbations etc.

    If this completes OK, then that copy is moved over the original if
    OVERWRITE_PERT_FILES is True
    '''

    def extract_check_dz_soil_levels(soil_fields, smc_ff):

        """
        Extract the delta z soil levels from the soil field, and do a quick check
        :param soil_fields: (dictionary) soil fields from smc file
        :param smc_ff: (field) the smc fields file
        :return: DZ_SOIL_LEVELS: (list) soil delta z levels
        """

        DZ_SOIL_LEVELS = {}
        for level in soil_fields[STASH_SMC].keys():
            DZ_SOIL_LEVELS[level] = soil_fields[STASH_SMC][level].blev
            # now check it's not missing data:
            if DZ_SOIL_LEVELS[level] == smc_ff.level_dependent_constants.MDI:
                raise ValueError('Input dump field contains missing data for level {}'.format(level))

        return DZ_SOIL_LEVELS

    def derive_landice_mask(landice_field):

        """
        Derive numpy array land ice mask, using the land ice field.
        :param landice_field: mule field of land ice.
        :return: landice_mask: Numpy array of bools (True = ice and on land)
        """

        # create landice mask (1.0 = ice, 0 or bdmi (missing data) are not ice)
        # from that dump data, a few masks are applied to all members:
        landice_data = landice_field.get_data()
        # at the moment, landice fractions are defined as being 0.0 or 1.0, or missing data.
        # we use this to find land ice points, so this code will need revisiting if that
        # ever changes:
        valid_icefrac_vals = np.array([landice_field.bmdi, 0.0, 1.0])
        # now chek the land_ice fraction data:
        land_ice_unique = np.sort(np.unique(landice_data))
        if len(land_ice_unique) != 3:
            raise ValueError('Land ice fractions are not 0.0, 1.0 (or bmdi)')
        if not np.all(land_ice_unique == valid_icefrac_vals):
            raise ValueError('Land ice fractions are not 0.0, 1.0 (or bmdi)')
        # now check the land ice:
        landice_mask = np.logical_and(landice_data == 1.0,
                                      landice_data != landice_field.bmdi)

        return landice_mask

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
    bpert_landsea_field = bpert_fields[STASH_LAND_SEA_MASK]

    # now set the DZ_SOIL_LEVELS from the field headers for each level (blev for soil = soil depth)
    # Assume the same soil depths for soil moisture content and soil temperature
    # DZ_SOIL_LEVELS = extract_check_dz_soil_levels(soil_fields, smc_ff)

    # Derive the land ice mask numpy array (True = ice and on land).
    #landice_field = soil_fields[STASH_LANDFRAC][PSEUDO_LEVEL_LANDICE]
    #landice_mask = derive_landice_mask(landice_field)

    # Derive snow mask (is snow present and above critical threshold)
    # snow_data = snow_field.get_data()
    # snow_mask = np.logical_and(snow_data > SNOW_AMOUNT_THRESH,
    #                           snow_data != snow_field.bmdi)

    # now combine snow and ice mask:
    #snow_or_ice_mask = np.logical_or(landice_mask, snow_mask)

    # Combined perturbation from breeding method (bpert) with the EKF pertubation, for each soil level
    soil_comb_pert = {stash: {} for stash in STASH_TO_MAKE_PERTS}
    adder = mule.operators.AddFieldsOperator(preserve_mdi=True)
    for stash in STASH_TO_MAKE_PERTS:
        for level in bpert_fields[stash].iterkeys():
            bpert_l = bpert_fields[stash][level]
            ekf_pert_l = soil_fields[stash][level]
            soil_comb_pert[stash][level] = adder([bpert_l, ekf_pert_l])

    return soil_comb_pert


# processing functions

def smc_check(perts, dump_data, snow_or_ice_mask, DZ_SOIL_LEVELS):
    """
    Checks the soil moisture content field WITH perturbation to prevent it creating
    points that, when applied, are outside the following conditions:

    1) Volumetric soil moisture content > saturation
    2) Volumetric soil moisture content < wilting point * scaling (currently 0.1)

    Soil moisture perturbations are removed at points specified
    by the input snow_or_ice_mask.

    :param perts: (dictionary) ensemble perturbation fields
    :param dump_data: (dictionary) dump of soil parameters valid for t+0 of current cycle, that the
             [perts] can be checked against e.g. wilting point
    :param snow_or_ice_mask:
    :param DZ_SOIL_LEVELS:
    :return: perts: (dictionary) perts field with corrected data after going through the checks
    """

    class LimitByFields(mule.DataOperator):
        """
        Limits an input field to max and min values given by input
        max_field and min_field fields.

        The minimum field is applied first, then the max. The min and max are not checked
        against each other.

        Missing data points are ignored.
        """

        def __init__(self):
            self.max_field = None
            self.min_field = None

        def new_field(self, source_field, max_field=None, min_field=None):
            # copy the input field code:
            newfld = source_field.copy()
            # and assign the input fields into the object for use in transform, later on:
            self.max_field = max_field
            self.min_field = min_field
            return newfld

        def transform(self, source_field, newfld):
            # firstly, get the data as a numpy array:
            src_data = source_field.get_data()

            # work out missing data mask:
            src_valid = src_data != source_field.bmdi

            # now do the minimum first
            if self.min_field is not None:
                # get the min data:
                min_data = self.min_field.get_data()
                # find where it is valid to do a minimum (both min and src are valid):
                min_valid = np.logical_and(min_data != self.min_field.bmdi, src_valid)
                # at these points, set the src data to the min:
                src_to_min = np.logical_and(min_valid, src_data < min_data)
                # now apply those:
                if np.any(src_to_min):
                    src_data[src_to_min] = min_data[src_to_min]

            # and the max:
            if self.max_field is not None:
                # get the max data:
                max_data = self.max_field.get_data()
                # find where it is valid to do a maximum (both max and src are valid):
                max_valid = np.logical_and(max_data != self.max_field.bmdi, src_valid)
                # at these points, set the src data to the max:
                src_to_max = np.logical_and(max_valid, src_data > max_data)
                # now apply those:
                if np.any(src_to_max):
                    src_data[src_to_max] = max_data[src_to_max]

            # now return the source data, after being modified
            return src_data

    # define the operators we need to use:
    # an add/subtract fields:
    adder = mule.operators.AddFieldsOperator(preserve_mdi=True)
    subber = mule.operators.SubtractFieldsOperator(preserve_mdi=True)

    # apply wilting point scaling:
    # ToDo - really compare to 10 % of the wilting factor? Not wilting factor + or - 10 %
    scalewiltingpoint = mule.operators.ScaleFactorOperator(WILTING_POINT_SCALING_FACTOR)
    # and save the result in memory rather than recalculating it each time:
    cahceoperator = CachingOperator()
    # limit a field, but by other fields, not constant values:
    limit_by_fields = LimitByFields()

    for level in sorted(perts[STASH_SMC].keys()):
        # extract SMC for this level with both perturbations applied
        smc_level = perts[STASH_SMC][level]

        # convert from soil moisture content (mass) to volumetric soil moisture content
        # (which the ancillaries are specified), anc vice-versa:
        # fieldcalc lines from the UM version:
        # 191      WHERE ( (smc_new(k) % RData /= smc_new(k) % Hdr % bmdi))
        # 192        smc_new(k) % RData = smc_new(k) % RData/(RHO_WATER * DZ_SOIL_LEVELS(k))
        smc_to_vol_smc = mule.operators.ScaleFactorOperator(1.0 / (RHO_WATER * DZ_SOIL_LEVELS[level]))
        vol_smc_to_smc = mule.operators.ScaleFactorOperator(RHO_WATER * DZ_SOIL_LEVELS[level])

        # construct a soil moisture content that the dump will have, if this raw perturbation
        # were to be applied:
        # smc_new = adder([dump_data[STASH_SMC][level], perts[STASH_SMC][level]])

        # now convert that to a volumetric smc:
        vol_smc_new = smc_to_vol_smc(smc_level)
        # store it as a temporary space:
        vol_smc_b4_bounds = vol_smc_new.get_data()
        # Check within bounds <= saturation, >= wilting point
        vol_smc_new = limit_by_fields(vol_smc_new,
                                      min_field=scalewiltingpoint(dump_data[STASH_VSMC_WILT]),
                                      max_field=dump_data[STASH_VSMC_SAT])
        vol_smc_aft_bounds = vol_smc_new.get_data()
        # print 'Level %s, limiting smc pert by wilting and satruation changed %s points' % (level,
        #                                        np.sum(vol_smc_b4_bounds != vol_smc_aft_bounds))

        # now convert vol smc back to an smc:
        smc_level = vol_smc_to_smc(vol_smc_new)

        # convert that back into a perturbation (remember, pert is the state of the ensemble member,
        # minus the background value):
        # tmp_pert = subber([smc_level, dump_data[STASH_SMC][level]])

        # now apply masks to that perturbation, masking out points where
        # the snow and land ice fields are relevant;
        tmp_pert_data = smc_level.get_data()
        tmp_pert_data[snow_or_ice_mask] = 0.0

        # now put that data back into the smc_level object:
        array_provider = mule.ArrayDataProvider(tmp_pert_data)
        smc_level.set_data_provider(array_provider)

        # now set the perts to use the tmp_pert, as we're done with it:
        perts[STASH_SMC][level] = smc_level

    return perts


def tsoil_check(perts, dump_data, snow_or_ice_mask):
    """
    Soil temperature perturbations ONLY (not with the TSOIL field) are removed at points specified
    by the input snow_or_ice_mask.
    :param perts: (dictionary) perturbed ensemble data (bpert and EKF)
    :param dump_data: (dictionary) soil variables used in checking perts e.g. wilting point
    :param snow_or_ice_mask: (np.array of booleans) mask for if snow or ice is present
    :return: perts (dictionary) same as input [perts] but with data checked and modified if necessary
    """

    levels = sorted(perts[STASH_TSOIL].keys())
    if sorted(dump_data[STASH_TSOIL].keys()) != levels:
        raise ValueError('Inconsistent soil levels in perturbation and dump')
    # loop over the levels:
    for level in levels:
        # get the data for this level's perturbation:
        tmp_pert_data = perts[STASH_TSOIL][level].get_data()
        # set the relevant parts to zero:
        tmp_pert_data[snow_or_ice_mask] = 0.0
        # limit perturbations to physically sensible bounds
        tsoil_max_val = 20.0
        tsoil_min_val = -20.0
        tmp_pert_data = np.clip(tmp_pert_data, tsoil_min_val, tsoil_max_val)
        # and set that back in the field:
        array_provider = mule.ArrayDataProvider(tmp_pert_data)
        perts[STASH_TSOIL][level].set_data_provider(array_provider)

    return perts


# save


def save_comb_pert(soil_comb_pert):

    # now write out the perturbation on top of the fields in the actual perturbation file.
    output_pert_file = ENS_SMC_DIR
    tmp_output_pert_file = ENS_SMC_DIR + '_tmp.ff'

    pert_ff_in = mule.AncilFile.from_file(output_pert_file, remove_empty_lookups=True)  # the full existing file
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
            if not np.all(field.get_data() == soil_comb_pert[STASH_LAND_SEA_MASK].get_data()):
                raise ValueError('Inconsistent land sea masks in perturbation files')
            pert_ff_out.fields.append(field)
        else:
            pert_ff_out.fields.append(field)
            if template_time_field is None:
                template_time_field = field
    # # does the pert_ff_out have a land sea mask already?
    if not out_pert_has_lsm:
        # add it:
        pert_ff_out.fields.append(bpert_landsea_field)

    # Now add in the perturbations:
    for stash in STASH_TO_MAKE_PERTS:
        if stash in MULTI_LEVEL_STASH:
            # write out each level (the order now matters!):
            for level in sorted(soil_comb_pert[stash]):
                # # modify the headers for time:
                set_time_metadata(soil_comb_pert[stash][level], template_time_field)
                # and append:
                pert_ff_out.fields.append(soil_comb_pert[stash][level])
        else:
            # set_time_metadata(soil_comb_pert[stash], template_time_field)
            pert_ff_out.fields.append(soil_comb_pert[stash])

    # pert_ff_out.validate
    pert_ff_out.to_file(tmp_output_pert_file)

    # if we are done, and this id being run 'for real'
    # then we want to overwrite the real pert file with the temporary one:
    if OVERWRITE_PERT_FILES:
        shutil.move(tmp_output_pert_file, output_pert_file)

    return


if __name__ == '__main__':
    # load the breeding perturbation data, EFK data and combine them
    soil_comb_pert = load_ekf_bpert()

    # save combined perturbations in the original perturbation file
    save_comb_pert(soil_comb_pert)


