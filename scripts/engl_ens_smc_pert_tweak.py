#!/usr/bin/env python
'''
Generates SMC (soil moisture content) and TSOIL (temp soil) perturbations using the breeding method
Differences at T+6h forecasts from the previous cycle are carried to the new cycle

Usage: engl_ens_smc_pert.py

Environemnt variables required:
  * ROSE_DATACPT6H   - rose data directory for cycle 6 hours previous to current
  * ENS_SMC_DIR      - ensemble soil moisture directory to write out diagnostic means
  * ENS_PERTS_DIR    - ensemble perturbation directory for the current cycle's ensemble
  * ENS_ANAL_FILE    - ensemble analysis to be used for the current cycle's ensemble
  * ENS_SNOW_FILE    - ensemble T+6h to be used for masking out perturbations under snow
  * NUM_PERT_MEMBERS - number of ensemble members. - remove? Change to a single member name (know which file to put
                        perts into

Version 1.0 : 20160918 Malcolm.E.Brooks@metoffice.gov.uk
Version 2.0: 20191119 elliott.warren@metoffice.gov.uk
'''

import os, pdb, datetime, shutil
import numpy as np

import mule
import mule.operators
from mule import FieldsFile

# Environement variables:
# Directories:
ROSE_DATACPT6H = os.getenv('ROSE_DATACPT6H')
ENS_SMC_DIR = os.getenv('ENS_SMC_DIR')
ENS_PERTS_DIR = os.getenv('ENS_PERTS_DIR')
# Files:
# initial conditions of unperturbed ensemble forecast:
ENS_ANAL_FILE = os.getenv('ENS_ANAL_FILE')

# global dump for masking perturbations under snow:
ENS_SNOW_FILE = os.getenv('ENS_SNOW_FILE')

# number of ensemble members to do perturbations for
NUM_PERT_MEMBERS = os.getenv('NUM_PERT_MEMBERS')

# when called from a suite, picking up environment variables, then
# we don't write out diagnostic files of each stage to disk:
WRITE_DIAG_FILES = False

# actually overwrite the main perturbation files at the end of the script
OVERWRITE_PERT_FILES = True

if ROSE_DATACPT6H is None:
    # if not set, then this is being run for development, so have canned variable settings to hand:
    ROSE_DATACPT6H = "/home/d03/freb/cylc-run/u-ai506/share/cycle/20150629T1200Z"
    ENS_SMC_DIR = "/home/d03/freb/cylc-run/u-ai506/share/cycle/20150629T1800Z/engl_etkf/smc"
    ENS_PERTS_DIR = "/home/d03/freb/cylc-run/u-ai506/share/cycle/20150629T1800Z/engl_etkf/perts"
    ENS_ANAL_FILE = "/home/d03/freb/cylc-run/u-ai506/share/cycle/20150629T1800Z/engl_atmanl"
    ENS_SNOW_FILE = "/home/d03/freb/cylc-run/u-ai506/share/cycle/20150629T1800Z/engl_um_000/englaa_da006"
    NUM_PERT_MEMBERS = "44"
    NUM_PERT_MEMBERS = "3"
    WRITE_DIAG_FILES = False
    OVERWRITE_PERT_FILES = False

# conversions to type:
NUM_PERT_MEMBERS = int(NUM_PERT_MEMBERS)
MEMBERS_PERT_INTS = range(1, NUM_PERT_MEMBERS+1)

print 'ROSE_DATACPT6H = "{}"'.format(ROSE_DATACPT6H)
print 'ENS_SMC_DIR = "{}"'.format(ENS_SMC_DIR)
print 'ENS_PERTS_DIR = "{}"'.format(ENS_PERTS_DIR)
print 'NUM_PERT_MEMBERS = "{}"'.format(NUM_PERT_MEMBERS)

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

# layer depth of each soil layer, to be set once input files are read in:
DZ_SOIL_LEVELS = {}
# density of pure water (kg/m3)
RHO_WATER = 1000.0


class Timers:
    'some very simple timers (from the trui package on linux, but this is on the HPC)'
    def __init__(self, timer_name, timer_list):
        'initialises a list of timers'
        self.name = timer_name
        self.timers = {}
        for key in timer_list:
            self.timers[key] = [None, 0.0]

    def __repr__(self):
        'string representation of the timer information'#
        outstr = 'Timer information "{}":\n'.format(self.name)
        # extract and sort the key names and times:
        sorted_times = sorted([(self.timers[x][1], x) for x in self.timers], reverse=True)
        # and make the outstr:
        for (this_time, this_key) in sorted_times:
            outstr += '  "{}" took {} s\n'.format(this_key, this_time)
        return outstr

    def start(self, key):
        'starts the clock on a timer'
        if key in self.timers.keys():
            if self.timers[key][0] is None:
                # this time is not currently active:
                self.timers[key][0] = datetime.datetime.now()
            else:
                # this timer has been started and not ended
                raise ValueError('Timer "{}" is being started but it is alredy active'.format(key))
        else:
            self.timers[key] = [datetime.datetime.now(), 0.0]

    def end(self, key):
        'ends the clock on a timer, and appends the time to the counter'
        self.timers[key][1] += (datetime.datetime.now() - self.timers[key][0]).total_seconds()
        # set the start time back to None, so it can be restarted without raising an error:
        self.timers[key][0] = None

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

class LimitByFields(mule.DataOperator):
    '''
    Limits an iput field to max and min values given by input
    max_field and min_field fields.

    The minimum field is applied first, then the max. The min and max are not checked
    against each other.

    Missing data points are ignored.
    '''
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


def load_prev_ens_data():
    '''
    Loads in input data from previous ensemble run. This is from a large
    number of files, and sorted into a dict strucure by field, level to
    produce a list containing the data for ensemble members.
    '''
    # set up the returned data structure...
    prev_ens_data = {}
    for stash in STASH_TO_LOAD:
        if stash in MULTI_LEVEL_STASH:
            prev_ens_data[stash] = {}
        else:
            prev_ens_data[stash] = []
    # and the fieldsfile objects, which we can use as templates later on:
    prev_ens_ff_files = []

    # load in the required data for each file:
    for member in MEMBERS_PERT_INTS:
        smc_src_file = engl_prev_cycle_6hr_dump(member)
        print 'reading in file: %s' % smc_src_file
        # data file to open:
        TIMER.start('opening file')
        ff_file_in = mule.load_umfile(smc_src_file)
        if ff_file_in.fixed_length_header.dataset_type not in (1,2,3):
            raise ValueError("wrong file type")
        ff_file_in.remove_empty_lookups()
        TIMER.end('opening file')

        # pull out the fields:
        # SMC on multiple heights, therefore has multiple fields
        TIMER.start('checking/extracting input data')
        for field in ff_file_in.fields:
            # is this stash code we want?
            if field.lbuser4 in STASH_TO_LOAD:
                if field.lbuser4 in MULTI_LEVEL_STASH:
                    # multi-level fields are a dict, with a list for each level
                    if field.lblev in prev_ens_data[field.lbuser4].keys():
                        prev_ens_data[field.lbuser4][field.lblev].append(field)
                    else:
                        prev_ens_data[field.lbuser4][field.lblev] = [field]
                else:
                    # single level fields are a flat list:
                    prev_ens_data[field.lbuser4].append(field)
        TIMER.end('checking/extracting input data')
        # keep the fieldsfile objects for reference:
        prev_ens_ff_files.append(ff_file_in)

    # verify that all the data is included:
    TIMER.start('verifying loaded fields')
    for stash in STASH_TO_LOAD:
        if stash in MULTI_LEVEL_STASH:
            for level in prev_ens_data[stash]:
                n_mems = len(prev_ens_data[stash][level])
                if n_mems != NUM_PERT_MEMBERS:
                    raise ValueError('{} fields found in ensemble for STASH code {}, level {}, '
                                     'but NUM_PERT_MEMBERS ({}) is expected'.format(
                                     n_mems, stash, level, NUM_PERT_MEMBERS))
        else:
            if len(prev_ens_data[stash]) != NUM_PERT_MEMBERS:
                raise ValueError('Not all data found!')
    TIMER.end('verifying loaded fields')

    return prev_ens_data, prev_ens_ff_files

def engl_prev_cycle_6hr_dump(member):
    'locates T+6 hour start dump from the engl_prev_cycle for an ensemble member'
    return '{0}/engl_um_{1}/englaa_da006'.format(ROSE_DATACPT6H, mem_to_str(member))

def engl_pert(member):
    'locates the output perturbation file for this cycle'
    return '{0}/engl_pert_{1}'.format(ENS_PERTS_DIR, mem_to_str(member))

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

def load_from_single_ff(stash_and_constraints, ff_file, cache=False):
    '''
    loads required data from a single file, supplied as a file name to load
    or a ff object, into a simple dictionary.
    
    Returns the requested fieldsfile data, and the fieldsfile object 
    '''

    if isinstance(ff_file, str):
        ff_obj = mule.load_umfile(ff_file)
        if ff_obj.fixed_length_header.dataset_type not in (1,2,3):
            raise ValueError("wrong file type")
        ff_obj.remove_empty_lookups()
    else:
        ff_obj = ff_file
        ff_obj.remove_empty_lookups()

    if cache:
        cacheoperator = CachingOperator()
    ff_data = {}
    TIMER.start('reading input ff')
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
                    # fields without psuedo levels:
                    #
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

    TIMER.end('reading input ff')
    return ff_data, ff_obj

def ens_structure_to_single(ens_data, member):
    '''
    extracts the data for a given ensemble member, from all fields
    in a data strcuture from load_prev_ens_data
    '''

    this_member = {}
    for stash in ens_data:
        # whether the data is stored in a dict, or not, depends on whether it's multi-level
        if isinstance(ens_data[stash], dict):
            if stash not in MULTI_LEVEL_STASH:
                raise ValueError('At the moment, only MULTI_LEVEL_STASH should be a dict')
            this_member[stash] = {}
            for level in ens_data[stash]:
                this_member[stash][level] = ens_data[stash][level][member-1]
        elif isinstance(ens_data[stash], list):
            if stash  in MULTI_LEVEL_STASH:
                raise ValueError('At the moment, MULTI_LEVEL_STASH should be a dict')
            this_member[stash] = ens_data[stash][member-1]
    return this_member

def mean_prev_ens_data(in_data, in_files):
    'produces the mean of the required fields, input as in_data'
    TIMER.start('Meaning fields')

    mean_data = {}

    # define the operators we need to use:
    # add the fields up:
    adder = mule.operators.AddFieldsOperator(preserve_mdi=True)
    # dived them by the number:
    divver = mule.operators.ScaleFactorOperator(1.0 / NUM_PERT_MEMBERS)
    # and save the result in memory rather than recalculating it each time:
    cahceoperator = CachingOperator()

    for stash in STASH_TO_MAKE_PERTS:
        if stash in MULTI_LEVEL_STASH:
            mean_data[stash] = {}
            for level in in_data[stash]:
                # add them up:
                sum_field = adder(in_data[stash][level])
                # now divied:
                mean_field = divver(sum_field)
                # and store that in the output strcuture, as cached data:
                mean_data[stash][level] = cahceoperator(mean_field)
                # now use get_data to actually do the meaning at this point
                # and cache the result. This means the timer is accurate.
                _ = mean_data[stash][level].get_data()
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
    TIMER.end('Meaning fields')

    if WRITE_DIAG_FILES:
        mean_ff = in_files[0].copy(include_fields=False)
        for stash in STASH_TO_MAKE_PERTS:
            if stash in MULTI_LEVEL_STASH:
                for level in in_data[stash]:
                    mean_ff.fields.append(mean_data[stash][level])
            else:
                mean_ff.fields.append(mean_data[stash])

        # now add the land/sea mask:
        mean_ff.fields.append(in_data[STASH_LAND_SEA_MASK][0])

        mean_file = '%s/mean.ff' % ENS_SMC_DIR
        print 'writing out diagnostic mean file "%s"' % mean_file
        TIMER.start('writing diagnostic mean smc file')
        mean_ff.to_file(mean_file)
        TIMER.end('writing diagnostic mean smc file')

    return mean_data

def in_data_minus_mean(in_data, mean_data, in_files):
    'subtracts the mean of a set of fields from the input data, to produce a perturbation'

    subber = mule.operators.SubtractFieldsOperator(preserve_mdi=True)

    TIMER.start('Applying subtract opertator to ensemble')
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
    # now add the land sea masks on as well:
    pert_data[STASH_LAND_SEA_MASK] = in_data[STASH_LAND_SEA_MASK]
    TIMER.end('Applying subtract opertator to ensemble')

    if WRITE_DIAG_FILES:
        TIMER.start('Writing out diagnostic smc perturbations')
        # write out a file, for each member:
        diag_pert_files = [engl_pert(x) + '_smc_diag_unchecked_pert.ff' for x in MEMBERS_PERT_INTS]
        write_pert_diag_files(pert_data, diag_pert_files, in_files)
        TIMER.end('Writing out diagnostic smc perturbations')

    return pert_data

def pert_check_and_output(first_guess_perts, in_files):
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
    # STASH codes to load in from the analysis dump:
    # stash codes as dict key, then a dict of constraints, if any:
    stash_from_dump = {STASH_SMC: None,
                       STASH_TSOIL: None,
                       STASH_VSMC_WILT: None,
                       STASH_VSMC_SAT: None,
                       STASH_SNOW_AMNT: None,
                       STASH_LAND_SEA_MASK: None,
                       STASH_LANDFRAC: {'lbuser5': [PSEUDO_LEVEL_LANDICE]},
                      }
    TIMER.start('opening file')
    dump_data, dump_ff = load_from_single_ff(stash_from_dump, ENS_ANAL_FILE, cache=True)
    TIMER.end('opening file')

    # Read snow from a forecast dump, because the reconfigured dump does not have sensible
    # snow on STASH=23 on the first time-step
    stash_from_dump2 = {STASH_SNOW_AMNT: None,
                      }
    TIMER.start('opening snow file')
    dump2_data, dump2_ff = load_from_single_ff(stash_from_dump2, ENS_SNOW_FILE, cache=True)
    TIMER.end('opening snow file')
    
    # check the soil levels are the same as the dump, for all input data files:
    for test_in_file in in_files:
        if not np.all(dump_ff.level_dependent_constants.soil_thickness ==
                      test_in_file.level_dependent_constants.soil_thickness):
            raise ValueError('Soil level mismatch between ensemble input files and ENS_ANAL_FILE')
    # now set the DZ_SOIL_LEVELS from the dump header data:
    for level in dump_data[STASH_SMC].keys():
        DZ_SOIL_LEVELS[level] = dump_ff.level_dependent_constants.soil_thickness[level-1]
        # now check it's not missing data:
        if DZ_SOIL_LEVELS[level] == dump_ff.level_dependent_constants.MDI:
            raise ValueError('Input dump level_dependent_constants.soil_thickness contains '
                             'missing data for level {}'.format(level))
                                                    
    # from that dump data, a few masks are applied to all members:
    landice_data = dump_data[STASH_LANDFRAC][PSEUDO_LEVEL_LANDICE].get_data()
    # at the moment, landice fractions are defined as being 0.0 or 1.0, or missing data.
    # we use this to find land ice points, so this code will need revisiting if that
    # ever changes:
    valid_icefrac_vals = np.array([dump_data[STASH_LANDFRAC][PSEUDO_LEVEL_LANDICE].bmdi, 0.0, 1.0])
    # now chek the land_ice fraction data:
    land_ice_unique = np.sort(np.unique(landice_data))
    if len(land_ice_unique) != 3:
        raise ValueError('Land ice fractions are not 0.0, 1.0 (or bmdi)')
    if not np.all(land_ice_unique == valid_icefrac_vals):
        raise ValueError('Land ice fractions are not 0.0, 1.0 (or bmdi)')
    # now check the land ice:
    landice_mask = np.logical_and(landice_data == 1.0,
                    landice_data != dump_data[STASH_LANDFRAC][PSEUDO_LEVEL_LANDICE].bmdi)

    ## as a test, load the critical smc, and use that as a masl
    #vcrit_data = dump_data[STASH_VSMC_CRIT].get_data()
    #vcrit_mask = np.logical_and(vcrit_data == 0.0, vcrit_data !=  dump_data[STASH_VSMC_CRIT].bmdi)
    #print 'Do land ice masks, and crit smc masks agree: %s' % np.all(vcrit_mask == landice_mask)
    # They do agree. Using land ice fraction as it's not hacky, but a direct input for land ice:

    # Derive snowmask by combining snow from both reconfigured file and T+6h forecast file
    snow_data = dump2_data[STASH_SNOW_AMNT].get_data() + dump_data[STASH_SNOW_AMNT].get_data()
    snow_mask = np.logical_and(snow_data > SNOW_AMOUNT_THRESH,
                               snow_data != dump2_data[STASH_SNOW_AMNT].bmdi)
    # now combine those:
    snow_or_ice_mask = np.logical_or(landice_mask, snow_mask)

    # now loop through, ensemble member at a time, working on the perturbations:
    for member in MEMBERS_PERT_INTS:
        print 'Checking and output pert for member %s' % member
        # Soil moisutre checks on this member:
        #
        # pull out the first guess perturbations for this member:
        this_perts = ens_structure_to_single(first_guess_perts, member)
        # update those smc perturbations with the checks, soil moisture first:
        TIMER.start('smc_checks')
        smc_check(this_perts, dump_data, snow_or_ice_mask)
        TIMER.end('smc_checks')
        # Soil temperature checks, on this member:
        TIMER.start('tsoil_checks')
        tsoil_check(this_perts, dump_data, snow_or_ice_mask)
        TIMER.end('tsoil_checks')

        # write out diagnostic file:
        if WRITE_DIAG_FILES:
            diag_file_name = engl_pert(member) + '_smc_diag_checked_pert.ff'
            # open a fieldfile object:
            diag_ff = in_files[member-1].copy(include_fields=False)
            # add the fields:
            for stash in this_perts:
                if stash in MULTI_LEVEL_STASH:
                    for level in this_perts[stash]:
                        diag_ff.fields.append(this_perts[stash][level])
                else:
                    diag_ff.fields.append(this_perts[stash])
            print 'writing out diagnostic smc pert file "%s"' % diag_file_name
            TIMER.start('writing diagnostic mean smc file')
            diag_ff.to_file(diag_file_name)
            TIMER.end('writing diagnostic mean smc file')

        # now write out the perturbation on top of the fields in the actual perturbation file.
        TIMER.start('Writing output perturbation files')
        output_pert_file = engl_pert(member)
        tmp_output_pert_file = engl_pert(member) + '_tmp.ff'

        if OVERWRITE_PERT_FILES:
            print 'appending soil perturbations to %s' % output_pert_file
        else:
            print 'appending soil perturbations to create %s' % tmp_output_pert_file

        pert_ff_in = FieldsFile.from_file(output_pert_file, remove_empty_lookups=True)
        pert_ff_out = pert_ff_in.copy(include_fields=False)

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
                pass
            elif field.lbuser4 == STASH_LAND_SEA_MASK:
                out_pert_has_lsm = True
                if not np.all(field.get_data() == this_perts[STASH_LAND_SEA_MASK].get_data()):
                    raise ValueError('Inconsistent land sea masks in perturbation files')
                pert_ff_out.fields.append(field)
            else:
                pert_ff_out.fields.append(field)
                if template_time_field is None:
                    template_time_field = field
        # does the pert_ff_out have a land sea mask already?
        if not out_pert_has_lsm:
            # add it:
            pert_ff_out.fields.append(this_perts[STASH_LAND_SEA_MASK])

        # Now add in the perturbations:
        for stash in STASH_TO_MAKE_PERTS:
            if stash in MULTI_LEVEL_STASH:
                # write out each level (the order now matters!):
                for level in sorted(this_perts[stash]):
                    # modify the headers for time:
                    set_time_metadata(this_perts[stash][level], template_time_field)
                    # and append:
                    pert_ff_out.fields.append(this_perts[stash][level])
            else:
                set_time_metadata(this_perts[stash], template_time_field)
                pert_ff_out.fields.append(this_perts[stash])
        # and write that out to the tmp file:
        pert_ff_out.to_file(tmp_output_pert_file)
        # if we are done, and this id being run 'for real'
        # then we want to overwrite the real pert file with the temporary one:
        if OVERWRITE_PERT_FILES:
            shutil.move(tmp_output_pert_file, output_pert_file)
        TIMER.end('Writing output perturbation files')

def smc_check(perts, dump_data, snow_or_ice_mask):
    '''
    Checks a soil moisture perturbation to prevent it creating
    points that, when applied, are outside the following conditions:

    1) Volumetric soil moisture content > satruation
    2) Volumetric soil moisture content < wilting point * scaling (currently 0.1)

    Soil moisture perturbations are removed at points specified
    by the intput snow_or_ice_mask.
    '''
    # define the operators we need to use:
    # an add/subtract fields:
    adder = mule.operators.AddFieldsOperator(preserve_mdi=True)
    subber = mule.operators.SubtractFieldsOperator(preserve_mdi=True)

    # apply wilting point scaling:
    scalewiltingpoint = mule.operators.ScaleFactorOperator(WILTING_POINT_SCALING_FACTOR)
    # and save the result in memory rather than recalculating it each time:
    cahceoperator = CachingOperator()
    # limit a field, but by other fields, not constant values:
    limit_by_fields = LimitByFields()

    # check the land sea masks are identical at this point:
    if not np.all(dump_data[STASH_LAND_SEA_MASK].get_data() == perts[STASH_LAND_SEA_MASK].get_data()):
        raise ValueError('Inconsistent land sea masks in perturbation and dump')

    for level in sorted(perts[STASH_SMC].keys()):
        # convert from soil moisture content (mass) to volumetric soil moisture content
        # (which the ancillaries are specified), anc vice-versa:
        # fieldcalc lines from the UM version:
        #191      WHERE ( (smc_new(k) % RData /= smc_new(k) % Hdr % bmdi))
        #192        smc_new(k) % RData = smc_new(k) % RData/(RHO_WATER * DZ_SOIL_LEVELS(k))
        smc_to_vol_smc = mule.operators.ScaleFactorOperator(1.0 / (RHO_WATER * DZ_SOIL_LEVELS[level]))
        vol_smc_to_smc = mule.operators.ScaleFactorOperator(RHO_WATER * DZ_SOIL_LEVELS[level])

        # construct a soil moisture content that the dump will have, if this raw perturbation
        # were to be applied:
        smc_new = adder([dump_data[STASH_SMC][level], perts[STASH_SMC][level]])

        # now convert that to a volumetric smc:
        vol_smc_new = smc_to_vol_smc(smc_new)
        # store it as a temporary space:
        vol_smc_b4_bounds = vol_smc_new.get_data()
        # Check within bounds <= saturation, >= wilting point
        vol_smc_new = limit_by_fields(vol_smc_new,
                                      min_field=scalewiltingpoint(dump_data[STASH_VSMC_WILT]),
                                      max_field=dump_data[STASH_VSMC_SAT])
        vol_smc_aft_bounds = vol_smc_new.get_data()
        print 'Level %s, limiting smc pert by wilting and satruation changed %s points' % (level,
                                                np.sum(vol_smc_b4_bounds != vol_smc_aft_bounds))

        # now convert vol smc back to an smc:
        smc_new = vol_smc_to_smc(vol_smc_new)

        # convert that back into a perturbation (remember, pert is the state of the ensemble member,
        # minus the background value):
        tmp_pert = subber([smc_new, dump_data[STASH_SMC][level]])
        # now apply masks to that perturbation, masking out points where
        # the snow and land ice fields are relevant;
        tmp_pert_data = tmp_pert.get_data()
        tmp_pert_data[snow_or_ice_mask] = 0.0

        # now put that data back into the tmp_pert object:
        array_provider = mule.ArrayDataProvider(tmp_pert_data)
        tmp_pert.set_data_provider(array_provider)

        # now set the perts to use the tmp_pert, as we're done with it:
        perts[STASH_SMC][level] = tmp_pert

def tsoil_check(perts, dump_data, snow_or_ice_mask):
    '''
    Soil temperature perturbations are removed at points specified
    by the intput snow_or_ice_mask.
    Checks
    '''
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

def set_time_metadata(data_field, time_template_field):
    'sets the time metadata of an input data field to that of a template field'
    data_field.lbyr = time_template_field.lbyr
    data_field.lbmon = time_template_field.lbmon
    data_field.lbdat = time_template_field.lbdat
    data_field.lbmin = time_template_field.lbmin
    data_field.lbsec = time_template_field.lbsec

    data_field.lbyrd = time_template_field.lbyrd
    data_field.lbmond = time_template_field.lbmond
    data_field.lbdatd = time_template_field.lbdatd
    data_field.lbmind = time_template_field.lbmind
    data_field.lbsecd = time_template_field.lbsecd

    data_field.lbtim = time_template_field.lbtim
    data_field.lbft = time_template_field.lbft

def write_pert_diag_files(pert_data, filename_list, in_files):
    '''
    Writes out a set of perturbation files of data within the pert_data dict
    to files, where the names are decided already in filename_list
    '''
    for member, diag_ff_filename in zip(MEMBERS_PERT_INTS, filename_list):
        # find the location:
        # use the input data files as a template file object
        # (the member-1 is because member is an int from 1 to N,
        #  but they are stored in a list, using indexing from zero)
        diag_ff = in_files[member-1].copy(include_fields=False)
        # loop through the files, adding them to the fieldsfile object:
        for stash in STASH_TO_MAKE_PERTS:
            if stash in MULTI_LEVEL_STASH:
                for level in pert_data[stash]:
                    diag_ff.fields.append(pert_data[stash][level][member-1])
            else:
                diag_ff.fields.append(pert_data[stash][member-1])
        # now add the land/sea mask:
        diag_ff.fields.append(pert_data[STASH_LAND_SEA_MASK][0])
        # and write it out:
        print 'writing out diagnostic file "%s"' % diag_ff_filename
        diag_ff.to_file(diag_ff_filename)

def smc_pert_top():
    '''
    Top level Soil Moisture Content (smc) perturbation routine
    1) Loads in data
    2) Produces the mean of the input fields
    3) Subtracts the mean from each ensemble member to create a perturbation
    4) Checks that the perturbations will cause push the smc in the new dump
       out of physical ranges.
    '''
    # load the input data:
    prev_ens_data, prev_ens_ff_files = load_prev_ens_data()
    # mean the required fields
    prev_ens_mean = mean_prev_ens_data(prev_ens_data, prev_ens_ff_files) #ToDo to become one script
    # subtract the mean from the input data to create a perturbation
    diff_data = in_data_minus_mean(prev_ens_data, prev_ens_mean, prev_ens_ff_files) #ToDo 1st part of 2nd script
    # now check these differences (first guess perturbations) are acceptible,
    # when applied to the current dump:
    pert_data = pert_check_and_output(diff_data, prev_ens_ff_files) #ToDo 2nd part of 2nd script

    # ToDo need to append the now checked data to the SURF file

if __name__ == '__main__':
    TIMER = Timers('engl_smc_perts', [])
    TIMER.start('Total time')
    smc_pert_top()
    TIMER.end('Total time')
    print 'engl_ens_smc_pert.py completed'
    print TIMER
    exit(0)
