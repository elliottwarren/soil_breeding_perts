"""
Quick script to help create some plots for Imtiaz

Namely an ensemble spread plot of soil temperature
"""

import numpy as np
import os
import subprocess
import datetime as dt

# ==============================================================================
# Setup
# ==============================================================================

# destination directory
datadir = '/scratch/ewarren/imtiaz'

# cycle to extract data for
YYYY = '2020'
MM = '12'
DD = '02'
cycle = '18'

# Full date and cycle
date_cycle = YYYY+MM+DD+'_'+cycle

# forecast lead times to plot for
forecast_lead_times = [0, 3, 6, 9]

if __name__ == '__main__':

    # ==============================================================================
    # Process
    # ==============================================================================

    for lead_time in forecast_lead_times:
        file = '/opfc/atm/mogreps-g/prods/{}{}.pp/prods_op_mogreps-g_{}_*_00{}.pp'.format(
            YYYY, MM, date_cycle, lead_time)

        # get data
        s = 'moo get moose:'+file+' '+datadir
        os.system(s)

        # do the stats, make the plot






    exit(0)