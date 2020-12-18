"""
Quick script to help create some plots for Imtiaz

Namely an ensemble spread plot of soil temperature
"""

import numpy as np
import os
import subprocess
import datetime as dt
import iris
import mule

import matplotlib.pyplot as plt

# ==============================================================================
# Setup
# ==============================================================================

# destination directory
datadir = '/scratch/ewarren/imtiaz'
plotdir = '/data/users/ewarren/R2O_projects/soil_moisture_pertubation/figures/imtiaz'

# cycle to extract data for
YYYY = '2020'
MM = '12'
DD = '02'
cycle = '18'

YYYYMMDD_ZZ = YYYY+MM+DD+'_'+cycle

# Full date and cycle
date_cycle = YYYY+MM+DD+'_'+cycle

# forecast lead times to plot for
forecast_lead_times = ['003', '006', '009']

if __name__ == '__main__':

    # ==============================================================================
    # Process
    # ==============================================================================

    for lead_time in forecast_lead_times:

        print('')
        print('working on lead time: '+lead_time)

        file = '/opfc/atm/mogreps-g/prods/{}{}.pp/prods_op_mogreps-g_{}_*_{}.pp'.format(
            YYYY, MM, date_cycle, lead_time)

        # get data
        s = 'moo get moose:'+file+' '+datadir
        os.system(s)

        # find all filenames and members saved to MASS
        onlyfiles = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f))]
        members = [f.split('_')[-2] for f in onlyfiles]

        ens_data = {}
        for member_i in members:

            print('... loading: '+member_i)

            # load
            fname = datadir + '/' + 'prods_op_mogreps-g_'+YYYYMMDD_ZZ+'_'+member_i+'_'+lead_time+'.pp'

            # soil_temp_cube.coord('depth').points => array([0.05 , 0.225, 0.675, 2.   ], dtype=float32)
            # first soil level
            con = iris.Constraint(name='soil_temperature', depth=0.05,
                                  coord_values={'latitude': lambda cell: 20.0 < cell < 40.0,
                                                'longitude': lambda cell: 100.0 < cell < 120.0})
            soil_temp = iris.load_cube(fname, con)

            # extract data and put into dictionary (nan missing values)
            data = soil_temp.data.data
            data[soil_temp.data.mask] = np.nan
            ens_data[member_i] = data


        # use whichever cube was the last read in to get the lon and lats
        lat = soil_temp.coord('latitude').points
        lon = soil_temp.coord('longitude').points

        all_data = np.stack(ens_data.values(), axis=2)

        # -----------------------------------
        print('... ... processing and plotting')

        # 1. plot stdev
        # take stdev of the data
        stdev = np.nanstd(all_data, axis=2)

        title = 'TSOIL depth=0.05 m, standard deviation of {} members (including control)\n'.format(len(ens_data.keys())) + \
                'OS44 MOGREPS-G '+YYYYMMDD_ZZ + ' t+' + lead_time

        savename = plotdir + '/stdev/stdev_'+YYYYMMDD_ZZ+'_t+'+lead_time+'.png'

        fig = plt.figure(figsize=(8, 5))
        # plt.pcolormesh(lon, lat, stdev, vmin=np.nanpercentile(stdev, 1), vmax=np.nanpercentile(stdev, 99))
        plt.pcolormesh(lon, lat, stdev)

        # prettify
        plt.suptitle(title)
        plt.xlabel('longitude [degrees]')
        plt.ylabel('latitude [degrees]')
        plt.colorbar()

        # save
        plt.savefig(savename)
        plt.close(fig)

        # -----------------------------------

        # 2. Range
        # calculate range for each cell
        range_array = np.empty((all_data.shape[0], all_data.shape[1]))
        range_array[:] = np.nan

        for i in np.arange(all_data.shape[0]):
            for j in np.arange(all_data.shape[1]):
                range_array[i, j] = np.max(all_data[i, j, :]) - np.min(all_data[i, j, :])

        title = 'TSOIL depth=0.05 m, range (max-min) of {} members (including control)\n'.format(len(ens_data.keys())) + \
                'OS44 MOGREPS-G '+YYYYMMDD_ZZ+' t+'+lead_time
        savename = plotdir + '/range/range_'+YYYYMMDD_ZZ+'_t+'+lead_time+'.png'

        fig = plt.figure(figsize=(8, 5))
        plt.pcolormesh(lon, lat, range_array)

        # prettify
        plt.suptitle(title)
        plt.xlabel('soil temperature [K]')
        plt.ylabel('frequency')
        plt.colorbar()

        # save
        plt.savefig(savename)
        plt.close(fig)

        # -----------------------------------

        # clean up
        #os.system('rm ' + datadir + '/*')

    exit(0)