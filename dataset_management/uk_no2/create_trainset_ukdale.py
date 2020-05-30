from ukdale_parameters import *
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
from functions import load_dataframe
import numpy as np

DATA_DIRECTORY = '../../data/refit/UKDALE/'
SAVE_PATH = 'kettle/'
AGG_MEAN = 522
AGG_STD = 814


def get_arguments():
    parser = argparse.ArgumentParser(description='sequence to point learning \
                                     example for NILM')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                          help='The directory containing the UKDALE data')
    parser.add_argument('--appliance_name', type=str, default='kettle',
                          help='which appliance you want to train: kettle,\
                          microwave,fridge,dishwasher,washingmachine')
    parser.add_argument('--aggregate_mean',type=int,default=AGG_MEAN,
                        help='Mean value of aggregated reading (mains)')
    parser.add_argument('--aggregate_std',type=int,default=AGG_STD,
                        help='Std value of aggregated reading (mains)')
    parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                          help='The directory to store the training data')
    return parser.parse_args()


args = get_arguments()
appliance_name = args.appliance_name
print(appliance_name)

def aggregate_app(df):
    a = df['aggregate']
    b = df[appliance_name]
    if a < b:
        b = a
    return b


def main():

    start_time = time.time()
    sample_seconds = 6
    training_building_percent = 0
    validation_percent = 20
    testing_percent = 20
    nrows = None
    debug = True

    train = pd.DataFrame(columns=['aggregate', appliance_name])

    for h in params_appliance[appliance_name]['houses']:
        print('    ' + args.data_dir + 'house_' + str(h) + '/'
              + 'channel_' +
              str(params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(h)]) +
              '.dat')

        mains_df = load_dataframe(args.data_dir, h, 1)
        app_df = load_dataframe(args.data_dir,
                                h,
                                params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(h)],
                                col_names=['time', appliance_name]
                                )

        mains_df['time'] = pd.to_datetime(mains_df['time'], unit='s')
        mains_df.set_index('time', inplace=True)
        mains_df.columns = ['aggregate']
        #############################
        # mains_df.resample(str(sample_seconds) + 'S').mean()
        mains_df.reset_index(inplace=True)

        if debug:
            print("    mains_df:")
            print(mains_df.head())
            print(mains_df.tail())
            plt.plot(mains_df['time'], mains_df['aggregate'])
            plt.savefig('original-mains.png')
            plt.show()

        # Appliance
        ############

        app_df['time'] = pd.to_datetime(app_df['time'], unit='s')


        if debug:
            print("app_df:")
            print(app_df.head())
            print(app_df.tail())
            plt.plot(app_df['time'], app_df[appliance_name])
            plt.savefig('original-{}.png'.format(appliance_name))
            plt.show()

        # the timestamps of mains and appliance are not the same, we need to align them
        # 1. join the aggragte and appliance dataframes;
        # 2. interpolate the missing values;
        mains_df.set_index('time', inplace=True)
        app_df.set_index('time', inplace=True)
        ######add 
        #app_df.resample(str(sample_seconds) + 'S').mean()


        df_align = mains_df.join(app_df, how='outer'). \
            resample(str(sample_seconds) + 'S').mean().fillna(method='backfill', limit=1)#fillna(method='backfill', limit=1)
        df_align = df_align.dropna()

        df_align.reset_index(inplace=True)
        
        
        if appliance_name == 'fridge':
            df_align[appliance_name] = df_align.apply(aggregate_app,axis=1)
            #df_align.to_csv('11111111_test_.csv', mode='a', index=False, header=False)

        del mains_df, app_df, df_align['time']


        if debug:
            # plot the dtaset
            print("df_align:")
            print(df_align.head())
            print(df_align.tail())
            # plt.plot(df_align['aggregate'].values)
            # plt.plot(df_align[appliance_name].values)
            # plt.savefig('{}.png'.format(appliance_name))
            # plt.show()
            test_len = int((len(df_align)/100)*testing_percent)

            # fig1 = plt.figure()
            # ax1 = fig1.add_subplot(111)

            # ax1.plot(df_align['aggregate'][-test_len:-1], color='#7f7f7f', linewidth=1.8)
            # ax1.plot(df_align[appliance_name][-test_len:-1], color='#d62728', linewidth=1.6)

            plt.subplot(211)
            plt.title(appliance_name)
            plt.plot(df_align['aggregate'][-test_len:])
            plt.yticks(np.linspace(0,5000,5,endpoint=True))

            plt.subplot(212)
            plt.plot(df_align[appliance_name][-test_len:])
            plt.yticks(np.linspace(0,5000,5,endpoint=True))
           
            
            # plt.subplots_adjust(bottom=0.2, right=0.7, top=0.9, hspace=0.3)
            plt.savefig('{}-_subplot.png'.format(args.appliance_name))
            # # ax1.plot(prediction,
            # #          color='#1f77b4',
            # #          #marker='o',
            # #          linewidth=1.5)
            # # plt.xticks([])
            # ax1.grid()
            # # ax1.set_title('Test results on {:}'.format(test_filename), fontsize=16, fontweight='bold', y=1.08)
            # ax1.set_ylabel(appliance_name)
            # ax1.legend(['aggregate', appliance_name],loc='upper left')

            # mng = plt.get_current_fig_manager()
            # #mng.resize(*mng.window.maxsize())
            # plt.savefig('{}.png'.format(args.appliance_name))

        # Normilization ----------------------------------------------------------------------------------------------
        mean = params_appliance[appliance_name]['mean']
        std = params_appliance[appliance_name]['std']

        df_align['aggregate'] = (df_align['aggregate'] - args.aggregate_mean) / args.aggregate_std
        df_align[appliance_name] = (df_align[appliance_name] - mean) / std

        # if h == params_appliance[appliance_name]['test_build']:
        #     # Test CSV
        #     df_align.to_csv(args.save_path + appliance_name + '_test_.csv', mode='a', index=False, header=False)
        #     print("    Size of test set is {:.4f} M rows.".format(len(df_align) / 10 ** 6))
        #     continue

        train = train.append(df_align, ignore_index=True)
        del df_align

    # Crop dataset
    if training_building_percent is not 0:
        train.drop(train.index[-int((len(train)/100)*training_building_percent):], inplace=True)

    test_len = int((len(train)/100)*testing_percent)
    val_len = int((len(train)/100)*validation_percent)

    #Testing CSV
    test = train.tail(test_len)

    test.reset_index(drop=True, inplace=True)
    train.drop(train.index[-test_len:], inplace=True)
    test.to_csv(args.save_path + appliance_name + '_test_.csv', mode='a', index=False, header=False)


    # Validation CSV
    val = train.tail(val_len)
    val.reset_index(drop=True, inplace=True)
    train.drop(train.index[-val_len:], inplace=True)
    # Validation CSV
    val.to_csv(args.save_path + appliance_name + '_validation_' + '.csv', mode='a', index=False, header=False)

    # Training CSV
    train.to_csv(args.save_path + appliance_name + '_training_.csv', mode='a', index=False, header=False)

    print("    Size of total training set is {:.4f} M rows.".format(len(train) / 10 ** 6))
    print("    Size of total validation set is {:.4f} M rows.".format(len(val) / 10 ** 6))
    print("    Size of total testing set is {:.4f} M rows.".format(len(test) / 10 ** 6))
    del train, val, test


    print("\nPlease find files in: " + args.save_path)
    print("Total elapsed time: {:.2f} min.".format((time.time() - start_time) / 60))


if __name__ == '__main__':
    main()