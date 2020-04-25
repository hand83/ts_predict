import sys
import os

valid_date_range = ('2016', '2018')
seed = 987654

if __name__ == '__main__':
    model_config_path = sys.argv[1]
    hyperparam_path = sys.argv[2]
    data_path = 'data/TCS_20200303.csv'
    result_dir = 'results'
    if 'gpu' in sys.argv:
        print('--- GPU ENABLED ---')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print('--- GPU DISABLED ---')
    from ts_validation import nn_validation_runner


    nn_validation_runner(model_config_path=model_config_path, 
                         hyperparam_path=hyperparam_path,
                         data_path=data_path,
                         valid_date_range=valid_date_range,
                         result_dir=result_dir,
                         seed=seed)
                         
    print('finished: {} {}'.format(model_config_path, hyperparam_path))
