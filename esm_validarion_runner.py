import sys
from ts_esm import esm_validation_runner

valid_date_range = ('2016', '2018')


if __name__ == '__main__':
    hyperparam_path = sys.argv[1]
    history_limit = int(sys.argv[2])

    if history_limit == 0:
        history_limit = None
        
    if 'diff' in sys.argv:
        diff=True
    else:
        diff=False

    if 'fixed' in sys.argv:
        fixed_history=True
    else:
        fixed_history=False

    if 'parallel' in sys.argv:
        parallel = True
    else:
        parallel = False

    data_path = 'data/TCS_20200303.csv'
    result_dir = 'results'

    esm_validation_runner(hyperparam_path=hyperparam_path,
                          data_path=data_path,
                          valid_date_range=valid_date_range,
                          history_limit=history_limit,
                          fixed_history=fixed_history,
                          diff=diff,
                          result_dir=result_dir,
                          parallel=parallel)

    print('finished: ESM {}'.format(hyperparam_path))
