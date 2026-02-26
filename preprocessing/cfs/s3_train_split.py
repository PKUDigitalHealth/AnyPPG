import pandas as pd
from os.path import join, basename

if __name__ == '__main__':

    dataset_dir = './preprocessing/cfs/Proc_CFS_Patient_Level'
    train_ratio, valid_ratio, test_ratio = 0.8, 0.1, 0.1

    df = pd.read_csv('./preprocessing/cfs/logs/log.csv')
    shuffled_df = df[['file', 'num_records']].sample(frac=1, random_state=927)

    num_train, num_valid = int(train_ratio * len(df)), int(valid_ratio * len(df))
    num_test = len(df) - num_train - num_valid

    split = ['train'] * num_train + ['valid'] * num_valid + ['test'] * num_test
    shuffled_df['Split'] = split

    shuffled_df['Path'] = shuffled_df['file'].apply(
        lambda x: join(dataset_dir, basename(x))
    )

    shuffled_df.to_csv('./preprocessing/cfs/logs/split.csv', index=False)