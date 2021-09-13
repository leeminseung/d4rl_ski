import tqdm
import pandas as pd
import numpy as np

def get_filtered_indice():
    raw_df = pd.read_csv("DataSet_ARO1_LHT_LC010.CSV", low_memory=False)
    raw_df = raw_df.rename(columns={'Unnamed: 0': 'TIME', 'Unnamed: 5': 'TARGET', 'Unnamed: 6': 'CURRENT'})
    raw_df = raw_df.iloc[2:-1].reset_index(drop=True)
    raw_df = raw_df.iloc[:,1:].astype(np.float32)

    ### data filtering
    # env1_normal_indice = (3 < raw_df.loc[:, 'ENV1']) & (raw_df.loc[:, 'ENV1'] < 9)
    # env3_normal_indice = (6.95 < raw_df.loc[:, 'ENV3']) & (raw_df.loc[:, 'ENV3'] < 7.05)
    # # target_normal_indice = (29 < raw_df.loc[:, 'TARGET']) & (raw_df.loc[:, 'TARGET'] < 31)
    target_normal_indice = (raw_df.loc[:, 'TARGET'] == 30)
    # current_normal_indice = (24 < raw_df.loc[:, 'CURRENT']) & (raw_df.loc[:, 'CURRENT'] < 35)
    # agent_normal_indice = (raw_df.loc[:, 'AGENT'] < 40)
    # reward_normal_indice = (-5 < raw_df.loc[:, 'REWARD']) & (raw_df.loc[:, 'REWARD'] < 5)
    
    env1_q3 = raw_df['ENV1'].quantile(q = 0.98)
    env1_normal_indice = raw_df['ENV1'] <= env1_q3

    reward_q1 = raw_df['REWARD'].quantile(q = 0.001)
    reward_q3 = raw_df['REWARD'].quantile(q = 0.99)
    reward_normal_indice = (raw_df['REWARD'] >= reward_q1) & (raw_df['REWARD'] <= reward_q3)

    agent_q1 = raw_df['AGENT'].quantile(q = 0.001)
    agent_q3 = raw_df['AGENT'].quantile(q = 0.995)
    agent_normal_indice = (raw_df['AGENT'] >= agent_q1) & (raw_df['AGENT'] <= agent_q3)
    
    current_q1 = raw_df['CURRENT'].quantile(q = 0.001)
    current_q3 = raw_df['CURRENT'].quantile(q = 0.99)
    current_normal_indice = (raw_df['CURRENT'] >= current_q1) & (raw_df['CURRENT'] <= current_q3)

    # selected_bool_indice = env1_normal_indice & env3_normal_indice & target_normal_indice & current_normal_indice & agent_normal_indice & reward_normal_indice
    selected_bool_indice = env1_normal_indice & target_normal_indice & current_normal_indice & agent_normal_indice & reward_normal_indice
    return raw_df, selected_bool_indice

def get_ski_data(sequential_size=0, sampling_frequency=1):
    raw_df, selected_bool_indice = get_filtered_indice()
    
    selected_indice = np.arange(selected_bool_indice.shape[0])[selected_bool_indice]
    shifted_selected_indice = np.roll(selected_indice + sequential_size, sequential_size)
    
    selected_sequential_bool_indice = (selected_indice == shifted_selected_indice)[sequential_size:]
    if not sequential_size:
        selected_sequential_indice = selected_indice[selected_sequential_bool_indice]
    else:
        selected_sequential_indice = selected_indice[:-sequential_size][selected_sequential_bool_indice]

    filtered_df = raw_df.iloc[selected_sequential_indice]
    filtered_and_scaled_df = (filtered_df - filtered_df.mean(0)) / filtered_df.std(0)
    # except for target
    filtered_and_scaled_df = filtered_and_scaled_df.loc[:, ['ENV1', 'ENV2', 'ENV3', 'ENV4', 'CURRENT', 'AGENT']].values

    data = list()
    for row_index in tqdm.tqdm(range(0, filtered_and_scaled_df.shape[0] - sequential_size, sampling_frequency)):
        data.append(filtered_and_scaled_df[row_index : row_index + sequential_size].tolist())

    data = np.array(data)
    np.save('step_size_{},sequential_length_{}.npy'.format(sampling_frequency, sequential_size), data)
    print("Saved Successfully with data shape: {}".format(data.shape))

    print(filtered_df.mean(0))
    print(filtered_df.std(0))
    return data, filtered_df.mean(0), filtered_df.std(0)

if __name__ == '__main__':
    get_ski_data(sequential_size=55, sampling_frequency=1)
