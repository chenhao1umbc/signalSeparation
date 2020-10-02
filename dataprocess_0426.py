# Data and module loading
import norbert
import h5py
import numpy as np
import pickle
import glob
output_data = []
plt.rcParams['figure.dpi'] = 200
#Load mixture PSD image
for file_path in glob.glob('./PsdAll/*.mat'):
    all_psd_file = h5py.File(file_path)
    label_array = []
    label = file_path[-8:-4]
    for let in label:
        label_array.append(int(let))
    label_array = np.array(label_array)
    all_psds = all_psd_file['allPsd']
    for i in range(2000):
        output_data.append({'mixture':np.log(all_psds[i*20:i*20+20, :]), 'label':label_array})

        
output_file = open('./mixture_data_14000.pickle', 'wb')
pickle.dump(output_data, output_file)
output_file.close()


#%%
import pickle
blt_file_path = './component/Blt.mat'
zigbee_file_path = './component/Zigbee.mat'
zigbeeASK_file_path = './component/ZigbeeASK.mat'
zigbeeBPSK_file_path = './component/ZigbeeBPSK.mat'

dataset_name_blt = 'bltPsd'
dataset_name_zigbee = 'zigbeePsd'
dataset_name_zigbeeASK = 'zigbeeASKPsd'
dataset_name_zigbeeBPSK = 'zigbeeBPSKPsd'

file_paths = [(blt_file_path,dataset_name_blt), (zigbee_file_path,dataset_name_zigbee), (zigbeeASK_file_path,dataset_name_zigbeeASK), (zigbeeBPSK_file_path,dataset_name_zigbeeBPSK)]




def label_data_process(label_data_path, dataset_name):
    label_data = []
    label_data_file = h5py.File(label_data_path)
    label_psds = label_data_file[dataset_name]
    for i in range(2000):
        label_data.append(np.log(label_psds[i*20:i*20+20, :]))
    output_file = open(label_data_path + '.pickle', 'wb')
    pickle.dump(label_data, output_file)
    output_file.close()
    return label_data

for (label_data_path, dataset_name) in file_paths:
    output_label = label_data_process(label_data_path, dataset_name)