import librosa as lr
import json
import os

ds_path = "dataset"
json_path = "data.json"
samples_to_consider = 22050     #this is one second worth of sound for librosa loading audio file

def prep_ds(dataset_path, json_path, n_mfcc = 13, hop_length = 512, n_fft = 2048):
    # print("here")
    data = {
        "mappings":[],
        "labels": [],
        "MFCCs":[],
        "files":[]
    }
    for i, (dir_path,dir_name,file_name) in enumerate(os.walk(ds_path)):
        if dir_path is not dataset_path:
            print(dir_path)
            category = dir_path.split(os.sep)[-1]
            data["mappings"].append(category)

            for f in file_name:
                 file_path = os.path.join(dir_path,f)
                 signal, sr = lr.load(file_path)
                 print(f"processing: {category}")

                 if len(signal) >= samples_to_consider:
                     signal = signal[:samples_to_consider]
                     MFCCs = lr.feature.mfcc(signal,n_mfcc=n_mfcc,hop_length=hop_length,n_fft=n_fft)
                     data["labels"].append(i-1)
                     data["MFCCs"].append(MFCCs.T.tolist())
                     data["files"].append(file_path)
                     print(f"{file_path}:{i-1}")


    with open(json_path,"w") as fp:
        json.dump(data,fp,indent=4)

if __name__ == "__main__":
    prep_ds(ds_path,json_path)