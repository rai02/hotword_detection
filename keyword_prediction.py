import tensorflow.keras as keras
import numpy as np
import librosa
MODEL_PATH = "model.h5"
NUM_SAMPLES = 22050



class _keyword_prediction:
    model = None
    _mappings = ["down",
        "go",
        "happy",
        "left",
        "no",
        "off",
        "on",
        "right",
        "stop",
        "up",
        "yes"]
    _instance = None

    def predict(self, file_path):
        # get MFCCs
        mfccs = self.preprocess(file_path)  # no of segments, #coefficient
        #convert 2d to 4d array => (# samples we want to predict,no of segments, #coefficient,  #channels = 1)
        mfccs = mfccs[np.newaxis,...,np.newaxis]

        #make prediction
        predictions = self.model.predict(mfccs)
        preds_index = np.argmax(predictions)
        f_preds = self._mappings[preds_index]
        return f_preds

    def preprocess(self,file_path,n_mfccs=13,n_fft=2048,hop=512):
        signal,sr = librosa.load(file_path,sr=16000)
        if len(signal) > NUM_SAMPLES:
            print(f'size {len(signal)}')
            signal = signal[:NUM_SAMPLES]
        MFCCs = librosa.feature.mfcc(signal,n_mfcc=n_mfccs,n_fft=n_fft,hop_length=hop)
        return MFCCs.T



def keyword_prediction():
    if _keyword_prediction._instance is None:
        _keyword_prediction._instance = _keyword_prediction()
        _keyword_prediction.model = keras.models.load_model(MODEL_PATH)
    return _keyword_prediction._instance


if __name__ == "__main__":
    kss = keyword_prediction()
    # test1 = kss.predict("/home/devashish/projects/speech/test/test1.wav")
    # test2 = kss.predict("/home/devashish/projects/speech/test/test2.wav")
    test3 = kss.predict("/home/devashish/projects/speech/test/random2.wav")

    print(test3)