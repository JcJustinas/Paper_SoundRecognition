import sys
import numpy as np
import pandas as pd
import librosa

def extract_all_features(add_to_feature_file_name, sample_numb, path_to_samples):
    Chord_labels = ["Am", "A", "Bm", "B", "C", "Dm", "D", "Em", "E", "F", "G"]
    Chromas = []
    MEL = []
    Stft = []
    for i in range(len(Chord_labels)):
        for x in range(sample_numb):
            file_path =  path_to_samples + "\\" + Chord_labels[i] + "\\" + Chord_labels[i] + str(x+1) + ".wav"
            
            print("Extracting " + file_path) 
            samples , sampling_rate = librosa.load(file_path, sr = None, mono = True, offset = 0.0, duration = None)
            sam_abs = np.abs(librosa.stft(samples))
            
            stft_one = np.mean(librosa.amplitude_to_db(sam_abs).T, axis=0)
            Stft.append(stft_one)
            
            chromagram = np.mean(librosa.feature.chroma_stft(S = sam_abs, sr=sampling_rate).T, axis=0)
            Chromas.append(chromagram)
            
            mel_spek = librosa.feature.melspectrogram(samples, sr=sampling_rate, n_mels=128, fmax=1024)
            mel_one = np.mean(librosa.power_to_db(mel_spek, ref=np.max).T, axis=0)
            
            MEL.append(mel_one) 
            
    np.set_printoptions(threshold=sys.maxsize)
            
    Chroma_csv = pd.DataFrame(Chromas)
    Chroma_csv.to_csv('chr'+ add_to_feature_file_name +'.csv', index=False)
    
    MEL_csv = pd.DataFrame(MEL)
    MEL_csv.to_csv('mel'+ add_to_feature_file_name +'.csv', index=False)  
    
    Stft_csv = pd.DataFrame(Stft)
    Stft_csv.to_csv('Stft'+ add_to_feature_file_name +'.csv', index=False)        
    
    
    

extract_all_features("", 60, "Samples")
extract_all_features("Val", 6, "Samples\Validation")