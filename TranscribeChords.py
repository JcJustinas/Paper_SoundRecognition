import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from librosa import display
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
import pickle
from hmmlearn import hmm

def cut_song_features(full_features, frame_count, feature_type):
    cut_song_features = []
    
    start_hop = 0
    loop = True
    end_hop = frame_count-1
    while loop:
        
        if feature_type=="chroma" :
            cut_song_features.append(np.mean(full_features[:,start_hop:end_hop].T, axis=0))
        elif (feature_type=="mel") or (feature_type=="stft") :
            cut_song_features.append(np.mean(librosa.power_to_db(full_features[:,start_hop:end_hop], ref=np.max).T, axis=0))
        else:
            print("Bad feature type!")
            return []
        
        if end_hop == len(full_features[0]):
            break
        start_hop = end_hop+1
        end_hop = start_hop+frame_count-1
        if end_hop >= len(full_features[0]):
            end_hop = len(full_features[0])
            
    return cut_song_features

def plot_chords(chords):
    pitches = list(chords.keys())
    fig, ax = plt.subplots()
    
    y_tick_range = []
    y_tick_rng = 1
    for x in range(len(pitches)):
        ax.broken_barh(chords[pitches[x]], (y_tick_rng-0.25,0.5))
        y_tick_range.append(y_tick_rng)
        y_tick_rng+=1
    
    ax.set_yticks(y_tick_range)
    ax.set_ylim(0, y_tick_range[len(pitches)-1]+1)
    ax.set_yticklabels(pitches)    
    
    plt.xlabel("Kadrai")
    plt.ylabel("Akordas")     
    return plt.show()

    

def transcribe_chords(classifier_type, feature_type, model_path, music_file_path, frame_cut_count):
    
    samples , sampling_rate = librosa.load(music_file_path, sr = None, mono = True, offset = 0.0, duration = None)
    
    if feature_type=="chroma" :
        sam_abs = np.abs(librosa.stft(samples))
        full_song_features = librosa.feature.chroma_stft(S = sam_abs, sr=sampling_rate, hop_length=1024)
    elif feature_type=="mel" :
        full_song_features = librosa.feature.melspectrogram(samples, sr=sampling_rate, n_mels=128, fmax=1024)
    elif feature_type=="stft" :
        full_song_features = np.abs(librosa.stft(samples))
    else:
        print("Bad feature type!")
        return
    
    pitches =  ["Am", "A", "Bm", "B", "C", "Dm", "D", "Em", "E", "F", "G"]
    predicted_chords = []    
    model = []    
    
    cutup_song_features = cut_song_features(full_song_features, frame_cut_count, feature_type)

    if classifier_type == "ann" :
        model = load_model(model_path)
        predict = model.predict_classes((pd.DataFrame(cutup_song_features)))
        
        for x in range(np.shape(cutup_song_features)[0]):
            predicted_chords.append(pitches[predict[x]])        
        
        
    elif classifier_type == "hmm" :
        for x in range(len(pitches)):
            model.append(pickle.load( open(  model_path + "\\" + pitches[x] + ".pkl", "rb" ) ))  
        
        for x in range (np.shape(cutup_song_features)[0]):
                scores=[]
                for i in range(len(model)):
                    hmm_model, label = model[i]
                    score = hmm_model.score(cutup_song_features[x:x+1])
                    scores.append(score)
                n=np.array(scores).argmax()        
                predicted_chords.append(model[n][1])        
        
    else:
        print("Bad classifier type!")
        return
    
    det_chord_ranges = {}
    
    frame_st = 0
    for x in range(len(predicted_chords)):
        det_chord_ranges[predicted_chords[x]] = []
        
    keys = list(det_chord_ranges.keys())
    print(keys)
    
    
    for x in range(len(predicted_chords)):
        det_chord_ranges[predicted_chords[x]].append((frame_st,frame_cut_count-1))
        frame_st+=frame_cut_count
    
    plot_chords(det_chord_ranges)    
    

#Classifier types: ann; hmm
#Feature types: chroma; mel, stft
transcribe_chords("ann", "chroma", "ANN\ChromaModelH", "Samples\TestMelody\Am-C.wav", 43)