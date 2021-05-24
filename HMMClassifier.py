from hmmlearn import hmm
import numpy as np
import pandas as pd
import librosa
import pickle
import os

def create_hmm_classifier(components, cov_type, n_iter, tol, min_cov, verbose, parameters, feature_path, model_save_path):

    pitches = ["Am", "A", "Bm", "B", "C", "Dm", "D", "Em", "E", "F", "G"]
    feat=pd.read_csv(feature_path)    
    hmm_chord_models = []
    
    inc = 0
    for x in range (len(pitches)):
        label = pitches[x]
        train_data = np.array([])
        for i in range (inc, inc+60):
            if len(train_data) == 0:
                train_data = feat[i:i+1]
            else:
                train_data = np.append(train_data, feat[i:i+1], axis=0)

        inc += 60        
        hmm_model = hmm.GaussianHMM(n_components=components, covariance_type=cov_type, n_iter=n_iter, tol=tol, min_covar = min_cov, verbose=verbose, params=parameters )
        hmm_model.fit(train_data)
        hmm_chord_models.append((hmm_model, label))
        hmm_model = None
  
  
    os.makedirs(os.path.dirname(model_save_path+"\\"), exist_ok=True)
    for x in range(len(hmm_chord_models)):
        with open( model_save_path + "\\" + pitches[x] + ".pkl", "wb") as file: pickle.dump(hmm_chord_models[x], file)
    
def eval_single_chord_hmm(feature_path, model_path, chord_count):
    
    pitches = ["Am", "A", "Bm", "B", "C", "Dm", "D", "Em", "E", "F", "G"]
    features = pd.read_csv(feature_path)
    hmm_chord_models = []
    predicted_chords = []
    
    for x in range(len(pitches)):
        hmm_chord_models.append(pickle.load( open(  model_path + "\\" + pitches[x] + ".pkl", "rb" ) ))
    
    for x in range (chord_count*len(pitches)):
            scores=[]
            for i in range(len(hmm_chord_models)):
                hmm_model, label = hmm_chord_models[i]
        
                score = hmm_model.score(features[x:x+1])
                scores.append(score)
            n=np.array(scores).argmax()        
            predicted_chords.append(hmm_chord_models[n][1])
    
    good_chord = 0
    start = 0
    end = chord_count
    
    for x in range(len(pitches)):
        for i in range(start, end):
            if predicted_chords[i] == pitches[x]:
                good_chord+=1
        start+=chord_count
        end+=chord_count
    
    
    print("Correctly predicted: ", good_chord)
    print("Total chords: ", len(predicted_chords))
    print(predicted_chords)    


#create_hmm_classifier(4, "tied", 40, 0.0001, 0.0001, True, "mts", "Features\chr.csv", "HMM\ChrHMM_n4")
eval_single_chord_hmm("Features\chrVal.csv", "HMM\ChrHMM_n4", 6)