import numpy as np
import pandas as pd
import librosa
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam


def create_ann_classifier(hidden_layers, input_dimentions, train_batch_size, train_epoch_count, train_verbose, feature_path, model_save_path):
    
    pitches =  ["Am", "A", "Bm", "B", "C", "Dm", "D", "Em", "E", "F", "G"]    
    features=pd.read_csv(feature_path)
    
    labels = []
    
    for i in range(len(pitches)):
        lab = [0] * len(pitches)
        lab[i] = 1
        for k in range(60):
            labels.append(lab)
    
    train_data = pd.DataFrame(labels)
    
    model = Sequential()
    
    if(hidden_layers):
        model.add(Dense(128, input_dim = input_dimentions, activation = 'relu'))
        model.add(Dense(64, activation = 'relu'))
        model.add(Dense(32, activation = 'relu'))
        model.add(Dense(11, activation = 'softmax'))
    else:
        model.add(Dense(100, input_dim = input_dimentions, activation = 'relu'))
        model.add(Dense(11, activation = 'sigmoid'))        
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    print(model.summary())    
    
    model.fit(features.values, train_data.values, batch_size=train_batch_size, epochs=train_epoch_count, verbose=train_verbose)
    
    scores = model.evaluate(features.values, train_data.values)
    print(scores)    
    
    model.save(model_save_path)

def eval_single_chord_ann(feature_path, model_path, chord_count):
    
    pitches =  ["Am", "A", "Bm", "B", "C", "Dm", "D", "Em", "E", "F", "G"]
    predicted_chords = []
    
    features = pd.read_csv(feature_path)
    model = load_model(model_path)    
    
    predict = model.predict_classes(features)
    for x in range(len(predict)):
        predicted_chords.append(pitches[predict[x]])    

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
    

#Feature imput dimentions: Chroma: 12; Mel: 128; Stft: 1025

#create_ann_classifier(False, 12, 16, 600, 0, "Features\chr.csv", "ChromaModel")
eval_single_chord_ann("Features\chrVal.csv", "ANN\ChromaModel", 6)