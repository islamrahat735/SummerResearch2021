# -*- coding: utf-8 -*-
"""
Created on Mon May 24 20:44:32 2021

@author: Rahat
"""
from __future__ import print_function
import csv
import glob
import cv2
import argparse
from tqdm import tqdm
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten, LSTM, Reshape, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Input, Lambda
from keras import backend as K
from keras.layers import multiply, add, concatenate
import tensorflow as tf
import seaborn as sn


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

image_size = 256

def draw_cm(trueP, modelP, nm, cmap=plt.cm.Purples, gestureType = 2):
    import itertools

    matrix = confusion_matrix(trueP, modelP)
    print(matrix)
    matrix = np.array(matrix, dtype='float')
    for i in range(len(matrix)):
        matrix[i] = matrix[i]/np.sum(matrix[i])
        matrix[i] = np.around(matrix[i], decimals = 2)
   
    column_labels = []
    if (gestureType == 0):    
        column_labels=['A1_1', 'A1_2', 'A1_3', 'A1_4', 'A1_5',
                            'A2_1', 'A2_3', 'A2_4', 'A2_5',
                            'S1_1', 'S1_2', 'S1_3',
                            'S2_2', 'S2_3', 'S2_4']
    elif (gestureType == 1):
        column_labels=['A2_2',
                      'S1_4', 'S1_5',
                      'S2_1',
                      'P1_1', 'P1_2', 'P1_3', 'P1_4', 'P1_5',
                      'P2_1', 'P2_2', 'P2_3', 'P2_4', 'P2_5']
    # print (column_labels)
    df_cm = pd.DataFrame(matrix, column_labels, column_labels)
    fig = plt.figure(figsize=(11,8))
    sn.set(font_scale=1) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})
    if gestureType == 0:    
        plt.title("Static Action Confusion Matrix", fontsize = 15)
    if gestureType == 1:    
        plt.title("Dynamic Action Confusion Matrix", fontsize = 15)
    
    plt.xlabel('Predicted', fontsize = 10)
    plt.ylabel('True', fontsize = 10)
    
    # plt.show()
    from matplotlib.backends.backend_pdf import PdfPages
    
        #saves confusion matrix to "nm"
    with PdfPages(nm) as pdf:
        pdf.savefig(fig,bbox_inches='tight')



class LabelInfo:

    def __init__(self, label, start, end):
        self.label = label
        self.start = start
        self.end = end
        self.begin = parseTime(start)
        self.last = parseTime(end)
    
    
class PatientInfo:
    
    def __init__(self, path):
        self.timestamps =[]
        temp = path.rindex("_")
        self.number = int(path[temp+1:temp+3:1])
        with open (path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.timestamps.append(LabelInfo(row[0], row[1], row[2]))

def parseTime(time):
        components = time.split('-')
        components[0] = float(components[0])*3600
        components[1] = float(components[1])*60
        components[2] = float(components[2])
        return components[0] + components[1] + components[2]


class data_process(object):
    def __init__(self, root_path):
        #obtain the paths to all images in folder
        image_path = []
        for p in root_path:
            image_path.append(self.get_path(p))

        skeletonPath = image_path[1].copy()
        for x in range(len(skeletonPath)):
            skeletonPath[x] = skeletonPath[x].replace("RGB", "Skeleton")
            skeletonPath[x] = skeletonPath[x][0 : len(skeletonPath[x])-4]
        
        image_path.append(skeletonPath)     
        
        self.path = image_path
        self.labelToNum = {'A1_1':0, 'A1_2':1, 'A1_3':2, 'A1_4':3, 'A1_5':4,
                           'A2_1':5, 'A2_2':6, 'A2_3':7, 'A2_4':8, 'A2_5':9,
                           'S1_1':10, 'S1_2':11, 'S1_3':12, 'S1_4':13, 'S1_5':14,
                           'S2_1':15, 'S2_2':16, 'S2_3':17, 'S2_4':18,
                           'P1_1':19, 'P1_2':20, 'P1_3':21, 'P1_4':22, 'P1_5':23,
                           'P2_1':24, 'P2_2':25, 'P2_3':26, 'P2_4':27, 'P2_5':28}
        
        self.labelToNumStatic = {'A1_1':0, 'A1_2':1, 'A1_3':2, 'A1_4':3, 'A1_5':4,
                           'A2_1':5, 'A2_3':6, 'A2_4':7, 'A2_5':8,
                           'S1_1':9, 'S1_2':10, 'S1_3':11,
                           'S2_2':12, 'S2_3':13, 'S2_4':14}
        
        self.labelToNumDynamic = {
                           'A2_2':0,
                           'S1_4':1, 'S1_5':2,
                           'S2_1':3,
                           'P1_1':4, 'P1_2':5, 'P1_3':6, 'P1_4':7, 'P1_5':8,
                           'P2_1':9, 'P2_2':10, 'P2_3':11, 'P2_4':12, 'P2_5':13}
                           
        
        self.labelToGestureType = {'A1_1':0, 'A1_2':0, 'A1_3':0, 'A1_4':0, 'A1_5':0,
                           'A2_1':0, 'A2_2':1, 'A2_3':0, 'A2_4':0, 'A2_5':0,
                           'S1_1':0, 'S1_2':0, 'S1_3':0, 'S1_4':1, 'S1_5':1,
                           'S2_1':1, 'S2_2':0, 'S2_3':0, 'S2_4':0,
                           'P1_1':1, 'P1_2':1, 'P1_3':1, 'P1_4':1, 'P1_5':1,
                           'P2_1':1, 'P2_2':1, 'P2_3':1, 'P2_4':1, 'P2_5':1}
        #self.type = {'mask':0, 'unmask':1}

    #get sorted path based on image number
    def get_path(self, path):
        p = glob.glob(path)
        return p
    
    def get_label(self, image_path, PatientList):
        label = "random"
        temp = image_path[image_path.rindex('T')+1 : image_path.rindex('-')]
        temp = parseTime(temp)
        patient =int(image_path[image_path.index('_')+1: image_path.index('_')+ 3 ])
        for timestamp in PatientList[patient-1].timestamps:
            if( temp>= timestamp.begin and temp <= timestamp.last):
                label = timestamp.label
                break
        return label
    
    def get_img(self, image_pathS):
        with open(image_pathS, 'r') as f:
                lines = f.readlines()
            
        imageS = []
        
        for n in range(2):
            imageS.append(lines[n].split()[6:14])
        
        
        for x in range(len(imageS)):
            for y in range(len(imageS[x])):
                index = imageS[x][y].index("e")
                base = float(imageS[x][y][:index])
                expo = int(imageS[x][y][index+1:])
                imageS[x][y] = np.float32(base * 10**expo)
        x = imageS[0]
        y = imageS[1]
        t = x[6]
        s = y[6]
        for elem in range(0,8):
            x[elem] -= t
            y[elem] -= s
        arr =[]
        arr.append(x)
        arr.append(y)
        arr = np.array(arr)
        reshaped = np.reshape(arr, 16)
        return reshaped

    #load each image and create an array
    def load_images(self,PatientList, start=0, end =100, timestep = 1, gestureType = 2):
        image_listD = []
        image_listR = []
        image_listS = []
        label_list = []
        diction = self.labelToNum
        if(gestureType == 0):
            diction = self.labelToNumStatic
        elif(gestureType == 1):
            diction = self.labelToNumDynamic
        #show a progress bar to see how many images are loaded
        prog = tqdm(range(start, end))

        #loop to load each image
        for path_index in prog:
            #get path to image
            image_path = self.path[0][path_index]
            image_pathS = self.path[2][path_index]
            

            label = self.get_label(image_path, PatientList)
            if(gestureType==0 and self.labelToGestureType[label]!=0):
                continue
            if(gestureType==1 and self.labelToGestureType[label]!=1):
                continue
            
            #
            #
            #
            if (path_index + timestep <= end):
                if( label == self.get_label(self.path[0][path_index - 1 + timestep], PatientList)):
                    sequence = []
                    for x in range(path_index, path_index + timestep):
                        p = self.path[2][x]
                        sequence.append(self.get_img(p))
                    image_listS.append(np.array(sequence))
                    label_list.append(diction[label])
            
        return np.array(image_listD), np.array(image_listR), np.array(image_listS), label_list
    
def loadSkeleton(tr_st, tr_en, val_st, val_en, te_st, te_en, frames, gestureType = 2):
    print("skeleton only")
    csvPaths = glob.glob(r"timestamps/labels_time_stamps_release/*")
    PatientList = []
    for x in range(0, 58):
        PatientList.append(PatientInfo(csvPaths[x]))
    

    depthPath = r'Data/*/Depth/*'
    rgbPath = r'Data/*/RGB/*'
    bgrImagePath = r'Data/*/Skeleton/*'

    data = data_process((depthPath,rgbPath))
    
    x, y = getData(tr_st, tr_en, data.path[0])
    a, b = getData(val_st, val_en, data.path[0])
    w, z = getData(te_st, te_en, data.path[0])
    print(f"training indeces are ({x},{y})")
    print(f"val indeces are ({a},{b})")
    print(f"testing indeces are ({w},{z})")
    imagesD, imagesR, x_train, y_train = data.load_images(PatientList,x, y , frames, gestureType)
    imagesD, imagesR, x_val, y_val = data.load_images(PatientList,a, b, frames, gestureType)
    _, _, x_test, y_test = data.load_images(PatientList,w, z, frames, gestureType)

    print("finished successfully!")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def getData(start, end, paths):
    
    counter = 0
    starting_index = 0
    ending_index = 0
    
    for image_path in paths:
        patient =int(image_path[image_path.index('_')+1: image_path.index('_')+ 3 ])
        if (patient == start):
            starting_index = counter
            break
        counter+=1
    
    counter = 0
    if (end == 58):
        ending_index = 200930
    else:
        for image_path in paths:
            patient =int(image_path[image_path.index('_')+1: image_path.index('_')+ 3 ])
            if (patient == end+1):
                ending_index = counter
                break
            counter+=1
    
    return starting_index, ending_index
    
    

if __name__ == "__main__":
    _argparser = argparse.ArgumentParser(
            description='Gesture Recognition',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument(
        '--timestep', type=int, default=1, metavar='INTEGER',
        help='Time step in network')
    _argparser.add_argument(
        '--lr', type=float, default=0.001, metavar='INTEGER',
        help='learning rate of network')
    _argparser.add_argument(
        '--epochs', type=int, default=50, metavar='INTEGER',
        help='repititions in network')
    _argparser.add_argument(
        '--trainingFold', type=int, default=1, metavar='INTEGER',
        help=' select fold for training, other two folds are for testing/validation')
    _argparser.add_argument(
        '--customSelection', type=int, default=0, metavar='INTEGER',
        help=' option to choose specific training, testing and validation sets')
    _argparser.add_argument(
        '--trPatientSt', type=int, default=1, metavar='INTEGER',
        help=' starting patient for training')
    _argparser.add_argument(
        '--trPatientEn', type=int, default=1, metavar='INTEGER',
        help=' ending patient for training')
    _argparser.add_argument(
        '--valPatientSt', type=int, default=2, metavar='INTEGER',
        help=' starting patient for validation')
    _argparser.add_argument(
        '--valPatientEn', type=int, default=2, metavar='INTEGER',
        help=' ending patient for validation')
    _argparser.add_argument(
        '--tePatientSt', type=int, default=3, metavar='INTEGER',
        help=' starting patient for validation')
    _argparser.add_argument(
        '--tePatientEn', type=int, default=3, metavar='INTEGER',
        help=' ending patient for validation')
    _argparser.add_argument(
        '--model', type=int, default=0, metavar='INTEGER',
        help=' machine learning model')
    _argparser.add_argument(
        '--gesture', type=int, default=0, metavar='INTEGER',
        help=' machine learning model')
    _argparser.add_argument(
        '--fcUnits', type=int, default=512, metavar='INTEGER',
        help=' machine learning model')
    _args = _argparser.parse_args()
    batch_size = 128
    num_classes = 29
    epochs = _args.epochs
    fc_units = _args.fcUnits
    num_point = 2*8  #assumes each sample tick
    num_frame = _args.timestep  #assume this to be in orders of time
    learning_rate = _args.lr
    model = _args.model
    gesture = _args.gesture
    
    folds = [ [1,16], [17,37], [38,58] ]
    foldSelection = [1,2,3]
    trainingParam = folds[_args.trainingFold - 1].copy()
    folds.remove(trainingParam)
    validationParam = folds[0]
    testParam = folds[1]
    
    if (gesture == 0):
        num_classes = 15
    elif (gesture == 1):
        num_classes = 14
    
    numToGesture = {0:'static', 1:'dynamic', 2:'all'}
    
    gestureInWords = numToGesture[gesture]
    
    print(f"using a timestep of {num_frame}, learning rate of {learning_rate} and there are {epochs} epochs")
    if _args.customSelection == 1:    
        print(f"training ranges from patients {_args.trPatientSt} to {_args.trPatientEn}")
        print(f"validation ranges from patients {_args.valPatientSt} to {_args.valPatientEn}")
        print(f"testing ranges from patients {_args.tePatientSt} to {_args.tePatientEn}")
    else:
        print(f"testing fold is {_args.trainingFold}. Remaining two folds are for validation and testing")
    print(f'using model {model}')
    print(f'looking at gesture types: {gestureInWords}')
    # the data, split between train and test sets
    
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = ([],[]),([],[]),([],[])
    
    if _args.customSelection == 1:
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = loadSkeleton(_args.trPatientSt, _args.trPatientEn,
                                                        _args.valPatientSt, _args.valPatientEn,
                                                        _args.tePatientSt, _args.tePatientEn,
                                                        num_frame, gesture)
        # print("combining val and test set...")
        # x_val= np.concatenate((x_val,x_test), axis=0)
        # y_val.extend(y_test)
        # x_test = x_val
        # y_test = y_val
        # print("complete")
    else:
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = loadSkeleton(trainingParam[0], trainingParam[1],
                                                        validationParam[0], validationParam[1],
                                                        testParam[0], testParam[1],
                                                        num_frame, gesture)
        print("combining val and test set...")
        x_val= np.concatenate((x_val,x_test), axis=0)
        y_val.extend(y_test)
        x_test = x_val
        y_test = y_val
        print("complete")
        
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    #recurrent neural network assuming each row of an image as time and each column as individual sample
    main_input = Input(shape=(num_frame, num_point))
    
    if(model == 1):        
        t1,t2 = Lambda(lambda x: tf.split(x,num_or_size_splits=2,axis=-1))(main_input)
        t1 = LSTM(256, return_sequences=True)(t1)
        t1 = LSTM(256, return_sequences=False)(t1)
        t2 = LSTM(256, return_sequences=True)(t2)
        t2 = LSTM(256, return_sequences=False)(t2)
        t = concatenate([t1, t2])
        t = BatchNormalization(axis=-1)(t)#before here
        t = Dropout(0.5)(t)
        t = Dense(fc_units, activation='relu',)(t)
        t = Dense(fc_units*2, activation='relu')(t)
        t = Dropout(0.5)(t)
        t = Dense(fc_units, activation='relu')(t)
        t = Dropout(0.5)(t)
        t = BatchNormalization(axis=-1)(t)
        t = Dense(num_classes, activation='softmax')(t)
    elif(model == 2):
        t1,t2 = Lambda(lambda x: tf.split(x,num_or_size_splits=2,axis=-1))(main_input)
        
        t1 = LSTM(256, return_sequences=True)(t1)
        t1 = LSTM(256, return_sequences=False)(t1)
        t1 = BatchNormalization(axis=-1)(t1)#before here
        t1 = Dropout(0.5)(t1)
        t1 = Dense(fc_units, activation='relu',)(t1)
        t1 = Dense(fc_units*2, activation='relu')(t1)
        t1 = Dropout(0.5)(t1)
        t1 = Dense(fc_units, activation='relu')(t1)
        t1 = Dropout(0.5)(t1)
        
        t2 = LSTM(256, return_sequences=True)(t2)
        t2 = LSTM(256, return_sequences=False)(t2)
        t2 = BatchNormalization(axis=-1)(t2)#before here
        t2 = Dropout(0.5)(t2)
        t2 = Dense(fc_units, activation='relu',)(t2)
        t2 = Dense(fc_units*2, activation='relu')(t2)
        t2 = Dropout(0.5)(t2)
        t2 = Dense(fc_units, activation='relu')(t2)
        t2 = Dropout(0.5)(t2)
        
        t = concatenate([t1, t2])
        t = BatchNormalization(axis=-1)(t)
        t = Dense(num_classes, activation='softmax')(t)
    elif(model == 3):
        t1,t2 = Lambda(lambda x: tf.split(x,num_or_size_splits=2,axis=-1))(main_input)
        
        t1 = LSTM(256, return_sequences=True)(t1)
        t1 = LSTM(256, return_sequences=False)(t1)
        t1 = BatchNormalization(axis=-1)(t1)#before here
        t1 = Dropout(0.5)(t1)
        t1 = Dense(fc_units, activation='relu',)(t1)
        t1 = Dense(fc_units*2, activation='relu')(t1)
        t1 = Dropout(0.5)(t1)
        t1 = Dense(fc_units, activation='relu')(t1)
        t1 = Dropout(0.5)(t1)
        t1 = BatchNormalization(axis=-1)(t1)
        t1 = Dense(num_classes, activation='softmax')(t1)
        
        t2 = LSTM(256, return_sequences=True)(t2)
        t2 = LSTM(256, return_sequences=False)(t2)
        t2 = BatchNormalization(axis=-1)(t2)#before here
        t2 = Dropout(0.5)(t2)
        t2 = Dense(fc_units, activation='relu',)(t2)
        t2 = Dense(fc_units*2, activation='relu')(t2)
        t2 = Dropout(0.5)(t2)
        t2 = Dense(fc_units, activation='relu')(t2)
        t2 = Dropout(0.5)(t2)
        t2 = BatchNormalization(axis=-1)(t2)
        t2 = Dense(num_classes, activation='softmax')(t2)
        
        t = multiply([t1, t2])
    else:
        t = LSTM(256, return_sequences=True)(main_input)
        t = LSTM(256, return_sequences=False)(t)
        t = BatchNormalization(axis=-1)(t)#before here
        t = Dropout(0.5)(t)
        t = Dense(fc_units, activation='relu',)(t)
        t = Dense(fc_units*2, activation='relu')(t)
        t = Dropout(0.5)(t)
        t = Dense(fc_units, activation='relu')(t)
        t = Dropout(0.5)(t)
        t = BatchNormalization(axis=-1)(t)
        t = Dense(num_classes, activation='softmax')(t)
        
    model = Model(inputs=main_input, output=t)
    model.summary()
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_val, y_val))
    
    
    prediction = model.predict(x_test)
    
    
    acc = np.sum((np.argmax(prediction,1) == np.argmax(y_test,1)))/len(prediction)
    print("Accuracy: " + str(acc))
    
    draw_cm(np.argmax(y_test,1), np.argmax(prediction,1), f"CM_fold{_args.trainingFold}_{numToGesture[gesture]}.pdf", gestureType=gesture)
    


    
    
    

             

