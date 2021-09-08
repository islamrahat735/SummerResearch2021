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
from keras.layers import Conv2D, MaxPooling2D, Input, Lambda, Conv1D, Activation, GlobalAveragePooling1D

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

from keras.regularizers import L1L2
l1 = 0
l2 = 0
activation_custom = 'relu'
def TCN_Block(inp, activation_custom, vals, jump=True, length=8):
    t = Conv1D(vals[0], length, padding='same')(inp)

    def sub_block(activation_custom, fc_units, stride, inp, length):
        t1 = Conv1D(fc_units, 1, strides=stride, padding='same', kernel_regularizer=L1L2(l1, l2))(inp)
        t = BatchNormalization(axis=-1)(inp)
        t = Activation(activation_custom)(t)
        t = Dropout(0.5)(t)
        t2 = Conv1D(fc_units, length, strides=(stride), dilation_rate=1, padding='causal', kernel_regularizer=L1L2(l1, l2))(t)
        t = add([t1, t2])

        t1 = Conv1D(fc_units, 1, strides=1, padding='same', kernel_regularizer=L1L2(l1, l2))(t)
        t = BatchNormalization(axis=-1)(t)
        t = Activation(activation_custom)(t)
        t = Dropout(0.5)(t)
        t2 = Conv1D(fc_units, length, strides=(1), dilation_rate=2, padding='causal', kernel_regularizer=L1L2(l1, l2))(t)
        t = add([t1, t2])

        t1 = Conv1D(fc_units, 1, strides=1, padding='same', kernel_regularizer=L1L2(l1, l2))(t)
        t = BatchNormalization(axis=-1)(t)
        t = Activation(activation_custom)(t)
        t = Dropout(0.5)(t)
        t2 = Conv1D(fc_units, length, strides=(1), dilation_rate=4, padding='causal', kernel_regularizer=L1L2(l1, l2))(t)
        t = add([t1, t2])
        return t

    tout1 = sub_block(activation_custom, vals[0],1,t, length)
    tout2 = sub_block(activation_custom, vals[1],jump+1,tout1, length)
    tout3 = sub_block(activation_custom, vals[2],jump+1,tout2, length)
    tout4 = sub_block(activation_custom, vals[3],jump+1,tout3, length)

    return tout1, tout2, tout3, tout4

def precustom(layer):
    layer = BatchNormalization(axis=-1)(layer)
    layer = Activation(activation_custom)(layer)
    layer = GlobalAveragePooling1D()(layer)
    return layer

def MLP(fc_units, t):
    t = Dense(fc_units, kernel_regularizer=L1L2(l1, l2))(t)
    t = BatchNormalization(axis=-1)(t)
    t = Activation(activation_custom)(t)
    t = Dropout(0.5)(t)
    t = Dense(fc_units, kernel_regularizer=L1L2(l1, l2))(t)
    t = BatchNormalization(axis=-1)(t)
    t = Activation(activation_custom)(t)
    t = Dropout(0.5)(t)
    return t

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
                if row[3] == '1': #only correct imitations will be forwarded -ADDED code
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
#        t = x[6]
#        s = y[6]
#        for elem in range(0,8):
#            x[elem] -= t
#            y[elem] -= s
        arr =[]
        arr.append(x)
        arr.append(y)
#        arr = np.array(arr)
#        reshaped = np.reshape(arr, 16)
        return arr

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
        skip = 1
        prog = tqdm(range(start, end,skip))

        #loop to load each image
        for path_index in prog:
            #get path to image
            image_path = self.path[0][path_index]
            image_pathS = self.path[2][path_index]


            label = self.get_label(image_path, PatientList)
            if label == 'random':
                continue
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
                    x_cor = 0
                    y_cor = 0
                    for x in range(path_index, path_index + timestep):
                        p = self.path[2][x]
                        seq = self.get_img(p)

                        seq_a = seq[0]
                        seq_b = seq[1]

                        if x == path_index:
                            x_cor = seq_a[6]
                            y_cor = seq_b[6]

                        d = []
                        ang = []
                        for ind in range(len(seq_a)):
#                            d.append(np.sqrt((seq_a[ind]-x)*(seq_a[ind]-x) + (seq_b[ind]-y)*(seq_b[ind]-y)))
#                            ang.append(np.arctan2(seq_a[ind]-x, seq_b[ind]-y))
                            seq_a[ind] = (seq_a[ind]-x_cor)/x_cor
                            seq_b[ind] = (seq_b[ind]-y_cor)/y_cor


                        final_seq = []
                        final_seq.append(seq_a)
                        final_seq.append(seq_b)

                        final_seq = np.array(final_seq)
                        final_seq = np.reshape(final_seq, 16)
                        sequence.append(final_seq)
                    image_listS.append(np.array(sequence))
                    label_list.append(diction[label])

        return np.array(image_listD), np.array(image_listR), np.array(image_listS), label_list

def loadSkeleton(tr_st, tr_en, val_st, val_en, te_st, te_en, frames, gestureType = 2):
    print("skeleton only")
    csvPaths = glob.glob(r"timestamps/labels_time_stamps_release/*")
    # print (csvPaths)
    PatientList = []
    for x in range(0, 58):
        # print(x)
        PatientList.append(PatientInfo(csvPaths[x]))

    # print(PatientList[0].timestamps[0].begin)
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


def fifty50Split(data, label, class_index):
    x = []
    y = []
    temp = np.where(np.array(label) == class_index, 1, 0).nonzero()
    indeces_of_ones = temp[0]
    for index in indeces_of_ones:
        x.append(data[index])
        y.append(label[index])
    count = len(y)
    temp = np.where(np.array(label) != class_index, 1, 0).nonzero()
    indeces_of_ones = temp[0]
    
    np.random.shuffle(indeces_of_ones)
    subset = indeces_of_ones[0:count]
    for index in subset:
        x.append(data[index])
        y.append(label[index])
    
    y = np.where(np.array(y) == class_index, 1, 0)
    return np.array(x), np.array(y)

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
    fc_units = 512#_args.fcUnits
    num_point = 2*8  #assumes each sample tick
    num_frame = 35#_args.timestep  #assume this to be in orders of time
    learning_rate = 0.001#_args.lr
    model_num = 1#_args.model
    gesture =2# _args.gesture

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
        print(f'using model {model_num}')
        print(f'looking at gesture types: {gestureInWords}')
    # the data, split between train and test sets

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = ([],[]),([],[]),([],[])

    if _args.customSelection == 1:
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = loadSkeleton(_args.trPatientSt, _args.trPatientEn,
                                                        _args.valPatientSt, _args.valPatientEn,
                                                        _args.tePatientSt, _args.tePatientEn,
                                                        num_frame, gesture)
        print("combining val and test set...")
        x_val= np.concatenate((x_val,x_test), axis=0)
        y_val.extend(y_test)
        x_test = x_val
        y_test = y_val
        print("complete")
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
#    y_train = keras.utils.to_categorical(y_train, num_classes)
#    y_val = keras.utils.to_categorical(y_val, num_classes)
#    y_test = keras.utils.to_categorical(y_test, num_classes)
    accuracies=[]
    total_prediction=[]
    #recurrent neural network assuming each row of an image as time and each column as individual sample
    print(np.unique(y_train))
    for i in range(len(np.unique(y_train))):
        train_x, train_y = fifty50Split(x_train, y_train, i)
        val_x, val_y = fifty50Split(x_val, y_val, i)
        test_x = val_x
        test_y = val_y
        
        # val_y = np.where(np.array(y_val) == i, 1, 0)
        
        main_input = Input(shape=(num_frame, num_point))

        if(model_num == 0):
            vals = [4, 8, 16, 32]
            _, _, _, t = TCN_Block(main_input, 'relu', vals, jump=True, length=6)

            t = precustom(t)
            t = MLP(1024, t)
            t = Dense(1,activation='sigmoid', name='out')(t)

        else:
            t = LSTM(256, return_sequences=True)(main_input)
            t = LSTM(256, return_sequences=False)(t)
            t = BatchNormalization(axis=-1)(t)#before here
            t = Dropout(0.5)(t)
            t = Dense(fc_units*2, activation='relu',)(t)
            t = Dense(fc_units, activation='relu')(t)
            t = Dropout(0.5)(t)
            t = Dense(fc_units*2, activation='relu')(t)
            t = Dropout(0.5)(t)
            t = BatchNormalization(axis=-1)(t)
            t = Dense(1, activation='sigmoid', name='out')(t)

        model = Model(inputs=main_input, output=t)
        model.summary()

        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
        #saving weights from best model
        import os
        filepath = 'save' + str(num_frame) + '\\'
        os.makedirs(filepath, exist_ok=True)
        filepath = 'save' + str(num_frame) + '\\' + str(model_num) + 'best' + str(i) +'.hdf5'
        from keras.callbacks import ModelCheckpoint
        #checkpoint1 is for saving on windows systems
        checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy',
                                     verbose=0, save_best_only=True, save_weights_only=True,
                                     mode='max')
        #checkpoint2 is for saving on linux (arc) systems
        checkpoint2 = ModelCheckpoint(filepath, monitor='val_accuracy',
                                     verbose=0, save_best_only=True, save_weights_only=True,
                                     mode='max')

        #create a list of callbacks, having both checkpoint1 and checkpoint2 is fine
        callbacks_list = [checkpoint1, checkpoint2]

        model.fit(train_x, train_y, shuffle=True,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2,
                  validation_data=(val_x, val_y), 
                  callbacks=callbacks_list)

        model.load_weights(filepath)
        prediction = model.predict(x_test)
        a = np.rint(prediction)
        acc_array = []
        for index in range (len(a)):
            if a[index] == test_y[index]:
                acc_array.append(1)
            else:
                acc_array.append(0)
        acc = np.sum(acc_array)/len(acc_array)
        accuracies.append(acc)
        total_prediction.append(prediction)

        f = open('acc.txt', 'a')
        f.write("Model: %d, class: %d, Accuracy: %.2f%%\n"
                % (model_num, i, acc))
        f.close()

        #draw_cm(np.argmax(y_test,1), np.argmax(prediction,1), f"CM_fold_{i}_{_args.trainingFold}_{numToGesture[gesture]}.pdf", gestureType=gesture)

        del model
        K.clear_session()
    
    average_acc= np.sum(np.array(accuracies))/len(accuracies)
    
    # final_prediction =np.argmax(total_prediction, 0)
    # results = []
    # for i in range(len(y_test)):
    #     if final_prediction[i] == y_test[i]:
    #         results.append(1)
    #     else:
    #         results.append(0)
    # proper_acc = np.sum(results)/ len(results)
    print(f"average accuracy is {average_acc}")
    # print(f"proper accuracy is {proper_acc}")
    print(accuracies)






