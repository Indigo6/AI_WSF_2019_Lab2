import keras
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from data_process import get_subject

def write_score(cvscores, average_score, file_name):
    cvscores_file = open(file_name,'w')
    for i in cvscores:
        cvscores_file.write("%.4f         " % (i))
    cvscores_file.write("\n%.4f" % (average_score))
    cvscores_file.close()

def MLP_id(root_dir):
    splits_num = 5
    #先获取全部数据
    x = []
    y = []
    for i in range(splits_num):     
        #从文件获取数据
        subjects,videos = get_subject(root_dir+'subject/subject_video_'+str(i)+'.txt')
        features = np.loadtxt(root_dir+'feature/EEG_feature_'+str(i)+'.txt')
        va_labels = np.loadtxt(root_dir+'valabel/valence_arousal_label_'+str(i)+'.txt',dtype=int)
        if 'HCI' in root_dir:
            emotions = np.loadtxt(root_dir+'emotion/EEG_emotion_category_'+str(i)+'.txt',dtype=int)

        #归一化
        videos = videos / np.max(videos)
        features /= np.max(features)
        if 'HCI' in root_dir:
            emotions = emotions / np.max(emotions)
        va_labels = va_labels / np.array([np.max(va_labels[:,0]),np.max(va_labels[:,1])])

        x_temp = []
        y_temp = []
        length = len(subjects)
        for j in range(length):
            x_temp.append([])
            x_temp[j].append(videos[j])
            x_temp[j].extend(va_labels[j])
            if 'HCI' in root_dir:
                x_temp[j].append(emotions[j])
            x_temp[j].extend(features[j])
            y_temp.append(subjects[j]-1)
        x.append(x_temp)
        y_temp = to_categorical(y_temp)
        y.append(y_temp)

    cvscores = []
    #交叉验证
    for i in range(splits_num):
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for j in range(splits_num):
            if j==i:
                x_test.extend(x[j])
                y_test.extend(y[j])
            else:
                x_train.extend(x[j])
                y_train.extend(y[j])
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        input_shape = len(x_train[0])
        output_shape = len(y_train[0])
        # create model
        model = Sequential()
        model.add(Dense(12, input_dim=input_shape, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(output_shape, activation='softmax'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        model.fit(x_train, y_train,  validation_data=(x_test,y_test), 
                    epochs=50, batch_size=10, verbose=0)
        # evaluate the model
        scores = model.evaluate(x_test, y_test, verbose=0)
        print("     %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    
    average_score = sum(cvscores)/len(cvscores)
    write_score(cvscores,average_score,root_dir+'MLP/subject_id_cvscores.txt')
    return average_score
        

if __name__ == "__main__":
    root_dir = './data/DEAP/'
    print('Cross-Validation trainning on DEAP dataset')
    MLP_id(root_dir)
    root_dir = './data/HCI/'
    print('\nCross-Validation trainning on MAHNOB-HCI dataset')
    MLP_id(root_dir)