import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from data_process import process_DEAP,process_HCI
import SVM



def valabel_to_value(valabel):
    return (valabel[0]-1)*1 + (valabel[1]-1)*2

def write_score(cvscores, average_score, file_name):
    cvscores_file = open(file_name,'w')
    for i in range(cvscores.shape[1]):
        cvscores_file.write("%.4f         " % (cvscores[0][i]))
        cvscores_file.write("%.4f         \n" % (cvscores[1][i]))
    cvscores_file.write("\n\n%.4f         " % (average_score[0]))
    cvscores_file.write("%.4f" % (average_score[1]))
    cvscores_file.close()

def SVM_valabel(root_dir):
    splits_num = 5
    #先获取全部数据
    x = []
    y1 = []
    y2 = []
    for i in range(splits_num):     
        #从文件获取数据
        subjects = np.loadtxt(root_dir+'subject/subject_video_'+str(i)+'.txt',dtype=int)
        features = np.loadtxt(root_dir+'feature/EEG_feature_'+str(i)+'.txt')
        va_labels = np.loadtxt(root_dir+'valabel/valence_arousal_label_'+str(i)+'.txt',dtype=int)
        if 'HCI' in root_dir:
            emotions = np.loadtxt(root_dir+'emotion/EEG_emotion_category_'+str(i)+'.txt',dtype=int)

        # #归一化
        # subjects = subjects / np.array([np.max(subjects[:,0]),np.max(subjects[:,1])]) 
        # features /= np.max(features)
        # if 'HCI' in root_dir:
        #     emotions /= np.max(emotions)
        # # va_labels = va_labels / np.array([np.max(va_labels[:,0]),np.max(va_labels[:,1])])

        x_temp = []
        y1_temp = []
        y2_temp = []
        length = len(subjects)
        for j in range(length):
            x_temp.append([])
            x_temp[j].extend(subjects[j])
            if 'HCI' in root_dir:
                x_temp[j].append(emotions[j])
            x_temp[j].extend(features[j])
            y1_temp.append(va_labels[j][0])
            y2_temp.append(va_labels[j][1])
        x.append(x_temp)
        y1.append(y1_temp)
        y2.append(y2_temp)
    # y = np.array(y,dtype=int)
    # y_valenc,y_arousal = np.split(y, 2, axis = 1)
    cvscores = [[],[]]
    #交叉验证
    for i in range(splits_num):
        x_train = []
        y1_train = []
        y2_train = []
        x_test = []
        y1_test = []
        y2_test = []
        for j in range(splits_num):
            if j==i:
                x_test.extend(x[j])
                y1_test.extend(y1[j])
                y2_test.extend(y2[j])
            else:
                x_train.extend(x[j])
                y1_train.extend(y1[j])
                y2_train.extend(y2[j])

        x_train = np.array(x_train)
        y1_train = np.array(y1_train)
        y2_train = np.array(y2_train)
        x_test = np.array(x_test)
        y1_test = np.array(y1_test)
        y2_test = np.array(y2_test)

    
        # svm = SVC(C=10)
        svm = SVM.SVM_dual(100,['rbf',1],C=6,toler=1e-3)
        svm.fit(x_train,y1_train)
        score1 = svm.score(x_test,y1_test)
        print("     on v_label, %s: %.2f%%" % ('acc', score1*100))
        cvscores[0].append(score1)

        
        # svm = SVC(C=10)
        svm = SVM.SVM_dual(100,['rbf',1],C=6,toler=1e-3)
        svm.fit(x_train,y2_train)
        score2 = svm.score(x_test,y2_test)
        print("     on a_label, %s: %.2f%%\n" % ('acc', score2*100))
        cvscores[1].append(score2)
    
    cvscores = np.array(cvscores)
    average_score = np.sum(cvscores,axis=1) / splits_num
    write_score(cvscores,average_score,root_dir+'SVM/valabel_cvscores.txt')
    return average_score
        

if __name__ == "__main__":
    root_dir = './data/DEAP/'
    print('Cross-Validation trainning on DEAP dataset')
    SVM_valabel(root_dir)
    print('\nCross-Validation trainning on MAHNOB-HCI dataset')
    root_dir = './data/HCI/'
    SVM_valabel(root_dir)