import numpy as np
from sklearn.naive_bayes import GaussianNB
import Bayes


#降低y的维度
def category_to_index(category):
    if category in [11,12]:
        return category-4
    else:
        return category


def write_score(cvscores, average_score, file_name):
    cvscores_file = open(file_name,'w')
    for i in cvscores:
        cvscores_file.write("%.4f         " % (i))
    cvscores_file.write("\n%.4f" % (average_score))
    cvscores_file.close()

def Bayes_emotion(root_dir):
    global video_type_num, feature_dim
    global id_type_num, valabel_type_num
    splits_num = 5
    #先获取全部数据
    x = []
    y = []
    for i in range(splits_num):     
        #从文件获取数据
        subjects = np.loadtxt(root_dir+'subject/subject_video_'+str(i)+'.txt',dtype=int)
        features = np.loadtxt(root_dir+'feature/EEG_feature_'+str(i)+'.txt')
        va_labels = np.loadtxt(root_dir+'valabel/valence_arousal_label_'+str(i)+'.txt',dtype=int)
        emotions = np.loadtxt(root_dir+'emotion/EEG_emotion_category_'+str(i)+'.txt',dtype=int)

        #归一化
        subjects = subjects / np.array([np.max(subjects[:,0]),np.max(subjects[:,1])]) 
        features /= np.max(features)
        va_labels = va_labels / np.array([np.max(va_labels[:,0]),np.max(va_labels[:,1])])

        #收集每个维度的可选数目，0代表连续值
        attri_option_nums = []

        x_temp = []
        y_temp = []
        length = len(subjects)
        for j in range(length):
            x_temp.append([])
            x_temp[j].extend(subjects[j])
            attri_option_nums.append(id_type_num)
            attri_option_nums.append(video_type_num)
            x_temp[j].extend(va_labels[j])
            attri_option_nums.extend(valabel_type_num)
            x_temp[j].extend(features[j])
            attri_option_nums.extend([0]*feature_dim)

            y_temp.append(category_to_index(emotions[j]))
        x.append(x_temp)
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

        model = Bayes.Mix_NB()
        model.fit(x_train,y_train,attri_option_nums)
        score = model.score(x_test,y_test)
        print("     %s: %.2f%%" % ('acc', score*100))
        cvscores.append(score * 100)

        # model = GaussianNB() 
        # model.fit(x_train,y_train)
        # score = model.score(x_test,y_test)
        # print("     %s: %.2f%%" % ('acc', score*100))
        # cvscores.append(score * 100)
    
    average_score = sum(cvscores)/len(cvscores)
    write_score(cvscores,average_score,root_dir+'NB/emotion_category_cvscores.txt')
        

if __name__ == "__main__":
    root_dir = './data/HCI/'
    (video_type_num,feature_dim,valabel_type_num,id_type_num) = \
        (38,160,[2,2],27)
    Bayes_emotion(root_dir)