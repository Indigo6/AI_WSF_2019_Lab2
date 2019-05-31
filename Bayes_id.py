import numpy as np
from data_process import get_subject
from sklearn.naive_bayes import GaussianNB
import Bayes
import cProfile

def write_score(cvscores, average_score, file_name):
    cvscores_file = open(file_name,'w')
    for i in cvscores:
        cvscores_file.write("%.4f         " % (i))
    cvscores_file.write("\n%.4f" % (average_score))
    cvscores_file.close()

def Bayes_id(root_dir):
    global video_type_num, feature_dim
    global emotion_type_num, valabel_type_num
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
        # features /= np.max(features)

        #收集每个维度的可选数目，0代表连续值
        attri_option_nums = []

        x_temp = []
        y_temp = []
        length = len(subjects)
        for j in range(length):
            x_temp.append([])
            x_temp[j].append(videos[j])
            attri_option_nums.append(video_type_num)
            x_temp[j].extend(va_labels[j])
            attri_option_nums.extend(valabel_type_num)
            if 'HCI' in root_dir:
                x_temp[j].append(emotions[j])
                attri_option_nums.append(emotion_type_num)
            x_temp[j].extend(features[j])
            attri_option_nums.extend([0]*feature_dim)
            y_temp.append(subjects[j]-1)
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
        # temp = model.predict_log_proba(x_test)
        print("     %s: %.2f%%" % ('acc', score*100))
        cvscores.append(score * 100)

        # model = GaussianNB() 
        # model.fit(x_train,y_train)
        # score = model.score(x_test,y_test)
        # print("     %s: %.2f%%" % ('acc', score*100))
        # cvscores.append(score * 100)
    
    average_score = sum(cvscores)/len(cvscores)
    write_score(cvscores,average_score,root_dir+'NB/subject_id_cvscores.txt')
    return average_score
        

if __name__ == "__main__":
    root_dir = './data/DEAP/'
    (video_type_num,feature_dim,valabel_type_num) = (38,160,[2,2])
    print('Cross-Validation trainning on DEAP dataset')
    # cProfile.run("Bayes_id(root_dir)",sort='cumtime')
    Bayes_id(root_dir)
    

    root_dir = './data/HCI/'
    (video_type_num,feature_dim,valabel_type_num) = (20,160,[2,2])
    emotion_type_num = 9
    print('\nCross-Validation trainning on MAHNOB-HCI dataset')
    Bayes_id(root_dir)