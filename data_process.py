import numpy as np
import random

#解析subject_video文件
def get_subject(file_name):
    subject_file = open(file_name,'r')
    line = subject_file.readline()
    subjects = []
    videos = []
    while line:
        line = [int(x) for x in line.strip().split()]
        subjects.append(line[0])
        videos.append(line[1])
        line = subject_file.readline()
    subject_file.close()
    return subjects,videos

#写subject_video子数据集
def write_subject(testers_num,tested_ids,data,file_name,index,length):
    subject_file = open(file_name,'w')
    for i in range(testers_num):
        if i+1 not in tested_ids:
            continue
        for j in range(index[i],index[i]+length[i]):
            subject_file.write(str(i+1)+'       '+str(data[i][j][0])+'\n')
    subject_file.close()

#写EEG_feature子数据集
def write_feature(testers_num,tested_ids,data,file_name,index,length):
    feature_file = open(file_name,'w')
    dim = len(data[0][0][1])
    for i in range(testers_num):
        if i+1 not in tested_ids:
            continue
        for j in range(index[i],index[i]+length[i]):
            for k in range(dim):
                feature_file.write(str(data[i][j][1][k])+'	    ')
            feature_file.write('\n')
    feature_file.close()

#写valence_arousal_label子数据集
def write_valabel(testers_num,tested_ids,data,file_name,index,length):
    valabel_file = open(file_name,'w')
    dim = 2
    for i in range(testers_num):
        if i+1 not in tested_ids:
            continue
        for j in range(index[i],index[i]+length[i]):
            for k in range(dim):
                valabel_file.write(str(data[i][j][2][k])+'      ')
            valabel_file.write('\n')
    valabel_file.close()

#写EEG_emotion_category子数据集
def write_emotion(testers_num,tested_ids,data,file_name,index,length):
    emotion_file = open(file_name,'w')
    for i in range(testers_num):
        if i+1 not in tested_ids:
            continue
        for j in range(index[i],index[i]+length[i]):
            emotion_file.write(str(data[i][j][3])+'\n')
    emotion_file.close()

def process_DEAP():
    root_dir = './data/DEAP/'
    testers_num = 32                    #被测者数量
    videos_num = 38                     #视频种类
    data_num = testers_num*videos_num   #HEAP的总数据量是规则的
    splits_num = 5                      #分成5个子数据集
    #feature_dim = 126

    #每个被测者每个组(即子数据集)应该有的数据量
    temp1 = [int(videos_num/splits_num)+1]*testers_num
    temp2 = [(videos_num - (int(videos_num/splits_num)+1)*(splits_num-1))]*testers_num  
    splitted_data_num = [temp1]*(splits_num-1)
    splitted_data_num.append(temp2)
    splitted_data_num = np.array(splitted_data_num,dtype=int)
      
    testers_data = []                   #总数据集
    testers_data_num = [0]*testers_num
    for i in range(testers_num):
        testers_data.append([])

    #从文件获取数据
    subjects,videos = get_subject(root_dir+'subject_video.txt')
    features = np.loadtxt(root_dir+'EEG_feature.txt').tolist()
    va_labels = np.loadtxt(root_dir+'valence_arousal_label.txt',dtype=int).tolist()

    #确认有被实验的用户id
    tested_ids = set(subjects)   

    #整理数据
    #testers_data[testers_id][record_index][video_id,feature,valabels,emotion_category]
    for i in range(data_num):
        testers_id = subjects[i]-1
        testers_data_num[testers_id] += 1
        testers_data[testers_id].append([])
        index = len(testers_data[testers_id]) - 1
        testers_data[testers_id][index].append(videos[i])
        testers_data[testers_id][index].append(features[i])
        testers_data[testers_id][index].append(va_labels[i])

    #随机化
    for i in range(testers_num):
        random.shuffle(testers_data[i])

    #如果不放心shuffle之后对应关系有没有错，可以在下面这句设断点查看对应关系
    #index 存储每个被测者的数据已经存了多少了
    #legnth 表示每个被测者在这个子数据集中要存多少数据
    index = np.zeros((testers_num),dtype=int)
    for i in range(splits_num):
        length = splitted_data_num[i]
        write_subject(testers_num,tested_ids,testers_data,
                        root_dir+'subject/subject_video_'+str(i)+'.txt',index,length)
        write_feature(testers_num,tested_ids,testers_data,
                        root_dir+'feature/EEG_feature_'+str(i)+'.txt',index,length)
        write_valabel(testers_num,tested_ids,testers_data,
                        root_dir+'valabel/valence_arousal_label_'+str(i)+'.txt',index,length)
        index += length


#两个数据集的处理仅在 “每个被测者的数据量、以及他们的5个子数据集的数据量” 不同
#因为MAHNOB-HCI的数据是不规则的，有些被测者未被记录，有些被测者对于某几个视频的结果未被记录
def process_HCI():
    root_dir = './data/HCI/'
    testers_num = 30    #id最大为30，有3个人没被登记
    videos_num = 20     
    data_num = 0
    splits_num = 5
    #feature_dim = 126
    #每个组应该有的数据量
    splitted_data_num = np.zeros((splits_num,testers_num),dtype=int)    
    #每个被测者已随机分配的数据量                         
    #splitted_data_count = [[0]*splits_num]*testers_num    
    testers_data = []
    testers_data_num = np.zeros((testers_num),dtype=int)
    for i in range(testers_num):
        testers_data.append([])

    #从文件获取数据
    subjects,videos = get_subject(root_dir+'subject_video.txt')
    features = np.loadtxt(root_dir+'EEG_feature.txt').tolist()
    va_labels = np.loadtxt(root_dir+'valence_arousal_label.txt',dtype=int).tolist()
    emotions = np.loadtxt(root_dir+'EEG_emotion_category.txt',dtype=int).tolist()

    data_num = len(subjects)
    #确认有被实验的用户id
    tested_ids = set(subjects)   

    #整理数据
    #testers_data[testers_id][record_index][video_id,feature,valabels,emotion_category]
    for i in range(data_num):
        testers_id = subjects[i]-1
        testers_data_num[testers_id] += 1
        testers_data[testers_id].append([])
        index = len(testers_data[testers_id]) - 1
        testers_data[testers_id][index].append(videos[i])
        testers_data[testers_id][index].append(features[i])
        testers_data[testers_id][index].append(va_labels[i])
        testers_data[testers_id][index].append(emotions[i])
    
    #因为HCI数据有缺失，所以要确定每个用户每组有多少数据
    testers_average_num = np.array(testers_data_num/splits_num,dtype=int)
    for i in range(splits_num-1):
        for j in range(testers_num):
            splitted_data_num[i][j] = testers_average_num[j]
    i = splits_num-1
    for j in range(testers_num):
        splitted_data_num[i][j] = testers_data_num[j] - testers_average_num[j]*i

    #随机化
    for i in range(testers_num):
        random.shuffle(testers_data[i])

    #如果不放心shuffle之后对应关系有没有错，可以在下面这句设断点查看对应关系
    #index 存储每个被测者的数据已经存了多少了
    #legnth 表示每个被测者在这个子数据集中要存多少数据
    index = np.zeros((testers_num),dtype=int)
    length = np.zeros((testers_num),dtype=int)
    for i in range(splits_num):
        length = splitted_data_num[i]
        write_subject(testers_num,tested_ids,testers_data,
                        root_dir+'subject/subject_video_'+str(i)+'.txt',index,length)
        write_feature(testers_num,tested_ids,testers_data,
                        root_dir+'feature/EEG_feature_'+str(i)+'.txt',index,length)
        write_valabel(testers_num,tested_ids,testers_data,
                        root_dir+'valabel/valence_arousal_label_'+str(i)+'.txt',index,length)
        write_emotion(testers_num,tested_ids,testers_data,
                        root_dir+'emotion/EEG_emotion_category_'+str(i)+'.txt',index,length)
        index += length

if __name__ == "__main__":
    process_DEAP()
    process_HCI()



