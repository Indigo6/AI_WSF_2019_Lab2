import numpy as np

class Mix_NB:
    def __init__(self):
        self.attri_option_nums = None
        self.dim = 0                    #输入维数
        self.label_num = 0              #标签数量
        self.sample_num = 0             #样本总数量
        self.num_per_label = None       #每个标签有多少样本
        self.labels = None              #标签集
        self.label_index = dict()   
        self.label_back = dict()
        # 考虑到label可能缺失，如HCI数据集无8,9,10号emotion_category
        # 则label_index[11] = 7, label_back[7] = 11

        self.discr_num = 0              #离散属性数
        self.conti_num = 0              #连续属性数
        self.conti_start = 0

        self.wrong_predict = 0
        self.right_preidct = 0
        self.accuracy = 0
        self.predict_result = None

        
        #x_mus[label][i] = average(x_i | label), x_i代表x的第i维
        self.mus = None             #一阶矩mu
        self.sigmas = None          #二阶矩sigma
        self.priori = None          #先验概率
        self.miss_prob = None       #缺失属性值的类条件概率 = 1/(|D| + Ni)
        self.cprob = []           
        #离散属性的类条件概率
        #       cprob[label][discrete_dim][option] : 
        #           第discrete_dim维离散属性取 (option值 & label) 标签的概率

    def analysis(self):
        for i in range(self.label_num):
            self.cprob.append([])

        for i in range(self.dim):
            option_nums = self.attri_option_nums[i]
            if option_nums==0:
                # 开始进入样本的连续值的那些维数
                self.conti_start = i
                self.conti_num = self.dim - self.conti_start
                break
            else:
                for j in range(self.label_num):
                    self.cprob[j].append(dict())
                self.discr_num += 1
        
        self.miss_prob = np.ones((self.conti_start),dtype=float)
        self.miss_prob /= np.array(self.attri_option_nums[:self.conti_start])

    #合法的数据形式是numpy array
    def fit(self,x,y,attri_option_nums):
        self.attri_option_nums = attri_option_nums[:]
        self.dim = len(x[0])
        self.sample_num = len(y)
        self.labels = list(set(y))
        self.label_num = len(self.labels)
        self.analysis()

        self.mus = np.zeros((self.label_num,self.conti_num),dtype=float)
        self.sigmas = np.zeros((self.label_num,self.conti_num),dtype=float)
        self.num_per_label = np.zeros((self.label_num),dtype=int)
        self.priori = np.zeros((self.label_num),dtype=float)

        #初始化raw_label和分类器label之间的映射
        for i in range(self.label_num):
            self.label_index[self.labels[i]] = i
            self.label_back[i] = self.labels[i]

        self.update(x,y)

    def update(self,x,y):
        for i in range(self.sample_num):
            label = self.label_index[y[i]]
            self.num_per_label[label] += 1

            for j in range(self.conti_start):
                if x[i][j] in self.cprob[label][j]:
                    self.cprob[label][j][x[i][j]] += 1
                else:
                    self.cprob[label][j][x[i][j]] = 0
            
            self.mus[label] += x[i][self.conti_start:]
        
        #update priori, 带有拉普拉斯修正
        self.priori = (self.num_per_label + 1) / (self.sample_num + self.label_num)

        #update mus
        for i in range(self.label_num):
            self.mus[i] /= self.num_per_label[i]

        #update sigmas
        for i in range(self.sample_num):
            label = self.label_index[y[i]]
            temp = (x[i][self.conti_start:]-self.mus[label])**2
            self.sigmas[label] += temp

        for i in range(self.label_num):
            self.sigmas[i] /= self.num_per_label[i]
        self.sigmas = np.sqrt(self.sigmas)   

        #update condition_prob for discrete attribute
        for i in range(self.label_num):
            for j in range(self.conti_start):
                for key in self.cprob[i][j]:
                    #拉普拉斯修正
                    self.cprob[i][j][key] += 1
                    self.cprob[i][j][key] /= (self.sample_num 
                                + self.attri_option_nums[j])

    def score(self,x,y):
        self.wrong_predict = 0      #预测错误数
        self.right_preidct = 0      #预测正确数
        test_num = len(y)           #测试样本数目
        #预测结果
        self.predict_result = np.zeros((test_num),dtype=int)


        for i in range(test_num):
            predict_probs = []
            for j in range(self.label_num):
                label = self.label_index[self.labels[j]]
                prob = 0
                for k in range(self.conti_start):
                    #该属性值在样本中存在
                    if x[i][k] in self.cprob[label][k]:
                        prob += np.log(self.cprob[label][k][x[i][k]])
                    #该属性值在样本中不存在
                    else:
                        prob += np.log(self.miss_prob[k])

                prob += self.Gaussian(self.mus[label],self.sigmas[label],
                                    x[i][self.conti_start:])

                prob += np.log(self.priori[label])  
                predict_probs.append(prob)

            #将预测的标签还原成真实raw标签
            self.predict_result[i] = self.label_back[np.argmax(predict_probs)]
            if self.predict_result[i]==y[i]:
                self.right_preidct += 1
            else:
                self.wrong_predict += 1
        self.accuracy = self.right_preidct/(self.right_preidct+self.wrong_predict)
        return self.accuracy


    #获取概率 P(x_i | label), 默认使用 log 
    #此处label已映射
    def Gaussian(self,mu,sigma,x):
        cond_probs = np.ones((self.conti_num),dtype=float)
        cond_probs /= (sigma * np.sqrt(2*np.pi))
        cond_probs = np.log(cond_probs)
        temp = -((x-mu)**2 /2 /(sigma**2))
        cond_probs += temp
        return cond_probs.sum()

    def predict_log_proba(self,x):
        test_num = x.shape[0]           #测试样本数目
        log_proba = []

        for i in range(test_num):
            predict_probs = []
            for j in range(self.label_num):
                label = self.label_index[self.labels[j]]
                prob = 0
                for k in range(self.conti_start):
                    #该属性值在样本中存在
                    if x[i][k] in self.cprob[label][k]:
                        prob += np.log(self.cprob[label][k][x[i][k]])
                    #该属性值在样本中不存在
                    else:
                        prob += np.log(self.miss_prob[k])

                prob += self.Gaussian(self.mus[label],self.sigmas[label],
                                    x[i][self.conti_start:])

                prob += np.log(self.priori[label])  
                predict_probs.append(prob)

            log_proba.append(predict_probs)

        return log_proba