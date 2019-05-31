import numpy as np 

class SVM_dual:
    def __init__(self,max_iter,ktup,C=1,toler=1e-3):
        self.C = C                      #Penalty parameter C of the error term
        self.max_iter = max_iter        #max_iteration_times
        self.toler = toler              #tolerance

        self.sample_num = 0             #样本总数量
        self.dim = 0                    #输入维数
        self.alphas = None
        self.labels = None              #标签集
        self.label_index = dict()
        self.label_back = dict()
        # 考虑到label可能缺失，如HCI数据集无8,9,10号emotion_category
        # 则label_index[11] = 7, label_back[7] = 11

        self.b = 0
        self.omega = None
        self.data = None				#训练数据
        self.label_array = None			#训练数据标签
        
        self.K = None					#核函数转换矩阵
        self.ktup = ktup                #kernel type
        self.eCache = None				#误差缓存

        self.predict_y = None			#预测结果
        self.accuracy = 0				#准确率

    #提前计算好核函数转换矩阵
    def kernel_trans(self): 
        if self.ktup[0]=='linear':      #linear kernel
            self.K = np.dot(self.data, self.data.T)
        
        elif self.ktup[0]=='rbf':       #gaussian kernel
            for i in range(self.sample_num):
                for j in range(i,self.sample_num):
                    #根据公式 ||xi-xj|| 计算
                    deltaRow = self.data[j] - self.data[i]
                    self.K[i][j] = np.dot(deltaRow, deltaRow.T)
                    self.K[j][i] = self.K[i][j]
        self.K = np.exp(self.K/(-1*self.ktup[1]**2)) 

    def analysis(self,x,y):
        self.labels = list(set(y))
        if len(self.labels)!=2:
            print('Fault!')

        #将标签映射成 1 或 -1
        raw_label = self.labels[0]
        self.label_index[raw_label] = -1
        self.label_back[-1] = raw_label
        raw_label = self.labels[1]
        self.label_index[raw_label] = 1
        self.label_back[1] = raw_label

        #根据映射关系，把传入的训练数据的标签改成 1 或1-1
        self.label_array = np.zeros((self.sample_num),dtype=int)
        for i in range(self.sample_num):
            if self.labels[0]==y[i]:
                self.label_array[i] = -1
            else:
                self.label_array[i] = +1

        #计算和函数转换矩阵
        self.K = np.zeros((self.sample_num,self.sample_num))
        self.kernel_trans()

    def fit(self,x,y):
        self.sample_num = len(y)
        self.dim = len(x[0])
        self.alphas = np.zeros((self.sample_num),dtype=float)
        self.eCache = np.zeros((self.sample_num,2)) 
        self.data = np.array(x)
        self.analysis(x,y)

        iter = 0
        alphaPairsChanged = 0
        entireSet = True

        #使用 PlattSMO 算法进行 SVM 分类器的训练
        while (iter < self.max_iter) and ((alphaPairsChanged > 0) or (entireSet)):  
            alphaPairsChanged = 0  
            #先遍历所有样本
            if entireSet:   #go over all  
                for i in range(self.sample_num):          
                    alphaPairsChanged += self.inner_loop(i)  
                iter += 1  
            #再遍历非支持向量
            else:
                nonBoundIs = np.nonzero((self.alphas > 0) * (self.alphas < self.C))[0]  
                for i in nonBoundIs:  
                    alphaPairsChanged += self.inner_loop(i)   
                iter += 1  
            if entireSet: 
                entireSet = False #toggle entire set loop  
            elif (alphaPairsChanged == 0): 
                entireSet = True    
            # print ("iteration number: %d" % iter ) 
        
        self.omega = np.dot(self.data.T,(self.alphas*self.label_array).T)
        
    def score(self,x,y):
        num = len(y)
        wrong_num = 0
        right_num = 0
        self.accuracy
        self.predict_y = np.zeros((num),dtype=int)
        for i in range(num):
            f_xi = np.dot(self.omega,x[i])
            if f_xi>=0:
                self.predict_y[i] = 1
            else:
                self.predict_y[i] = -1
            temp = self.label_back[self.predict_y[i]]
            if temp==y[i]:
                right_num += 1
            else:
                wrong_num += 1
        self.accuracy = right_num/num
        return self.accuracy

    def selectJ_rand(self,i):
        j=i #we want to select any J not equal to i
        while j==i:
            j = np.random.randint(self.sample_num)
        return j

    #启发式选择第 2 个 alpha
    def selectJ(self,i,Ei): 
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        self.eCache[i] = [1,Ei] 
        validEcacheList = np.nonzero(self.eCache[:,0])[0]
        if (len(validEcacheList)) > 1:
            #遍历有效的误差缓存表，找到能让 |Ei-Ej| 最大化的 alpha_j
            for k in validEcacheList:   
                if k == i: 
                    continue     
                Ek = self.calc_Ek(k)
                deltaE = abs(Ei - Ek)
                if (deltaE > maxDeltaE):
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        #无可选 alpha
        else:   
            j = self.selectJ_rand(i)
            Ej = self.calc_Ek(j)
            return j, Ej

    def inner_loop(self,i):
        Ei = self.calc_Ek(i)
        if ((self.label_array[i]*Ei < -self.toler) and (self.alphas[i] < self.C)) \
            or ((self.label_array[i]*Ei > self.toler) and (self.alphas[i] > 0)):

            #启发式选择会让参数更新最大化的第二个 alpha 
            j,Ej = self.selectJ(i,Ei) 
            
            #准备更新
            alpha_i_old = self.alphas[i].copy()
            alpha_j_old = self.alphas[j].copy()

            #获取裁剪上下限
            if (self.label_array[i] != self.label_array[j]):
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            else:
                L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                H = min(self.C, self.alphas[j] + self.alphas[i])
            if L==H: 
                return 0

            #即公示的的分母 η 
            eta = 2.0 * self.K[i,j] - self.K[i,i] - self.K[j,j] #changed for kernel
            if eta >= 0: 
                return 0

            #更新 alpha_j
            self.alphas[j] -= self.label_array[j]*(Ei - Ej)/eta
            #裁剪，确保 0<alpha_j<C，因为要确保符合 KKT 条件
            self.clip_alpha(j,H,L)
            self.update_Ek(j) #added this for the Ecache
            if (abs(self.alphas[j] - alpha_j_old) < 0.00001): 
                return 0
            
            #更新 alpha_i
            self.alphas[i] += self.label_array[j]*self.label_array[i]*(alpha_j_old - self.alphas[j])
            self.update_Ek(i)

            #更新 b
            b1 = self.b - Ei- self.label_array[i] * (self.alphas[i] - alpha_i_old) * self.K[i,i] \
                    - self.label_array[j] * (self.alphas[j] - alpha_j_old) * self.K[i,j]
            b2 = self.b - Ej- self.label_array[i] * (self.alphas[i] - alpha_i_old) * self.K[i,j] \
                    - self.label_array[j] * (self.alphas[j] - alpha_j_old) * self.K[j,j]
            if (0 < self.alphas[i]) and (self.C > self.alphas[i]): 
                self.b = b1
            elif (0 < self.alphas[j]) and (self.C > self.alphas[j]): 
                self.b = b2
            else: 
                self.b = (b1 + b2)/2.0
            return 1
        else: 
            return 0
    
    def calc_Ek(self,k):
        f_xk = np.dot(self.alphas*self.label_array, np.dot(self.data,self.data[k].T))
        Ek = f_xk - float(self.label_array[k])
        return Ek

    def clip_alpha(self,j,H,L):
        if self.alphas[j] > H: 
            self.alphas[j] = H
        if L > self.alphas[j]:
            self.alphas[j] = L

    #after any alpha has changed update the new value in the cache
    def update_Ek(self,k):
        Ek = self.calc_Ek(k)
        self.eCache[k] = [1,Ek]