# -*-coding:utf8 -*-
from __future__ import unicode_literals #compatible with python3

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation   
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import os
import codecs
from sklearn import metrics
import numpy as np
import jieba
from remove_stop.remove_stopwords import remove_stopwords
from time import time
import random
import sys

reload(sys)
sys.setdefaultencoding('utf8')

t0 = time()

#data_test = load_files(test_path, encoding='utf-8', decode_error='ignore')
#print(train_file)
#print(train_file.data)
#print(type(train_file.data))

#read train file
def _read_train_file(name,text):
    train_path = "test_news_data/"+name
    #test_path = "test0001.txt"
    #去掉shuffle=True,
    data_train = load_files(train_path, encoding='utf-8', decode_error='ignore')
    #单个文件测试
    words = ['']
    #with codecs.open(test_path, 'r', 'utf-8') as test_file:  
    for line in text:
        words[0] += line.strip()
	data_test = words
    return data_train, data_test

def _write_blank_file_to_others(file_path='news_data'):
	#由于某些others目录没有文件，需要添加以空白文件，防止训练出错,只在首次将
	#新闻写到目录中才用到，其他情况少用或者不用。
	dir_list =  os.listdir(file_path)
	same_sum = 0
	tag = 0
	for d in dir_list:
		tag = 0
		p = file_path+'/'+d
		#print (os.listdir(p))
		same_sum = len(os.listdir(p))
		for dd in os.listdir(p):
			if len(os.listdir(p+'/'+dd))>20:
				tag += 1
		#if 'others' not in os.listdir(p):
		'''
		if os.path.exists(p+'/others/000000.txt'):
			num += 1
			os.remove(p+'/others/000000.txt')
			#os.makedirs(p+'/others')
			with codecs.open(p+'/others/'+'000000.txt', 'w', 'utf-8') as wf:
				wf.write('h')
			num += 1
		'''
		if same_sum - tag == 1:
			print (d)
	#print(num)

#preprocess train and test data, include remove stopwords and oversample or undersample
def _preprocess(data_train, data_test=[], standard_len=20):
	data_train.data = remove_stopwords(data_train.data)
	X_train = data_train.data
	y_train = data_train.target
	return X_train, y_train
	'''
	#计算各个样本数目
	class_num = {}
	class_dict = {}
	for i, file_name in enumerate(data_train.filenames):
		if data_train.target[i] not in class_num:
			class_num[data_train.target[i]] = 0
		class_num[data_train.target[i]] += 1
		if data_train.target[i] not in class_dict:
			class_dict[data_train.target[i]] = []
		class_dict[data_train.target[i]].append(data_train.data[i])
	#print(len(class_dict[1]))
	max_num = 0 
	min_num = 1000000       
	for s in class_num.values():
		if max_num < s:
			max_num = s
		if min_num > s:
			min_num = s
	print(min_num)
	print(max_num)        
	#判断过采样或者欠采样
	if min_num > standard_len:
		#欠采样，全部样本超过标准，选择全部样本欠拟合
		X_train = np.array(data_train.data).reshape(-1, 1)
		y_train = data_train.target
		rus = RandomUnderSampler()
		X_train, y_train = rus.fit_sample(X_train, y_train)
		print("好的样本good_sample!!!!@@@@#####$$$%%%%")
		return X_tranin, y_train

	elif max_num < standard_len:
		#过采样,选择最大数目的样本，随机过采样到标准数目，然后对所有的样本随机过拟合
		for target_index, s in class_num.items(): 
			if s != max_num and s>3:
				m_sum = s
				while m_sum < standard_len:
					ran_num = random.randint(0, s-1)
					data_train.data.append(class_dict[target_index][ran_num])
					data_train.target = np.concatenate((data_train.target, np.array([target_index])))
					m_sum += 1
			elif s<=2:
				standard_len = 3
				m_sum = s
				while m_sum < standard_len:
					ran_num = random.randint(0, s-1)
					data_train.data.append(class_dict[target_index][ran_num])
					data_train.target = np.concatenate((data_train.target, np.array([target_index])))
					m_sum += 1	
		X_train = np.array(data_train.data).reshape(-1, 1)
		y_train = data_train.target
		rus = RandomOverSampler()
		X_train, y_train = rus.fit_sample(X_train, y_train)
		return X_train, y_train

	else:
		#某些样本小于标准个数，全部随机过采样到标准的数目   
		#如果   
		 
		if max_num / standard_len > 10:
			standard_len = 100
		for index, s in class_num.items(): 
			if s < standard_len and s>3:
				m_sum = s
				while m_sum < standard_len:
					rand_num = random.randint(0, s-1)
					data_train.data.append(class_dict[index][rand_num])
					data_train.target = np.concatenate((data_train.target, np.array([index])))
					m_sum += 1
			elif s<=2:
				standard_len = 3
				m_sum = s
				while m_sum < standard_len:
					ran_num = random.randint(0, s-1)
					data_train.data.append(class_dict[index][ran_num])
					data_train.target = np.concatenate((data_train.target, np.array([index])))
					m_sum += 1	
		X_train = np.array(data_train.data).reshape(-1, 1)
		y_train = data_train.target 
		rus = RandomUnderSampler()
		X_train, y_train = rus.fit_sample(X_train, y_train)
		return X_train, y_train
		#delete encounter problems
		for index, s in class_num.items():
			if s > 50:
				while len(data_train.target) > 50:
					rand_num = random.randint(0, s-1)
					del data_train.data[rand_num]
					del data_train.target[rand_num]
		X_train = data_train.data
		y_train = data_train.target
		return X_train, y_train	
		
	'''		

def train_clf(data_train, X_train, y_train, X_test):
	#不平衡注释掉
	#X_train = X_train.tolist()
	#X_train = [j for i in X_train for j in i]
	'''
	vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
	X_train = vectorizer.fit_transform(X_train)
	#X_test = vectorizer.transform(data_test.data)
	#单个文件
	X_test = vectorizer.transform(X_test)
	feature_names = vectorizer.get_feature_names()
	ch2 = SelectKBest(chi2, k=2000)
	X_train = ch2.fit_transform(X_train, y_train)
	X_test = ch2.transform(X_test)
	#print(ch2.scores_[0])
	feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
	with codecs.open("feature.txt", 'w', 'utf-8') as f:
		print("############################")
		print(len(feature_names))
		for index,feat in enumerate(feature_names):
			f.write(feat)
			f.write('  '+str(ch2.scores_[index]))
			f.write('\n')
	'''
	
	clf = RandomForestClassifier()
	pipeline= Pipeline([
			('vect', CountVectorizer()),
			('tfidf', TfidfTransformer()),
			('rfc', clf)
	])
	parameters = {
				'vect__max_df':(0.5,0.75),
				'rfc__n_estimators':(100,125)
			}
	clf1 = GridSearchCV(pipeline, parameters).fit(X_train, y_train)
	clf1_best_parameters,clf1_score,_ = max(clf1.grid_scores_,key = lambda x:x[1])
	for param_name in sorted(parameters.keys()):
		print("%s: %r" %(param_name,clf1_best_parameters[param_name]))
	print('clf1 score: '+str(clf1_score))
	print(clf1.predict_proba(X_test))
	for i in clf1.predict(X_test):
		print(data_train.target_names[i])
	
	'''	
	for clf in (MultinomialNB(), SGDClassifier(loss='log', random_state = 42)):
		pipeline = Pipeline([
						('vect', CountVectorizer()),
						('tfidf', TfidfTransformer()),
						('clf1', clf)
		])
		parameters = {
				'vect__max_df': (0.5, 0.75, 1.0),
				'vect__ngram_range':[(1,1),(1,2)],
				'tfidf__use_idf':(True,False),
				'tfidf__norm': ('l1', 'l2'),
				'clf1__alpha':(1e-2,1e-3)
		}
		#分类器 贝叶斯
		#clf1 = MultinomialNB(alpha=.01).fit(X_train, y_train, sample_weight)
		#clf1 = MultinomialNB()
		clf1 = GridSearchCV(pipeline, parameters).fit(X_train, y_train)
		clf1_best_parameters,clf1_score,_ = max(clf1.grid_scores_,key = lambda x:x[1])
		for param_name in sorted(parameters.keys()):
			print("%s: %r" %(param_name,clf1_best_parameters[param_name]))
		print('clf1 score: '+str(clf1_score))
		print(clf1.predict_proba(X_test))
		#预测
		predicted1 = clf1.predict(X_test)
		for p in predicted1:
			print(p)
			print( data_train.target_names[p] )
	'''
	
	#支持向量机
	#Cs = np.logspace(0, 2, 10)
	svc = SVC(probability=True)
	pipeline = Pipeline([
					('vect', CountVectorizer()),
					('tfidf', TfidfTransformer()),
					('clf1', svc)
		])
	parameters = {
			'clf1__kernel':('linear','rbf'), 
			'clf1__C':(1,20)
			}
	clf1 = GridSearchCV(pipeline, parameters).fit(X_train, y_train)
	clf1_best_parameters,clf1_score,_ = max(clf1.grid_scores_,key = lambda x:x[1])
	for param_name in sorted(parameters.keys()):
		print("%s: %r" %(param_name,clf1_best_parameters[param_name]))
	print('clf1 score: '+str(clf1_score))
	print(clf1.predict_proba(X_test))
	#预测
	predicted1 = clf1.predict(X_test)
	for p in predicted1:
		print(p)
		print( data_train.target_names[p] )
	#梯度下降 
	#clf3 = SGDClassifier().fit(X_train, y_train)
	#预测
	#predicted3 = clf3.predict(X_test)

	'''
	alpha = np.logspace(-3,-1,10)
	#clf4 = SVC(C=10.0, kernel='rbf', gamma=2.0, tol=1e-3, class_weight=class_weight).fit(X_train, y_train) 
	clf4.fit(X_train, y_train)
	#print(clf4.best_score_)
	#print(clf4.best_params_ )
	#print(clf4.best_estimator_)
	#预测
	predicted4 = clf4.predict(X_test)
	#单个文件输出结果
	for p in predicted1:
		print(p)
		print( data_train.target_names[p] )
	for p in predicted3:
		print(p)
		print( data_train.target_names[p] )
	for p in predicted4:
		print(p)
		print( data_train.target_names[p] )
	'''

def find_cmp_entities(p_name, c_name=''):
	with codecs.open('all_persons_news/cmp-name.txt') as f:
		cmp_entities = [line.split(":")[2:] for line in f.readlines() \
						if line.split(':')[0]==p_name and c_name in line.split(":")[2:]] \
						if c_name!='' else [line.split(':')[2:] for line in f.readlines() if line.split(':')[0] == p_name]
		cmp_entities = [j for i in cmp_entities for j in i]

	return cmp_entities

def _find_namelist(text):
	with codecs.open('all_persons_news/cmp-name.txt', 'r', 'utf-8') as f:
		namelist = set([''.join(name_and_num.split(':')[0]) for name_and_num in f.readlines()])
		namelist = [name for name in namelist if name in text]
	#print(' '.join(namelist))
	return 	namelist

def identify_entity(text_path):
	with codecs.open(text_path, 'r', 'utf-8') as f:
		text = f.read()
	#print (text)
	namelist = _find_namelist(text)
	print ('人名：'+' '.join(namelist))
	for name in namelist:
		cmp_entities = find_cmp_entities(name)
		#print (' '.join(cmp_entities))
		#use the rules first
                '''
		for entity in cmp_entities:
			#print (entity)
			if entity.strip() in text:
				print ( 'By rules: name:' +name+"cmp"+ entity )
				return entity
                '''
		#train classifier to classify
		print("By clf:")
		data_train, data_test = _read_train_file(name, text)
		X_train, y_train = _preprocess(data_train, data_test=data_test) 
		entity = train_clf(data_train, X_train, y_train, data_test)
		print(entity)

	t = time()-t0
	print("time:%0.3f" % t)

if __name__ == '__main__':
	#_write_blank_file_to_others()
	#text_path = 'test0001.txt'
	text_path = 'test0002.txt'
	identify_entity(text_path)
	#print ( entity )
	#cmp_list = find_cmp_entities("刘弘", c_name='布丁酒店浙江股份有限公司')
	#print (' '.join(cmp_list))
	#_find_namelist('a')

'''
for index, f in enumerate(train_file.data):
    print(index)
    print(f.encode("utf-8").decode('gb2312'))
'''
#将公司名称分词后添加到分词字典中

#with codecs.open('aaaaaa.txt', 'w', 'utf-8')as f:
#    f.write(str(data_train.data))
#print(data_train.target.shape)

#X_train = np.array(data_train.data).reshape(-1, 1)

#单个文件测试注释
#data_test.data = remove_stopwords(data_test.data)
#print(text_out_stopwords)
#一段文字
#words = [""]
#y_train, y_test = data_train.target, data_test.target
#y_train = data_train.target
#过采样
#ros = RandomOverSampler()
#X_train, y_train = ros.fit_sample(X_train, y_train)

#sm = SMOTE(kind='svm')
#X_train, y_train = sm.fit_sample(X_train, y_train)

#sm = SMOTEENN()
#X_train, y_train = sm.fit_sample(X_train, y_train)
#欠采样
#cnn = CondensedNearestNeighbour()
#X_train, y_train = cnn.fit_sample(X_train, y_train)

#nm1 = NearMiss(version=3)
#X_train, y_train = nm1.fit_sample(X_train, y_train)

#rus = RandomUnderSampler()
#X_train, y_train = rus.fit_sample(X_train, y_train)

#X_train_temp = []
'''
with codecs.open('aaaaaa.txt', 'w', 'utf-8')as f:
    for i in X_train:
        for j in i:
            X_train_temp.append(j)
            f.write(j+'\n')
'''
#X_train = X_train_temp
#print("**************************")
#print(y_train)
'''
class_sum = len(class_num.items())
X_ttrain = []
y_ttrain = []
X_ttest = []
y_ttest = []

for i in range(class_sum):
    index = i*standard_len
    X_train_slice = X_train[0+index : index+standard_len//2]
    y_train_slice = y_train[0+index : index+standard_len//2]
    X_test_slice = X_train[index+standard_len//2 : index+standard_len]
    y_test_slice = y_train[index+standard_len//2 : index+standard_len]
    X_ttrain.extend(X_train_slice)
    y_ttrain.extend(y_train_slice)
    X_ttest.extend(X_test_slice)
    y_ttest.extend(y_test_slice)
    print(len(X_ttrain))
'''    
#X_train, y_train = X_ttrain, y_ttrain
#X_test, y_test = X_ttest, y_ttest
#print(str(len(y_train))+"!!!!!"+str(len(y_test)))
#X_temp.clear()
#词频矩阵
#print(words)
#count_vect = CountVectorizer()
#权重矩阵
#tfidf_transformer = TfidfTransformer()
#训练数据
#x_train_counts = count_vect.fit_transform(train_file.data)
#x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
#测试数据
#x_test_counts = count_vect.transform(words)	#单个文件测试
#x_test_counts = count_vect.transform(test_file.data)
#x_test_tfidf = tfidf_transformer.transform(x_test_counts)

#with codecs.open('中文停用词表.txt', 'r', 'utf-8') as f_stop:
#		stopwords = [line.strip() for line in f_stop.readlines()]

#print(feature_names)
#print("#################name###################")
#print(''.join(data_train.filenames).split('\\')[-1])
'''
sample_weight = []
class_weight = {}
for i, file_name in enumerate(data_train.filenames):
    if data_train.target[i] not in class_weight:
        class_weight[data_train.target[i]] = 0
    class_weight[data_train.target[i]] += 1 
    if "_" in file_name.split('\\')[-1]:
        sample_weight.append(2.)
    else:
        sample_weight.append(1.)
'''
'''
max_num = 0        
for s in class_weight.values():
    if max_num < s:
        max_num = s
for s,i in class_weight.items():
    class_weight[s] = max_num / i
'''
#print(class_weight)
#print(sample_weight)
#if (''.join(data_train.filenames)).split('\\')[-1] == 'baike':
#	print("#################baike###################")
#print(vectorizer.vocabulary_)


#print(x_test_tfidf.shape)
#伯努利
#clf2 = BernoulliNB(alpha=.01).fit(X_train, y_train, sample_weight)
#clf2 = BernoulliNB(alpha=.01).fit(X_train, y_train)
#预测
#predicted2 = clf2.predict(X_test)
#支持向量机
#clf4 = LinearSVC(dual=False, tol=1e-3).fit(X_train, y_train)
#预测
#predicted4 = clf4.predict(X_test)
#print(clf4.predict_proba(X_test))

#决策树
#clf5 = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)
#predicted5 = clf5.predict(X_test)

#kfold = cross_validation.KFold(len(X_train.data), n_folds=3)
#print(cross_validation.cross_val_score(clf4, X_train, y_train))
'''
y_test = data_test.target
target_names = data_train.target_names

score = metrics.accuracy_score(y_test, predicted1)
print("accuracy:   %0.3f" % score)
score = metrics.accuracy_score(y_test, predicted2)
print("accuracy:   %0.3f" % score)
score = metrics.accuracy_score(y_test, predicted3)
print("accuracy:   %0.3f" % score)
score = metrics.accuracy_score(y_test, predicted4)
print("accuracy:   %0.3f" % score)
print(metrics.classification_report(y_test, predicted1, target_names=target_names))

print(metrics.classification_report(y_test, predicted3, target_names=target_names))
print(clf4.n_support_)
'''
'''
target_names = data_train.target_names
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(metrics.classification_report(y_test, predicted1, target_names=target_names))
print(metrics.classification_report(y_test, predicted2, target_names=target_names))
print(metrics.classification_report(y_test, predicted3, target_names=target_names))
print(metrics.classification_report(y_test, predicted4, target_names=target_names))
print(metrics.classification_report(y_test, predicted5, target_names=target_names))
'''
'''
    
#单个文件输出结果
for p in predicted2:
    print(p)
    print( data_train.target_names[p] )

#单个文件输出结果
for p in predicted3:
    print(p)
    print( data_train.target_names[p] )
    
#单个文件输出结果
for p in predicted4:
    print(p)
    print( data_train.target_names[p] )    
    
    #单个文件输出结果
for p in predicted5:
    print(p)
    print( data_train.target_names[p] )  
'''   
