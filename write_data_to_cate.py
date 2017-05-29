# -*-coding:utf-8 -*-
from __future__ import unicode_literals #compatible with python3 unicode coding
import codecs
import os
import sys
import re
import shutil
from clf_and_rules import find_cmp_entities

reload(sys)
sys.setdefaultencoding('utf-8')

pkg_path = os.path.dirname(os.path.abspath(__file__))

def _listdir(cateName='news_data', path=''):
	return  os.listdir(cateName) if path=='' else os.listdir(cateName+'/'+path)

def _write_file(content, cateName='news_data', path=''):
	with codecs.open(cateName+'/'+path, 'w', 'utf-8') as f:
		f.write(content)

def _create_cate(cateName='news_data', path=''):
	if os.path.exists(pkg_path+'/'+cateName+'/'+path):
		print('others cate is exits')
	else:
		os.makedirs(pkg_path+'/'+cateName+'/'+path)

def _delet_cate(cateName='news_data', path=''):
	if os.path.exists(pkg_path+'/'+cateName+'/'+path):
		shutil.rmtree(pkg_path+'/'+cateName+'/'+path)

def _choose_cate_to_write(personName, companyName):
	pnameAndCompanyName = _listdir(path=personName)
	print ( ' '.join(pnameAndCompanyName) )
	for name in pnameAndCompanyName:
		companyInSameCate = _listdir(path=personName+'/'+name)
		companyInSameCate = [n.replace('.txt','') for n in companyInSameCate]
		if personName+'_'+companyName in companyInSameCate:
			print ( 'name' + name )
			return name

def _getCompanyName(name):
	companyNameList = []
	cateList = _listdir()
	if name in cateList:
		print ( name )
		companyAndNameList = _listdir(path=name)
		#print ( ' '.join(companyAndNameList) )
		for single in companyAndNameList:
			companyInSameCate = _listdir(path=name+'/'+single)
			for line in companyInSameCate:
				if '_' in line:
					companyNameList.append(line.split('_')[1].replace('.txt', ''))
	print ( ' '.join(companyNameList) )
	return companyNameList

def delete_file_recu(cate_name='news_data'):
	cate_list = _listdir()
	reg = '^\d+\.txt'
	for cate in cate_list:
		cate_with_cmp_list =  _listdir(path=cate)
		for c in cate_with_cmp_list:
			file_list = _listdir(path=cate+'/'+c)
			for f in file_list:
				m = re.match(re.compile(reg), f)
				if m:
					os.remove(cate_name+'/'+cate+'/'+c+'/'+f)

def writeToCate():
	#before write, delete the text file match '^\d+\.txt'
	#if need delete all txt in cate
	#delete_file_recu()
	#get names needed distinguish
	with codecs.open('all_persons_news/sorted_news_num.txt', 'r', 'utf-8') as f:
		for p_name in f.readlines():
			p_name = p_name.split('\t')[0]
			#print (name)
			companyNameList = _getCompanyName(p_name)
			with codecs.open('all_persons_news/news_UTF8/'+p_name+'.txt', 'r', 'utf-8') as newsf:
				content = newsf.read()
				newsList = content.split('@@@')
				for new in newsList:
					tag = True
					#print (new)
					for c_name in companyNameList:
						cmp_list = find_cmp_entities(p_name, c_name=c_name)
						cmp_in_new = False
						for cp in cmp_list:
							#print (cp.strip()+'!!!!!!!!!!!!!!!!!!!!!')
							if cp.strip() in new:
								cmp_in_new = True
								break
						#print (cmp_in_new)
						if cmp_in_new:
							print (p_name +'##########'+c_name )
							companyAndName = _choose_cate_to_write(p_name, c_name)
							print ( companyAndName )
							fileNumInCate = len(_listdir(path=p_name+'/'+companyAndName))
							index = str(fileNumInCate).zfill(6)
							_write_file(new, path=p_name+'/'+companyAndName+'/'+index+'.txt')
							newsList.remove(new)
							tag = False
							break
					#return
					#file have not company name in list, add file to others cate
					if tag :
						_create_cate(path=p_name+'/others')
						index = str(len(_listdir(path=p_name+'/others'))).zfill(6)
						_write_file(new, path=p_name+'/others/'+index+'.txt')


			#break
	#print (pkg_path)


#wtc = WriteToCate()
#writeToCate = wtc.writeToCate()

#the codes below just for test, import model cannot show the process in these codes
if __name__ == '__main__':
	#global companyNameList
	#writeToCate()
	
	#wtc = WriteToCate()
	#writeToCate = wtc._writeToCate()
