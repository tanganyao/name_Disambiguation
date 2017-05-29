# -*- coding:utf-8 -*-

from __future__ import unicode_literals

import sys, os
import codecs
import shutil

reload(sys)
sys.setdefaultencoding('utf8')
try:
    pkg_path = os.path.dirname(os.path.abspath(__file__))
except Exception, e:
    print('file path does not exists, need mount')
def _mkdir(cate_name='test_news_data', path=''):
    if os.path.exists(pkg_path+'/'+cate_name+'/'+path):
        print ('exists cate')
        _delete_cate(path=path)
    os.makedirs(pkg_path+'/'+cate_name+'/'+path)

#delete cate, use carefully
def _delete_cate(cate_name='test_news_data', path=''):
    if os.path.exists(pkg_path+'/'+cate_name+'/'+path):
        shutil.rmtree(pkg_path+'/'+cate_name+'/'+path)

def _delete_file(file_path_list):
    for f_path in file_path_list:
        os.remove(f_path)
 
def _renew_cate(name, keylist, c_name, cate_name='test_news_data'):
    print (name)
    #print(pkg_path)
    need_delete_file_list = []
    a_path = cate_name+'/'+name
    file_list = os.listdir(a_path+'/others')
    print (len(file_list))
    for f_name in file_list:
        with codecs.open(a_path+'/others'+'/'+f_name, 'r', 'utf-8') as f:
            with codecs.open(a_path+'/'+c_name+'/'+f_name, 'w', 'utf-8') as wf:
                c = f.read()
                for key in keylist:
                    if key in c:
                        wf.write(c)
                        need_delete_file_list.append(pkg_path+'/'+a_path+'/others'+'/'+f_name) 
    _delete_file(need_delete_file_list)

if __name__ == '__main__':
    name = '马云'
    keylist = ['阿里巴巴', '淘宝', '支付宝', '蚂蚁金服']
    c_name = name+'_阿里巴巴网络技术有限公司' 
    _mkdir(path=name+'/'+c_name)
    _renew_cate(name, keylist, c_name)
