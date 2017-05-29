综述：按照顺序使用下面的python程序， 如果已经生成取出全部同名为同一个个体的简历信息文件，那么只需运行2与3中的文件即可。如果也已经将新闻文件写到对应的目录中，那么只需运行3来识别即可。
运行方法：1. python samename_out_sameperson.py -f <输入简历信息文件> -o <输出文件>
		  2. 可以在其他模块调用，调用distin()
		  3. python write_data_to_cate.py (谨慎使用，可能需要重新全部覆盖重新写入新闻) 
		  4. python clf_and_rule.py 
1. samename_out_sameperson.py 目的是将简历相同的同名的人物全部指一个个体的信息删除，因为他们没有消岐的意义。程序的输出为samenam	  e_out_sameperson.txt, 该文件在write_to_cate中使用。

2. create_train_cate.py将待处理同名人物文件分别创建目录，例如：创建‘王石/王石_万科企业股份有限公司/王石_万科企业股份有限公司.txt’, 如果王石有其他的公司，也将被分到该目录下。
	使用方法：
	传入需要取出相同人名的文件，默认是/news_data/output_data/samename_out_sameperson.txt
	处理后，在news_data下创建人名分类目录

3. 使用 write_data_to_cate.py 将个体对应文件，例如：王石.txt，判断对应的企业，分类到对应的类别，例如：包含“万科、万科企业、万科企业股份有限公司”的文本将被分到‘/news_data/王石/王石_万科企业股份有限公司/’目录下。
	使用方法：
	import write_data_to_cate 
	write_data_to_cate.write_to_cate()

4. clf_and_rule.py 先规则判断，如果能识别出个体，则结束，如果不能通过训练分类器来识别个体。
	使用方法：
	python clf_and_rule.py

目录说明：
all_persons_news 目录包含所有未处理的新闻数据、人名和公司名和简称,主要为news_num_200_400.txt，具有200到400新闻量的人物
news_data 该目录包含所有的训练数据
remove_stop 该目录处理分词
output_data 该目录包含所有的去同名的简历信息，主要文件为samename_out_sameperson.txt
