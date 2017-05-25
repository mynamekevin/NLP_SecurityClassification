一、文件说明：
code # 存放代码
	security_classification.py #  算法代码

config data  # 配置数据文件
	stopword.txt  # 停用词词典

prediction data  # 存放测试集测试结果

train data  # 存放训练文件

test data  # 存放测试文件

二、配置要求
package     |version
————————————————————
window	    |win-64
Anaconda3   |4.3.14
jieba 	    |0.38 
scikit-learn|0.18.1（包含在Anaconda中）

三、测试方法

cd code/
python security_classification.py 