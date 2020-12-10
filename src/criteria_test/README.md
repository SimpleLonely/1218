# DRTest
Deep Neural Network, Robustness, Testing

We source our code as a open platform including all the data and algorithms for further researching.  
We provide the supplementary of our submission	"There is Limited Correlation between Coverage and Robustness for Deep Neural Networks (#387)" of ASPLOS2020 in supplementary.pdf. 



本份代码numtant_开头的文件夹大多为作者自行封装的函数/类。启动代码是train_models.py，运行本文件后将在目录外生成models文件夹，为后续计算criteria做准备，再去运行coverage_criteria中的py函数。

GoldModel定义在nmutant_model\tutorial_models.py最后，GoldModel的数据输入定义在nmutant_data/gold.py中。train_model的具体实现写在nmutant_model/model_operation中。

尝试：

试图将keras生成的模型用tf导入但一个是.h5一个保存的是session，未能找到合适解法...

