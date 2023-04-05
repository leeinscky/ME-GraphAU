import numpy as np
import os

list_path = '../data/BP4D/list'
class_num = 12

for i in range(1,4):
    read_list_name = 'BP4D_train_label_fold'+str(i)+'.txt'
    save_list_name = 'BP4D_train_AU_relation_fold'+str(i)+'.txt'
    aus = np.loadtxt(os.path.join(list_path,read_list_name)) # aus.shape = (n,12)
    le = aus.shape[0] # n
    new_aus = np.zeros((le, class_num * class_num)) # 初始化一个n x 144的矩阵
    for j in range(class_num): # 12
        for k in range(class_num): # 12
            new_aus[:,j*class_num+k] = 2 * aus[:,j] + aus[:,k] 
            # new_aus[:,j*class_num+k]表示new_aus数组的第j*class_num+k列，即当前处理的AU组合所在的列。一共有12*12=144列，每一列都是一个AU组合
            # aus[:,j]表示aus数组的第j列，即所有图像中的第j个AU的标签。aus[:,k]表示aus数组的第k列，即所有图像中的第k个AU的标签。
            # 2 * aus[:,j] + aus[:,k]表示将第j个AU的标签的值乘以2，再加上第k个AU的标签的值。这是因为new_aus的每个元素都是由两个AU的标签值相加而得到的。具体来说，这个表达式表示：将第j个AU和第k个AU的存在关系都视为重要，并且重要性相等。它们的和将被放置在new_aus中的相应位置，以便后续分析。
            # 综上所述，new_aus[:,j*class_num+k] = 2 * aus[:,j] + aus[:,k]将当前处理的两个AU的关系求和，并将结果存储在new_aus的相应列中，以便后续分析。
            # 将aus[:,j]乘以2是为了在不同AU之间加入权重，以便更好地表示它们之间的关系。所有可能的情况: 2*0+0=0, 2*0+1=1, 2*1+0=2, 2*1+1=3
    np.savetxt(os.path.join(list_path,save_list_name),new_aus,fmt='%d')