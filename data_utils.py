import torch.utils.data as data
import json
import pandas
import pandas as pd
import numpy as np
import scipy.sparse as sp
import random

def load_all(test_num=100):
    filename='C:/Users/Dingwenzhao/Desktop/Arts_Crafts_and_Sewing_5.json'
    data=pd.read_json(filename,lines=True)
    data=data[['reviewerID','asin']]
    nums_user,nums_common=0,0
    #用户ID和用户编号的字典
    users=dict()
    for user in data['reviewerID']:
        if user not in users:
            users[user]=nums_user
            nums_user+=1
    #用编号替换ID
    for id in users:
        data['reviewerID'].loc[data['reviewerID']==id]=users[id]

    #商品的ID替换成编号
    commons=dict()
    for common in data['asin']:
        if common not in commons:
            commons[common]=nums_common
            nums_common+=1
    for id in commons:
        data['asin'].loc[data['asin']==id]=commons[id]

    #创建用户-商品矩阵
    data = data.values.tolist()
    train_mat = sp.dok_matrix((nums_user, nums_common), dtype=np.float32)
    for x in data:
        train_mat[x[0], x[1]] = 1.0

    #计算每个用户有几个正样本，取x倍负样本（在没看过的商品里取）  x采样率
    negative_list=[]
    for i in range(nums_user):
        temp=0  #记录user有几个正样本
        unsee_item = []
        for j in range(nums_common):
            if (i,j) in train_mat:
                temp+=1
            else:
                unsee_item.append([i,j])
        #从unsee_item里面取出x*temp个添加到negative_list 采样率设为2
        for _ in random.sample(unsee_item, 2 * temp):
            negative_list.append(_)
    return data, negative_list, nums_user, nums_common, train_mat


class NCFData(data.Dataset):
	def __init__(self, features, 
				num_item, train_mat=None, num_ng=0, is_training=None):
		super(NCFData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features_ps = features
		self.num_item = num_item
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training
		self.labels = [0 for _ in range(len(features))]

	def ng_sample(self):
		assert self.is_training, 'no need to sampling when testing'

		self.features_ng = []
		for x in self.features_ps:
			u = x[0]
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while (u, j) in self.train_mat:
					j = np.random.randint(self.num_item)
				self.features_ng.append([u, j])

		labels_ps = [1 for _ in range(len(self.features_ps))]
		labels_ng = [0 for _ in range(len(self.features_ng))]

		self.features_fill = self.features_ps + self.features_ng
		self.labels_fill = labels_ps + labels_ng

	def __len__(self):
		return (self.num_ng + 1) * len(self.labels)

	def __getitem__(self, idx):
		features = self.features_fill if self.is_training \
					else self.features_ps
		labels = self.labels_fill if self.is_training \
					else self.labels

		user = features[idx][0]
		item = features[idx][1]
		label = labels[idx]
		return user, item ,label

