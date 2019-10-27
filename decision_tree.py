import math
import random
import numpy as np
import pandas as pd
import copy


class Node:
    def __init__(self, predict, parent_feature_val =None, data=None, values=None):
        if values is None:
            values = {}
        self.data = data
        self.values = values
        self.parent_feature_val = parent_feature_val
        self.predict = predict


def create_data(k, m):
    col_names = []
    weights = []
    sum = 0
    for i in range(k):
        col_names.append("X" + str(i + 1))
    col_names.append("Y")
    for u in range(2, k+1):
        sum += pow(0.9, u)
    for l in range(1, k):
        weights.append(pow(0.9, l+1) / sum)
    data = []

    for i in range(m):
        weighted_avg = 0
        lst = []
        lst.append(random.randint(0, 1))
        for j in range(1, k):
            choices = [lst[j-1], 1-lst[j-1]]
            prob = [0.75, 0.25]
            lst.append(np.random.choice(choices, p=prob))

        for o in range(len(weights)):
            weighted_avg+= weights[o]*lst[o+1]
        if weighted_avg>=0.5:
            lst.append(lst[0])
        else:
            lst.append(1-lst[0])
        data.append(lst)

    return pd.DataFrame(data, columns=col_names)




class DecisionTreeID3():

    def __init__(self,tree=None):
        self.tree=tree

    def _entropy(self,y):

        res=0
        prob=dict(y.value_counts(normalize=True))
        for key in prob:
            res-= prob[key]*np.log2(prob[key])
        return res


    def _max_info_gain(self,data):
        info_gain={}
        for i in list(data)[:-1]:
            res=0
            entr={}
            x=dict(data[i].value_counts(normalize=True))
            for key in x:
                data_split=self._split(data,key,i)
                entr[key]=self._entropy(data_split["Y"])
                res+=x[key]*entr[key]
            info_gain[i]=self._entropy(data["Y"])-res

        return info_gain

    def _split(self, data, key, feature):
        return data[data[feature] == key]

   # initial_distribution= dict(x["Y"].value_counts(normalize=True))
    #for v in initial_distribution:
     #   current_info_gain-=initial_distribution[v]*np.log2(initial_distribution[v])
    #tree=node(-1,-1)
    #print(max_info_gain(x))

    def fit_tree(self,data, tree = None):
        if tree == None:
            tree = Node(predict =-1)
            self.tree = tree
        if len(data["Y"].unique())==1:
            tree.data=-1
            tree.predict=int(data["Y"].unique())
            return
        #if len(data["Y"].unique())==0:
        #    return
        info_gain=self._max_info_gain(data)
        feature=max(info_gain,key= lambda k:info_gain[k])
        tree.data=feature
        for i in data[feature].unique():
            temp = Node(predict = -1)
            tree.values[i]= temp
        for j in tree.values:
            #if len(data["Y"].unique()!=1 and tree.data!=-1):
            data_split=self._split(data,j,feature)
            self.fit_tree(data_split,tree.values[j])


    def _predict_row(self,tree,row):
        feature = tree.data
        while(feature!=-1):
            x=row[tree.data]
            tree=tree.values[int(row[tree.data])]
            feature = tree.data
        return tree.predict


    def predict (self, data):
        y = []

        for i in range(data.shape[0]):
            predicted_val = (self._predict_row(self.tree, data.iloc[[i]]))
            y.append(predicted_val)
        return y

    def error (self, test_data):
        wrong_prediction_count = 0
        predicted_values = self.predict(test_data)
        for i in range(len(predicted_values)):
            if predicted_values[i] != test_data["Y"][i]:
                wrong_prediction_count+= 1
        #print(predicted_values)
        #print(test_data["Y"])
        return wrong_prediction_count/len(test_data)





x=create_data(4,100)
model = DecisionTreeID3()
model.fit_tree(x)
print("Hello")
#model.predict(x)
print(model.error(x))
x['y_pred'] = model.predict(x)
print(x)
"""
y=[]
count=0
for i in x:
    predicted_val=(predict(tree,i))
    y.append(predicted_val)
    if predicted_val==i[-1]:
        count+=1
print(count/(len(x)))
"""

#print(model.tree.data)
#print(x.drop(columns=["X1"]))






















