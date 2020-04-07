import numpy as np
import matplotlib.pyplot as plt
'main function'
def decision_tree_regression(X,y,max_level,level=0):

    size = X.shape
    try:
        n_features = size[1]
    except IndexError:
        n_features = 1
        X = X[:,None]

    'let the algorithm choose the feature'
    p_list = np.zeros(n_features)
    MSE_list = np.zeros(n_features)
    for feature in range(n_features): #for each feature, find the best partition
        feature_grid = np.linspace(np.max(X[:,feature]),np.min(X[:,feature]),1000)
        MSE_grid = []
        for p in feature_grid: #for each partition, find the MSE
            'subset 1'
            X1 = X[X[:,feature]>=p]
            y1 = y[X[:,feature]>=p]
            if len(y1)>0:
                MSE1 = np.linalg.norm(y1-np.mean(y1))
            else:
                MSE1 = 0
            'subset 2'
            X2 = X[X[:,feature]<p]
            y2 = y[X[:,feature]<p]
            if len(y2)>0:
                MSE2 = np.linalg.norm(y2-np.mean(y2))
            else:
                MSE2 = 0
            'MSE of the partition'
            MSE_grid.append(MSE1+MSE2)

        idx = np.argmin(MSE_grid)
        p_list[feature] = feature_grid[idx]
        MSE_list[feature] = MSE_grid[idx]

    feature = np.argmin(MSE_list)
    optimal_p = p_list[feature]

    Tree = [optimal_p]
    feature_Tree = [feature]

    'two subsets'
    X1,y1 = X[X[:,feature]>=optimal_p], y[X[:,feature]>=optimal_p] # subset 1
    X2,y2 = X[X[:,feature]<optimal_p], y[X[:,feature]<optimal_p] # subset 2

    values_Tree = []
    if len(y1)>0: #if y1 is not empty
        values_Tree.append(np.mean(y1))
    else:
        values_Tree.append(0)
    if len(y2)>0:
        values_Tree.append(np.mean(y2))
    else:
        values_Tree.append(0)


    X_subsets = [X1,X2]
    y_subsets = [y1,y2]

    if level<max_level:
        level = level+1

        Tree_next_level = []
        values_Tree_next_level = []
        feature_Tree_next_level = []
        for i in range(2):
            if len(y_subsets[i])>0: #if the set is not empty
                new_tree_list,new_values_tree,new_feature_list = decision_tree_regression(X_subsets[i],
                                                                               y_subsets[i],
                                                                               max_level=max_level,
                                                                               level=level)
                Tree_next_level.append(new_tree_list)
                values_Tree_next_level.append(new_values_tree)
                feature_Tree_next_level.append(new_feature_list)
            else:
                Tree_next_level.append('stop')
                values_Tree_next_level.append('stop')
                feature_Tree_next_level.append('stop')

        Tree.append(Tree_next_level)
        values_Tree.append(values_Tree_next_level)
        feature_Tree.append(feature_Tree_next_level)
    return Tree,values_Tree,feature_Tree

'predictor function'
def tree_predictor(Tree,values_Tree,feature_Tree,new_point,max_level):
    next_level = True
    level = 0
    try: #check whether new_point is a scalar or a vector
        new_point[0]
    except IndexError: #new_point is a scalar
        new_point = new_point[None] #new_point is a 1-component vector
    while level<max_level:
        feature = feature_Tree[0]
        p = Tree[0]
        if new_point[feature]>=p:
            y_predicted = values_Tree[0]
            Tree = Tree[1][0]
            values_Tree = values_Tree[2][0]
            feature_Tree = feature_Tree[1][0]
            if values_Tree == 'stop':
                return y_predicted
        else:
            y_predicted = values_Tree[1]
            Tree = Tree[1][1]
            values_Tree = values_Tree[2][1]
            feature_Tree = feature_Tree[1][1]
            if values_Tree == 'stop':
                return y_predicted
        level = level + 1

    'deepest level'
    p = Tree[0]
    feature = feature_Tree[0]
    if new_point[feature]>=p:
        y_predicted = values_Tree[0]
    else:
        y_predicted = values_Tree[1]
    return y_predicted

'entropy function'
def entropy(p):
    if p!=0:
        return -p*np.log2(p)
    else:
        return 0

from collections import Counter
def proportions(labels):
    total = len(labels)
    return [count/total for count in Counter(labels).values()]

'entropy of a subset function'
def subset_entropy(proportions):
    return np.sum([entropy(p) for p in proportions])

'entropy of a partition function'
def entropy_partition(subsets):
    'returns the entropy from this partion of data into subsets'
    total_count = sum(len(subset) for subset in subsets)
    return sum(subset_entropy(subset)*len(subset)/total_count for subset in subsets)

'main function'
def decision_tree(X,labels,level=0,max_level=1):

    _,n_features = X.shape

    'let the algorithm choose the feature'
    p_list = np.zeros(n_features)
    entropy_list = np.zeros(n_features)
    for feature in range(n_features):

        feature_grid = np.linspace(np.max(X[:,feature]),np.min(X[:,feature]),100)
        entropy_grid = []
        for p in feature_grid:
            subset1 = labels[X[:,feature]>=p]
            subset2 = labels[X[:,feature]<p]
            subsets =[proportions(subset1), proportions(subset2)] #list of subset lists
            entropy_grid.append(entropy_partition(subsets))
        idx = np.argmin(entropy_grid)
        p_list[feature] = np.round(feature_grid[idx],2)
        entropy_list[feature] = entropy_grid[idx]

    feature = np.argmin(entropy_list)
    optimal_p = p_list[feature]

    Tree = [optimal_p]
    feature_Tree = [feature]

    #split subset X into two subsets
    X1,labels1 = X[X[:,feature]>=optimal_p], labels[X[:,feature]>=optimal_p]
    X2,labels2 = X[X[:,feature]<optimal_p], labels[X[:,feature]<optimal_p]

    label_Tree = []
    if len(labels1)>0: #if labels1 is not empty
        label_Tree.append(Counter(labels1).most_common()[0][0])
    else:
        label_Tree.append(9999)
    if len(labels2)>0:
        label_Tree.append(Counter(labels2).most_common()[0][0])
    else:
        label_Tree.append(9999)


    X_subsets = [X1,X2]
    labels_subsets = [labels1,labels2]

    if level<max_level:
        level = level+1

        Tree_next_level = []
        label_Tree_next_level = []
        feature_Tree_next_level = []
        for i in range(2):
            if len(labels_subsets[i])>0 and len(Counter(labels_subsets[i]))>1: #if nonempty and more than one class
                tree_list1,label_tree1,feature_list1 = decision_tree(X_subsets[i],labels_subsets[i],level=level,max_level=max_level)
                Tree_next_level.append(tree_list1)
                label_Tree_next_level.append(label_tree1)
                feature_Tree_next_level.append(feature_list1)
            else:
                Tree_next_level.append('stop')
                label_Tree_next_level.append('stop')
                feature_Tree_next_level.append('stop')

        Tree.append(Tree_next_level)
        label_Tree.append(label_Tree_next_level)
        feature_Tree.append(feature_Tree_next_level)
    return Tree,label_Tree,feature_Tree

def draw_partitions(Tree,feature_Tree,xlim,ylim,level=0,max_level=1):
    'only for bidimensional (two features) datasets'


    p = Tree[0]
    feature = feature_Tree[0]
    'draw the line'

    if feature==0: #vertical line
        plt.plot([p,p],ylim,'k')
    else: #horizontal line
        plt.plot(xlim,[p,p],'k')


    'go one level deeper'
    if level<max_level:
        if feature==0:
            level = level + 1
            Tree1 = Tree[1][0]
            Tree2 = Tree[1][1]
            xlim1 = [p,xlim[1]]
            xlim2 = [xlim[0],p]
            feature_Tree1 = feature_Tree[1][0]
            feature_Tree2 = feature_Tree[1][1]
            if Tree1!='stop':
                draw_partitions(Tree1,feature_Tree1,xlim1,ylim,level=level,max_level=max_level)
            if Tree2!='stop':
                draw_partitions(Tree2,feature_Tree2,xlim2,ylim,level=level,max_level=max_level)
        else:
            level = level + 1
            Tree1 = Tree[1][0]
            Tree2 = Tree[1][1]
            ylim1 = [p,ylim[1]]
            ylim2 = [ylim[0],p]
            feature_Tree1 = feature_Tree[1][0]
            feature_Tree2 = feature_Tree[1][1]
            if Tree1 != 'stop':
                draw_partitions(Tree1,feature_Tree1,xlim,ylim1,level=level,max_level=max_level)
            if Tree2 != 'stop':
                draw_partitions(Tree2,feature_Tree2,xlim,ylim2,level=level,max_level=max_level)

'classifier function'
def tree_classifier(tree,label_tree,feature_tree,new_point,max_level):
    next_level = True
    level = 0
    while level<max_level:
        feature = feature_tree[0]
        p = tree[0]
        if new_point[feature]>=p:
            new_label = label_tree[0]
            tree = tree[1][0]
            label_tree = label_tree[2][0]
            feature_tree = feature_tree[1][0]
            if label_tree == 'stop':
                return new_label
        else:
            new_label = label_tree[1]
            tree = tree[1][1]
            label_tree = label_tree[2][1]
            feature_tree = feature_tree[1][1]
            if label_tree == 'stop':
                return new_label
        level = level + 1

    'deepest level'
    p = tree[0]
    feature = feature_tree[0]
    if new_point[feature]>=p:
        new_label = label_tree[0]
    else:
        new_label = label_tree[1]
    return new_label
