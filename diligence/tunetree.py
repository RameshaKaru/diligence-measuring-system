from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from .tree import hpTree2, hpTreeFull

class TuneTree:

    def __init__(self, cnfg, x_data, kmeans, num_clusters, verbose=False):
        self.x_data = x_data
        self.kmeans = kmeans
        self.num_clusters = num_clusters
        self.verbose = verbose
        self.cnfg = cnfg
        self.a = cnfg['tree_param']['a']
        self.b = cnfg['tree_param']['b']
        self.c = cnfg['tree_param']['c']
        self.t1 = cnfg['tree_param']['t1']
        self.t2 = cnfg['tree_param']['t2']
        self.allowed_num_mistakes = cnfg['tree_param']['allowed_mistake_proportion'] * len(x_data)
        self.min_levels = cnfg['tree_param']['min_levels']
        self.iteration = 0

    def tune(self):
        print("Tuning the hyperplane tree")
        print("ITERATION ", self.iteration)

        levels, mistakes, num_leaves, kb_tree = self.draw_tree(a=self.a, b=self.b, c=self.c, t1=self.t1, t2=self.t2)
        self.iteration += 1

        if (levels < self.min_levels) or (num_leaves < self.num_clusters):
            self.a = self.a / 5
            self.tune()
        elif mistakes > self.allowed_num_mistakes:
            self.c = self.c * 5
            self.tune()
        else:
            print("Hyperplane tree is drawn")
            # kb_tree.count_reduced(0.1)
            kb_tree.weights_csv()


    def draw_tree(self, a=500, b=10, c=500, t1=0.05, t2=5):

        kb_tree = hpTree2(k=self.num_clusters, a=a, b=b, c=c, t1=t1, t2=t2, verbose=self.verbose)
        kb_tree = kb_tree.fit(self.x_data, self.kmeans)
        kb_tree.plot("intermediate_outputs/kb_tree")

        levels = kb_tree.height(kb_tree.tree)
        mistakes = kb_tree.total_mistakes
        num_leaves = kb_tree.num_leaves
        print("Levels:", levels, "Mistakes:", mistakes, "Num_leaves:", num_leaves)

        return levels, mistakes, num_leaves, kb_tree

    def draw_full_tree(self):
        kb_tree_full = hpTreeFull(k=self.num_clusters)
        kb_tree_full = kb_tree_full.fit(self.x_data, self.kmeans)
        kb_tree_full.plot("full_tree")