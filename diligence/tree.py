import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cvxpy as cp
import pandas as pd
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpMinimize, getSolver
from copy import deepcopy
import itertools

# import pulp as pl
try:
    from graphviz import Source

    graphviz_available = True
except Exception:
    graphviz_available = False

# print(listSolvers(onlyAvailable=True))
solver = getSolver('CPLEX_PY')


def plot_kmeans2(kmeans, x_data):
    cmap = plt.cm.get_cmap('RdYlBu')

    k = kmeans.n_clusters
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .1),
                         np.arange(y_min, y_max, .1))

    values = np.c_[xx.ravel(), yy.ravel()]

    ########### K-MEANS Clustering ###########
    plt.figure(figsize=(6, 6))
    Z = kmeans.predict(values)
    Z = Z.reshape(xx.shape)
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=cmap,
               aspect='auto', origin='lower', alpha=0.4)

    y_kmeans = kmeans.predict(x_data)
    plt.scatter([x[0] for x in x_data], [x[1] for x in x_data], c=y_kmeans, s=20, edgecolors='black', cmap=cmap)
    for c in range(k):
        center = x_data[y_kmeans == c].mean(axis=0)
        plt.scatter([center[0]], [center[1]], c="white", marker='$%s$' % c, s=350, linewidths=.5, zorder=10,
                    edgecolors='black')

    plt.xticks([])
    plt.yticks([])
    plt.title("Kmeans", fontsize=14)

    return plt


def plot_all_hyperplanes(w):
    for i in range(len(w)):
        lx = np.linspace(-15, 10, 1000)
        ly = - w[i][0] / w[i][1] * lx - w[i][2] / w[i][1]

        plt.plot(lx, ly, label=str(i))

    plt.ylim(-15, 19)
    plt.title("All hyperplanes")
    plt.savefig("all_hyperplanes.png")
    plt.show


def plot_hyperplanes(w):
    for i in range(len(w)):
        lx = np.linspace(-15, 10, 1000)
        ly = - w[i][0] / w[i][1] * lx - w[i][2] / w[i][1]

        plt.plot(lx, ly, label=str(i))

    plt.ylim(-15, 19)
    plt.title("Relevant hyperplanes")
    plt.savefig("rel_hyperplanes.png")
    plt.show


def plot_1hyperplane(w):
    lx = np.linspace(-15, 10, 1000)
    ly = - w[0] / w[1] * lx - w[2] / w[1]

    plt.plot(lx, ly)

    plt.ylim(-15, 19)
    plt.show


def create_data(n=1000, d=2, k=5):
    x_data, _ = make_blobs(n, d, k, 2.5, random_state=42)

    kmeans = KMeans(k, random_state=42)
    kmeans.fit(x_data)
    centers = kmeans.cluster_centers_

    # plt = plot_kmeans2(kmeans, x_data)
    # plt.savefig("kmeans")

    return x_data, kmeans, centers


def find_hyperplane(c1, c2):
    c1.reshape(len(c1), 1)
    c2.reshape(len(c2), 1)

    mid_point = (c1 + c2) / 2
    w = c2 - c1

    c = -np.dot(w, mid_point)

    return w, c


def find_all_hyperplanes(centers):
    wi = []
    k = len(centers)
    combinations_idx = list(itertools.combinations(np.arange(k), 2))
    # print(combinations_idx)

    for i in range(len(combinations_idx)):
        c1 = centers[combinations_idx[i][0]]
        c2 = centers[combinations_idx[i][1]]
        w, c = find_hyperplane(c1, c2)
        w = np.append(w, c)
        wi.append(w)

    return np.asarray(wi), combinations_idx


def find_relevant_hyperplanes(w, combinations_idx, centers):
    solver = getSolver('CPLEX_PY')
    rel_p = []
    rel_p_ids = []
    rel_pij = {}
    for id in range(len(w)):
        pij = w[id]
        i = combinations_idx[id][0]
        j = combinations_idx[id][1]
        ci = centers[i]
        cj = centers[j]
        set_x = range(0, len(ci))

        model = LpProblem(name="rel_hyperplane", sense=LpMaximize)
        X = LpVariable.dicts("X", set_x, cat='Continuous')
        model += (lpSum([pij[v] * X[v] for v in set_x]) + pij[-1] == 0, "hyperplane_point")

        for c in range(len(centers)):
            if c != i and c != j:
                # ||x - c_c|| > ||x - c_i||
                model += (lpSum([(ci[v] - centers[c][v]) * X[v] +
                                 (centers[c][v] ** 2 - ci[v] ** 2) / 2 for v in set_x]
                                ) >= 0.000001, str(c))

        obj_func = 0
        model += obj_func

        status = model.solve(solver)
        # print("STATUS",status)
        x1 = model.variables()[0].value()
        y1 = model.variables()[1].value()

        if status == 1:
            rel_p.append(pij)
            rel_p_ids.append([i, j])
            rel_pij[str(i) + '_' + str(j)] = pij

    return rel_p, rel_p_ids, rel_pij


def sides_intersects(centers, rel_p, rel_p_ids, center_ids, remaining_rel_ids, verbose=False):
    intersect_all = {}
    sides_all = {}

    for l in remaining_rel_ids:
        i = rel_p_ids[l][0]
        j = rel_p_ids[l][1]
        intersect_k = find_intersect(rel_p[l], centers, i=rel_p_ids[l][0],
                                     j=rel_p_ids[l][1], center_ids=center_ids)
        intersect_all[str(i) + "_" + str(j)] = intersect_k

        sides = find_sides(rel_p[l], centers, intersect_k, center_ids=center_ids)
        sides_all[str(i) + "_" + str(j)] = sides

    if verbose:
        print("Intersects: ", intersect_all)
        print("Sides divided: ", sides_all)

    return sides_all


def find_intersect(pij, centers, i, j, center_ids):
    solver = getSolver('CPLEX_PY')
    intersect_k = []
    for k in center_ids:
        ck = centers[k]
        set_x = range(0, len(ck))
        if k != i and k != j:
            model = LpProblem(name="intersect", sense=LpMaximize)
            X = LpVariable.dicts("X", set_x, cat='Continuous')

            # ||x - c_k|| < ||x - c_c||
            for c in range(len(centers)):
                if c != i and c != j and c != k:
                    model += (lpSum([(ck[v] - centers[c][v]) * X[v]
                                     + (centers[c][v] ** 2 - ck[v] ** 2) / 2 for v in set_x]) >= 0.0000001, str(c))
                elif c == i or c == j:
                    model += (lpSum([(ck[v] - centers[c][v]) * X[v]
                                     + (centers[c][v] ** 2 - ck[v] ** 2) / 2 for v in set_x]) >= 0.0000001, str(c))

            model += (lpSum([pij[v] * X[v] for v in set_x]) + pij[-1] == 0, "hyperplane_point")
            obj_func = 0
            model += obj_func

            status = model.solve(solver)
            # print("STATUS", status)
            x1 = model.variables()[0].value()
            y1 = model.variables()[1].value()
            if status == 1:
                intersect_k.append(k)

    return intersect_k


def find_sides(pij, centers, intersect_k, center_ids):
    w = pij[:-1]
    right = []
    left = []
    for k in center_ids:
        if k not in intersect_k:
            side = np.dot(w, centers[k]) + pij[-1]
            if side >= 0:
                left.append(k)
            else:
                right.append(k)

    right = right + intersect_k
    left = left + intersect_k

    return [left, right]


def minimize_max(sides_all):
    max_list = []
    min_list = []
    for r in sides_all:
        clust_num = []
        for d in range(len(sides_all[r])):
            clusters = len(sides_all[r][d])
            clust_num.append(clusters)
        max_list.append(np.max(clust_num))
        min_list.append(np.min(clust_num))

    min_of_max_clusters = np.min(max_list)
    argmins_of_max_clusters = [i for i, j in enumerate(max_list) if j == min_of_max_clusters]
    if len(argmins_of_max_clusters) == 1:
        sel_pij = list(sides_all)[argmins_of_max_clusters[0]]
    else:
        second_max = [min_list[i] for i in argmins_of_max_clusters]
        min_of_second_max = argmins_of_max_clusters[np.argmin(second_max)]
        sel_pij = list(sides_all)[min_of_second_max]

    return sel_pij


def step(sides_all, verbose=False):
    sel_pij = minimize_max(sides_all)
    if verbose:
        print("Selected pij: ", sel_pij)
        print("Sides: ", sides_all[sel_pij])
    clusters_left = sides_all[sel_pij][0]
    clusters_right = sides_all[sel_pij][1]

    copy_sides_all = deepcopy(sides_all)
    copy_sides_all.pop(sel_pij)

    sides_left = deepcopy(copy_sides_all)
    sides_right = deepcopy(copy_sides_all)

    for s in sides_left:
        l = set(sides_left[s][0]).intersection(set(clusters_left))
        r = set(sides_left[s][1]).intersection(set(clusters_left))
        sides_left[s] = [list(l), list(r)]

    for s in sides_right:
        l = set(sides_right[s][0]).intersection(set(clusters_right))
        r = set(sides_right[s][1]).intersection(set(clusters_right))
        sides_right[s] = [list(l), list(r)]

    return sel_pij, sides_left, sides_right, clusters_left, clusters_right


####

def div_data(x_data, w, labels):
    ones = np.ones(len(x_data))
    x_datanew = np.append(x_data, ones.reshape(len(ones), 1), axis=1)
    dot = x_datanew * w
    dot_sum = np.sum(dot, axis=1)
    ind_left = np.where(dot_sum >= 0)
    ind_right = np.where(dot_sum < 0)
    x_data_left = x_data[ind_left]
    x_data_right = x_data[ind_right]
    labels_left = labels[ind_left]
    labels_right = labels[ind_right]

    return x_data_left, x_data_right, labels_left, labels_right


def find_mistakes(labels, clusters):
    ind = np.array([])
    for i in clusters:
        ind = np.append(ind, np.where(labels == i)[0])
    mistakes = len(labels) - len(ind.flatten())

    return mistakes


class NodeFull:
    def __init__(self):
        self.samples = None
        self.mistakes = None
        self.left = None
        self.right = None
        self.pij = None
        self.clusters = None

    def is_leaf(self):
        return (self.left is None) and (self.right is None)


class hpTreeFull:

    def __init__(self, k, verbose=False, n_jobs=None):

        self.k = k
        self.tree = None
        self.verbose = verbose
        self.n_jobs = n_jobs if n_jobs is not None else 1

    def build_tree(self, sides_all, clusters):
        """
        Build a tree.
        :param x_data: The input samples.
        :param y: Clusters of the input samples, according to the kmeans classifier given (or trained) by fit method.
        :param valid_centers: Boolean array specifying which centers should be considered for the tree creation.
        :param valid_cols: Boolean array specifying which columns should be considered fot the tree creation.
        :return: The root of the created tree.
        """

        node = NodeFull()
        if len(clusters) == 1:
            node.clusters = clusters[0]
            return node
        else:
            sel_pij, sides_left, sides_right, clusters_left, clusters_right = step(sides_all, self.verbose)
            node.pij = sel_pij
            node.clusters = clusters
            node.left = self.build_tree(sides_left, clusters_left)
            node.right = self.build_tree(sides_right, clusters_right)

            print("Node: ", node.pij, "Left: ", clusters_left, "Right: ", clusters_right)

            return node

    def fit(self, x_data, kmeans=None):
        """
        Build a threshold tree from the training set x_data.
        :param x_data: The training input samples.
        :param kmeans: Trained model of k-means clustering over the training data.
        :return: Fitted threshold tree.
        """

        if kmeans is None:
            if self.verbose > 0:
                print('Finding %d-means' % self.k)
            kmeans = KMeans(self.k, n_jobs=self.n_jobs, verbose=self.verbose, max_iter=40, random_state=0)
            kmeans.fit(x_data)

        else:
            assert kmeans.n_clusters == self.k

        centers = kmeans.cluster_centers_
        wi, combinations_idx = find_all_hyperplanes(centers)
        print("Number of all hyperplanes: ", len(wi))

        rel_p, rel_p_ids, self.rel_pij = find_relevant_hyperplanes(wi, combinations_idx, centers)
        print("Number of relevant hyperplanes: ", len(rel_p))

        if x_data.shape[1] == 2:
            plt = plot_kmeans2(kmeans, x_data)
            plt.savefig("kmeans")
            plt = plot_kmeans2(kmeans, x_data)
            plot_all_hyperplanes(wi)
            plt = plot_kmeans2(kmeans, x_data)
            plot_hyperplanes(rel_p)

        sides_all = sides_intersects(centers, rel_p, rel_p_ids, center_ids=np.arange(len(centers)),
                                     remaining_rel_ids=np.arange(len(rel_p)), verbose=self.verbose)

        self.tree = self.build_tree(sides_all, np.arange(len(centers)))

        return self

    def plot(self, filename="test", feature_names=None):
        if not graphviz_available:
            raise Exception("Required package is missing. Please install graphviz")

        if self.tree is not None:
            dot_str = ["digraph ClusteringTree {\n"]
            queue = [self.tree]
            nodes = []
            edges = []
            id = 0
            while len(queue) > 0:
                curr = queue.pop(0)
                if curr.is_leaf():
                    label = "cluster=\%d" % (curr.clusters)
                else:
                    clusters = ','.join(str(item) for item in curr.clusters)
                    label = "pij=%s\nclusters=\%s" % (curr.pij, clusters)
                    queue.append(curr.left)
                    queue.append(curr.right)
                    edges.append((id, id + len(queue) - 1))
                    edges.append((id, id + len(queue)))
                nodes.append({"id": id,
                              "label": label,
                              "node": curr})
                id += 1
            for node in nodes:
                dot_str.append("n_%d [label=\"%s\"];\n" % (node["id"], node["label"]))
            for edge in edges:
                dot_str.append("n_%d -> n_%d;\n" % (edge[0], edge[1]))
            dot_str.append("}")
            dot_str = "".join(dot_str)
            try:
                s = Source(dot_str, filename=filename + '.gv', format="png")
                s.view()
            except Exception as e:
                print(dot_str)
                raise e


# tree2 = hpTreeFull(k=5)
# x_data, kmeans, centers = create_data(k=5, d=2)
# tree2 = tree2.fit(x_data, kmeans)
# tree2.plot("5clusters_2d")




class Node:
    def __init__(self):
        self.samples = None
        self.mistakes = None
        self.left = None
        self.right = None
        self.pij = None
        self.clusters = None
        self.x_data = None
        self.labels = None
        self.count = None

    def is_leaf(self):
        return (self.left is None) and (self.right is None)

    def set_condition(self, clusters, pij):
        self.clusters = clusters
        self.pij = pij


class hpTree2:

    def __init__(self, k, verbose=False, n_jobs=None, a=100000, b=10, c=0.001, t1=0.05, t2=5):

        self.k = k
        self.tree = None
        self.verbose = verbose
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self.new_pij = {}
        self.all_data = None
        self.total_mistakes = None
        self.num_leaves = None
        self.t1 = t1
        self.t2 = t2
        self.a = a
        self.b = b
        self.c = c

    def build_tree(self, sides_all, clusters, x_data, labels):
        """
        Build a tree.
        :param x_data: The input samples.
        :param y: Clusters of the input samples, according to the kmeans classifier given (or trained) by fit method.
        :param valid_centers: Boolean array specifying which centers should be considered for the tree creation.
        :param valid_cols: Boolean array specifying which columns should be considered fot the tree creation.
        :return: The root of the created tree.
        """

        node = Node()

        if len(clusters) == 1:
            node.clusters = clusters
            node.x_data = x_data
            node.labels = labels
            node.count = len(labels)
            node.mistakes = find_mistakes(labels, clusters)
            return node
        else:
            node.clusters = clusters
            node.x_data = x_data
            node.labels = labels
            node.count = len(labels)

            node.mistakes = find_mistakes(labels, clusters)

            sel_pij, sides_left, sides_right, clusters_left, clusters_right = step(sides_all, self.verbose)
            node.pij = sel_pij

            w = self.red_weights(x_data, self.rel_pij[sel_pij], self.a, self.b, self.c)
            # CHECK
            # self.new_pij[sel_pij] = w
            x_data_left, x_data_right, labels_left, labels_right = div_data(x_data, w, labels)

            if len(x_data_left) / len(x_data) <= self.t1 or len(x_data_left) <= self.t2:
                node = self.build_tree(sides_right, clusters_right, x_data, labels)
            elif len(x_data_right) / len(x_data) <= self.t1 or len(x_data_right) <= self.t2:
                node = self.build_tree(sides_left, clusters_left, x_data, labels)
            else:
                self.new_pij[sel_pij] = w
                node.left = self.build_tree(sides_left, clusters_left, x_data_left, labels_left)
                node.right = self.build_tree(sides_right, clusters_right, x_data_right, labels_right)

            return node

    def fit(self, x_data, kmeans=None):
        """
        Build a threshold tree from the training set x_data.
        :param x_data: The training input samples.
        :param kmeans: Trained model of k-means clustering over the training data.
        :return: Fitted threshold tree.
        """
        self.all_data = x_data
        if kmeans is None:
            if self.verbose > 0:
                print('Finding %d-means' % self.k)
            kmeans = KMeans(self.k, n_jobs=self.n_jobs, verbose=self.verbose, max_iter=40, random_state=0)
            kmeans.fit(x_data)

        else:
            assert kmeans.n_clusters == self.k

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        wi, combinations_idx = find_all_hyperplanes(centers)
        print("Number of all hyperplanes: ", len(wi))

        rel_p, rel_p_ids, self.rel_pij = find_relevant_hyperplanes(wi, combinations_idx, centers)
        print("Number of relevant hyperplanes: ", len(rel_p))

        if x_data.shape[1] == 2:
            plt = plot_kmeans2(kmeans, x_data)
            plt.savefig("kmeans")
            plt = plot_kmeans2(kmeans, x_data)
            plot_all_hyperplanes(wi)
            plt = plot_kmeans2(kmeans, x_data)
            plot_hyperplanes(rel_p)

        sides_all = sides_intersects(centers, rel_p, rel_p_ids, center_ids=np.arange(len(centers)),
                                     remaining_rel_ids=np.arange(len(rel_p)), verbose=self.verbose)

        self.tree = self.build_tree(sides_all, np.arange(len(centers)), x_data, labels)

        if x_data.shape[1] == 2:
            plt = plot_kmeans2(kmeans, x_data)
            wi = np.array(list(self.new_pij.values()))
            plot_all_hyperplanes(wi)

        return self

    def plot(self, filename="test", feature_names=None):
        if not graphviz_available:
            raise Exception("Required package is missing. Please install graphviz")

        if self.tree is not None:
            total_mistakes = 0
            num_leaves = 0
            dot_str = ["digraph ClusteringTree {\n"]
            queue = [self.tree]
            nodes = []
            edges = []
            id = 0
            while len(queue) > 0:
                curr = queue.pop(0)
                if curr.is_leaf():
                    total_mistakes += curr.mistakes
                    num_leaves += 1
                    clusters = ','.join(str(item) for item in curr.clusters)
                    label = "cluster=\%s\ntotal=%d\nmistakes=\%d" % (clusters, curr.count, curr.mistakes)
                else:
                    clusters = ','.join(str(item) for item in curr.clusters)
                    label = "pij=%s\nclusters=\%s" % (curr.pij, clusters)
                    queue.append(curr.left)
                    queue.append(curr.right)
                    edges.append((id, id + len(queue) - 1))
                    edges.append((id, id + len(queue)))
                nodes.append({"id": id,
                              "label": label,
                              "node": curr})
                id += 1
            for node in nodes:
                dot_str.append("n_%d [label=\"%s\"];\n" % (node["id"], node["label"]))
            for edge in edges:
                dot_str.append("n_%d -> n_%d;\n" % (edge[0], edge[1]))
            dot_str.append("}")
            dot_str = "".join(dot_str)
            try:
                s = Source(dot_str, filename=filename + '.gv', format="png")
                s.view()
            except Exception as e:
                print(dot_str)
                raise e

            self.total_mistakes = total_mistakes
            self.num_leaves = num_leaves
            if self.verbose:
                print("Total Mistakes: ", total_mistakes)
                print("Number of leaves: ", num_leaves)

    def red_weights(self, x_data, W, a, b, c):
        ones = np.ones(len(x_data))
        x_datanew = np.append(x_data, ones.reshape(len(ones), 1), axis=1)
        w = cp.Variable(len(W))
        err = cp.logistic(-b * (cp.sum_entries(x_datanew[0] * w)) * (cp.sum_entries(x_datanew[0] * W)))
        for j in range(len(x_data)):
            err += cp.logistic(-b * (cp.sum_entries(x_datanew[j] * w)) * (cp.sum_entries(x_datanew[j] * W)))
        obj = cp.sum_squares(w - W) + a * cp.sum_entries(cp.abs(w)) + c * err
        prob = cp.Problem(cp.Minimize(obj))
        prob.solve(verbose=False, solver='SCS')
        if self.verbose:
            print(prob.status)
            print("The optimal value is", prob.value)
            print("The optimal w is")
            print(w.value)

        w_val = np.asarray(w.value)

        return w_val.reshape(len(w_val))

    def count_reduced(self, val=0.00000001):
        wi = np.array(list(self.new_pij.values()))
        flat_list = np.array([abs(item) for sublist in wi for item in sublist])
        non_zero = flat_list[np.where(flat_list > val)]

        if self.verbose:
            print("mean: ", np.mean(flat_list))
            print("reduced :", len(flat_list) - len(non_zero))
            print(len(non_zero) / len(flat_list))

    def weights_csv(self):
        wi = self.new_pij
        df = pd.DataFrame(wi)
        df.to_csv('intermediate_outputs/weights.csv')

    def height(self, tree):
        if tree is None:
            return 0
        else:
            return max(self.height(tree.left), self.height(tree.right)) + 1

# tree3 = hpTree2(k=5, a=500,c=100)
# x_data, kmeans, centers = create_data(k=5, d=2)
# tree3 = tree3.fit(x_data, kmeans)
# tree3.plot("5clusters_2d")
# print(tree3.height(tree3.tree))
