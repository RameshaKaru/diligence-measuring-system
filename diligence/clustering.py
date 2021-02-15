from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from .tunetree import TuneTree


class Clustering:

    def __init__(self, cnfg):
        print("Initializing the clustering class")
        self.cnfg = cnfg
        self.do_clustering = self.check_clustering()

    def check_clustering(self):
        if 'clustering_history' in self.cnfg.keys():
            if (self.cnfg['clustering_history']) is None:
                self.write_history()
                return True
            else:
                duration = self.cnfg['reclustering_months']
                df = pd.read_csv(self.cnfg['clustering_history'])
                m = list(df['last_clustering_month'])

                last_date = datetime.strptime(m[0], '%d-%b-%Y')
                # last_date = datetime.strptime('01-Nov-2020', '%d-%b-%Y')

                now_data_date = datetime.strptime(self.cnfg['months'][-1], '%d-%b-%Y')

                dif = self.diff_month(now_data_date, last_date)
                print("Month difference: ", dif)

                if dif >= duration:
                    self.write_history()
                    return True
                else:
                    return False

        else:
            self.write_history()
            return True

    def write_history(self):
        m = self.cnfg['months'][-1]
        df = pd.DataFrame({'last_clustering_month': [m]})
        df.to_csv('clustering_history.csv')

    def diff_month(self, d1, d2):
        return (d1.year - d2.year) * 12 + d1.month - d2.month

    def predict_cluster(self, X):

        if self.do_clustering:
            print("Reclustering")
            labels = self.cluster_data(X)
            return labels
        else:
            print("Not reclustering")
            labels = self.get_labels(X)
            return labels

    def get_labels(self, X):
        df = pd.read_csv('intermediate_outputs/centers.csv')
        self.centers = np.asarray(df).T[1:]
        print("centers shape: ", self.centers.shape)
        self.num_clusters = self.centers.shape[0]

        norms = np.zeros((X.shape[0], self.centers.shape[0]))
        for i in range(self.centers.shape[0]):
            norms[:, i] = np.linalg.norm(X - self.centers[i], axis=1, ord=2)
        labels = np.argmin(norms, axis=1)

        return labels

    def cluster_data(self, X):
        self.num_clusters = self.cnfg['num_clusters']
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)
        labels = kmeans.fit_predict(X)
        self.centers = kmeans.cluster_centers_

        df_d = {}
        for i in range(self.num_clusters):
            df_d[i] = self.centers[i]
        df = pd.DataFrame(df_d)
        df.to_csv("intermediate_outputs/centers.csv")

        if self.cnfg['explain']['hyperplane_tree']:
            tuneTree = TuneTree(self.cnfg, X, kmeans, self.num_clusters)
            tuneTree.tune()

        return labels




