import numpy as np
import pandas as pd

class ProcessOutputData:

    def __init__(self, sc, clustering, processInput):
        print("Initializing the output processing class")
        self.sc = sc
        self.clustering = clustering
        self.processInput = processInput

    # norm distribution
    # anm, anm_id, kmeans, month_list, norm_scores
    def outputs(self, anm, labels, month_list, norm_scores, predicted_score_df):
        pos_threshold_percentage = 0.3
        neg_threshold_percentage = 0.3
        months_labels = self.processInput.months
        clust_matrix = np.zeros((len(self.processInput.anm_id), self.clustering.num_clusters))
        tot = []
        anm_mon = []
        clust_mon = [[], [], [], []]
        clust_norm = [[], [], [], []]
        anm_norm = []
        anm_wnorm = []
        ns_all = []
        anm_clust = []
        anm_clust_display = []
        trend = []
        anm_mon_display = []
        active_status = []
        last_month = len(months_labels) - 1

        for i in range(len(self.processInput.anm_id)):
            clust0 = labels[np.where(anm == self.processInput.anm_id[i])]
            mon0 = month_list[np.where(anm == self.processInput.anm_id[i])]
            ns0 = norm_scores[np.where(anm == self.processInput.anm_id[i])]

            if len(clust0) > 6:
                clust = clust0[-6:]
                mon = mon0[-6:]
                ns = ns0[-6:]
            else:
                clust = clust0
                mon = mon0
                ns = ns0

            if mon[-1] == last_month:
                active_status.append("Current")
            else:
                active_status.append(months_labels[mon[-1]])

            # ws 1,3,5...
            ws = np.arange(1, len(ns) * 2, 2)
            wnorm = np.dot(ns, ws) / np.sum(ws)

            tot.append(len(clust))
            anm_mon.append(mon)
            anm_norm.append(np.mean(ns))
            anm_wnorm.append(wnorm)
            ns_all.append(ns)
            anm_clust.append(clust)

        #   for j in range(self.clustering.num_clusters):
        #     clust_matrix[i][j] = (clust == j).sum()
        #     clust_mon[j].append(mon[np.where(clust == j)])
        #     clust_norm[j].append(ns[np.where(clust == j)])

        # display_clusters = get_clust_mean(clust_norm)
        display_clusters = self.sort_clusters()
        self.print_display_cluster_centers(display_clusters)
        display_clusters_t = np.argsort(display_clusters)

        for k in range(len(self.processInput.anm_id)):
            anm_clust_display.append(np.take(display_clusters_t, anm_clust[k]))

        for k in range(len(self.processInput.anm_id)):
            anm_mon_display.append(np.take(months_labels, anm_mon[k]))

        trend_value = np.asarray(anm_norm) - np.asarray(anm_wnorm)
        pos_trend_threshold = pos_threshold_percentage * np.max(trend_value)
        neg_trend_threshold = neg_threshold_percentage * np.min(trend_value)

        for k in range(len(self.processInput.anm_id)):
            if trend_value[k] >= pos_trend_threshold:
                trend.append("Improving")
            elif trend_value[k] <= neg_trend_threshold:
                trend.append("Degrading")
            else:
                trend.append("No significant trend")

        raw_df = pd.DataFrame({'sub_center_id': self.processInput.anm_id,
                               'Average_scores_of_past_6months': anm_norm,
                               'Average_scores_giving_more_weight_to_recent_months': anm_wnorm,
                               'Trend_value': trend_value,
                               'Scores_of_past_6_months': pd.Series(ns_all).values,
                               'Total_months_considered': tot,
                               'Worked_months': pd.Series(anm_mon_display).values,
                               'Clusters_in_past_6months_(Most_recent_in_the_right)': anm_clust_display})

        field_df = pd.DataFrame({'sub_center_id': self.processInput.anm_id,
                                 'Average_scores_of_past_6months': anm_norm,
                                 'Trend': trend,
                                 'Clusters_in_past_6months_(Most_recent_in_the_right)': anm_clust_display,
                                 'Active_status': active_status,
                                 'Predicted_score_for_the_next_month': predicted_score_df['predicted_score']
                                 })
        field_df.sort_values(by=['Average_scores_of_past_6months'], ascending=True, inplace=True)

        # anm_mon = pd.Series(anm_mon)
        # clust_df = clust_df.assign(months=anm_mon.values)
        raw_df.to_csv("outputs/raw_summary_output.csv")
        field_df.to_csv("outputs/summary_output.csv")
        # print(field_df.head())

        return display_clusters

    def get_clust_mean(self, clust_norm):
        clust_norm_mean = []
        for i in range(len(clust_norm)):
            c = clust_norm[i]
            flat_clust_norm = [item for sublist in c for item in sublist]
            flat_clust_norm = np.asarray(flat_clust_norm)
            clust_norm_mean.append(np.nanmean(flat_clust_norm))
        print(clust_norm_mean)
        ind = np.argsort(clust_norm_mean)
        print(ind)
        return ind

    def sort_clusters(self):
        probs = self.sc.get_fraud_probs(np.multiply(self.clustering.centers,100))
        probs_mean = np.mean(probs, axis=1)
        ind = np.argsort(probs_mean)

        return ind

    def print_display_cluster_centers(self, display_clusters):
        display_centers = {}
        for i in range(len(display_clusters)):
            display_centers['Center_of_cluster'+str(i)] = np.multiply(self.clustering.centers[display_clusters[i]], 100)
        display_cluster_ct_df = pd.DataFrame(display_centers)
        display_cluster_ct_df.to_csv('outputs/cluster_centers.csv')







