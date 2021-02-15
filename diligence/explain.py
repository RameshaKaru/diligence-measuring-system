import numpy as np
import pandas as pd


class Explain:

    def __init__(self, data, c, get_fraud_probs, cnfg, display_clusters, levels=3, l1=0.5, l2=0.15):
        print("Initializing the explain class")
        self.data = data
        self.explain_using_centers(c, get_fraud_probs, cnfg, display_clusters, levels, l1, l2)
        self.explain_using_weights(c, get_fraud_probs, cnfg, display_clusters, levels)

    def explain_using_centers(self, c, get_fraud_probs, cnfg, display_clusters, levels=3, l1=0.5, l2=0.15):
        c_p = np.multiply(c, 100)
        centers = get_fraud_probs(c_p)
        center_labels = []
        rules_labels = self.get_rule_names(cnfg)
        rule_ignore_std = cnfg['explain']['rule_ignore_std']
        clust_avg = np.mean(centers, axis=1)
        rule_avg = np.mean(centers, axis=0)

        max_min_dif = np.max(centers, axis=0) - np.min(centers, axis=0)
        l1_rules_idx = np.where(max_min_dif >= l1)
        l2_rules_idx = np.where((max_min_dif < l1) & (max_min_dif >= l2))
        l3_rules_idx = np.where(max_min_dif < l2)

        l1_rules_idx, l2_rules_idx, l3_rules_idx, less_important_rules_idx = self.less_important_rules(rule_ignore_std,
                                                                                                       l1_rules_idx[0],
                                                                                                       l2_rules_idx[0],
                                                                                                       l3_rules_idx[0])
        less_imp_rules_df = pd.DataFrame({"Less_important_rules": rules_labels[less_important_rules_idx]})

        label0 = ["Good" if i < np.mean(clust_avg) else "Bad" for i in clust_avg]
        center_labels.append(label0)
        print(label0)
        flag = centers > rule_avg

        if levels >= 1:
            label1 = []
            for i in range(len(centers)):
                s = []
                for j in l1_rules_idx:
                    if flag[i][j]:
                        l = "Bad in " + rules_labels[j]
                    else:
                        l = "Good in " + rules_labels[j]
                    s.append(l)
                label1.append(','.join(s))
            # print(label1)
            center_labels.append(label1)

        if levels >= 2:
            label2 = []
            for i in range(len(centers)):
                s = []
                for j in l2_rules_idx:
                    if flag[i][j]:
                        l = "Bad in " + rules_labels[j]
                    else:
                        l = "Good in " + rules_labels[j]
                    s.append(l)
                label2.append(','.join(s))
            # print(label2)
            center_labels.append(label2)

        if levels >= 3:
            label3 = []
            for i in range(len(centers)):
                s = []
                for j in l3_rules_idx:
                    if flag[i][j]:
                        l = "Bad in " + rules_labels[j]
                    else:
                        l = "Good in " + rules_labels[j]
                    s.append(l)
                label3.append(','.join(s))
            # print(label3)
            center_labels.append(label3)

        center_labels = (np.asarray(center_labels)).T
        # print(center_labels[3])

        df_d = {'Level': np.arange(levels + 1)}
        for k in range(len(centers)):
            display_k = display_clusters[k]
            df_d['cluster' + str(k)] = center_labels[display_k]
        df = pd.DataFrame(df_d)

        df_ex = pd.concat([df, less_imp_rules_df], ignore_index=False, axis=1)
        df_ex.to_csv("outputs/cluster_explain.csv", header=True, index=False)

    def get_rule_names(self, cnfg):
        rule_names = []
        for rule in cnfg['short_rules']:
            rule_names.append(rule['name'])
        for rule in cnfg['contra_rules']:
            rule_names.append(rule['name'])

        return np.asarray(rule_names)

    def explain_using_weights(self, c, get_fraud_probs, cnfg, display_clusters, levels=3):
        c_p = np.multiply(c, 100)
        centers = get_fraud_probs(c_p)
        center_labels = []
        rules_labels = self.get_rule_names(cnfg)
        rule_ignore_std = cnfg['explain']['rule_ignore_std']
        clust_avg = np.mean(centers, axis=1)
        rule_avg = np.mean(centers, axis=0)

        df = pd.read_csv('intermediate_outputs/weights.csv')
        w = np.asarray(df).T[1:, :-1]

        l1_rules_idx = list(set(np.argmax(w, axis=1)))

        sort_idx = np.argsort(abs(w), axis=1)
        sort_rules = np.zeros(w.shape)
        for i in range(len(w)):
            s = sort_idx[i]
            for j in range(w.shape[1]):
                k = np.where(s == j)
                sort_rules[i][j] = k[0][0]

        sort_sum = np.sum(sort_rules, axis=0)
        sort_idx = np.argsort(sort_sum)
        l3_count = int(w.shape[1] / 2)
        l3_rules_idx = np.setdiff1d(sort_idx[:l3_count], l1_rules_idx)

        l2_rules_idx = np.setdiff1d(np.arange(w.shape[1]), np.concatenate([l1_rules_idx, l3_rules_idx]))

        l1_rules_idx, l2_rules_idx, l3_rules_idx, less_important_rules_idx = self.less_important_rules(rule_ignore_std,
                                                                                                       l1_rules_idx,
                                                                                                       l2_rules_idx,
                                                                                                       l3_rules_idx)
        less_imp_rules_df = pd.DataFrame({"Less_important_rules": rules_labels[less_important_rules_idx]})

        label0 = ["Good" if i < np.mean(clust_avg) else "Bad" for i in clust_avg]
        center_labels.append(label0)
        print(label0)
        flag = centers > rule_avg

        if levels >= 1:
            label1 = []
            for i in range(len(centers)):
                s = []
                for j in l1_rules_idx:
                    if flag[i][j]:
                        l = "Bad in " + rules_labels[j]
                    else:
                        l = "Good in " + rules_labels[j]
                    s.append(l)
                label1.append(','.join(s))
            # print(label1)
            center_labels.append(label1)

        if levels >= 2:
            label2 = []
            for i in range(len(centers)):
                s = []
                for j in l2_rules_idx:
                    if flag[i][j]:
                        l = "Bad in " + rules_labels[j]
                    else:
                        l = "Good in " + rules_labels[j]
                    s.append(l)
                label2.append(','.join(s))
            # print(label2)
            center_labels.append(label2)

        if levels >= 3:
            label3 = []
            for i in range(len(centers)):
                s = []
                for j in l3_rules_idx:
                    if flag[i][j]:
                        l = "Bad in " + rules_labels[j]
                    else:
                        l = "Good in " + rules_labels[j]
                    s.append(l)
                label3.append(','.join(s))
            # print(label3)
            center_labels.append(label3)

        center_labels = (np.asarray(center_labels)).T
        # print(center_labels[3])

        df_d = {'Level': np.arange(levels + 1)}
        for k in range(len(centers)):
            display_k = display_clusters[k]
            df_d['cluster' + str(display_k)] = center_labels[k]
        df = pd.DataFrame(df_d)

        df_ex = pd.concat([df, less_imp_rules_df], ignore_index=False, axis=1)
        df_ex.to_csv("outputs/cluster_explain_validate.csv", header=True, index=False)

    def remove_all_good_bad(self, all_bad_rules_idx, all_good_rules_idx, l1_rules_idx, l2_rules_idx, l3_rules_idx):
        l1_rules_idx1 = np.setdiff1d(l1_rules_idx, all_bad_rules_idx)
        l1_rules_idx2 = np.setdiff1d(l1_rules_idx1, all_good_rules_idx)
        l2_rules_idx1 = np.setdiff1d(l2_rules_idx, all_bad_rules_idx)
        l2_rules_idx2 = np.setdiff1d(l2_rules_idx1, all_good_rules_idx)
        l3_rules_idx1 = np.setdiff1d(l3_rules_idx, all_bad_rules_idx)
        l3_rules_idx2 = np.setdiff1d(l3_rules_idx1, all_good_rules_idx)

        return l1_rules_idx2, l2_rules_idx2, l3_rules_idx2

    def less_important_rules(self, rule_ignore_std, l1_rules_idx, l2_rules_idx, l3_rules_idx):

        rule_std = np.std(self.data, axis=0)
        less_important_rules_idx = np.where(rule_std < rule_ignore_std)
        l1_rules_idx2 = np.setdiff1d(l1_rules_idx, less_important_rules_idx)
        l2_rules_idx2 = np.setdiff1d(l2_rules_idx, less_important_rules_idx)
        l3_rules_idx2 = np.setdiff1d(l3_rules_idx, less_important_rules_idx)

        return l1_rules_idx2, l2_rules_idx2, l3_rules_idx2, less_important_rules_idx

# explain_using_centers(clustering.centers, sc.get_fraud_probs,
#                       processInput.cnfg, display_clusters, levels=3)
# explain_using_weights(clustering.centers, sc.get_fraud_probs,
#                       processInput.cnfg, display_clusters, levels=3)
