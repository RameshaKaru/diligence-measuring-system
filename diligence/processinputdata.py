import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
import yaml


class ProcessInputData:

    def __init__(self):
        print("Initializing the input processing class")
        self.cnfg = self.get_configs()
        self.months = self.cnfg['months']
        self.threshold_each = self.cnfg['filtering']['threshold_each']
        self.threshold_all = self.cnfg['filtering']['threshold_all']
        self.num_clusters = self.cnfg['num_clusters']

    def get_configs(self):
        """
        Obtains the configurations from the config file

        Returns
        -------
        dictionary with configurations

        """
        with open('config.yaml', 'r') as stream:
            try:
                configs = yaml.safe_load(stream)
                return configs
            except yaml.YAMLError as exc:
                print(exc)

    def filter_anm_by_rule(self, rule):

        print(rule['name'])

        df = pd.read_csv(self.cnfg['location'] + rule['file_name'])

        grouped = df.groupby(['sub_center_id'])
        anmdf = grouped.agg(
            num_patients=pd.NamedAgg(column=rule['col_name_num_patients'], aggfunc=lambda x: list(x)),
            percentages=pd.NamedAgg(column=rule['col_name'], aggfunc=lambda x: list(x)),
            month=pd.NamedAgg(column='starting_date_month', aggfunc=lambda x: list(x))
        ).reset_index()

        print("Number of ANMs: ", len(anmdf))

        noise_list = []
        val = []
        val_id = []
        for i in range(len(anmdf)):
            p = np.array(anmdf.loc[i, 'num_patients'])
            id = anmdf.loc[i, 'sub_center_id']
            percent = anmdf.loc[i, 'percentages']

            if len(p) == len(np.where(p < self.threshold_each)):
                noise = 1
            elif sum(p) < (len(p) * self.threshold_all):
                noise = 1
            else:
                noise = 0
                for j in percent:
                    val.append(j)
                    val_id.append(id)

            noise_list.append(noise)

        anmdf = anmdf.assign(noise=noise_list)
        print("Removed ANM number: ", sum(noise_list))

        filtered_df = anmdf[anmdf['noise'] == 0]
        filtered_anm = np.asarray(filtered_df['sub_center_id'])

        return filtered_anm

    def remove_noisy_anm(self):

        for i in range(len(self.cnfg['short_rules'])):
            filtered_anm_per_rule = self.filter_anm_by_rule(self.cnfg['short_rules'][i])

            if i == 0:
                not_noisy_anm = filtered_anm_per_rule
            else:
                not_noisy_anm = np.intersect1d(not_noisy_anm, filtered_anm_per_rule)

        self.anm_id = not_noisy_anm
        self.anm_df = pd.DataFrame({'sub_center_id': not_noisy_anm})
        self.anm_df.to_csv('intermediate_outputs/sub_center_id_new.csv')
        print("Anm count after filtering: ", len(self.anm_df))

    def get_rule_bounds(self):
        bounds = []
        rule_type = []
        for rule in self.cnfg['short_rules']:
            p, r = self.calc_bounds(rule)
            bounds.append(p)
            rule_type.append(r)
        for rule in self.cnfg['contra_rules']:
            bounds.append([0, 100])
            rule_type.append('lower')

        return bounds, rule_type

    def calc_bounds(self, rule):
        if rule['good_range'] == "lower":
            p = [0, 100]
            r = 'lower'
        elif rule['good_range'] == "higher":
            p = [0, 100]
            r = 'higher'
        elif rule['good_range'] == "mid":
            s = rule['range']['start']
            e = rule['range']['end']
            p = [s, e]
            r = 'mid'
        else:
            print("Rule 'good range' field is not valid")
            print(rule['name'])

        return p, r

    def process_rules(self):
        percentages = []

        for rule in self.cnfg['short_rules']:
            p = self.shortrule(rule)
            percentages.append(p)
        for rule in self.cnfg['contra_rules']:
            p = self.contrarule(rule)
            percentages.append(p)

        percentages = np.asarray(percentages)
        # print(percentages.shape)
        percentages_new = percentages.swapaxes(0, 1)
        # print(percentages_new.shape)
        return percentages_new

    def contrarule(self, rule):
        print(rule['name'])
        df1 = pd.read_csv(self.cnfg['location'] + rule['file_name'])
        percentages_all = []
        for m in range(len(self.months)):
            dfm = df1[df1['starting_date_month'] == self.months[m]]
            dfnew = dfm[dfm['sub_center_id'].isin(self.anm_id)]
            # print(len(dfm), len(dfnew))

            df_merge = pd.merge(self.anm_df, dfnew, on='sub_center_id', how='left')
            df_merge[rule['col_name']].fillna(0, inplace=True)
            percentages = np.asarray(df_merge[rule['col_name']])
            percentages_all.append(percentages)

        percentages_all = np.asarray(percentages_all)
        return percentages_all

    def shortrule(self, rule):
        print(rule['name'])
        df1 = pd.read_csv(self.cnfg['location'] + rule['file_name'])
        percentages_all = []
        for m in range(len(self.months)):
            dfm = df1[df1['starting_date_month'] == self.months[m]]
            dfnew = dfm[dfm['sub_center_id'].isin(self.anm_id)]
            # print(len(dfm), len(dfnew))

            df_merge = pd.merge(self.anm_df, dfnew, on='sub_center_id', how='left')
            percentages = np.asarray(df_merge[rule['col_name']])
            percentages_all.append(percentages)

        percentages_all = np.asarray(percentages_all)
        return percentages_all

    def get_percentages(self, nan_fill=True):
        percentages = self.process_rules()
        print(percentages.shape)
        percentages_all = []
        anm_list = []
        month_list = []

        for m in range(len(self.months)):
            p_m = percentages[m].T

            # remove anm with nan for all rules in month
            nancount = np.count_nonzero(np.isnan(p_m), axis=1)
            valid_p = p_m[np.where(nancount < 5)]
            anm = self.anm_id[np.where(nancount < 5)]
            # print(len(valid_p))

            if nan_fill:
                # Place column means in the indices. Align the arrays using take
                col_mean = np.nanmean(valid_p, axis=0)
                inds = np.where(np.isnan(valid_p))
                valid_p[inds] = np.take(col_mean, inds[1])

            percentages_all.append(valid_p)
            anm_list.append(anm)
            month_list.append(np.ones(len(anm)) * m)

        flat_p = [item for sublist in percentages_all for item in sublist]
        flat_p = np.asarray(flat_p)
        print(flat_p.shape)

        flat_anm = [item for sublist in anm_list for item in sublist]
        flat_anm = np.asarray(flat_anm)

        flat_month = [int(item) for sublist in month_list for item in sublist]
        flat_month = np.asarray(flat_month)

        return flat_p, flat_anm, flat_month


    def print_reorganized_input(self, flat_p, flat_anm, flat_month, nan_fill=True):
        month_labels = np.asarray(self.months)
        org_dict = {'sub_center_id': flat_anm, 'month': month_labels[flat_month]}
        rules_labels = self.get_rule_names()
        for i in range(flat_p.shape[1]):
            org_dict[rules_labels[i]] = flat_p[:,i]
        org_df = pd.DataFrame(org_dict)
        org_df.sort_values(by='sub_center_id', inplace=True, ignore_index=True)

        if nan_fill:
            org_df.to_csv("extra_outputs/all_reorganized_data_with_nan_filled.csv")
        else:
            org_df.to_csv("extra_outputs/all_reorganized_data_without_nan_filled.csv")


        for i in self.anm_id:
            anm_df = org_df[org_df['sub_center_id']==i]
            if nan_fill:
                anm_df.to_csv("extra_outputs/nan_filled/"+str(int(i))+"_anm_data_nan_filled.csv")
            else:
                anm_df.to_csv("extra_outputs/without_nan_filled/" + str(int(i)) + "_anm_data_without_nan_filled.csv")


    def get_rule_names(self):
        rule_names = []
        for rule in self.cnfg['short_rules']:
            rule_names.append(rule['name'])
        for rule in self.cnfg['contra_rules']:
            rule_names.append(rule['name'])

        return np.asarray(rule_names)

    def elbow(self, X):
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(2, 15))

        visualizer.fit(X)  # Fit the data to the visualizer
        visualizer.show(outpath="fig/elbow.png")  # Finalize and render the figure
        visualizer.show()

    def calinski(self, X):
        model = KMeans()
        visualizer = KElbowVisualizer(model, metric='calinski_harabasz', k=(2, 15))

        visualizer.fit(X)  # Fit the data to the visualizer
        visualizer.show(outpath="fig/calinski.png")  # Finalize and render the figure
        visualizer.show()

    def silhouette(self, X):
        model = KMeans()
        visualizer = KElbowVisualizer(model, metric='silhouette', k=(2, 15))

        visualizer.fit(X)  # Fit the data to the visualizer
        visualizer.show(outpath="fig/silhouette.png")  # Finalize and render the figure
        visualizer.show()
