import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
from sklearn import metrics

class Predictor:

    def __init__(self):
        print("Initializing the simple linear predictor")


    def simple_predictor(self, train_x, train_y, test_x, test_y, test_anm):

        # #split dataset in train and testing set
        # X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)

        print("Total", len(train_x) + len(test_x))
        print("Train", train_x.shape)
        print("Test", test_x.shape)

        model = LinearRegression()
        model.fit(train_x, train_y)

        print('coefficient of determination:', model.score(train_x, train_y))

        y_pred = model.predict(test_x)
        print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(test_y, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, y_pred)))

        df = pd.DataFrame({'sub_center_id': test_anm, 'Actual': test_y.flatten(), 'Predicted': y_pred.flatten()})
        # df.to_csv('test1.csv')

        return model

    def model_predict(self, model, pred_x, last_p_x_anm, anm_df):
        pred_y = model.predict(pred_x)
        df = pd.DataFrame({'sub_center_id': last_p_x_anm, 'predicted_score': pred_y.flatten()})
        df_merge = pd.merge(anm_df, df, on='sub_center_id', how='left')
        df_merge.fillna("Not_enough_data", inplace=True)
        # df_merge.to_csv('test2.csv')

        return df_merge

    def get_predictor_input_data(self, anm, anm_id, labels, month_list, norm_scores):
        num_history = 6
        p_x = []
        p_y = []
        p_anm = []
        p_test_x = []
        p_test_y = []
        p_test_anm = []
        last_p_x = []
        last_p_x_anm = []

        for i in range(len(anm_id)):
            clust0 = labels[np.where(anm == anm_id[i])]
            mon0 = month_list[np.where(anm == anm_id[i])]
            ns0 = norm_scores[np.where(anm == anm_id[i])]

            if len(ns0) < num_history:
                continue
            elif len(ns0) == num_history:
                last_p_x.append(ns0)
                last_p_x_anm.append(anm_id[i])
            else:
                last_p_x.append(ns0[-6:])
                last_p_x_anm.append(anm_id[i])
                indexer = np.arange(num_history + 1)[None, :] + np.arange(len(ns0) - num_history)[:, None]
                slide = ns0[indexer]
                for j in range(slide.shape[0]):
                    if j == slide.shape[0] - 1:
                        p_test_x.append(slide[j][:-1])
                        p_test_y.append(slide[j][-1])
                        p_test_anm.append(anm_id[i])
                    else:
                        p_x.append(slide[j][:-1])
                        p_y.append(slide[j][-1])
                        p_anm.append(anm_id[i])

        last_p_x = np.asarray(last_p_x)
        last_p_x_anm = np.asarray(last_p_x_anm)
        p_x = np.asarray(p_x)
        p_y = np.asarray(p_y)
        p_anm = np.asarray(p_anm)
        p_test_x = np.asarray(p_test_x)
        p_test_y = np.asarray(p_test_y)
        p_test_anm = np.asarray(p_test_anm)

        # print(last_p_x.shape)
        # print(last_p_x_anm.shape)
        # print(p_x.shape)
        # print(p_y.shape)
        # print(p_anm.shape)

        return last_p_x, last_p_x_anm, p_x, p_y, p_anm, p_test_x, p_test_y, p_test_anm




