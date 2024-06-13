import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, auc, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from our_knn import *


def test_loop(X, y, k_list=[5], distances=[None], score_funcs=[None], n_splits=5, thresholds=[95], filter_inliers=[False]):
    X = np.array(X)
    y = np.array(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    f1_arr = []
    precision_arr = []
    recall_arr = []
    PR_AUC_arr = []
    distance_arr = []
    k_neighbors_arr = []
    cv_arr = []
    decisions_arr = []
    filter_arr = []
    threshold_arr = []

    for filter in filter_inliers:
        for th in thresholds:
            for k in k_list:
                for d in distances:
                    for scr in score_funcs:
                        detector = KNNAnomalyDetector(k=k, distance_func=d, score_func=scr, threshold_percentile=th)
                        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
                            X_train, X_test = X[train_index], X[test_index]
                            y_train, y_test = y[train_index], y[test_index]

                            # filter only inliers 
                            if filter:
                                inliers_index = np.where(y_train==1)
                                X_train = X_train[inliers_index]

                            detector.fit(X_train)
                            predictions = detector.predict(X_test, detector.threshold_)

                            scores = detector.decision_function(X_test)
                            precision, recall, _ = precision_recall_curve(y_test, -scores)
                            pr_auc = auc(recall, precision)
                            f1 = f1_score(y_test, predictions, average='binary')

                            f1_arr.append(f1)
                            precision_arr.append(precision)
                            recall_arr.append(recall)
                            PR_AUC_arr.append(pr_auc)
                            k_neighbors_arr.append(k)
                            cv_arr.append(i)
                            if d is not None:
                                distance_arr.append(d.__name__)
                            else:
                                distance_arr.append(None)
                            if scr is not None:
                                if type(scr) == str:
                                    decisions_arr.append(scr)  
                                else:
                                    decisions_arr.append(scr.__name__)
                            else:
                                decisions_arr.append(None)
                            filter_arr.append(filter)
                            threshold_arr.append(th)
                        
                
    result_df = pd.DataFrame({"f1":f1_arr, "precision":precision_arr, "recall":recall_arr, 
                              "PR_AUC":PR_AUC_arr, "distance":distance_arr, 
                              "K neighbors":k_neighbors_arr, "CV":cv_arr, 
                              "score function":decisions_arr, "filter inliers":filter_arr,
                              "percentile threshold":threshold_arr})
    
    return result_df