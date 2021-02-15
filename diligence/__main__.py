import time
import numpy as np
from .processinputdata import ProcessInputData
from .kde import KDEs
from .scores import Scores
from .clustering import Clustering
from .predictor import Predictor
from .processoutputdata import ProcessOutputData
from .explain import Explain

if __name__ == '__main__':
    start_time = time.time()

    # Process the input and Remove noisy ANM
    processInput = ProcessInputData()
    processInput.remove_noisy_anm()
    X, anm, month_list = processInput.get_percentages()
    # processInput.print_reorganized_input(X, anm, month_list)  #uncomment this to output nan filled data
    X_wo_fill, anm_wo_fill, month_list_wo_fill = processInput.get_percentages(nan_fill=False)
    processInput.print_reorganized_input(X_wo_fill, anm_wo_fill, month_list_wo_fill, nan_fill=False)
    bounds, rule_type = processInput.get_rule_bounds()
    percentages = X.T.tolist()

    # draw KDES
    kdecls = KDEs()
    kdes = kdecls.func_get_all_kdes(rule_type, bounds, percentages)

    # get non diligence scores
    sc = Scores(kdes=kdes, rule_type=rule_type, bounds=bounds,
                func_get_prob_mass_trans=kdecls.func_get_prob_mass_trans)
    fraud_prob = sc.get_fraud_probs(X)
    norm_scores = sc.get_simple_norm(fraud_prob)


    # clustering
    clustering = Clustering(processInput.cnfg)
    labels = clustering.predict_cluster(np.multiply(X,0.01))

    # simple predictor
    predictor_obj = Predictor()
    last_p_x, last_p_x_anm, p_x, p_y, p_anm, p_test_x, p_test_y, p_test_anm = predictor_obj.get_predictor_input_data(
        anm, processInput.anm_id, labels, month_list, norm_scores)
    regres = predictor_obj.simple_predictor(p_x, p_y, p_test_x, p_test_y, p_test_anm)
    predicted_score_df = predictor_obj.model_predict(regres, last_p_x, last_p_x_anm, processInput.anm_df)

    # outputs
    out = ProcessOutputData(sc,clustering,processInput)
    display_clusters = out.outputs(anm, labels, month_list, norm_scores, predicted_score_df)

    # explain
    explain = Explain(X, clustering.centers, sc.get_fraud_probs, processInput.cnfg, display_clusters, levels=3)



    print("end")
    print("--- %s seconds ---" % (time.time() - start_time))