import rpy2.robjects as robjects


class KDEs:
    """
    This class serves as the intermediary between R objects and Python.

    At initialization, the KDEs are obtained using R scripts.

    """

    def __init__(self):
        robjects.r('''
            source('diligence/rfunc/helpers.r')
        ''')

        self.func_get_prob_mass_trans = robjects.globalenv['func_get_prob_mass_trans']
        self.func_get_all_kdes = robjects.globalenv['func_get_all_kdes']


