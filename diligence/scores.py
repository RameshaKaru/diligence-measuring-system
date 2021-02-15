import numpy as np


class Scores:

    def __init__(self, kdes, rule_type, bounds, func_get_prob_mass_trans):
        print("Initializing the non diligence score calculation class")
        self.kdes = kdes
        self.rule_type = rule_type
        self.bounds = bounds
        self.func_get_prob_mass_trans = func_get_prob_mass_trans

    def rule_prob(self, p, r):
        probabilities = []
        if self.rule_type[r] == 'lower':
            for x in p:
                x = float(x)
                prob = self.func_get_prob_mass_trans(self.kdes[r], 0, x)
                probabilities.append(prob[0])
            return np.asarray(probabilities)

        elif self.rule_type[r] == 'higher':
            for x in p:
                x = float(x)
                prob = self.func_get_prob_mass_trans(self.kdes[r], x, 100)
                probabilities.append(prob[0])
            return np.asarray(probabilities)

        elif self.rule_type[r] == 'mid':
            s = self.bounds[r][0]
            e = self.bounds[r][1]
            for x in p:
                x = float(x)
                if (x < s):
                    probR = self.func_get_prob_mass_trans(self.kdes[r][0], x, s)
                    prob = probR[0]
                elif (x > e):
                    probR = self.func_get_prob_mass_trans(self.kdes[r][1], e, x)
                    prob = probR[0]
                else:
                    prob = 0.0
                probabilities.append(prob)
            return np.asarray(probabilities)

    def get_fraud_probs(self, percentages):
        fraud_prob = np.zeros(percentages.shape)

        for r in range(percentages.shape[1]):
            fraud_prob[:, r] = self.rule_prob(percentages[:, r], r)

        return fraud_prob

    def get_simple_norm(self, prob):
        norms = np.linalg.norm(prob, axis=1, ord=2)
        return norms