import matplotlib.pyplot as plt
from math import sqrt

class ROC():

    def __init__(self, normal_errs: list, anom_errs: list, upper_bound: float = 10.0, lower_bound: float = 0.0, delta: float = 1.0) -> None:
        threshold = upper_bound

        self.tnr_list = [] # True negative rate list
        self.fnr_list = [] # False negative rate list
        self.th_list = [] # Thresholds list

        while threshold >= lower_bound:

            true_negative = sum(i < threshold for i in normal_errs)
            false_negative = sum(i < threshold for i in anom_errs)

            tn_rate = (true_negative / len(normal_errs)) * 100
            fn_rate = (false_negative / len(anom_errs)) * 100

            self.tnr_list.append(tn_rate)
            self.fnr_list.append(fn_rate)

            self.th_list.append(threshold)

            threshold -= delta
    
    def AUC(self):
        auc = 0
        for i in range(len(self.th_list) - 2):
            X1 = (self.fnr_list[i], self.tnr_list[i])
            X2 = (self.fnr_list[i + 1], self.tnr_list[i + 1])
            
            base = (X1[0] - X2[0])
            h1 = X2[1]
            h2 = X1[1] - h1

            a1 = base * h1
            a2 = base * h2 / 2

            auc += (a1 + a2)

        return auc / 100

    def top_left(self):
        """
            Calcola il punto della curva più vicino a (0, 100) [angolo in alto a sinistra]
        """
        min_dist = None
        top_left_point = None
        top_left_th = None
        for i in range(len(self.th_list) - 1):
            x = self.fnr_list[i]
            y = self.tnr_list[i]
            dist = sqrt(x ** 2 + (100 - y) ** 2)
            if min_dist == None or dist < min_dist:
                min_dist = dist
                top_left_point = (x, y)
                top_left_th = self.th_list[i]

        return top_left_point, top_left_th

    def plot(self, draw_th: bool = True, line_type: str = 'bo-'):
        plt.plot(self.fnr_list, self.tnr_list, line_type)
        plt.xlabel("False Negative Rate")
        plt.ylabel("True Negative Rate")

        for i, t in enumerate(self.th_list) :

            x,y = self.fnr_list[i], self.tnr_list[i]
            label = "{:.1f}".format(t)
            
            if draw_th:
                plt.annotate(label, # this is the text
                            (x,y), # this is the point to label
                            textcoords="offset points", # how to position the text
                            xytext=(0,10), # distance from text to points (x,y)
                            ha='center') # horizontal alignment can be left, right or center
        plt.show()
