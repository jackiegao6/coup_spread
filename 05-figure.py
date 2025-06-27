import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import couponUsageRate


def figure(dataset = '',distribution = None,constantFactorDistri = None,monteCarlo_L = None,times = None,methods = []):
    seedNumList = couponUsageRate.get_seedNumList(dataset,10,1000)
    dataFile_prefix = 'res_coupon_deliverers_theroy'
    methods_labelName = ['ours', 'usage_rate', 'random', 'highDegree', 'pageRank']
    personalization_titleName = ['original','firstDiscard','firstUnuse']
    x_values = ['0.1%','0.2%','0.3%','0.4%','0.5%','0.6%','0.7%','0.8%','0.9%','1%']
    plt.yticks(np.linspace(0, 1, 30))
    colors = ['g', 'b', 'y', 'c', 'm', 'black', 'b', 'purple', 'r', 'lime', 'navy', 'chocolate', 'g', 'b', 'y', 'c',
              'm', 'black']
    marker = ["o", "x", "^", ".", "+", "1", "2", "3", "4", "5", "x", "^", "."]
    personalization_list = [None,'firstDiscard','firstUnused']
    line_num = ((len(methods)+1)*10)+1
    for personalization, i in zip(personalization_list, range(len(personalization_list))):
        plt.subplot(1, 3,i+1)
        usageRate_file = 'D:/{}/usageRate_{}_distri{}_constantFactor{}_monteCarloL{}_testTimes{}_seedNum{}_{}.txt' \
            .format(dataFile_prefix, dataset, distribution, constantFactorDistri, monteCarlo_L, times, seedNumList[-1],
                    personalization_list[i])
        plt.title(dataset+ '-' +str(personalization_titleName[i]))
        method_res = {key: [] for key in methods}
        with open(usageRate_file,'r') as f:
            count = 0
            while count <line_num:
                line = f.readline()
                count = count+1
                for j in range(10):
                    res_list = []
                    sub_first_line = True
                    for k in range(len(methods)+1):
                        line = f.readline()
                        count = count + 1
                        if sub_first_line:
                            sub_first_line = False
                            continue
                        method = line.split(":")[0].strip()
                        res = eval(line.split(":")[1].strip())
                        method_res[method].append(res[-1])
        for j,method in enumerate(methods):
            plt.yticks(np.linspace(0, 1, 100))
            plt.plot(x_values, method_res[method], label=methods_labelName[j], color=colors[j],marker=marker[0])
            plt.title(dataset + '-' + str(personalization_titleName[i]))
            plt.legend()
    plt.show()
    fig_name = dataset+'_'+distribution+'.png'
    plt.savefig('D:/coupon_spread/res_figs/{}'.format(fig_name))



        # plt.legend()
        #                 for l, method in enumerate(methods):
        #                     plt.plot(x_values[j], res_list[l], label=method)
        #         plt.legend()
        # plt.show()

if __name__ == '__main__':
    figure(dataset='Amherst',distribution = 'random',constantFactorDistri = 'random',monteCarlo_L = 1000,times = [500,1000],methods= ['deliverers_theroy','succPro','random','degreeTopM','pageRank'])