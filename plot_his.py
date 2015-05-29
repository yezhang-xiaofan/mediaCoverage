__author__ = 'zhangye'
#this program plots coefficient value distribution for top features
import matplotlib.pyplot as plt
import cPickle
#from BS_PR import texify_most_informative_features
import seaborn as sns
from pylab import *
def load_and_plot(dir):
    file = dir + "/coefficient.pkl"
    dict = cPickle.load(open(file,'rb'))
    w_lists = dict["w_list"]
    sort_p_lower = dict["sort_p_lower"]
    sort_p_upper = dict["sort_p_upper"]
    mean = dict["mean"]
    n = 50
    texify_most_informative_features(sort_p_lower,sort_p_upper,mean,n)

    #draw top features for positive instances

    #set which positive features to draw here
    plot_index = input("input positive index")
    plt.figure(1)
    ax1 = None
    for i in range(len(plot_index)):
        values = w_lists[:,sort_p_lower[plot_index[i]][2]]
        mean_value = mean[sort_p_lower[plot_index[i]][2]]
        if(i==0):
            ax1 = plt.subplot(2,2,i+1)
        else:
            current = plt.subplot(2,2,i+1,sharex=ax1)
        temp = sns.kdeplot(values)
        plt.axvline(x=0,color='r',linestyle='--',linewidth=1.0)
        plt.axvline(x=mean_value,color='g',linestyle='--',linewidth=1.0)
        plt.title(str(sort_p_lower[plot_index[i]][1]))
        if(i==2):
            plt.xlabel("Coefficient Value")
            plt.ylabel("Density")
        else:
            plt.tick_params(axis='x',bottom='off',labelbottom='off')
        plt.tick_params(axis='y',left='off',labelleft='off')
        #plt.savefig("BS_PR/"+vectorizer.get_feature_names()[sort_p_lower[i][2]]+"_L2.png")
        #plt.clf()
    #plt.show()
    plt.savefig(dir+"/top_fea_pos" + "_L2.png")

    #set which negative features to draw here
    plot_index = input("negative index")
    plt.figure(2)
    ax1 = None
    for i in range(len(plot_index)):
        values = w_lists[:,sort_p_upper[plot_index[i]][2]]
        mean_value = mean[sort_p_upper[plot_index[i]][2]]
        if(i==0):
            ax1 = plt.subplot(2,2,i+1)
        else:
            current = plt.subplot(2,2,i+1,sharex=ax1)
        temp = sns.kdeplot(values)
        plt.axvline(x=0,color='r',linestyle='--',linewidth=1.0)
        plt.axvline(x=mean_value,color='g',linestyle='--',linewidth=1.0)
        plt.title(str(sort_p_upper[plot_index[i]][1]))
        if(i==2):
            plt.xlabel("Coefficient Value")
            plt.ylabel("Density")
        else:
            plt.tick_params(axis='x',bottom='off',labelbottom='off')
        plt.tick_params(axis='y',left='off',labelleft='off')
    plt.savefig(dir+"/top_fea_neg" + "_L2.png")

def texify_most_informative_features(sort_p_lower,sort_p_upper,mean,n):
    out_str = [
        r'''\begin{table}
            \caption{top 50 features for press release positive prediction}
            \begin{tabular}{l c|l c}

        '''
    ]
    out_str.append(r"\multicolumn{2}{c}{\emph{negative}} & \multicolumn{2}{c}{\emph{positive}} \\")
    i = 0
    while i<n:
        bound1 = sort_p_upper[i][0]
        mean1 = mean[sort_p_upper[i][2]]
        diff1 = bound1 - mean1
        name1 = sort_p_upper[i][1]
        bound2 = sort_p_lower[i][0]
        mean2 = mean[sort_p_lower[i][2]]
        diff2 = mean2 - bound2
        name2 = sort_p_lower[i][1]
        out_str.append("%.3f$\pm$%.3f & %s & %.3f$\pm$%.3f & %s \\\\" % (mean1, diff1,name1,mean2,diff2,name2))
        i += 1

    out_str.append(r"\end{tabular}")
    out_str.append(r"\end{table}")

    feature_str = "\n".join(out_str)

    print "\n"
    print feature_str
def main():
    dir = "BS_PR"
    load_and_plot(dir)
if __name__ == '__main__':
    main()