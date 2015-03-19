__author__ = 'zhangye'
import cPickle as pickle
import matplotlib.pyplot as plt
def load_and_plot():
    with open('CI_hash_pos.p', 'rb') as fp:
        CI_hash_pos = pickle.load(fp)

    with open('CI_hash_neg.p','rb') as fp:
        CI_hash_neg = pickle.load(fp)

    '''
    uk = CI_hash_pos['uk']
    plt.hist(uk,bins=20)
    plt.title('uk')
    plt.show()

    '''
    study = CI_hash_neg['study']
    plt.hist(study,bins=20)
    plt.title('study')
    plt.show()


def main():
    load_and_plot()

main()


