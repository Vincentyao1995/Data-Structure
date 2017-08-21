import numpy as np
import pickle
import pandas as pd

def polyfit(x, y, degree):
    '''
    Calcluate the R-Square of the linear functions.
    :param x:
    :param y:
    :param degree:
    :return: the coefficients and the R-Square.
    '''
    results = {}

    coeffs = np.polyfit(x, y, degree)

    results['polynomial'] = coeffs.tolist()
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    results['determination'] = ssreg / sstot

    return results

def bestgroup(filename):
    # read the file from the csv file
    # df = pd.read_csv(filename, header=None, names=None)

    # just get the csv file.
    df = filename
    # print (df)
    elements = df.values[-1]
    choice = 0
    bestval = 0
    for index in np.arange(len(df.values) - 1):
        val = polyfit(df.values[index], elements, 1)["determination"]
        if(val > bestval):
            bestval = val
            choice = index

    results = {
        'index': choice,
        'RSquare': bestval,
    }
    print ('Choose the best group for the RSquare:', results)
    return results

def calgroupnum(filename, lineslist):
    # 计算在每个组中属于该分类的个数，调用这些原本文件的光谱和下标信息然后进行组合，也就是需要知道原来分出来的有多少是符合条件的。
    # 统计在某一组分属的个数以及进一步得到新的下标。
    bestInfo = bestgroup(filename)
    bestIndex = bestInfo["index"]

    # filename = cluster_members_lists
    results = list()
    lens = list()
    for t in np.arange(len(lineslist)):
        s = list()
        for p in filename[bestIndex]:
            if t!=0:
                if p < lineslist[t] and p >= lineslist[t-1]:
                    temp = p - lineslist[t-1]
                    s.append(temp)
            else:
                if p < lineslist[t]:
                    s.append(p)
        results.append(s)
        lens.append(len(s))
    data = pd.DataFrame()
    data['length'] = lens
    data['grouplist'] = results
    print ('calculate group information done.')
    return data



def calfilelines(filenames):
    lineslist = list()
    skiplines = 8
    for file in filenames:
        count = 0
        fp = open(file, "r")
        while 1:
            buffer = fp.read(8 * 1024 * 1024)
            if not buffer:
                break
            count += buffer.count('\n')
        lineslist.append(count - skiplines)
        fp.close()
    print ('Calculate File lines done.')
    print ('lineslist:', lineslist)
    # outputvalues: lineslist: [60073, 135928, 28556, 140962, 88237, 112695]
    return lineslist

def loadpklfile():
    with (open("clusters.pkl", "rb")) as openfile:
        data = pd.read_pickle(openfile)
    cluster_members_lists = data['cluster_members_lists']
    cluster_exemplars = data['cluster_exemplars']
    print('lenA:', len(cluster_members_lists))
    print('lenB:', len(cluster_exemplars))



    return data

def savenewdata(lineslist):
    elements = [2.8,2.4,1.6,1.2,1.9,1.4]
    data = loadpklfile()
    cluster_members_lists = data['cluster_members_lists']
    results = list()
    lens = list()
    for t in np.arange(len(lineslist)):
        s = list()
        temp = list()
        for i in np.arange(len(cluster_members_lists)):
            k = list()
            for p in cluster_members_lists[i]:
                if t != 0:
                    if p < lineslist[t] and p >= lineslist[t - 1]:
                        temp = p - lineslist[t - 1]
                        k.append(temp)
                else:
                    if p < lineslist[t]:
                        k.append(p)
            if t != 0:
                s.append(len(k) / (lineslist[t] - lineslist[t-1]))
            else:
                s.append(len(k) / lineslist[t])
            temp.append(len(k))
            # results = t * p; 6 * 25 dimension
            results.append(k)
            print ('%d: temp number:' %t, temp)
        lens.append(s)
    newdata = pd.DataFrame(lens)
    newdata['elements'] = elements
    # with open('groups_percents_elements.txt','rb'):
    newdata.to_csv('group_percents_elements.csv')
    print ('Save group_percents_elements table done.')
    print ('newdata:', newdata)
    return newdata


def cumsum(L):
    # if L's length is equal to 1
    if L[:-1] == []:
        return L

    ret = cumsum(L[:-1])
    ret.append(ret[-1] + L[-1])
    return ret


def main():
    filenames = ['Box120_SWIR_sample1.txt', 'Box120_SWIR_sample2.txt', 'Box120_SWIR_sample3.txt',
                 'Box120_SWIR_sample4.txt', 'Box120_SWIR_sample5.txt', 'Box120_SWIR_sample6.txt']
    lineslist = calfilelines(filenames)

    # translineslist = lineslist[2:] + lineslist[:2]
    translineslist = lineslist[-2:] + lineslist[:-2]
    print ('translineslist:', translineslist)

    sumlinelist = cumsum(translineslist)
    print ('sumlinelist:', sumlinelist)

    newdata = savenewdata(sumlinelist)
    # dfgroupinfo = calgroupnum(newdata, sumlinelist)
    # print ('dfgroupinfo:', dfgroupinfo)
    # filename = 'group_percents_elements.csv'
    # dfgroupinfo = calgroupnum(filename, sumlinelist)


if __name__ == '__main__':
    main()