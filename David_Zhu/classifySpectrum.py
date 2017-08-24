import numpy as np
import pickle
import pandas as pd
import math

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

    # calculate way 1 and 2 are almost the same.
    # calculate way 1:
    def calway1():
        p = np.poly1d(coeffs)
        yhat = p(x)
        ybar = np.sum(y)/len(y)
        ssreg = np.sum((yhat - ybar) ** 2)
        sstot = np.sum((y - ybar) ** 2)
        results['determination'] = ssreg / sstot

    # calculate way 2:
    def calway2():
        correlation = np.corrcoef(x,y)[0,1]
        results['correlation'] = correlation
        results['determination'] = correlation ** 2

    calway1()

    return results

def bestgroup(filename):
    '''
    To choose the best group spectrum list according to the value of R-Square (Maximum)
    :param filename: CSV file of precent-element dataframe.
    :return: The output of the best ones of index and value.
    '''
    getfromfile = True
    if getfromfile:
        # read the file from the csv file.
        df = pd.read_csv(filename)
    else:
        # the filename is the dataframe.
        df = filename

    elements = df['elements']

    choice = 0
    bestval = 0

    for index in np.arange(len(df.columns) - 2):
        val = polyfit(df[str(index)], elements, 1)["determination"]
        print ('val:', val)
        if(val > bestval):
            bestval = val
            choice = index

    results = {
        'index': choice,
        'RSquare': bestval,
    }
    print ('Choose the best group for the RSquare:', results)
    return results

def calgroupnum(filename, lineslist, pklfilename, savefilename, convert=False):
    '''
    Calculate kind of rock in different groups.
    :param filename: CSV file of percent-element dataframe.
    :param lineslist: The total lines in order to separate them away. [60073, 135928, 28556, 140962, 88237, 112695]
    :param pklfilename: PKL file to get the cluster_members_lists for each group.
    :param savefilename: Save the output to a different file.
    :param convert: Just useful to reorder from 5,6,1,2,3,4 to 1,2,3,4,5,6
    :return:
    '''
    bestInfo = bestgroup(filename)
    bestIndex = bestInfo["index"]
    # bestIndex = 3

    data = loadpklfile(pklfilename)
    cluster_members_lists = data['cluster_members_lists']
    results = list()
    lens = list()
    for t in np.arange(len(lineslist)):
        s = list()
        for p in cluster_members_lists[bestIndex]:
            if t!=0:
                if p < lineslist[t] and p >= lineslist[t-1]:
                    temp = p - lineslist[t-1]
                    s.append(temp)
            else:
                if p < lineslist[t]:
                    s.append(p)
        # when the value's length is not the same, it will make the process as the characters.
        results.append(s)
        lens.append(len(s))
    data = pd.DataFrame()
    # Convert to the right order.
    if convert:
        reorderlens = lens[2:] + lens[:2]
        reorderresults = results[2:] + results[:2]
    else:
        reorderlens = lens
        reorderresults = results

    # The grouplist length is not the same, so when read from csv it will take as characters.
    data['length'] = reorderlens
    data['grouplist'] = reorderresults
    print ('calculate group information done.')
    data.to_csv('../SWIR_Output/best-group-info1.csv')
    # np.savetxt('best-group-info.txt', data)

    # Save the best-group-info.txt files so that it can be read and dealt.
    f = open(savefilename, "w")
    for index in np.arange(len(reorderlens)):
        print (reorderlens[index], file = f)
        print (reorderresults[index], file = f)

    print ('save the best group information done.')
    return data

def calfilelines(filenames):
    '''
    Read and get the pixels of each pictures.
    :param filenames: different pictures files.
    :return: the number of each pixels in list.
    '''
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

def loadpklfile(filename):
    '''
    Get the cluster_members_list and cluster_exemplars.
    :param filename: PKL files.
    :return:
    '''
    with (open(filename, "rb")) as openfile:
        data = pd.read_pickle(openfile)
    cluster_members_lists = data['cluster_members_lists']
    cluster_exemplars = data['cluster_exemplars']
    print('lenA:', len(cluster_members_lists))
    print('lenB:', len(cluster_exemplars))

    return data

def savenewdata(lineslist, readfilename, savefilename, totlineslist, convert=False):
    '''
    Get the percent-element table.
    :param lineslist: The lines of the group ones for each rocks.
    :param readfilename: Load PKL files to get the cluster spectrums.
    :param savefilename: Save the file in the path.
    :param totlineslist: The total lines in order to separate them away. [60073, 135928, 28556, 140962, 88237, 112695]
    :param convert: Dicide whether it needs to convert the order.
    :return:
    '''
    elements = [2.8,2.4,1.6,1.2,1.9,1.4]
    data = loadpklfile(readfilename)
    cluster_members_lists = data['cluster_members_lists']
    results = list()
    lens = list()
    record = list()
    for t in np.arange(len(lineslist)):
        s = list()
        numrecord = list()
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
                s.append(len(k) / (totlineslist[t] - totlineslist[t-1]))
            else:
                s.append(len(k) / totlineslist[t])
            numrecord.append(len(k))

            # results = 6 * 25 dimension
            results.append(k)
            # print ('%d: record number:' %t, numrecord)
        lens.append(s)
        record.append(numrecord)
    numberdata = pd.DataFrame(record)
    print ('numberdata:', numberdata)

    # the before list is 5,6,1,2,3,4 and try to correct the order and output the values and test the number.
    def countnumber():
        # Test and compare whether it is right or wrong.
        if convert:
            rightordernumber = record[2:] + record[:2]
        else:
            rightordernumber = record
        print ('rightordernumber:', rightordernumber)
        data2 = pd.DataFrame(rightordernumber)
        data2['elements'] = elements
        data2.to_csv('../SWIR_Output/group_number_elements.csv')
        print ('Save group_number_elements table done.')

    print ('lens:', lens)
    if convert:
        rightorderlens = lens[2:] + lens[:2]
    else:
        rightorderlens = lens

    newdata = pd.DataFrame(rightorderlens)
    print ('rightorderlens:', rightorderlens)
    newdata['elements'] = elements

    newdata.to_csv(savefilename)
    print ('Save group_percents_elements table done.')
    # print ('newdata:', newdata)
    return newdata


def cumsum(L):
    '''
    To accumulate the sum of the lists.
    :param L:
    :return: New lists.
    '''
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

    translineslist = lineslist[-2:] + lineslist[:-2]
    print ('translineslist:', translineslist)

    sumlinelist = cumsum(translineslist)
    print ('sumlinelist:', sumlinelist)

    # get the percent-elements data and save them.
    newdata = savenewdata(sumlinelist, "../SWIR/clusters.pkl", "../SWIR_Output/group_percents_elements1.csv", sumlinelist, True)
    print ('newdata:', newdata)

    # # get the group information from the data directly.
    # dfgroupinfo = calgroupnum(newdata, sumlinelist, "clusters.pkl")
    # print ('dfgroupinfo:', dfgroupinfo)

    # get the group information from the filedata.
    filename = '../SWIR_Output/group_percents_elements1.csv'
    dfgroupinfo = calgroupnum(filename, sumlinelist, "../SWIR/clusters.pkl", "../SWIR_Output/best-group-info1.txt", True)
    print ('dfgroupinfo:', dfgroupinfo)


if __name__ == '__main__':
    main()