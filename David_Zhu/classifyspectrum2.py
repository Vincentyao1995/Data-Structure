import classifySpectrum as cs
import pandas as pd
import numpy as np
import math
import pickle

def loadidx(openfilename):
    '''
    Deal with the best-group-list data and find the best one.
    :param openfilename: "best-group-info1.txt"
    :return: Length and group of each list.
    '''
    f = open(openfilename, 'r')
    lenslist = list()
    grouplist = list()
    for line in f.readlines():
        line = line.strip()
        group = list()
        if len(line) < 50 and len(line) != 0:
            lens = list(map(int, line.split(sep=',')))
            lenslist.extend(lens)
        elif len(line) > 50:
            newlines = line[1:-1]
            group.append(list(map(int, newlines.split(sep=','))))
            grouplist.extend(group)
    print ('load index group list done.')
    return lenslist, grouplist


def loaddata(filename):
    '''
    Load all spectrums and find the best group spectrums.
    :param filename:
    :return: The best group spectrums.
    '''
    # while read the csv file, it needs to treat it as number rather than characters. so just save it as txt and reload it.
    lenslist, grouplist = loadidx(filename)

    filenames = ['../SWIR/Box120_SWIR_sample1.txt', '../SWIR/Box120_SWIR_sample2.txt', '../SWIR/Box120_SWIR_sample3.txt',
                 '../SWIR/Box120_SWIR_sample4.txt', '../SWIR/Box120_SWIR_sample5.txt', '../SWIR/Box120_SWIR_sample6.txt']
    xbegin = 3

    # Load all the spectrums to the files.
    totfilespectrums = list()
    for index in np.arange(len(filenames)):
        df = pd.read_table(filenames[index], header=None, skiprows=[0, 1, 2, 3, 4, 5, 6, 7], sep='\s+')
        col_value = df.values
        value_list = list(col_value)

        totfilespectrums.append(value_list)

    # # Save the whole spectrums files.
    # f = open("totfilespectrums.txt",'w')
    # for index in np.arange(len(totfilespectrums)):
    #     print (totfilespectrums[index], file=f)

    print ('load all file spectrums done.')

    # Load the important spectrum to the files.
    spectrums = list()
    for index in np.arange(len(grouplist)):
        idx = grouplist[index]
        totspectrums = totfilespectrums[index]
        ss = list()
        for p in idx:
            spectrum = totspectrums[p][xbegin:]
            ss.append(spectrum)
        # print ('data:', data)
        spectrums.extend(ss)

    print('loaddata all best spectrums done.')
    np.savetxt("../SWIR_Output/best-group-spectrums2.txt", spectrums)
    print ('save best-group-spectrums done.')
    return spectrums

def angle(sp1, sp2):
    '''
    Find the angle of two different spectrums between sp1 and sp2.
    :param sp1:
    :param sp2:
    :return: Angle.
    '''
    UpSide = 0
    DownSide1 = 0
    DownSide2 = 0
    assert len(sp1) == len(sp2), 'your input two spectrum have different bands. please re-input'
    deepth = len(sp1)

    for d in range(deepth):
        bandValue_sp1 = sp1[d]
        bandValue_sp2 = sp2[d]

        UpSide += bandValue_sp1 * bandValue_sp2

        DownSide1 += bandValue_sp1 ** 2
        DownSide2 += bandValue_sp2 ** 2

    angle = UpSide / (DownSide1 ** 0.5 * DownSide2 ** 0.5)
    # It may raise error because of the float number such as 1.0002 > 1.0 and error will happen.
    def clean_cos(cos_angle):
        return min(1, max(cos_angle, -1))
    angle = math.acos(clean_cos(angle))  # value range of acos is [0, pi]
    return angle

def main():
    '''
    To cluster the new data with the last best group spectrum.
    :return:
    '''
    spectrums = np.loadtxt("../SWIR_Output/best-group-spectrums2.txt")
    print ('loadfiles done.')
    cluster_exemplars = [spectrums[0]]
    cluster_members_lists = [[]]
    # use 0.06 will have 25 clusters and the R square is much larger.
    ANGLE_THRESHOLD = 0.06
    for pdIdx in np.arange(len(spectrums)):
        spectrum = spectrums[pdIdx]
        angelsToClusters = np.array(list(map(lambda exemplar: angle(spectrum, exemplar), cluster_exemplars)))
        bestClusterIdx = angelsToClusters.argmin()
        bestAngle = angelsToClusters[bestClusterIdx]

        if (bestAngle < ANGLE_THRESHOLD):
            cluster_members_lists[bestClusterIdx].append(pdIdx)
            _members = cluster_members_lists[bestClusterIdx]

        else:
            cluster_members_lists.append([pdIdx])
            cluster_exemplars.append(spectrum)
            print ('[%d/%d] nClusters = %d' %(pdIdx, len(spectrums), len(cluster_exemplars)))
    toSave = {
        'cluster_members_lists': cluster_members_lists,
        'cluster_exemplars': cluster_exemplars,
    }
    pickle.dump(toSave, open('../SWIR/savefile006.pkl', 'wb'))
    f = open('../SWIR_Output/savefile6.txt', 'w')
    for i in np.arange(len(cluster_exemplars)):
        print(cluster_exemplars[i], file=f)
        print(cluster_members_lists[i], file=f)

    print('saved')

def calculateRSquare():
    '''
    Calculate the R-Square and the new group information.
    :return:
    '''
    filenames = ['../SWIR/Box120_SWIR_sample1.txt', '../SWIR/Box120_SWIR_sample2.txt',
                 '../SWIR/Box120_SWIR_sample3.txt',
                 '../SWIR/Box120_SWIR_sample4.txt', '../SWIR/Box120_SWIR_sample5.txt',
                 '../SWIR/Box120_SWIR_sample6.txt']
    lineslist = cs.calfilelines(filenames)

    totsumlinelist = cs.cumsum(lineslist)
    print('sumlinelist:', totsumlinelist)

    lenslist, grouplist = loadidx("../SWIR_Output/best-group-info1.txt")
    print ('lenslist:', lenslist)
    sumlenslist = cs.cumsum(lenslist)
    print ('sumlenslist:', sumlenslist)

    # get the precent-elements data and save them.
    newdata = cs.savenewdata(sumlenslist, "../SWIR/savefile006.pkl", "../SWIR_Output/group_percents_elements2.csv", totsumlinelist)
    print ('newdata:', newdata)

    # # get the group information from the data directly.
    # dfgroupinfo = calgroupnum(newdata, sumlinelist)
    # print ('dfgroupinfo:', dfgroupinfo)

    # get the group information from the filedata.
    filename = "../SWIR_Output/group_percents_elements2.csv"
    dfgroupinfo = cs.calgroupnum(filename, sumlenslist, "../SWIR/savefile006.pkl", "../SWIR_Output/best-group-info2.txt")
    print('dfgroupinfo:', dfgroupinfo)


if __name__ == '__main__':
    # loadidx()
    # Save "best-group-spectrums2.txt"
    loaddata("../SWIR_Output/best-group-info1.txt")
    # Cluster and save "savefile006.pkl"
    main()
    # Save the new information of "best-group-info2.txt"
    calculateRSquare()

