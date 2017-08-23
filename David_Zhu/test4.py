import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
from scipy import signal
from scipy import optimize

'''
the LRTable is the different windows of the spectrum from the library.
'''
ComplexLRTable = [
    [[22, 72, 138, 219, 298, 432, 472],
     [62, 136, 219, 298, 401, 460, 489], ],
    [[32, 147, 226, 315, 454],
     [147, 226, 315, 450, 505], ],
    [[54, 90, 143, 218, 312, 463],
     [86, 131, 218, 312, 402, 494], ],
    # 432--442 seems not important
    # [[54, 90, 143, 218, 312, 432, 463],
    # [86, 131, 218, 312, 402, 442, 494],],
    [[16, 147, 224, 307, 426, 473],
     [138, 224, 307, 421, 449, 485], ]
]

LRTable = [
    [[138, 219, 298, ],
     [219, 298, 401, ], ],
    [[147, 226, 315],
     [226, 315, 450], ],
    [[143, 218, 312, ],
     [218, 312, 402, ], ],
    [[147, 224, 307, ],
     [224, 307, 421, ], ]
]


# Finding the envelop of the spectrum
def qhull(sample):
    '''
    Finding the envelop of the spectrum
    qhull(
        sample    numpy.ndarry of [wavelength, spectrum]
        )
    '''
    link = lambda a, b: np.concatenate((a, b[1:]))
    edge = lambda a, b: np.concatenate(([a], [b]))

    def dome(sample, base):
        h, t = base
        dists = np.dot(sample - h, np.dot(((0, -1), (1, 0)), (t - h)))
        outer = np.repeat(sample, dists > 0, axis=0)

        if len(outer):
            pivot = sample[np.argmax(dists)]
            return link(dome(outer, edge(h, pivot)),
                        dome(outer, edge(pivot, t)))
        else:
            return base

    if len(sample) > 2:
        axis = sample[:, 0]
        base = np.take(sample, [np.argmin(axis), np.argmax(axis)], axis=0)
        return link(dome(sample, base),
                    dome(sample, base[::-1]))
    else:
        return sample


def changehull(hull, name_list):
    '''
    change the hull to fit all the data within [0-1] by considering the endpoints.
    :param hull:
    :param name_list:
    :return:
    '''
    c = 0
    for index in range(len(hull)):
        if hull[index][0] == name_list[-1]:
            c = index
            break
    hull = hull[:c + 1]
    hull = np.vstack((hull, hull[0]))

    return hull


def inputData4Lib(testnumber):
    '''
    Input the standard library of the 4 REE spectrum: VNIR: 396nm -- 1003nm
    :return:
    '''
    # Read the file and store in pandas
    df = pd.read_excel('nameslib4data.xlsx', sheetname='Sheet1')
    col_name = df.columns
    name_list = list(col_name)
    # xbegin = name_list.index(396.329987)
    xbegin = name_list.index(600.820007)
    wavelengths = np.array(name_list[xbegin:])

    col_value = df.values
    value_list = list(col_value)
    spectrum = col_value[testnumber][xbegin:]
    return wavelengths, spectrum, name_list


def inputsampledata():
    '''
    Input the samll sampledata of the Rock from: 600.820007
    :param testnumber: the input line of the rock spectrum
    :return:
    '''
    headers = list(['ID', 'X', 'Y'])
    with open('header.txt', 'r') as f:
        data = f.readlines()
        for line in data:
            spec = line.split(sep=',')
            headers.extend(map(float, spec))
            # headers.extend(spec)

    name_list = list(headers)
    xbegin = name_list.index(600.820007)

    # df = pd.read_table('Box120_VNIR_sample1.txt', header=None, skiprows=[0, 1, 2, 3, 4, 5, 6, 7], sep='\s+')
    df = pd.read_table('VNIR_sample1_25points.txt', header=None, skiprows=[0, 1, 2, 3, 4, 5, 6, 7], sep='\s+')
    # print (df.head())
    wavelengths = np.array(name_list[xbegin:])

    col_value = df.values
    value_list = list(col_value)
    # global col_length
    col_length = len(value_list)
    # spectrum = col_value[testnumber][xbegin:]
    return wavelengths, col_length, col_value, xbegin


def hull_to_spectrum(hull, wavelengths):
    f = interp1d(hull['wavelength'], hull['intensity'])
    return f(wavelengths)


def callibpeaknum(testnumber):
    '''
    calculate the peak of the library infromation
    :param testnumber: from the library
    :return: peakcounts,  the peak number of the windows
             peaklist,  the peak index of the windows
             peakheights,  the peak heights of the windows
             extermeminpeaks, the exterme minimum of the windows
    '''
    wavelengths, spectrum, name_list = inputData4Lib(testnumber)
    sample = np.array([wavelengths, spectrum]).T
    hull = qhull(sample)
    hull = changehull(hull, name_list)
    hull = pd.DataFrame(hull, columns=['wavelength', 'intensity'])
    hull_spectrum = hull_to_spectrum(hull[:-1], wavelengths)
    spectrum2 = spectrum / hull_spectrum

    newspectrum = signal.savgol_filter(spectrum2, 17, 4, deriv=0)
    # get the peak values of the minimum
    peakind_tuple = signal.argrelmin(newspectrum)
    peakind = peakind_tuple[0][:]
    # print ('peakind:', peakind)

    newspectrum2 = signal.savgol_filter(spectrum2, 17, 4, deriv=1)
    # get the 1 derivative value of spectrum
    peakargmin = signal.argrelmin(newspectrum2)
    peakargmax = signal.argrelmax(newspectrum2)

    lvalue = LRTable[testnumber][0]
    rvalue = LRTable[testnumber][1]
    # print ('lvalue:', lvalue)
    # print ('rvalue:', rvalue)

    peakcounts = list()
    peaklist = list()
    peakheights = list()
    plotpeak = list()
    extermeminpeaks = list()
    for index in np.arange(len(lvalue)):
        peaks = [val for val in peakind[peakind > lvalue[index]] if val in peakind[peakind < rvalue[index]]]
        # print ('peaks:', peaks)

        extermeminpeak_temp = np.argmin(newspectrum[peaks])
        extermeminpeak = peaks[extermeminpeak_temp]
        extermeminpeaks.append(extermeminpeak)

        plotpeak.extend(peaks)
        peakcounts.append(len(peaks))
        peaklist.append(np.array(peaks))
        predict_val = (newspectrum[rvalue[index]] + newspectrum[lvalue[index]]) / 2
        peakheights.append(np.array(np.repeat(predict_val, len(peaks)) - newspectrum[peaks]))

    def printval():
        print('Peakcounts:', peakcounts)
        print('Peaklist:', peaklist)
        print('peakheights:', peakheights)
        print('extermeminpeaks:', extermeminpeaks)

    def showlipic():
        plt.figure()
        plt.title('The Library Spectrum:')
        plt.plot(wavelengths, spectrum2, c='k', label='Removed Spectrum')
        plt.scatter(wavelengths[plotpeak], spectrum2[plotpeak], c='r', label='Found peaks')
        plt.scatter(wavelengths[lvalue], spectrum2[lvalue], c='g', label='Left endpoints')
        plt.scatter(wavelengths[rvalue], spectrum2[rvalue], c='b', label='right endpoints')
        plt.legend()

        plt.figure()
        plt.plot(wavelengths, spectrum2, c='k', label='Removed Spectrum')
        plt.scatter(wavelengths[plotpeak], spectrum2[plotpeak], c='r', label='Found peaks')
        plt.scatter(wavelengths[peakargmax], spectrum2[peakargmax], c='g', label='peakargmax endpoints')
        plt.scatter(wavelengths[peakargmin], spectrum2[peakargmin], c='b', label='peakargmin endpoints')
        plt.legend()

        plt.show()

    # printval()
    # showlibpic()
    return peakcounts, peaklist, peakheights, extermeminpeaks



def isPeakExist(libnumber, rockspectrum, libspectrum):
    '''
    Judge whether peak exists according to the rock spectrum and library spectrum
    :param testnumber:  the library spectrum number
    :param rockspectrum: the rock spectrums
    :param libspectrum: the library spectrums
    :return: judge the similarity and the scaling params and RMS
    '''
    # print('Calculate the information of the library peaks:')
    libpeakcounts, libpeaklist, libpeakheights, libextermeminpeaks = callibpeaknum(libnumber)
    lbands = LRTable[libnumber][0]
    rbands = LRTable[libnumber][1]

    # smoothrockspectrum = signal.savgol_filter(rockspectrum, 33, 4, deriv=0)
    smoothrockspectrum = rockspectrum
    rockpeakind_tuple = signal.argrelmin(smoothrockspectrum)
    rockpeakind = rockpeakind_tuple[0][:]

    judgethreshold = 12
    judgeflag = False
    distance = 0
    liballpeaks = list()
    libminpeakdepth = list()
    predrockmin = list()
    predrockdepth = list()
    rockmin = list()

    for index in np.arange(len(lbands)):
        # get each window of the library and rock spectrum.
        band = smoothrockspectrum[lbands[index]: rbands[index]]
        rockbandpeakind = [num for num in rockpeakind if (num > lbands[index] and num < rbands[index])]

        # print('rockbandpeakind:', rockbandpeakind)
        if len(rockbandpeakind) == 0:
            print('peaks not exists')
            return False, 0, 0, 0

        # get the library peaks index
        libpeak = libpeaklist[index]
        liballpeaks.extend(libpeak)

        # Use the shift of the peak distance to calculate the similarity
        libmin = libextermeminpeaks[index]

        # print ('libmin:', libmin)
        depth_temp = (libspectrum[lbands[index]] + libspectrum[rbands[index]]) / 2 - libspectrum[libmin]
        libminpeakdepth.append(depth_temp)

        # get the really minimum peaks in rock and calculate distance use the really minimum values.
        minval = min([smoothrockspectrum[item] for item in rockbandpeakind])
        for item in rockbandpeakind:
            if smoothrockspectrum[item] == minval:
                distance += abs(libmin - item)
                rockmin.append(item)

        # get the minimum peaks from the rocks and calculate distance use the nearby one peak values.
        # distance += min([abs(libmin - item) for item in rockbandpeakind])

        # print('distance:', distance)

        # Calculate the value which close to the peaks seems to be the extreme min values of the band
        minval = min([abs(libmin - item) for item in rockbandpeakind])
        for item in rockbandpeakind:
            if abs(libmin - item) == minval:
                predrockmin.append(item)
                depth = (rockspectrum[lbands[index]] + rockspectrum[rbands[index]]) / 2 - rockspectrum[item]
                predrockdepth.append(depth)
        # print ('predrockdepth:', predrockdepth)


        # Calculate depth of the original rock peaks
        rockdepth = []
        for peakindex in np.arange(len(rockbandpeakind)):
            depth = (rockspectrum[lbands[index]] + rockspectrum[rbands[index]]) / 2 - rockspectrum[
                rockbandpeakind[peakindex]]
            rockdepth.append(depth)
            # print ('rockdepth:', rockdepth)

    # print('predrockmin: ', predrockmin)
    # print ('predrockdepth: ', predrockdepth)

    # Try1: Only use the minimum value to see whether it works 使用的是最靠近这个最小值的峰来进行判断的
    def calscalling_extrememindepth():
        print('Try1: Calculate Scalling with only the exterme minimum depth:')
        scaling = 1.0

        # print ('libminpeakdepth:', libminpeakdepth)
        # print ('predrockdepth:', predrockdepth)
        def residuals(p, libminpeakdepth, predrockdepth):
            result = 0
            for item in np.arange(len(libminpeakdepth)):
                result += abs(libminpeakdepth[item] * p - predrockdepth[item])
            return result

        plsq = optimize.leastsq(residuals, scaling, args=(libminpeakdepth, predrockdepth))
        scaling = plsq[0]
        rms = residuals(scaling, libminpeakdepth, predrockdepth)
        print('Scaling: %f, RMS: %f' % (scaling, rms))

    # calscalling_extrememindepth()

    # Try2: Use the same index to calculate depth 使用的是相同的index来计算的值
    def calscalling_costantmindepth():
        print('Try2: Calculate Scalling with only the constant depth:')
        scaling = 1.0

        # print ('libextermeminpeaks:', libextermeminpeaks)
        # print ('libdepth:', libminpeakdepth)

        def residuals(p, libextermeminpeaks):
            result = 0
            for item in np.arange(len(libextermeminpeaks)):
                rockdepth = (smoothrockspectrum[lbands[item]] + smoothrockspectrum[rbands[item]]) / 2 - \
                            smoothrockspectrum[libextermeminpeaks[item]]
                libdepth = libminpeakdepth[item]
                result += abs(rockdepth - p * libdepth)
            return result

        plsq = optimize.leastsq(residuals, scaling, args=(libextermeminpeaks))
        scaling = plsq[0]
        rms = residuals(scaling, libextermeminpeaks)
        print('Scaling: %f, RMS: %f' % (scaling, rms))

    # calscalling_costantmindepth()

    weight = [0.45, 0.45, 0.1]

    # Try3: Use the multiple peaks of the library, the peakind is the costant number since it is not far away from those ones.
    def calscaling_multiconstant_mindepth():
        # print('Try3: Calculate Scalling with the multiply constant minimum depth:')
        scaling = 1.0

        # print ('libpeaklist:', libpeaklist)
        def residuals(p, libpeaklist):
            result = 0
            for item in np.arange(len(libpeaklist)):
                for t in libpeaklist[item]:
                    rockdepth = (smoothrockspectrum[lbands[item]] + smoothrockspectrum[rbands[item]]) / 2 - \
                                smoothrockspectrum[t]
                    libdepth = (libspectrum[lbands[item]] + libspectrum[rbands[item]]) / 2 - libspectrum[t]
                    result += abs(rockdepth - p * libdepth) * weight[item]
            return result

        plsq = optimize.leastsq(residuals, scaling, args=(libpeaklist))
        scaling = plsq[0]
        rms = residuals(scaling, libpeaklist)
        # print ('Scaling: %f, RMS: %f' %(scaling, rms))
        return scaling, rms

    # calscaling_multiconstant_mindepth()

    # Try4: 使用库中最接近的多个峰进行判断和拟合，使用的非定值，但也不能完全的非定值，只能先找到最低点的，然后对于原来库中的进行相对的偏移来进行判断拟合
    def calscaling_multivariable_mindepth():
        print('Try4: Calculate Scalling with the multiply variable minimum depth:')
        scaling = 1.0

        # print ('libpeaklist:', libpeaklist)
        # print ('libextermeminpeaks:', libextermeminpeaks)
        # print ('predrockmin:', predrockmin)
        def residuals(p, libextermeminpeaks, predrockmin, libpeaklist):
            result = 0
            for index in np.arange(len(libextermeminpeaks)):
                distance = libextermeminpeaks[index] - predrockmin[index]
                for t in libpeaklist[index]:
                    rockdepth = (smoothrockspectrum[lbands[index]] + smoothrockspectrum[rbands[index]]) / 2 - \
                                smoothrockspectrum[t - distance]
                    libdepth = (libspectrum[lbands[index]] + libspectrum[rbands[index]]) / 2 - libspectrum[t]
                    result += abs(rockdepth - p * libdepth)
            return result

        plsq = optimize.leastsq(residuals, scaling, args=(libextermeminpeaks, predrockmin, libpeaklist))
        scaling = plsq[0]
        rms = residuals(scaling, libextermeminpeaks, predrockmin, libpeaklist)
        print('Scaling: %f, RMS: %f' % (scaling, rms))

    # calscaling_multivariable_mindepth()


    def calscalling_samebandsweights():
        # use the same index to calculate depth and the no consideration about the weights of all bands
        print('liballpeaks:', liballpeaks)
        scaling = [1.0]

        def residuals(p, liballpeaks):
            result = 0
            for item in liballpeaks:
                # should use the depth to calculate the results.
                result += abs(smoothrockspectrum[item] - p * libspectrum[item])
            return result

        plsq = optimize.leastsq(residuals, scaling, args=(liballpeaks))
        scaling = plsq[0]
        rms = residuals(scaling, liballpeaks)
        print('Scaling: %f, RMS: %f' % (scaling, rms))

    averagedistance = distance / len(lbands)
    print('averagedistance:', averagedistance)

    scaling, rms = calscaling_multiconstant_mindepth()
    if averagedistance < judgethreshold:
        # print ('Peak Exists.')

        return True, averagedistance, scaling, rms
    else:
        # print ('Peak not exist.')
        return False, averagedistance, scaling, rms


# 后面设计时需要考虑到： 对于不同的band段所设置的权值可能不一样； 对于同一band段但是根据峰的高度来设定不同的权重； 如何确定那个band更加的重要
def calscallingwithgaussian():
    pass


def calscallingwithsignaldepth():
    pass


def getspectrum2(wavelengths, spectrum, name_list):
    '''
    Get the removed spectrum from the original spectrum
    :param testnumber: from the library
    :return: return the removed spectrum
    '''
    sample = np.array([wavelengths, spectrum]).T
    hull = qhull(sample)

    hull = changehull(hull, name_list)
    # print (hull)
    hull = pd.DataFrame(hull, columns=['wavelength', 'intensity'])

    hull_spectrum = hull_to_spectrum(hull[:-1], wavelengths)
    spectrum2 = spectrum / hull_spectrum
    return spectrum2

def tottest(rockspectrum):
    # rocknumber = int(input('Please input the rock test number:'))
    libnumber = 0

    wavelengths, libspectrum, libname_list = inputData4Lib(libnumber)

    rocksmoothspectrum = signal.savgol_filter(rockspectrum, 131, 4, deriv=0)
    # plt.figure()
    # plt.plot(wavelengths, rocksmoothspectrum, label='original spectrum')
    #
    # plt.legend()
    # plt.show()

    libspectrum2 = getspectrum2(wavelengths, libspectrum, libname_list)
    rockspectrum2 = getspectrum2(wavelengths, rocksmoothspectrum, libname_list)

    # Calculate the lib peak information.
    # print('Calculate the rock peak information:')
    # calrockpeaknum(libnumber, rocknumber)
    judgeflag, averagedistance, scaling, rms = isPeakExist(libnumber, rockspectrum2, libspectrum2)
    return judgeflag, averagedistance, scaling, rms
    # plt.show()


if __name__ == "__main__":

    # if the scaling value is less than 0 then it also can say the peak doesn't exist.
    wavelengths, col_length, col_value, xbegin = inputsampledata()
    with open('Box120_VNIR_sample1_output.txt', 'wt') as f:
        for number in np.arange(col_length):
            rockspectrum = col_value[number][xbegin:]
            flag, averagedistance, scaling, rms = tottest(rockspectrum)
            if (flag == True):
                print('Number %d has peaks.' % number)
            
                print('Number %d has peaks.' % number, file=f)
                print ('Averagedistance %.3f' %averagedistance, file=f)
                print('Scaling: %.3f; RMS: %.3f\n' % (scaling, rms), file=f)


            else:
                print('Number %d doesnot has peaks.' % number)


            

