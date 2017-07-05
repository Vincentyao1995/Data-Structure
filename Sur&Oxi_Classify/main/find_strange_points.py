#swtich ~~~~~~~~~~~~~~
switch_find_stange_points = 0
switch_write_ori_acc = 0
switch_write_para_acc = 1

threshold_ratio = 3

def find_strange_points(SAM_single_double = 0):
    
    filePath = 'E:\College\Code\Python\DataAnalyzing\DataAnalyzing\data/'
    fileName = '6paras_accuracy_normalized'
    fileName_filter = fileName + '_filtered.txt'
    fileName += '.txt'
    file_acc = open(filePath + fileName,'r')
    file_acc_filter = open(filePath + fileName_filter,'w')
    lines = [line for line in file_acc]
    lines = sorted(lines)
    print(lines[0:-1])
    count_medium = 0
    count_high = 0
    count_low = 0

    #count medium acc
    if SAM_single_double == 1:
        file_acc_filter.write('medium acc:\n')
        for line in lines[0:-1]:
            name, single_acc, all_acc = line.split('\t')

            if (float(single_acc) < 0.90 and  float(single_acc) >= 0.80) and (float(all_acc) < 0.90 and  float(all_acc) >= 0.80):
                file_acc_filter.write("%s \t %s \t %s" % (name, single_acc, all_acc))
                count_medium += 1

        #count high acc
        file_acc_filter.write('\nlow acc:\n')
        for line in lines[0:-1]:
        
            name, single_acc, all_acc = line.split('\t')

            if (float(single_acc) < 0.80 or  float(all_acc) < 0.80) and (float(single_acc) < 0.90 and  float(all_acc) < 0.90):

                file_acc_filter.write("%s \t %s \t %s" % (name, single_acc, all_acc))
                count_low += 1

        file_acc_filter.write('\nhigh acc:\n')
        for line in lines[0:-1]:
        
            name, single_acc, all_acc = line.split('\t')

            if float(single_acc) >= 0.90 or  float(all_acc) >= 0.90:
                file_acc_filter.write("%s \t %s \t %s" % (name, single_acc, all_acc))
                count_high += 1


        allNum = len(lines) - 1
        file_acc_filter.write("\nTotal acc  \t  >0.9  %f  \t <0.8  %f \t 0.8-0.9  %f \n" % ((allNum - count_low - count_medium)/allNum ,count_low / allNum, count_medium / allNum))
        #print("Total acc  \t  >0.9  %f  \t <0.8  %f \t 0.8-0.9  %f \n" % ((allNum - count_low - count_medium)/allNum ,count_low / allNum, count_medium / allNum))
        file_acc_filter.close()
        print('done')

        return 0
    else:
        file_acc_filter.write('medium acc:\n')
        for line in lines[0:-1]:
            name, acc = line.split('\t')

            if (float(acc) < 0.90 and  float(acc) >= 0.80):
                file_acc_filter.write("%s \t %s" % (name, acc))
                count_medium += 1

        #count high acc
        file_acc_filter.write('\nlow acc:\n')
        for line in lines[0:-1]:
        
            name, acc = line.split('\t')

            if float(acc) < 0.80:
                file_acc_filter.write("%s \t %s" % (name, acc))
                count_low += 1

        file_acc_filter.write('\nhigh acc:\n')
        for line in lines[0:-1]:
        
            name, acc = line.split('\t')

            if float(acc) >= 0.90:
                file_acc_filter.write("%s \t %s" % (name, acc))
                count_high += 1


        allNum = len(lines) - 1
        file_acc_filter.write("\nTotal acc  \t  >0.9  %f  \t <0.8  %f \t 0.8-0.9  %f \n" % ((allNum - count_low - count_medium)/allNum ,count_low / allNum, count_medium / allNum))
        #print("Total acc  \t  >0.9  %f  \t <0.8  %f \t 0.8-0.9  %f \n" % ((allNum - count_low - count_medium)/allNum ,count_low / allNum, count_medium / allNum))
        file_acc_filter.close()
        print('done')

        return 0

def write_ori_acc_to_oneFile(filePath = 'data/acc_SAM/'):
    
    # open the acc files
    file_all_band = open(filePath + 'accuracy_SAM.txt','r')
    file_ABP_bands = open(filePath + 'ori_ABP_bands.txt','r')
    file_ABP_band1 = open(filePath + 'ori_ABP_band1.txt','r')
    file_ABP_band2 = open(filePath + 'ori_ABP_band2.txt', 'r')
    file_ABP_band3 = open(filePath + 'ori_ABP_band3.txt', 'r')
    
    lines_all_band = sorted([line for line in file_all_band])
    lines_ABP_bands = sorted([line for line in file_ABP_bands])
    lines_ABP_band1 = sorted([line for line in file_ABP_band1])
    lines_ABP_band2 = sorted([line for line in file_ABP_band2])
    lines_ABP_band3 = sorted([line for line in file_ABP_band3])
    
    assert ( len(lines_all_band) == len(lines_ABP_bands) == len(lines_ABP_band1) == len(lines_ABP_band2) ==len(lines_ABP_band3) ), 'your acc file did not match, there are some acc omits in acc files'
    
    file_new = open(filePath + 'ori_acc_res.txt', 'w')
    
    rate_bands = 0.0
    rate_band1 = 0.0
    rate_band2 = 0.0
    rate_band3 = 0.0
    count = 0

    for i in range(len(lines_all_band)):
        if i == 0:
            file_new.write('file_name \t\t\t\t all_bands \t\t\t\t ABP_bands \t\t\t\t ABP_band1 \t\t\t\t ABP_band2 \t\t\t\t ABP_band3 \t\t\t\t \n')
        if i == len(lines_all_band) - 1 :
            continue

        fileName = lines_all_band[i].split()[0]
        file_new.write('%s\t\t' % fileName)
        
        acc_all_band = float(lines_all_band[i].split()[2])
        acc_ABP_bands = float(lines_ABP_bands[i].split()[1])
        acc_ABP_band1 = float(lines_ABP_band1[i].split()[1])
        acc_ABP_band2 = float(lines_ABP_band2[i].split()[1])
        acc_ABP_band3 = float(lines_ABP_band3[i].split()[1])
        
        count += 1 
        rate_bands += acc_ABP_bands/ acc_all_band
        rate_band1 += acc_ABP_band1/ acc_all_band
        rate_band2 += acc_ABP_band2/ acc_all_band
        rate_band3 += acc_ABP_band3/ acc_all_band

        file_new.write('%.6f \t\t' % acc_all_band)
        file_new.write('%.6f (%.6f) \t\t' % (acc_ABP_bands, acc_ABP_bands/ acc_all_band))
        file_new.write('%.6f (%.6f) \t\t' % (acc_ABP_band1, acc_ABP_band1/ acc_all_band ))
        file_new.write('%.6f (%.6f) \t\t' % (acc_ABP_band2, acc_ABP_band2/ acc_all_band ))
        file_new.write('%.6f (%.6f) \t\t\n' % (acc_ABP_band3, acc_ABP_band3/ acc_all_band ))
    
    file_new.write('\nBand Average Contribution Rate: \t\t bands: %f   \tband1: %f   \tband2: %f   \tband3: %f \n'% (rate_bands/count , rate_band1/count, rate_band2/count, rate_band3/count))
    #attion: cast out acc_all_band <0.9 and cal average 

    print('done\n')
    
def write_para_acc_to_oneFile(filePath = 'data/acc_SAM/'):
        
    # open the acc files
    file_ABP_bands = open(filePath + 'ori_ABP_bands.txt','r')
    file_ABP_band1 = open(filePath + 'ori_ABP_band1.txt','r')
    file_ABP_band2 = open(filePath + 'ori_ABP_band2.txt', 'r')
    file_ABP_band3 = open(filePath + 'ori_ABP_band3.txt', 'r')
    
    lines_ABP_bands = sorted([line for line in file_ABP_bands])
    lines_ABP_band1 = sorted([line for line in file_ABP_band1])
    lines_ABP_band2 = sorted([line for line in file_ABP_band2])
    lines_ABP_band3 = sorted([line for line in file_ABP_band3])
    
    assert ( len(lines_ABP_bands) == len(lines_ABP_band1) == len(lines_ABP_band2) ==len(lines_ABP_band3) ), 'your acc file did not match, there are some acc omits in acc files'
    
    file_new = open(filePath + '3paras_acc_res.txt', 'w')
    
    rate_bands = 0.0
    rate_band1 = 0.0
    rate_band2 = 0.0
    rate_band3 = 0.0
    count = 0

    for i in range(len(lines_ABP_bands)):
        if i == 0:
            file_new.write('file_name \t\t\t\t ABP_bands \t\t\t\t ABP_band1 \t\t\t\t ABP_band2 \t\t\t\t ABP_band3 \t\t\t\t \n')
        if i == len(lines_ABP_bands) - 1 :
            continue

        fileName = lines_ABP_bands[i].split()[0]
        file_new.write('%s\t\t' % fileName)
        
        acc_ABP_bands = float(lines_ABP_bands[i].split()[1])
        acc_ABP_band1 = float(lines_ABP_band1[i].split()[1])
        acc_ABP_band2 = float(lines_ABP_band2[i].split()[1])
        acc_ABP_band3 = float(lines_ABP_band3[i].split()[1])
        
        count += 1 
 
        if acc_ABP_band1/ acc_ABP_bands < threshold_ratio:
            rate_band1 += acc_ABP_band1/ acc_ABP_bands
        if acc_ABP_band2/ acc_ABP_bands < threshold_ratio:
            rate_band2 += acc_ABP_band2/ acc_ABP_bands
        if acc_ABP_band3/ acc_ABP_bands < threshold_ratio:
            rate_band3 += acc_ABP_band3/ acc_ABP_bands



        file_new.write('%.6f \t\t' % (acc_ABP_bands))
        file_new.write('%.6f (%.6f) \t\t' % (acc_ABP_band1, acc_ABP_band1/ acc_ABP_bands ))
        file_new.write('%.6f (%.6f) \t\t' % (acc_ABP_band2, acc_ABP_band2/ acc_ABP_bands ))
        file_new.write('%.6f (%.6f) \t\t\n' % (acc_ABP_band3, acc_ABP_band3/ acc_ABP_bands ))
    
    file_new.write('\nBand Average Contribution Rate: \t\t   band1: %f   \tband2: %f   \tband3: %f \n'% ( rate_band1/count, rate_band2/count, rate_band3/count))
    #attion: cast out acc_all_band <0.9 and cal average 

    print('done\n')

if __name__ == '__main__':

    if switch_find_stange_points:
        find_strange_points(SAM_single_double = 0)
    if switch_write_ori_acc:
        write_ori_acc_to_oneFile()
    if switch_write_para_acc:
        write_para_acc_to_oneFile()
    