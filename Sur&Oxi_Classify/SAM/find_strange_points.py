def find_strange_points():
    
    filePath = 'E:\College\Code\Python\DataAnalyzing\DataAnalyzing\data/'
    fileName = 'accuracy_result.txt'
    fileName_filter = 'acc_filtered.txt'
    file_acc = open(filePath + fileName,'r')
    file_acc_filter = open(filePath + fileName_filter,'w')
    lines = [line for line in file_acc]


    
    count_medium = 0
    count_high = 0
    count_low = 0

    #count medium acc
    file_acc_filter.write('medium acc:\n')
    for line in lines[1:-1]:
        
        name, single_acc, all_acc = line.split('\t')

        if (float(single_acc) <= 0.90 and  float(single_acc) >= 0.80) and (float(all_acc) <= 0.90 and  float(all_acc) >= 0.80):
            file_acc_filter.write("%s \t %s \t %s\n" % (name, single_acc, all_acc))
            count_medium += 1

    #count high acc
    file_acc_filter.write('low acc:\n')
    for line in lines[1:-1]:
        
        name, single_acc, all_acc = line.split('\t')

        if float(single_acc) < 0.80 and  float(all_acc) < 0.80:
            file_acc_filter.write("%s \t %s \t %s\n" % (name, single_acc, all_acc))
            count_low += 1

    file_acc_filter.write('high acc:\n')
    for line in lines[1:-1]:
        
        name, single_acc, all_acc = line.split('\t')

        if float(single_acc) >= 0.90 or  float(all_acc) > 0.90:
            file_acc_filter.write("%s \t %s \t %s\n" % (name, single_acc, all_acc))
            count_high += 1


    allNum = len(lines) - 1
    file_acc_filter.write("Total acc  \t  >0.9  %f  \t <0.8  %f \t 0.8-0.9  %f \n" % ((allNum - count_low - count_medium)/allNum ,count_low / allNum, count_medium / allNum))
    #print("Total acc  \t  >0.9  %f  \t <0.8  %f \t 0.8-0.9  %f \n" % ((allNum - count_low - count_medium)/allNum ,count_low / allNum, count_medium / allNum))
    file_acc_filter.close()
    print('done')

    return 0
find_strange_points()
