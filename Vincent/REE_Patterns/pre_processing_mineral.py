from scipy import signal
from numpy import mean
import ModifiedGaussianModel as MGM
import numpy as np
import matplotlib.pyplot as plt

center_error = 6

#this function input the spectrum(rocks pixels'), centers' position and weight([863.5,80%]), and the judging method('general', 'modeling'). Return the percent this pixel is possible to be mineralA
def cal_centers_around(sp_testing, center, method = 'general'):
    
    if method == 'general':
        for i in range(len(sp_testing[:,0])):
            if sp_testing[:,0][i] <= center[0] and sp_testing[:,0][i+1] >= center[0]: 
                index = i
        left_higher_num = 0
        right_higher_num = 0
        # set acceptable error
        index_list = [index-2, index -1, index, index+1, index+2 ]
        res = False
        for index in sorted(index_list):
            for i in range(index-3, index+4):
                if i < index:
                    if sp_testing[:,1][i] >= center[1]:
                        left_higher_num += 1
                if i > index:
                    if sp_testing[:,1][i] >= center[1]:
                        right_higher_num += 1
            if (left_higher_num >= 1 and right_higher_num >= 2) or (left_higher_num >= 2 and right_higher_num >= 1):
                res = True
                break
            else:
                continue
        return res

    if method == 'modeling':
    # use Gaussian model to fit and got center.
        params_initial = []
        height = [0.01 for i in range(len(center))]
        width = [5. for i in range(len(center))]
        params_initial.extend(height)
        params_initial.extend(width)
        params_initial.extend(center)
        params_initial.extend([0])

        params_optimize = MGM.fitting_leastSquare(sp_testing, params_initial)

        num_params_group = int(len(params_optimize)/3)
        centers_modeling = params_optimize[-num_params_group - 1 : -1]
        res_list = []
        for i in range(len(center)):
            if centers_modeling[i] >= center[i] - center_error and centers_modeling[i] <= center[i] + center_error:
                res_list.append(1)
            else:
                res_list.append(0)
        return res_list
#this function input centers position, return a dict, key is centers' position and value is weight. [720.056: 0.3, 760.58: 0.7]
def cal_centers_weight(centers_position, mineral_type = 'bastnas'):
    centers_weight = {}

    #in this for loop, u could read different mineral centers info from a txt file. 
    for center in sorted(centers_position):
        if mineral_type == 'bastnas':
            if int (sum(centers_position)/ len(centers_position)) in range(705,770):
                if center == 740:
                    centers_weight.setdefault(center, 0.9)
                else:
                    centers_weight.setdefault(center, float(0.1/5))
            if int (sum(centers_position)/ len(centers_position)) in range(770,833):
                if center == 791 or center == 797:
                    centers_weight.setdefault(center, 0.45)
                else:
                    centers_weight.setdefault(center, 0.1/4)
            if int (sum(centers_position)/ len(centers_position)) in range(854,880):
                if center == 863 :
                    centers_weight.setdefault(center, 0.9)
                else:
                    centers_weight.setdefault(center, 0.1/2)
            if int (sum(centers_position)/ len(centers_position)) in range(880,900):
                if center == 880 :
                    centers_weight.setdefault(center, 1.0)
                else:
                    centers_weight.setdefault(center, 0.0)
    return centers_weight

# input the reference spectrum info(a list), including this mineral spectrum's main feature, like centers position and depth. And testing spectrum need to be tested. return the simlarity(0-100%) between ref and testing. This function is to make sure whether this spectrum is possible to be mineralA (reference mineral spectrum)
def cal_similarity(reference_info, sp_testing, depth_threshold = 0.0075, method = 'general'):
    #initial part and scoring system: only use center to score and evaluate similarity.
    
    num_param_group = int(len(reference_info['params_initial'])/3)
    centers_position = reference_info['params_initial'][-num_param_group - 1 : -1]
    #there, users should input the weigth of each center: double ABP: double abp center 90%, other centers occupies 10%; single abp center 80% other centers shares 20%.
    centers = cal_centers_weight(centers_position)

    
    index_minimum = sp_testing[:,1].argmin()
    index_mark = 0
    if index_minimum == 0:
        index_mark = 1
    absorption_depth = (mean(sp_testing[:,1][0:3]) + mean(sp_testing[:,1][-4:-1]))/2 - mean(sp_testing[:,1][index_minimum-1:index_minimum+2])
    
    sim_percent = 0.
    if absorption_depth < depth_threshold or index_mark == 1:
        sim_percent = 0.
        return sim_percent

    else:
        if method == 'general':
            for center_position in sorted(centers.keys()):
                if cal_centers_around(sp_testing, [center_position, centers[center_position]], method = method):
                    sim_percent += centers[center_position]
        elif method == 'modeling':
            res_list = cal_centers_around(sp_testing, sorted(list(centers.keys())), method = method)
            for i in range(len(res_list)):
                if res_list[i] == 1:
                    center_position = sorted(centers.keys())[i]
                    sim_percent += centers[center_position]

    return sim_percent

# this method need pre-geologist knowledge, so cal this info auto-matically is kind of hard.
# The most similar: use MGM to simulate reference spectrum and use 'Gaussian params' as reference info.
def cal_reference_info(sp_reference):
    
    return reference_info

# this function input a spectrum(band), return its ABP depth, alg is DT's 'depth - proxy value'. 
def cal_absorption_depth(spectrum):
    return depth







