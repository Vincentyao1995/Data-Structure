import SP_paras
import test_algorithm as ta
#this experiment could extract paras from SP and then input these paras into SAM and got res.  Switch discard_AAWP control whether use AA AW AP or not.(much bigger than AS's)

# Switch~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
discard_AAWP = 1
check_paras = 1
check_ori_ABP_bands = 0

#this is dataProcess alg that could process SP_testing(array or list.)
def extract_SP_paras(SP_testing, discard_AAWP = discard_AAWP):
    # if you extract paras from SP and use this paras to classify.
    para_testing_dict = SP_paras.SP_paras(SP_testing)
    if discard_AAWP == 1:
        for i in para_testing_dict:
            para_testing_dict[i].pop('AA')
            para_testing_dict[i].pop('AW')
            para_testing_dict[i].pop('AP')

    para_testing_list = SP_paras.dict_to_list(para_testing_dict)
    SP_testing = para_testing_list

    return SP_testing


SP_training_sulf = ta.load_training_SP('sulfuro')
SP_training_oxi = ta.load_training_SP('oxido')

# algorithm that process sp data to ABP ori bands.
def ori_ABP_bands(sp):
    #########################################this is for testing original sp ABP bands' result################################################################
    sp = SP_paras.choose_ABP_bands(sp, choose_band = [1,0,0])
    spList = []

    for i in range(len(sp)):
        for j in range(len(sp[i])):
            spList.append(sp[i][j])
    return spList

# algorithm that process sp data to ABP paras
def para_ABP_bands(sp):
    para_dict = SP_paras.SP_paras(sp,choose_band = [0,0,1])
    
    if discard_AAWP == 1:
        for i in para_dict :
            para_dict[i].pop('AA')
            para_dict[i].pop('AP')
            para_dict[i].pop('AW')
    para_list = SP_paras.dict_to_list(para_dict)
    return para_list

if check_ori_ABP_bands == 1:
    SP_training_sulf = ori_ABP_bands(SP_training_sulf)
    SP_training_oxi = ori_ABP_bands(SP_training_oxi)

if check_paras == 1:
    SP_training_sulf = para_ABP_bands(SP_training_sulf)
    SP_training_oxi = para_ABP_bands(SP_training_oxi)

if check_ori_ABP_bands ==1 :
    ta.check(SP_training_oxi, SP_training_sulf, check_all = 1, dataProcess_alg = ori_ABP_bands)
if check_paras == 1:
    ta.check(SP_training_oxi, SP_training_sulf, check_all = 1, dataProcess_alg = para_ABP_bands,file_acc_name = '3paras_ABP_band3_SAM.txt')




