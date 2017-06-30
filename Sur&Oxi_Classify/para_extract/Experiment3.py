import SP_paras
import Classifier_Exp2 as exp2

def use_SP_paras(SP_testing, discard_AAWP = 1):
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


SP_training_sulf = SP_paras.load_training_SP('sulfuro')
SP_training_oxi = SP_paras.load_training_SP('oxido')

#cal parameters of spectrum
para_sulf_dict = SP_paras.SP_paras(SP_training_sulf)
para_oxi_dict = SP_paras.SP_paras(SP_training_oxi, type = 'oxido')
#testing data is calculated in 'SP_paras.check() -- Exp2.Tranversing()'

for i in para_sulf_dict :
    para_sulf_dict[i].pop('AA')
    para_sulf_dict[i].pop('AP')
    para_sulf_dict[i].pop('AW')
    para_oxi_dict[i].pop('AA')
    para_oxi_dict[i].pop('AP')
    para_oxi_dict[i].pop('AW')
#convert format from dict to list
para_sulf_list = SP_paras.dict_to_list(para_sulf_dict)
para_oxi_list = SP_paras.dict_to_list(para_oxi_dict)
#testing data is converted in 'SP_paras.check() -- Exp2.Tranversing()' 

# normalize the data list, 1.6,1600,53,50 --- all convert to 0-1  
# This is done in 'SP_paras.check() -- Exp2.Tranversing()'




#  use_SP_paras should genggai, Tranversing img_tesing, img_tesing could have an image that is written by alg.
SP_paras.check(para_oxi_list, para_sulf_list, check_all = 1, dataProcess_alg = use_SP_paras)

#attention: absorption window didn't match. Pixel's ABP window extraction manually is wrong. So AA AW SAI is not right.




