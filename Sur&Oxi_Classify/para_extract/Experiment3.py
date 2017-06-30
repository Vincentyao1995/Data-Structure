import SP_paras
import test_algorithm as ta
#this experiment could extract paras from SP and then input these paras into SAM and got res.  Switch discard_AAWP control whether use AA AW AP or not.(much bigger than AS's)

# Switch~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
discard_AAWP = 1

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

#cal parameters of spectrum
para_sulf_dict = SP_paras.SP_paras(SP_training_sulf)
para_oxi_dict = SP_paras.SP_paras(SP_training_oxi, type = 'oxido')
#testing data is calculated in 'SP_paras.check() -- Exp2.Tranversing()'

if discard_AAWP == 1:
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
ta.check(para_oxi_list, para_sulf_list, check_all = 1, dataProcess_alg = extract_SP_paras)





