import SP_paras
import Classifier_Exp2 as exp2


SP_training_sulf = SP_paras.load_training_SP('sulfuro')
SP_training_oxi = SP_paras.load_training_SP('oxido')

#cal parameters of spectrum
para_sulf_dict = SP_paras.SP_paras(SP_training_sulf)
para_oxi_dict = SP_paras.SP_paras(SP_training_oxi, type = 'oxido')
#testing data is calculated in 'SP_paras.check() -- Exp2.Tranversing()'

#convert format from dict to list
para_sulf_list = SP_paras.dict_to_list(para_sulf_dict)
para_oxi_list = SP_paras.dict_to_list(para_oxi_dict)
#testing data is converted in 'SP_paras.check() -- Exp2.Tranversing()' 

# normalize the data list, 1.6,1600,53,50 --- all convert to 0-1  
# This is done in 'SP_paras.check() -- Exp2.Tranversing()'

#  use_SP_paras should genggai, Tranversing img_tesing, img_tesing could have an image that is written by alg.
SP_paras.check(para_oxi_list, para_sulf_list, check_all = 1, use_SP_paras = 1)

#attention: absorption window didn't match. Pixel's ABP window extraction manually is wrong. So AA AW SAI is not right.




