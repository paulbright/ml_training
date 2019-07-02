# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:30:27 2019

@author: a142400
"""

import pickle as pickle 

loaded_model = pickle.load(open("linear_reg.sav", 'rb'))
print(loaded_model.predict(8))

lnreg2_model = pickle.load(open("linear_reg2.sav", 'rb'))
polyreg_model = pickle.load(open("polyreg.sav", 'rb'))

print(lnreg2_model.predict(polyreg_model.fit_transform(8)))
