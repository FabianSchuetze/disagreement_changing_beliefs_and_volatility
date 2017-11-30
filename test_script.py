import os 

getcwd = os.getcwd() 
path_data = os.path.join(getcwd, 'Data/')
from Python.Disagreement_Data import disagreement_data
disagreement = disagreement_data(path_data)
survey, summary = disagreement.data(method='arma')
independent = ['disagreement', 'risk']
regression_output = disagreement.regression(summary, 'vol', independent)
regression_output.summary()
