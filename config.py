import os
import ConfigParser

configParser = ConfigParser.RawConfigParser()
configFilePath = os.path.abspath('./Plots//')
configParser.read(configFilePath)

variable = configParser.get(configFilePath, 'variable')

def construct_df(data, number_rol, number_col, col_labels):
    pass
