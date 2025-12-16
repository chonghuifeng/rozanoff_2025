"""
Module to define the paths used in the code
"""

###########
# Imports #
###########

import os

#########
# Paths #
#########

# .\rozanoff_env\Scripts\Activate.ps1

# Root
PATH_ROOT_FOLDER = os.path.dirname(os.path.dirname(__file__))
PATH_DATA_FOLDER = os.path.join(PATH_ROOT_FOLDER, "original_data")
PATH_FLAMELENGTH_FOLDER = os.path.join(PATH_ROOT_FOLDER, "flame_length_pred")

# Data
PATH_DATASETV1 = os.path.join(PATH_DATA_FOLDER, "DataSetV1.xlsx")
PATH_DATA_FLAMELENGTH = os.path.join(PATH_DATA_FOLDER, "FlameLength.xlsx")
PATH_DATA_FLAMELENGTH_NEW = os.path.join(PATH_DATA_FOLDER, "DatasetFlameComplet.xlsx")
