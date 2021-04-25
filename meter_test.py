# importing libraries
import sys, getopt
import math
import pandas as pd
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Input, concatenate
from tensorflow.keras.layers import LeakyReLU
import cv2
import numpy as np

import matplotlib.pyplot as plt


def load_and_process( filename, input_shape, row, model,  column_list ):
    img  = cv2.imread( filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize( gray, ( input_shape[0], input_shape[1]  ) )/255.0
    tensor_image = np.expand_dims( gray, axis=0 )
    results = model.predict( tensor_image )
    results = results[0]
    return  results


def model_test( csv, model_file ):
    # Lonad CSV Fi4
    df=pd.read_csv(csv, sep=',')

    nb_train_samples      = len(df.index)
    nb_validation_samples = len(df.index)

    column_list = []
    index = 0
    for col in df.columns:
        if index != 0:
            column_list.append( col)
        index+=1

    # Save model
    model = tensorflow.keras.models.load_model(model_file)
    model.summary()

    # Model preparation
    input_shape = model.get_layer('input_1').input_shape[0][1:]

    for index, row in df.iterrows():
        result = load_and_process( row['filename'] , input_shape, row, model,  column_list )
        print( "File ", row['filename'], " Result ", result, " - ",  ( result*(90+2) -1 ) )


def main( argv ):
    """
    :param argv: Command line arguments
    """
    csv = None # CSV File
    model_file = None
    try:
        opts, args = getopt.getopt(argv,"hc:m:",["csv", "model"])
    except getopt.GetoptError:
        print('python meter_test.py -c <csv> -m <model>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('python meter_test.py -c <csv> -m <model>')
            sys.exit()
        elif opt in ("-c", "--csv"):
            csv = arg
        elif opt in ("-m", "--model"):
            model_file = arg

    if csv is None or model_file is None:
        print('python meter_test.py -c <csv> -m <model>')
        exit(2)

    print(" Csv file ", csv, " Model ", model_file  )
    model_test( csv, model_file )


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print('python meter_test.py -c <csv> -m <model>')
        sys.exit(2)
    main(sys.argv[1:])
