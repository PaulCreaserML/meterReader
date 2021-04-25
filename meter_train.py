# importing libraries
import sys, getopt
import math
import random
import pandas as pd
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Conv2D, MaxPooling2D, Average, Add, Multiply
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Input, concatenate
from tensorflow.keras.layers import LeakyReLU
import numpy as np

import matplotlib.pyplot as plt

img_width, img_height = 100, 100

def model_build( img_height, img_width, depth=1 ):
    # Model build
    input_img = Input( shape=( img_height, img_width, depth ) )
    x = Conv2D( 64, (5, 5), use_bias=False, activation='relu')( input_img )
    x = MaxPooling2D( (2, 2) )( x )

    x = Conv2D(64, (3, 3), use_bias=False, activation='relu')( x )
    x = MaxPooling2D( (2, 2) )( x )

    x = Conv2D( 32, (3, 3), use_bias=False, activation='relu' )( x )
    x = MaxPooling2D( (2, 2) )( x )

    # Minute hand features feature1_3x3 ) #
    x  = Flatten()( x )
    x  = Dense(32, activation='relu')( x )
    # x  = tensorflow.keras.layers.Dropout(.01)( x )
    #
    x  = Dense(16, activation='relu')( x )
    # Last
    output = Dense(1, activation='sigmoid')( x )
    return  Model( input_img, output )

def meter_train( csv, test_csv, epochs=200, batch_size=2, saved_model=None, checkpoint_dir=None ):
    df      = pd.read_csv(csv, sep=',')
    test_df = pd.read_csv(test_csv, sep=',')

    nb_train_samples      = len(df.index)
    nb_validation_samples = len(df.index)

    # Train Data
    column_list = []
    index = 0
    for col in df.columns:
        if index != 0:
            column_list.append( col )
            print("Col ", col )
        index+=1

    # Test Data
    test_column_list = []
    index = 0
    for col in test_df.columns:
        if index != 0:
            test_column_list.append( col )
        index+=1

    print("Datafrrame ",  df )
    print("Column list ", column_list )

    model = model_build( img_height, img_width, depth=1 )
    model.compile(optimizer='adam', loss='mse', metrics =['mae'])
    model.summary()

    # Image preprocessing
    train_datagen = ImageDataGenerator(
        rescale = 1. / 255,
        horizontal_flip = False,
        #brightness_range=[0.8,1.2],
        #zoom_range=[0.95, 1.05],
        #height_shift_range=[-2, 2],
        #width_shift_range=[-2, 2]
        )

    test_datagen = ImageDataGenerator(
        rescale = 1. / 255,
        horizontal_flip = False
    )

    # Training dataset
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df,
        x_col='filename',
        color_mode='grayscale',
        y_col= column_list,
        target_size =( img_height, img_width ),
        batch_size = batch_size,
        shuffle = True,
        class_mode = 'raw' )

    # Validation dataset
    validation_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filename',
        color_mode='grayscale',
        y_col= test_column_list,
        target_size =( img_height, img_width ),
        batch_size =  batch_size,
        class_mode ='raw' )

    model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    # Fit Model
    model.fit(train_generator,
        steps_per_epoch = nb_train_samples // batch_size,
        epochs = epochs, validation_data = validation_generator,
        validation_steps = nb_validation_samples // batch_size,
        callbacks=[model_checkpoint_callback] )

    # Save model
    if saved_model is not None:
        tensorflow.saved_model.save( model, saved_model)


def main( argv ):
    """
    :param argv: Command line arguments
    """
    train_csv      = None # CSV File
    model_file     = None
    checkpoint_dir = None
    test_csv       = None

    try:
        opts, args = getopt.getopt(argv,"hc:m:k:t:",["csv", "model", "checkpoint", "test_csv" ])
    except getopt.GetoptError:
        print('python meter_train.py -c <csv> -t <test_csv> -m <model> -k <checkpoint>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('python meter_train.py -c <csv> -t <test_csv> -m <model> -k <checkpoint>')
            sys.exit()
        elif opt in ("-c", "--csv"):
            train_csv = arg
        elif opt in ("-m", "--model"):
            model_file = arg
        elif opt in ("-k", "--checkpoint"):
            checkpoint_dir = arg
        elif opt in ("-t", "--test_csv"):
            test_csv = arg

    if train_csv is None or test_csv is None or model_file is None or checkpoint_dir is None:
        print('python meter_train.py -c <csv> -t <test_csv> -m <model> -k <checkpoint>')
        exit(2)

    print(" --------------------------------------------------------------" )
    print(" Train csv file ", train_csv, " Test CSV ", test_csv, " Model ", model_file  )
    print(" --------------------------------------------------------------" )
    meter_train( train_csv, test_csv, epochs=120, batch_size=8, saved_model=model_file, checkpoint_dir=checkpoint_dir )


if __name__ == "__main__":
    if len(sys.argv) != 9:
        print('python meter_train.py -c <csv> -t <test_csv> -m <model> -k <checkpoint>')
        sys.exit(2)
    main(sys.argv[1:])
