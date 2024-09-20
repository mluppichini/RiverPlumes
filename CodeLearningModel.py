import numpy as np
import h5py
import pandas as pd
#from osgeo import gdal, ogr
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
from shapely import geometry
from shapely.geometry import Point
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import LeakyReLU, RNN
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import History, ModelCheckpoint
from time import time
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras import callbacks
from tensorflow.keras.models import model_from_json
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, MaxPooling2D, Dense, Flatten

import tensorflow as tf
import keras
from urllib.request import urlretrieve
import shutil
import zipfile
import geopandas as gp
from shapely.geometry import LineString
from shapely.ops import split, snap
import numpy as np
from sklearn.metrics import *
import seaborn as sns
import ee
ee.Initialize()
ee.Authenticate()

def CreateDataset(parametri, pathLayer, path_out ):

    '''
    Function used to create the input dataset in category 0,1,2

    :param parametri: a dictionary with the function input data
    :param pathLayer: it's parametri['path_Sediments_shp'] --> the path of the shapefile that contains area category 0,1,2
    :param path_out: it's parametri['path_Sediments'] --> output file
    :return:
    '''
    # pathLayer = parametri['path_Sediments_shp']
    # path_out = parametri['path_Sediments']

    service = parametri['service']
    if 'COPERNICUS' in service:
        band_request = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
    elif 'LANDSAT/LC08/C02/T1' == service:
        band_request = ['B%s' % i for i in range(1, 12)]
    else:
        band_request = ['SR_B%s' % i for i in range(1, 8)]

    NDSSI = False
    scale = parametri['scale']

    #layer = QgsVectorLayer(pathLayer, "old", "ogr")
    layer = gp.read_file(pathLayer)
    areas = layer['area'].unique()
    areas.sort()
    values_tot = []
    dates_tot = []
    for area_bbox in areas:
        sel = layer[layer['area'] == area_bbox]
        bounds = sel.geometry.bounds
        xMin = bounds['minx'].min()
        xMax = bounds['maxx'].max()
        yMin = bounds['miny'].min()
        yMax = bounds['maxy'].max()

        dates = list(sel.date.unique())
        area = ee.Geometry.Polygon([[xMin, yMin], \
                                    [xMax, yMin], \
                                    [xMax, yMax], \
                                    [xMin, yMax], \
                                    [xMin, yMin]])
        addTime = lambda x: x.set('Date', ee.Date(x.get('system:time_start')).format("YYYY-MM-dd"))
        collection = ee.ImageCollection(service) \
            .filterBounds(area) \
            .select(band_request) \
            .map(addTime) \
            .filter(ee.Filter.inList('Date', ee.List(dates)))

        nImages = collection.size().getInfo()
        imgs = collection.toList(collection.size())
        list_imgs = [ee.Image(imgs.get(i)) for i in range(nImages)]

        dates_unique = list(set(dates))
        dates_unique.sort()
        # check if I have to do a mosaic
        list_imgs = []
        dates = []
        for date in dates_unique:
            dates.append(date)
            date = ee.Date(date)
            filtered = collection.filterDate(date, date.advance(1, 'day'))
            image = ee.Image(filtered.mosaic())
            list_imgs.append(image)
        collection = ee.ImageCollection.fromImages(list_imgs)
        # self.nImages = collection.size().getInfo()
        valori = []
        nspace = int(parametri['model_setting']['nDim'] / 2)
        #print ("nspace", nspace)
        points = []
        for i, img in enumerate(list_imgs):
            if i % 5 == 0: print (i)
            date = dates[i]
            lat, lon, data = LatLonImg(img, band_request, area, scale, NDSSI)
            lats = toImage(lat, lon, lat)
            lons = toImage(lat, lon, lon)
            buffer_size = (lon[1] - lon[0])/2
            images = []
            values = {}
            sel2 = sel[sel['date'] == date]
            dict_bands =  {i: toImage(lat, lon, column_band) for i, column_band in enumerate(data)}
            for feat in sel2.geometry:
                for nrow in range (lats.shape[0]):
                    for ncol in range (lats.shape[1]):
                        point = Point(lons[nrow][ncol], lats[nrow][ncol]) #.buffer(buffer_size)
                        if feat.intersects(point):
                            values = []
                            for nband, image in dict_bands.items():
                                matrix_values = image[nrow - nspace: nrow + nspace + 1, ncol - nspace : ncol+ nspace +1]
                                #print("matrix shape", matrix_values.shape, "wanted", (parametri['model_setting']['nDim'], parametri['model_setting']['nDim']))
                                if matrix_values.shape != (parametri['model_setting']['nDim'], parametri['model_setting']['nDim']):
                                    break
                                values.append(matrix_values)
                            values = np.array(values)
                            if values.shape == (len(band_request), parametri['model_setting']['nDim'],parametri['model_setting']['nDim']):
                                values_tot.append(values)
                                dates_tot.append(date)

    values_tot = np.array(values_tot)
    with open(path_out, 'wb') as f:
        np.save(f, values_tot)
    Dates = pd.DataFrame(dates_tot, columns = ['date'])
    path_out2 = path_out.replace(".npy", "_dates.csv")
    Dates.to_csv(path_out2, index = False)



def LearnClassificationModel(parametri):
    n = 150000
    plt.close()
    xNo = np.load(parametri["path_noSediments"])
    print('N. No data:', len(xNo))
    no_indexes = [i for i in range (len(xNo))]
    no_indexes = np.random.choice(no_indexes, size=n)
    xNo = xNo[no_indexes]
    print('N. No data:', len(xNo))
    xSi = np.load(parametri["path_Sediments"])
    print('N. Si data:', len(xSi))
    si_indexes = [i for i in range (len(xSi))]
    si_indexes = np.random.choice(si_indexes, size=n)
    xSi = xSi[si_indexes]
    print('N. Si data:', len(xSi))
    xNi = np.load(parametri["path_clearwater"])
    print('N. Ni data:', len(xNi))
    ni_indexes = [i for i in range (len(xNi))]
    ni_indexes = np.random.choice(ni_indexes, size=n)
    xNi = xNi[ni_indexes]
    print('N. Ni data:', len(xNi))

    for matrix in xSi:
        if np.all(matrix == 0):
            print("All elements are zero")
        else:
            pass

    ySi = [2 for i in range (len(xSi))]
    yNi = [1 for i in range (len(xNi))]
    yNo = [0 for i in range (len(xNo))]
    xNoTrain, xNoTest, yNoTrain, yNoTest = train_test_split(xNo, yNo, test_size = 0.2, random_state=42)
    xNoTrain, xNoVal, yNoTrain, yNoVal = train_test_split(xNoTrain, yNoTrain, test_size = 0.25, random_state=42)

    xSiTrain, xSiTest, ySiTrain, ySiTest = train_test_split(xSi, ySi, test_size = 0.2, random_state=42)
    xSiTrain, xSiVal, ySiTrain, ySiVal = train_test_split(xSiTrain, ySiTrain, test_size = 0.25, random_state=42)

    xNiTrain, xNiTest, yNiTrain, yNiTest = train_test_split(xNi, yNi, test_size=0.2, random_state=42)
    xNiTrain, xNiVal, yNiTrain, yNiVal = train_test_split(xNiTrain, yNiTrain, test_size=0.25, random_state=42)

    xTrain = np.concatenate([xNoTrain, xSiTrain, xNiTrain])
    yTrain = np.concatenate([yNoTrain, ySiTrain, yNiTrain])
    xVal = np.concatenate([xNoVal, xSiVal, xNiVal])
    yVal = np.concatenate([yNoVal, ySiVal, yNiVal])
    xTest = np.concatenate([xNoTest, xSiTest, xNiTest])
    yTest = np.concatenate([yNoTest, ySiTest, yNiTest])

    print (xTrain.shape, yTrain.shape, xVal.shape, yVal.shape)
   # return

    # max_val = np.amax(xTrain)
    # min_val = np.amin(xTrain)
    #
    # xTrain = (xTrain - min_val) / (max_val - min_val)
    # xTest = (xTest - min_val) / (max_val - min_val)

    # MODEL
    patience = parametri['model_setting']['patience']
    earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                            mode="min", patience=patience,
                                            restore_best_weights=True)

    bst_model_path = os.path.join( parametri["folder_out"] , "model.sav")
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)


    model = Sequential()
    model.add(Conv1D(filters=parametri['model_setting']['nNodi'], kernel_size=2, input_shape=xTrain.shape[1:]))
    model.add(Conv1D(filters=parametri['model_setting']['nNodi'] * 2, kernel_size=2))
    model.add(MaxPooling2D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(50))
    model.add(Dense(1))
    model.add(Dense(3, activation=tf.nn.softmax))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(xTrain, yTrain, epochs=10000, batch_size=1000,
                        validation_data=(xVal, yVal), verbose=1,
                        shuffle=False, validation_split=0, callbacks=[earlystopping, model_checkpoint])

    model.load_weights(bst_model_path)
    bst_model_path_json = bst_model_path.replace(".h5", ".json")
    save_deep_learning_model(model, bst_model_path_json, False)

    model.load_weights(bst_model_path)

    # keras.backend.clear_session()

    yhat = model.predict(xTest, verbose=1)
    labels_predicted = [np.argmax(perc) for i, perc in enumerate(yhat)]

    dfRes = pd.DataFrame()
    dfRes['obs'] = yTest
    dfRes['label_pred'] = labels_predicted
    # dfRes.loc[dfRes['pred'] <= 0.5, 'label_pred'] = 0
    # dfRes.loc[dfRes['pred'] > 0.5, 'label_pred'] = 1

    #plt.scatter(dfRes['obs'], dfRes['label_pred'])
    cf_matrix = confusion_matrix(dfRes['obs'], dfRes['label_pred']) #tn, fp, fn, tp: true false positive negative
   # group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0: 0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0: .2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v2}\n{v3}" for v2, v3 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(3, 3)
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap = 'Blues')
    plt.savefig(os.path.join( parametri["folder_out"] , "confusion_matrix.png"))

    import json
    with open(os.path.join( parametri["folder_out"] , "parametri.json"), "w") as f:
        json.dump(parametri, f)


######################              INPUT PARAMETERS              ####################
parameters = {"service" :  "COPERNICUS/S2_HARMONIZED",
              'scale': 50,
                "path_noSediments": r".\No_Sediments.npy",
                 "path_Sediments": r".\Sediments.npy",
                 "path_noSediments_shp": r".\clouds.shp",
                 "path_terra_shp": r".\terra.shp",
              'path_clearwater_shp':r".\clearwater.shp",
                'path_clearwater': r".\CleanWater.npy",
                 "maxDateTerra": 3,
                "path_Sediments_shp": r".\sediments.shp",
                "folder_out": r".\sentinel2_Imagenet",
                'model_setting': {
                                   "nNodi": 2 ** 5,
                                   'patience': 200,
                                   'nDim': 11},
                }


##############          CREATE 3 DATASET        #######################
CreateDataset(parameters, parameters['path_Sediments_shp'], parameters['path_Sediments'])
CreateDataset(parameters, parameters['path_noSediments_shp'], parameters['path_noSediments'])
CreateDataset(parameters, parameters['path_clearwater_shp'], parameters['path_clearwater'])


##############          LEARNING MODEL       #######################

LearnClassificationModel(parameters)
