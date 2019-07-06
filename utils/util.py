import pickle
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
from skimage import transform

def save_everything(model,file_name, history):
    """
    Saves given:
    model: ANN model weights
    history: training history of model
    score: Score of the model on test data
    """
    with open('history/' + str(file_name) + '_hist', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    model_json = model.to_json()
    
    with open('models/' + str(file_name) + '.json', "w") as json_file:
        json_file.write(model_json)
    model.save_weights('models/' + str(file_name) + '.h5')
    
    
def flatten_data(train, val, test, shape=784):
    """
    Flattens given arrays
    """
    train_flat = train.flatten().reshape(train.shape[0], 784)
    val_flat = val.flatten().reshape(val.shape[0], 784)
    test_flat = test.flatten().reshape(test.shape[0], 784)
    return train_flat, val_flat, test_flat


def resize_data(images, imx, imy):
    images = images.reshape((-1, 28, 28,1))
    images_res = np.zeros((images.shape[0], imx, imy,1))
    for i in range(images.shape[0]):
        #cv2.imwrite('r/'+str(i)+'.jpg', cv2.resize(images[i, ..., 0].astype('float32'), (imx, imy)))
        #images_res[i, ..., 0] = transform.resize(images[i,...,0], (imx, imy))
        images_res[i, ..., 0] = cv2.resize(images[i, ..., 0].astype('float32'), (imx, imy), interpolation = cv2.INTER_CUBIC)

    #cv2.imwrite('org.jpg', images[0,...,0])
    #cv2.imwrite('res.jpg', images_res[0,...,0])
    
    
    return images_res
        
        
def prepare_data(X,y,X_test28,resx=56,resy=56):
    """
    Resizes and normalizes given image lists
    X: train_data
    y: train_label
    X_test: test data
    y_test: test_label
    """
    img_rows, img_cols = 28, 28

    #Here we split validation data to optimize classifier during training
    X_train28, X_val28, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)

    #Resize images for bigger networks
    X_train_res = resize_data(X_train28,resx,resy)
    X_val_res = resize_data(X_val28,resx,resy)
    X_test_res = resize_data(X_test28,resx,resy)
    
    X_train28 = X_train28.reshape(X_train28.shape[0], img_rows, img_cols, 1)
    X_test28 = X_test28.reshape(X_test28.shape[0], img_rows, img_cols, 1)
    X_val28 = X_val28.reshape(X_val28.shape[0], img_rows, img_cols, 1)

       
    #Normalize data
    X_train28 = X_train28.astype('float32')
    X_test28 = X_test28.astype('float32')
    X_val28 = X_val28.astype('float32')
    X_train28 /= 255
    X_test28 /= 255
    X_val28 /= 255

    #Normalize data
    X_train_res = X_train_res.astype('float32')
    X_test_res = X_test_res.astype('float32')
    X_val_res = X_val_res.astype('float32')
    X_train_res /= 255
    X_test_res /= 255
    X_val_res /= 255
    return X_train28, X_val28, X_test28, X_train_res, X_val_res, X_test_res, y_train, y_val
    
    
    
