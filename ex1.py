import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.spatial import distance


def init_centroids(X, K):
    """
    Initializes K centroids that are to be used in K-Means on the dataset X.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    K : int
        The number of centroids.

    Returns
    -------
    centroids : ndarray, shape (K, n_features)
    """
    if K == 2:
        return np.asarray([[0., 0., 0.],
                           [0.07843137, 0.06666667, 0.09411765]])
    elif K == 4:
        return np.asarray([[0.72156863, 0.64313725, 0.54901961],
                           [0.49019608, 0.41960784, 0.33333333],
                           [0.02745098, 0., 0.],
                           [0.17254902, 0.16862745, 0.18823529]])
    elif K == 8:
        return np.asarray([[0.01568627, 0.01176471, 0.03529412],
                           [0.14509804, 0.12156863, 0.12941176],
                           [0.4745098, 0.40784314, 0.32941176],
                           [0.00784314, 0.00392157, 0.02745098],
                           [0.50588235, 0.43529412, 0.34117647],
                           [0.09411765, 0.09019608, 0.11372549],
                           [0.54509804, 0.45882353, 0.36470588],
                           [0.44705882, 0.37647059, 0.29019608]])
    elif K == 16:
        return np.asarray([[0.61568627, 0.56078431, 0.45882353],
                           [0.4745098, 0.38039216, 0.33333333],
                           [0.65882353, 0.57647059, 0.49411765],
                           [0.08235294, 0.07843137, 0.10196078],
                           [0.06666667, 0.03529412, 0.02352941],
                           [0.08235294, 0.07843137, 0.09803922],
                           [0.0745098, 0.07058824, 0.09411765],
                           [0.01960784, 0.01960784, 0.02745098],
                           [0.00784314, 0.00784314, 0.01568627],
                           [0.8627451, 0.78039216, 0.69803922],
                           [0.60784314, 0.52156863, 0.42745098],
                           [0.01960784, 0.01176471, 0.02352941],
                           [0.78431373, 0.69803922, 0.60392157],
                           [0.30196078, 0.21568627, 0.1254902],
                           [0.30588235, 0.2627451, 0.24705882],
                           [0.65490196, 0.61176471, 0.50196078]])
    else:
        print('This value of K is not supported.')
        return None


def closest_centroid_assign(centerArr, point):
    """
    this function return the closest centroid to the given point
    """
    closest_center = centerArr[0]
    min_dist = distance.euclidean(point, centerArr[0])
    for index, center in enumerate(centerArr):
        temp = distance.euclidean(point, center)
        if temp < min_dist:
            min_dist = temp
            closest_center = centerArr[index]
    return closest_center


def update_centroid(centroid_arr, clusters, picture):
    """update each centroid to be the average of the points in the cluster"""
    for j, centroid in enumerate(centroid_arr):
        counter = 0
        sum = 0
        for index, point in enumerate(clusters):
            if np.array_equal(point, centroid):
                sum += picture[index]
                counter += 1
        if counter != 0:
            centroid_arr[j] = sum / counter
    return centroid_arr


def print_centroids(centroids):
    """print first iteration"""

    data = ""
    for i, point in enumerate(centroids):
        data += "["
        for num in point:
            num_str = str(np.floor(num * 100) / 100)
            if num_str == "0.0":
                num_str = "0."
            data += num_str + ", "
        data = data[:-2]  # trim the last ", "
        data += "], "
    data = data[:-2]  # trim the last ", "
    print(data)


# data preparation (loading, normalizing, reshaping)
path = 'dog.jpeg'
A = imread(path)
A_norm = A.astype(float) / 255.
img_size = A_norm.shape
X = A_norm.reshape(img_size[0] * img_size[1], img_size[2])
Y = np.copy(X)

kArray = [2, 4, 8, 16]
for k in kArray:
    print("k=" + str(k) + ":")
    centroids = init_centroids([0], k)
    print("iter 0: ", end="")
    print_centroids(centroids);
    for i in range(1, 11):
        print("iter " + str(i) + ": ", end="")
        # the loop assign each point to the closest centroid
        for index, point in enumerate(X):
            Y[index] = closest_centroid_assign(centroids, point)
        # update each centroid to be the average of the points in the cluster
        centroids = update_centroid(centroids, Y, X)
        print_centroids(centroids)

    Z = Y.reshape(128, 128, 3)
    plt.imshow(Z)
    plt.grid(False)
    plt.show()

