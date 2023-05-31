import numpy as np
import random
import math
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float
from skimage import color
# from sklearn.neighbors import KNeighborsClassifier
import multiprocessing

# Clustering Methods for 1-D points


def kmeans(features, k, num_iters=500):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """
    print(features.shape)
    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)
    for i in range(len(idxs)):
        assignments[idxs[i]] = i + 1

    def get_distance(a, b):
        return np.sum((a - b) ** 2)

    def get_neighbors(idx: int):
        dis = np.array([get_distance(features[idx], features[i])
                       for i in range(N)])
        dis[idx] = 999999999
        s = np.argsort(dis)
        f = s[assignments[s] > 0]
        # print(f'[idx={idx}] f', f[:k], assignments[f][:k], dis[f][:k])
        return np.array(f[:k])

    # assign each point to the closest center
    for i in range(N):
        assignments[i] = np.argmin(
            [get_distance(features[i], centers[j]) for j in range(k)]) + 1

    def classify(idx: int) -> int:
        neighbors = get_neighbors(idx)
        # print('neighbors', neighbors)
        b = np.bincount(assignments[neighbors])
        r = 0
        if np.sum(b == np.max(b)) > 1:
            r = assignments[neighbors[np.where(
                (b == np.max(b)) == True)[0][0] - 1]]
        else:
            r = np.argmax(b)
        # print('b', b, 'r', r)
        return r

    for n in range(num_iters):
        done = True
        li = list(range(N))
        random.shuffle(li)
        for i in li:
            c = classify(i)
            if assignments[i] != c and c != 0:
                done = False
                assignments[i] = c
                print('set', i, c)
        # move new center to the center of the cluster
        for i in range(k):
            if i + 1 in assignments:
                centers[i] = np.mean(features[assignments == i + 1], axis=0)
        if done:
            break
    assignments = assignments[assignments > 0] - 1

    # for n in range(num_iters):
    #     model = KNeighborsClassifier(k)
    #     model.fit(features[np.where(assignments > 0)[0]], assignments[assignments > 0])
    #     for i in range(N):
    #         if assignments[i] == 0:
    #             assignments[i] = model.predict([features[i]])[0]

    print('assignments', assignments)
    return assignments

# Clustering Methods for colorful image


def color_get_distance(a, b):
    return np.sum((a - b) ** 2)


def color_get_neighbors(x: int, y: int, N: int, M: int, features, k: int, D: int = 10):
    dis_others = np.full((N, M), 99999999, dtype=np.uint32)
    target_shape = dis_others[max(0, x - D):min(N, x + D + 1),
                             max(0, y - D):min(M, y + D + 1)].shape
    dis = np.array([[color_get_distance(features[x, y], features[i, j])
                    for i in range(max(0, x - D), min(N, x + D + 1))] for j in range(max(0, y - D), min(M, y + D + 1))], dtype=np.uint32)\
        .reshape(target_shape)
    dis_others[max(0, x - D):min(N, x + D + 1),
              max(0, y - D):min(M, y + D + 1)] = dis
    dis = dis_others
    dis[x, y] = 999999999
    # print('dis', dis.shape, dis, assignments.shape)
    s = np.dstack(np.unravel_index(
        np.argsort(dis.ravel()), (N, M))).reshape(-1, 2)
    return np.array(s[:k])


def color_classify(x: int, y: int, N: int, M: int, features, k: int, assignments) -> int:
    neighbors = color_get_neighbors(x, y, N, M, features, k)
    # print('neighbors', neighbors)
    # print('assignments[neighbors[:, 0], neighbors[:, 1]]',
    #       assignments[neighbors[:, 0], neighbors[:, 1]])
    b = np.bincount(assignments[neighbors[:, 0], neighbors[:, 1]])
    r = 0
    if np.sum(b == np.max(b)) > 1:
        npi = np.where((b == np.max(b)) == True)[0][0] - 1
        # print('npi', npi)
        r = assignments[neighbors[npi, 0], neighbors[npi, 1]]
    else:
        r = np.argmax(b)
    # print('b', b, 'r', r)
    return r


def color_handle_process(i) -> int:
    x, y, N, M, features, k, assignments, n = i
    r = assignments[x, y]
    c = color_classify(x, y, N, M, features, k, assignments)
    # print('c', c, 'assignments[x, y]', assignments[x, y])
    if r != c and c != 0:
        # print(f'[n={n}] set ({x}, {y}) = {c}')
        return c
    return r


def kmeans_color(features, k, num_iters=3):
    print(features.shape)
    N, M, _ = features.shape
    print(f'N={N}, M={M}')
    assignments = np.zeros((N, M), dtype=np.uint32)
    # Like the kmeans function above
    # YOUR CODE HERE
    idxs = np.random.choice(N, size=(k, 2), replace=False)
    print('idxs', idxs)

    centers = features[idxs[:, 0], idxs[:, 1]]
    print('centers', centers)
    for i in range(len(idxs)):
        assignments[idxs[i][0], idxs[i][1]] = i + 1

    # assign each point to the closest center
    for j in range(M):
        for i in range(N):
            dis = [color_get_distance(features[i, j], centers[l])
                   for l in range(k)]
            # print('dis', dis)
            m = np.argmin(dis)
            assignments[i, j] = m + 1

    for n in range(num_iters):
        done = True
        # assignments_copy = assignments.copy()
        li = [[(x, y, N, M, features, k, assignments, n) for x in range(N)]
              for y in range(M)]
        li = [item for sublist in li for item in sublist]
        result = np.zeros((N * M), dtype=np.uint32)
        with multiprocessing.Pool(16) as pool:
            result = np.array(pool.map(color_handle_process, li))
        result = result.reshape(N, M)
        print(f'[n={n}] result', result)
        done = np.sum(assignments != result) == 0
        assignments = result
        if done:
            break
    assignments = assignments - 1
    # print(assignments)

    # END YOUR CODE

    return assignments


# 找每个点最后会收敛到的地方（peak）
def findpeak(data, idx, r):
    t = 0.01
    shift = np.array([1])
    data_point = data[:, idx]
    dataT = data.T
    data_pointT = data_point.T
    data_pointT = data_pointT.reshape(1, 3)

    # Runs until the shift is smaller than the set threshold
    while shift.all() > t:
        # 计算当前点和所有点之间的距离
        # 并筛选出在半径r内的点，计算mean vector（这里是最简单的均值，也可尝试高斯加权）
        # 用新的center（peak）更新当前点，直到满足要求跳出循环
        # YOUR CODE HERE
        pass
        # END YOUR CODE

    return data_pointT.T


# Mean shift algorithm
# 可以改写代码，鼓励自己的想法，但请保证输入输出与notebook一致
def meanshift(data, r):
    labels = np.zeros(len(data.T))
    peaks = []  # 聚集的类中心
    label_no = 1  # 当前label
    labels[0] = label_no

    # findpeak is called for the first index out of the loop
    peak = findpeak(data, 0, r)
    peakT = np.concatenate(peak, axis=0).T
    peaks.append(peakT)

    # Every data point is iterated through
    for idx in range(0, len(data.T)):
        # 遍历数据，寻找当前点的peak
        # 并实时关注当前peak是否会收敛到一个新的聚类（和已有peaks比较）
        # 若是，更新label_no，peaks，labels，继续
        # 若不是，当前点就属于已有类，继续
        # YOUR CODE HERE
        pass
        # END YOUR CODE
    # print(set(labels))
    return labels, np.array(peaks).T


# image segmentation
def segmIm(img, r):
    # Image gets reshaped to a 2D array
    img_reshaped = np.reshape(img, (img.shape[0] * img.shape[1], 3))

    # We will work now with CIELAB images
    imglab = color.rgb2lab(img_reshaped)
    # segmented_image is declared
    segmented_image = np.zeros((img_reshaped.shape[0], img_reshaped.shape[1]))

    labels, peaks = meanshift(imglab.T, r)
    # Labels are reshaped to only one column for easier handling
    labels_reshaped = np.reshape(labels, (labels.shape[0], 1))

    # We iterate through every possible peak and its corresponding label
    for label in range(0, peaks.shape[1]):
        # Obtain indices for the current label in labels array
        inds = np.where(labels_reshaped == label + 1)[0]

        # The segmented image gets indexed peaks for the corresponding label
        corresponding_peak = peaks[:, label]
        segmented_image[inds, :] = corresponding_peak
    # The segmented image gets reshaped and turn back into RGB for display
    segmented_image = np.reshape(
        segmented_image, (img.shape[0], img.shape[1], 3))

    res_img = color.lab2rgb(segmented_image)
    res_img = color.rgb2gray(res_img)
    return res_img


# Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    # YOUR CODE HERE
    pass
    # END YOUR CODE

    return accuracy
