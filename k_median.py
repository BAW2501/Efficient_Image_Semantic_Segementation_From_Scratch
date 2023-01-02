import numpy as np
import matplotlib.pyplot as plt
import time


# from scipy.spatial.distance import cdist
# https://stackoverflow.com/questions/32154475/einsum-and-distance-calculations


def cluster_index(x_train, centers):
    # images are of type uint8, so we need to convert to int16 to avoid overflow when subtracting and squaring
    x_train = x_train.astype(np.int16)
    centers = centers.astype(np.int16)
    dist = np.sum((x_train[:, None, :] - centers)** 2, axis=-1)
    return np.argmin(dist, axis=1)


def k_median(x_train, n_cluster, max_iter=100):
    # Randomly initialize  n_clusters centers
    new_centers = x_train[np.random.choice(len(x_train), n_cluster, replace=False)]

    for _ in range(max_iter):
        # Assign each point to the closest center
        indices = cluster_index(x_train, new_centers)
        # Push current centers to previous, reassign centers as median of the points belonging to them
        old_centers = new_centers
        # new_centers = np.array([np.median(x_train[indices == j], axis=0) for j in range(n_cluster)], dtype=np.uint16)
        new_centers = np.array([np.median(x_train[indices == j], axis=0) for j in range(n_cluster)], dtype=np.uint8)
        # If no centers have changed, we're done
        if np.equal(new_centers, old_centers).all():
            return new_centers[indices], indices


if __name__ == '__main__':
    # Load image and flatten into array of pixels
    img = plt.imread('NaturalColorImages.jpg')
    # flatten all but last dim
    pixel_vals = img.reshape((-1, img.shape[-1]))
    # apply k_median
    times = []
    for i in range(10):
        start = time.time()
        new_pixel_vals, labels = k_median(pixel_vals, n_cluster=3)
        # reshape into image
        end = time.time()
        print(f'Time taken for {i:02d} iteration : {(end - start) * 1000:.2f} ms', )
        segmented_image = new_pixel_vals.reshape(img.shape)
        times.append(end - start)
        # plot
        plt.axis('off')
        plt.imshow(segmented_image)
        plt.show()
    # print statistics avr min max
    times = np.array(times)
    print(f'Average time taken: {np.mean(times) * 1000:.2f} ms')
    print(f'Min time taken: {np.min(times) * 1000:.2f} ms')
    print(f'Max time taken: {np.max(times) * 1000:.2f} ms')
