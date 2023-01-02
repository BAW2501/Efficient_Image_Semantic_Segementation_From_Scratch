import cython
cimport cython
import numpy as np
cimport numpy as np
from libc.stdint cimport int16, uint8

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def cluster_index(np.ndarray[uint8, ndim=2] x_train, np.ndarray[uint8, ndim=2] centers):
    # images are of type uint8, so we need to convert to int16 to avoid overflow when subtracting and squaring
    cdef np.ndarray[int16_t, ndim=2] x_train_typed = x_train.astype(np.int16)
    cdef np.ndarray[int16_t, ndim=2] centers_typed = centers.astype(np.int16)
    cdef np.ndarray[int16_t, ndim=3] dist = (x_train_typed[:, None, :] - centers_typed)** 2
    return np.argmin(np.sum(dist, axis=-1), axis=1)
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def k_median(np.ndarray[uint8, ndim=2] x_train, int n_cluster, max_iter=100):
    # Randomly initialize  n_clusters centers
    np.ndarray[uint8, ndim=2] new_centers = x_train[np.random.choice(len(x_train), n_cluster, replace=False)]

    for _ in range(max_iter):
        # Assign each point to the closest center
        indices = cluster_index(x_train, new_centers)
        # Push current centers to previous, reassign centers as median of the points belonging to them
        np.ndarray[uint8, ndim=2] old_centers = new_centers
        # new_centers = np.array([np.median(x_train[indices == j], axis=0) for j in range(n_cluster)], dtype=np.uint16)
        new_centers = np.array([np.median(x_train[indices == j], axis=0) for j in range(n_cluster)], dtype=uint8)
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

