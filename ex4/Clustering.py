import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

def circles_example():
    """
    an example function for generating and plotting synthetic data.
    """

    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.matrix([np.cos(t) + 0.1 * np.random.randn(length),
                         np.sin(t) + 0.1 * np.random.randn(length)])
    circle2 = np.matrix([2 * np.cos(t) + 0.1 * np.random.randn(length),
                         2 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle3 = np.matrix([3 * np.cos(t) + 0.1 * np.random.randn(length),
                         3 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle4 = np.matrix([4 * np.cos(t) + 0.1 * np.random.randn(length),
                         4 * np.sin(t) + 0.1 * np.random.randn(length)])
    circles = np.concatenate((circle1, circle2, circle3, circle4), axis=1)
    
    plt.plot(circles[0,:], circles[1,:], '.k')
    plt.show()

def get_circles():
    """
    an example function for generating and plotting synthetic data.
    """

    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.matrix([np.cos(t) + 0.1 * np.random.randn(length),
                         np.sin(t) + 0.1 * np.random.randn(length)])
    circle2 = np.matrix([2 * np.cos(t) + 0.1 * np.random.randn(length),
                         2 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle3 = np.matrix([3 * np.cos(t) + 0.1 * np.random.randn(length),
                         3 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle4 = np.matrix([4 * np.cos(t) + 0.1 * np.random.randn(length),
                         4 * np.sin(t) + 0.1 * np.random.randn(length)])
    circles = np.concatenate((circle1, circle2, circle3, circle4), axis=1)
    return np.array(circles.T)

def apml_pic_example(path='APML_pic.pickle'):
    """
    an example function for loading and plotting the APML_pic data.

    :param path: the path to the APML_pic.pickle file
    """

    with open(path, 'rb') as f:
        apml = pickle.load(f)

    plt.plot(apml[:, 0], apml[:, 1], '.')
    plt.show()


def microarray_exploration(data_path='microarray_data.pickle',
                            genes_path='microarray_genes.pickle',
                            conds_path='microarray_conds.pickle'):
    """
    an example function for loading and plotting the microarray data.
    Specific explanations of the data set are written in comments inside the
    function.

    :param data_path: path to the data file.
    :param genes_path: path to the genes file.
    :param conds_path: path to the conds file.
    """

    # loading the data:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(genes_path, 'rb') as f:
        genes = pickle.load(f)
    with open(conds_path, 'rb') as f:
        conds = pickle.load(f)

    # look at a single value of the data matrix:
    i = 128
    j = 63
    print("gene: " + str(genes[i]) + ", cond: " + str(
        conds[0, j]) + ", value: " + str(data[i, j]))

    # make a single bar plot:
    plt.figure()
    plt.bar(np.arange(len(data[i, :])) + 1, data[i, :])
    plt.show()

    # look at two conditions and their correlation:
    plt.figure()
    plt.scatter(data[:, 27], data[:, 29])
    plt.plot([-5,5],[-5,5],'r')
    plt.show()

    # see correlations between conds:
    correlation = np.corrcoef(np.transpose(data))
    plt.figure()
    plt.imshow(correlation, extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.show()

    # look at the entire data set:
    plt.figure()
    plt.imshow(data, extent=[0, 1, 0, 1], cmap="hot", vmin=-3, vmax=3)
    plt.colorbar()
    plt.show()


def euclid(X, Y):
    """
    return the pair-wise euclidean distance between two data matrices.
    :param X: NxD matrix.
    :param Y: MxD matrix.
    :return: NxM euclidean distance matrix.
    """
    return euclidean_distances(X, Y)



def euclidean_centroid(X):
    """
    return the center of mass of data points of X.
    :param X: a sub-matrix of the NxD data matrix that defines a cluster.
    :return: the centroid of the cluster.
    """
    return X.mean(axis=0)


    
def kmeans_pp_init(X, k, metric):
    """
    The initialization function of kmeans++, returning k centroids.
    :param X: The data matrix.
    :param k: The number of clusters.
    :param metric: a metric function like specified in the kmeans documentation.
    :return: kxD matrix with rows containing the centroids.
    """
    N, D = X.shape
    centroids = np.zeros((k, D))
    x_idx = np.random.choice(N)
    centroids[0] = X[x_idx]
    for i in range(1, k):
        data_centroid_distances = metric(X, centroids[:i,:])
        D_x = np.min(data_centroid_distances,axis=1)
        probs = D_x ** 2 / np.sum(D_x ** 2)
        x_idx = np.random.choice(N, p = probs)
        centroids[i] = X[x_idx]
    return centroids

def kmeans(X, k, iterations=10, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init):
    """
    The K-Means function, clustering the data X into k clusters.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param iterations: The number of iterations.
    :param metric: A function that accepts two data matrices and returns their
            pair-wise distance. For a NxD and KxD matrices for instance, return
            a NxK distance matrix.
    :param center: A function that accepts a sub-matrix of X where the rows are
            points in a cluster, and returns the cluster centroid.
    :param init: A function that accepts a data matrix and k, and returns k initial centroids.
    :param stat: A function for calculating the statistics we want to extract about
                the result (for K selection, for example).
    :return: a tuple of (clustering, centroids, statistics)
    clustering - A N-dimensional vector with indices from 0 to k-1, defining the clusters.
    centroids - The kxD centroid matrix.
    """
    centroids = init(X, k, metric)
    for _ in range(iterations):
        clustering = np.argmin(metric(X, centroids), axis=1)
        for i in range(k):
            cluster = X[clustering == i]
            centroids[i] = euclidean_centroid(cluster)
    return  clustering, centroids

def gaussian_kernel(X, sigma):
    """
    calculate the gaussian kernel similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param sigma: The width of the Gaussian in the heat kernel.
    :return: NxN similarity matrix.
    """
    return np.exp(-1 * X ** 2 / (2 * sigma ** 2))


def mnn(X, m):
    """
    calculate the m nearest neighbors similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param m: The number of nearest neighbors.
    :return: NxN similarity matrix.
    """
    N = X.shape[0]
    W = np.zeros(X.shape)
    m_nearest_neighbors = np.argsort(X, axis=1)[:,1:m+1]
    rows_idx = np.repeat(np.arange(N),m).reshape((N,m))
    W[rows_idx, m_nearest_neighbors] = 1
    W = np.logical_or(W, W.T).astype(int)
    return W

def get_laplacian(X, similarity_param, similarity):
    distance_matrix = euclid(X, X)
    W = similarity(distance_matrix, similarity_param)
    N = W.shape[0]
    D_diagonal = W.sum(axis=1)
    D_power_minus_half = np.diag(1 / (D_diagonal ** 0.5))
    L = np.eye(N) - np.dot(D_power_minus_half, W).dot(D_power_minus_half)
    return L

def spectral(X, k, similarity_param, similarity=gaussian_kernel):
    """
    Cluster the data into k clusters using the spectral clustering algorithm.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param similarity_param: m for mnn, sigma for the Gaussian kernel.
    :param similarity: The similarity transformation of the data.
    :return: clustering, as in the kmeans implementation.
    """
    L = get_laplacian(X, similarity_param, similarity)
    eigen_values, eigen_vectors = np.linalg.eigh(L)
    lowest_k_eigen_vectors = eigen_vectors[:,:k]
    rows_normalized =  np.linalg.norm(lowest_k_eigen_vectors, axis=1, ord=2)
    rows_normalized[rows_normalized == 0] = np.finfo(float).tiny
    normalized_lowest_k_eigen_vectors = lowest_k_eigen_vectors / rows_normalized[:,np.newaxis]
    clustering, centroids = kmeans(normalized_lowest_k_eigen_vectors, k)
    return clustering


def elbow(X, times, data_type, k_range=[2,3,4,6,7,8,9,11,13,15]):
    costs = []
    for k in k_range:
        clustering, centroids = kmeans(X, k)
        best_cost = np.inf
        for _ in range(times):
            cost = np.sum((X - centroids[clustering])**2)
            best_cost = min(best_cost,cost)
        costs.append(best_cost)
    plt.figure()
    plt.plot(k_range, costs, 'bo')
    plt.title("elbow for " + data_type)
    plt.savefig("elbow for " + data_type)
    
def silhouette(X, data_type, k_range=[2,3,4,6,7,8,9,11,13,15]):
    distance_matrix = euclid(X,X)
    N = X.shape[0]
    S = []
    for k in k_range:
        clustering, centroids = kmeans(X, k)
        a = []
        for i in range(k):
            cluster_distances = distance_matrix[clustering == i][:,clustering == i]
            sum_distances = cluster_distances.sum(axis=1)
            cluster_size = cluster_distances.shape[0]
            if cluster_size == 1:
                a_cluster = np.array([1])
            else:
                a_cluster = sum_distances / (cluster_size - 1 )
            a.append(a_cluster)
        a = np.concatenate(a)
        
        b = []
        for i in range(k):
            b_arrays = []
            for l in set(range(k)) - {i}:
                cluster_distances = distance_matrix[clustering == i][:,clustering == l]
                sum_distances = cluster_distances.sum(axis=1)
                cluster_size = cluster_distances.shape[1]
                if cluster_size == 1:
                    b_cluster = np.array([[1]])
                else:
                    b_cluster = sum_distances / cluster_size
                b_arrays.append(b_cluster)
            b.append(np.array(b_arrays).min(axis=0))
        b = np.concatenate(b)
        
        maximum_a_b = np.maximum(a,b)
        S.append(np.sum((b-a) / maximum_a_b) / N)
    
    plt.figure()
    plt.plot(k_range, S, 'bo')
    plt.title("silhouette for " + data_type)
    plt.savefig("silhouette for " + data_type)
    

        
def eigen_gap(X, data_type, similarity_param, sim_txt = "gaussian_kernel" ,similarity=gaussian_kernel):
    L = get_laplacian(X, similarity_param, similarity)
    eigen_values, _ = np.linalg.eigh(L)
    plt.figure()
    plt.plot(range(15), eigen_values[:15], 'bo')
    plt.title("eigen values of laplacian for " + data_type + " for " + sim_txt +" with param=" +str(similarity_param))
    plt.savefig("eigen values of laplacian for " + data_type + " for " + sim_txt +" with param=" +str(similarity_param), format='png')
    
def plot_microarray_data(microarray_data, clustering, k,algorithm="kmeans"):
    plt.figure()
    plt.title(algorithm + " for microarray data")
    for i in range(k):
        ax = plt.subplot(330+i+1)
        cluster = microarray_data[clustering == i]
        plt.imshow(cluster, extent=[0, 1, 0, 1], cmap="hot", vmin=-3, vmax=3)
        ax.set_title("cluster: " + str(i+1) + " size = "+str(len(cluster)))
        plt.colorbar()
    plt.tight_layout()
    plt.savefig(algorithm + " for microarray data")
        
if __name__ == '__main__':
    with open('APML_pic.pickle', 'rb') as f:
        apml = pickle.load(f)
    
    n_samples = 1000
    n_bins = 4  
    centers = [(-5, -5), (0, 0), (5, 5),(-5, 5)]
    X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                      centers=centers, shuffle=False, random_state=42)
    
    circles =  get_circles()
    
    
    # k selection for synthetic data
    elbow(X, 10, "synthetic data")
    silhouette(X, "synthetic data")
    eigen_gap(X, "synthetic data", 10, "mnn", mnn)
    eigen_gap(X, "synthetic data", 0.7)
    
    ######## clustering for synthetic data ##############
    clustering, centroids = kmeans(X, 4)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=clustering, cmap='rainbow')
    plt.title("kmeans on synthetic data k = 4")
    plt.savefig("kmeans on synthetic data k = 4")
    
    ##mnn for synthetic data
    clustering = spectral(X, 4, 10, mnn)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=clustering, cmap='rainbow')
    plt.title("spectral clustering on synthetic data k = 4 , with mnn, m = 10")
    plt.savefig("spectral clustering on synthetic data k = 4 , with mnn, m = 10")
    
    #gaussian_kernel for synthetic data
    clustering = spectral(X, 4, 0.7)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=clustering, cmap='rainbow')
    plt.title("spectral clustering on synthetic data k = 4 , with gaussian_kernel, sigma = 0.7")
    plt.savefig("spectral clustering on synthetic data k = 4 , with gaussian_kernel, sigma = 0.7", format='png')
    
    ###plotting similarity graph
    for similarity in [gaussian_kernel, mnn]:
        if similarity.__name__ == "mnn":
            similarity_param = 10
        else:
            similarity_param = 0.7
        data_shuffled = X.copy()
        np.random.shuffle(data_shuffled)
        distance_matrix = euclid(data_shuffled, data_shuffled)
        similiarty_matrix = similarity(distance_matrix, similarity_param)
        plt.figure()
        plt.imshow(similiarty_matrix)
        plt.title("similiarty_matrix for shuffled data with " + similarity.__name__)
        plt.savefig("similiarty_matrix for shuffled data with " + similarity.__name__)
        
        clustering = spectral(X, 4, similarity_param, similarity )
        sorted_data = X[np.argsort(clustering)]
        distance_matrix = euclid(sorted_data, sorted_data)
        similiarty_matrix = similarity(distance_matrix, similarity_param)
        plt.figure()
        plt.imshow(similiarty_matrix, extent=[0,1,0,1])
        plt.title("similiarty_matrix for sorted data with " + similarity.__name__)
        plt.savefig("similiarty_matrix for sorted data with " + similarity.__name__)

    ######## clustering for apml ##############
    clustering, centroids = kmeans(apml, 9)
    plt.figure()
    plt.scatter(apml[:, 0], apml[:, 1], c=clustering, cmap='rainbow')
    plt.title("kmeans on apml k = 9")
    plt.savefig("kmeans on apml k = 9")
    
    ##mnn for apml
    clustering = spectral(apml, 9, 15, mnn)
    plt.figure()
    plt.scatter(apml[:, 0], apml[:, 1], c=clustering, cmap='rainbow')
    plt.title("spectral clustering on apml k = 9 , with mnn, m = 15")
    plt.savefig("spectral clustering on apml k = 9 , with mnn, m = 15")
    
    #gaussian_kernel for apml
    clustering = spectral(apml, 9, 2)
    plt.figure()
    plt.scatter(apml[:, 0], apml[:, 1], c=clustering, cmap='rainbow')
    plt.title("spectral clustering on apml k = 9 , with gaussian_kernel, sigma = 2")
    plt.savefig("spectral clustering on apml k = 9 , with gaussian_kernel, sigma = 2")
    

    ######## clustering for circles ##############
    circles =  get_circles()
    clustering, centroids = kmeans(circles, 4)
    plt.figure()
    plt.scatter(circles[:, 0], circles[:, 1], c=clustering, cmap='rainbow')
    plt.title("kmeans on circles k = 4")
    plt.savefig("kmeans on circles k = 4")
    
    ##mnn for circles
    clustering = spectral(circles, 4, 5, mnn)
    plt.figure()
    plt.scatter(circles[:, 0], circles[:, 1], c=clustering, cmap='rainbow')
    plt.title("spectral clustering on circles k = 4 , with mnn, m = 5")
    plt.savefig("spectral clustering on circles k = 4 , with mnn, m = 5")
    
    #gaussian_kernel for circles
    clustering = spectral(circles, 4, 0.1)
    plt.figure()
    plt.scatter(circles[:, 0], circles[:, 1], c=clustering, cmap='rainbow')
    plt.title("spectral clustering on circles k = 4 , with gaussian_kernel, sigma = 0.1")
    plt.savefig("spectral clustering on circles k = 4 , with gaussian_kernel, sigma = 0.1", format='png')
    
    
    
    with open('microarray_data.pickle', 'rb') as f:
        microarray_data = pickle.load(f)
    #k selection for microarray_data
    elbow(microarray_data, 10, "microarray_data")
#    silhouette(microarray_data, "microarray_data")
    eigen_gap(microarray_data, "microarray_data", 10, "mnn", mnn)
#    eigen_gap(microarray_data, "microarray_data", 1.5)
    k=6
    clustering, centroids = kmeans(microarray_data, k)
    plot_microarray_data(microarray_data, clustering, k)
    clustering = spectral(microarray_data, k, 10, mnn)
    plot_microarray_data(microarray_data, clustering, k, algorithm="spectral")
    clustering = spectral(microarray_data, k, 1.5)
    plot_microarray_data(microarray_data, clustering, k, algorithm="spectral")
    
    #### t-SNE 
    n_samples = 1000
    n_bins = 4  
    centers = [(-5, -5), (0, 0), (5, 5),(-5, 5)]
    X, y = make_blobs(n_samples=n_samples, n_features=10, cluster_std=1.0,
                      centers=centers, shuffle=False, random_state=42)
    tsne_synthetic = TSNE(2).fit_transform(X, y)
    pca_synthetic = PCA(2).fit_transform(X,y)
    plt.figure()
    plt.subplot(211)
    plt.title("tsne on synthetic data")
    plt.scatter(tsne_synthetic[:, 0], tsne_synthetic[:, 1], c=y, cmap="rainbow")
    plt.colorbar()
    
    plt.subplot(212)
    plt.title("pca on synthetic data")
    plt.scatter(pca_synthetic[:, 0], pca_synthetic[:, 1], c=y, cmap="rainbow")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("tsne vs pca on synthetic data")
    
    digits = load_digits()
    tsne_MNIST = TSNE(2).fit_transform(digits.data, digits.target)
    pca_MNIST = PCA(2).fit_transform(digits.data, digits.target)
    plt.figure()
    plt.subplot(211)
    plt.title("tsne on MNIST")
    plt.scatter(tsne_MNIST[:, 0], tsne_MNIST[:, 1], c=digits.target, cmap="rainbow")
    plt.colorbar()
    
    plt.subplot(212)
    plt.title("pca on MNIST")
    plt.scatter(pca_MNIST[:, 0], pca_MNIST[:, 1], c=digits.target, cmap="rainbow")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("tsne vs pca on MNIST data")
    
    
    
    