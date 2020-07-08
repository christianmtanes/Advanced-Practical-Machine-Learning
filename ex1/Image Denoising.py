import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
import pickle
from skimage.util import view_as_windows as viewW
import time

def images_example(path='train_images.pickle'):
    """
    A function demonstrating how to access to image data supplied in this exercise.
    :param path: The path to the pickle file.
    """
    patch_size = (8, 8)

    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)

    patches = sample_patches(train_pictures, psize=patch_size, n=20000)

    plt.figure()
    plt.imshow(train_pictures[0])
    plt.title("Picture Example")

    plt.figure()
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(patches[:,i].reshape(patch_size), cmap='gray')
        plt.title("Patch Example")
    plt.show()


def im2col(A, window, stepsize=1):
    """
    an im2col function, transferring an image to patches of size window (length
    2 list). the step size is the stride of the sliding window.
    :param A: The original image (NxM size matrix of pixel values).
    :param window: Length 2 list of 2D window size.
    :param stepsize: The step size for choosing patches (default is 1).
    :return: A (heightXwidth)x(NxM) matrix of image patches.
    """
    return viewW(np.ascontiguousarray(A), (window[0], window[1])).reshape(-1,
                        window[0] * window[1]).T[:, ::stepsize]


def grayscale_and_standardize(images, remove_mean=True):
    """
    The function receives a list of RGB images and returns the images after
    grayscale, centering (mean 0) and scaling (between -0.5 and 0.5).
    :param images: A list of images before standardisation.
    :param remove_mean: Whether or not to remove the mean (default is True).
    :return: A list of images after standardisation.
    """
    standard_images = []

    for image in images:
        standard_images.append((0.299 * image[:, :, 0] +
                                0.587 * image[:, :, 1] +
                                0.114 * image[:, :, 2]) / 255)

    sum = 0
    pixels = 0
    for image in standard_images:
        sum += np.sum(image)
        pixels += image.shape[0] * image.shape[1]
    dataset_mean_pixel = float(sum) / pixels

    if remove_mean:
        for image in standard_images:
            image -= np.matlib.repmat([dataset_mean_pixel], image.shape[0],
                                      image.shape[1])

    return standard_images


def sample_patches(images, psize=(8, 8), n=10000, remove_mean=True):
    """
    sample N p-sized patches from images after standardising them.

    :param images: a list of pictures (not standardised).
    :param psize: a tuple containing the size of the patches (default is 8x8).
    :param n: number of patches (default is 10000).
    :param remove_mean: whether the mean should be removed (default is True).
    :return: A matrix of n patches from the given images.
    """
    d = psize[0] * psize[1]
    patches = np.zeros((d, n))
    standardized = grayscale_and_standardize(images, remove_mean)

    shapes = []
    for pic in standardized:
        shapes.append(pic.shape)

    rand_pic_num = np.random.randint(0, len(standardized), n)
    rand_x = np.random.rand(n)
    rand_y = np.random.rand(n)

    for i in range(n):
        pic_id = rand_pic_num[i]
        pic_shape = shapes[pic_id]
        x = int(np.ceil(rand_x[i] * (pic_shape[0] - psize[1])))
        y = int(np.ceil(rand_y[i] * (pic_shape[1] - psize[0])))
        patches[:, i] = np.reshape(np.ascontiguousarray(
            standardized[pic_id][x:x + psize[0], y:y + psize[1]]), d)

    return patches


def denoise_image(Y, model, denoise_function, noise_std, patch_size=(8, 8)):
    """
    A function for denoising an image. The function accepts a noisy gray scale
    image, denoises the different patches of it and then reconstructs the image.

    :param Y: the noisy image.
    :param model: a Model object (MVN/ICA/GSM).
    :param denoise_function: a pointer to one of the denoising functions (that corresponds to the model).
    :param noise_std: the noise standard deviation parameter.
    :param patch_size: the size of the patch that the model was trained on (default is 8x8).
    :return: the denoised image, after each patch was denoised. Note, the denoised image is a bit
    smaller than the original one, since we lose the edges when we look at all of the patches
    (this happens during the im2col function).
    :return: denoise_time, the time took for denoising one image
    """
    (h, w) = np.shape(Y)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))

    # split the image into columns and denoise the columns:
    noisy_patches = im2col(Y, patch_size)
    start = time.time()
    denoised_patches = denoise_function(noisy_patches, model, noise_std)
    end = time.time()
    denoise_time = end - start
    # reshape the denoised columns into a picture:
    x_hat = np.reshape(denoised_patches[middle_linear_index, :],
                       [cropped_h, cropped_w])

    return x_hat , denoise_time 


def crop_image(X, patch_size=(8, 8)):
    """
    crop the original image to fit the size of the denoised image.
    :param X: The original picture.
    :param patch_size: The patch size used in the model, to know how much we need to crop.
    :return: The cropped image.
    """
    (h, w) = np.shape(X)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))
    columns = im2col(X, patch_size)
    return np.reshape(columns[middle_linear_index, :], [cropped_h, cropped_w])


def normalize_log_likelihoods(X):
    """
    Given a matrix in log space, return the matrix with normalized columns in
    log space.
    :param X: Matrix in log space to be normalised.
    :return: The matrix after normalization.
    """
    h, w = np.shape(X)
    return X - np.matlib.repmat(logsumexp(X, axis=0), h, 1)

def test_denoising_with_plots(image, model, denoise_function,
                   noise_range=(0.01, 0.05, 0.1, 0.2), patch_size=(8, 8), k =0, iters =0, model_name = "mvn"):
    """
    it plots the the images vs denoised images, used to show example of 
    denoising on image.
    :param image: An image matrix.
    :param model: A trained model (MVN/ICA/GSM).
    :param denoise_function: The denoise function that corresponds to your model.
    :param noise_range: A tuple containing different noise parameters you wish
            to test your code on. default is (0.01, 0.05, 0.1, 0.2).

    :param patch_size: The size of the patches you've used in your model.
            Default is (8, 8).
    :param k number of gaussians
    :param iters number of iterations
    :param model_name name of model
    """
    h, w = np.shape(image)
    noisy_images = np.zeros((h, w, len(noise_range)))
    denoised_images = []
    cropped_original = crop_image(image, patch_size)

    # make the image noisy:
    for i in range(len(noise_range)):
        noisy_images[:, :, i] = image + (
        noise_range[i] * np.random.randn(h, w))

    # denoise the image:
    for i in range(len(noise_range)):
        denoised_image  = denoise_image(noisy_images[:, :, i], model, denoise_function,
                          noise_range[i], patch_size)[0]
        denoised_images.append(denoised_image)
    plt.figure()
    if k == 0:
        plt.title("example of image denoising for mvn")
    else:
        plt.title("example of image denoising for " + model_name + "with k = " + str(k) + " and iteration = " + str(iters))
    for i in range(len(noise_range)):
        plt.subplot(2, len(noise_range), i + 1)
        plt.imshow(noisy_images[:, :, i], cmap='gray')
        plt.subplot(2, len(noise_range), i + 1 + len(noise_range))
        plt.imshow(denoised_images[i], cmap='gray')
    if k == 0:
        plt.savefig("example of image denoising for mvn")
    else:
        plt.savefig("example of image denoising for " + model_name + "with k = " + str(k) + " and iteration = " + str(iters))
    
def test_denoising(image, model, denoise_function,
                   noise_range=(0.01, 0.05, 0.1, 0.2), patch_size=(8, 8)):
    """
    A simple function for testing your denoising code. You can and should
    implement additional tests for your code.
    :param image: An image matrix.
    :param model: A trained model (MVN/ICA/GSM).
    :param denoise_function: The denoise function that corresponds to your model.
    :param noise_range: A tuple containing different noise parameters you wish
            to test your code on. default is (0.01, 0.05, 0.1, 0.2).
    :param patch_size: The size of the patches you've used in your model.
            Default is (8, 8).
    :return:sum_denoise_time, overall denoise time on all noise range 
    :return: mse_denoised_array the mse of the denoised image for each noise    
    """
    h, w = np.shape(image)
    noisy_images = np.zeros((h, w, len(noise_range)))
    denoised_images = []
    cropped_original = crop_image(image, patch_size)

    # make the image noisy:
    for i in range(len(noise_range)):
        noisy_images[:, :, i] = image + (
        noise_range[i] * np.random.randn(h, w))

    # denoise the image:
    sum_denoise_time = 0
    for i in range(len(noise_range)):
        denoised_image, denoise_time  = denoise_image(noisy_images[:, :, i], model, denoise_function,
                          noise_range[i], patch_size) 
        denoised_images.append(denoised_image)
        sum_denoise_time += denoise_time
    mse_denoised_array = np.zeros(len(noise_range))
    # calculate the MSE for each noise range:
    for i in range(len(noise_range)):
#        print("noisy MSE for noise = " + str(noise_range[i]) + ":")
#        print(np.mean((crop_image(noisy_images[:, :, i],
#                                  patch_size) - cropped_original) ** 2))
#        print("denoised MSE for noise = " + str(noise_range[i]) + ":")
#        print(np.mean((cropped_original - denoised_images[i]) ** 2))
        mse_denoised_array[i] = np.mean((cropped_original - denoised_images[i]) ** 2)
    return sum_denoise_time, mse_denoised_array

class MVN_Model:
    """
    A class that represents a Multivariate Gaussian Model, with all the parameters
    needed to specify the model.

    mean - a D sized vector with the mean of the gaussian.
    cov - a D-by-D matrix with the covariance matrix.
    """
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov


class GSM_Model:
    """
    A class that represents a GSM Model, with all the parameters needed to specify
    the model.

    cov - a k-by-D-by-D tensor with the k different covariance matrices. the
        covariance matrices should be scaled versions of each other.
    mix - k-length probability vector for the mixture of the gaussians.
    """
    def __init__(self, cov, mix):
        self.cov = cov
        self.mix = mix


class ICA_Model:
    """
    A class that represents an ICA Model, with all the parameters needed to specify
    the model.

    P - linear transformation of the sources. (X = P*S)
    vars - DxK matrix whose (d,k) element corresponds to the variance of the k'th
        gaussian in the d'th source.
    mix - DxK matrix whose (d,k) element corresponds to the weight of the k'th
        gaussian in d'th source.
    """
    def __init__(self, P, vars, mix):
        self.P = P
        self.vars = vars
        self.mix = mix


def MVN_log_likelihood(X, model):
    """
    Given image patches and a MVN model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A MVN_Model object.
    :return: The log likelihood of all the patches combined.
    """
    # we insert X.T because because we need that last axis of `x` to denote the components
    return np.sum(multivariate_normal.logpdf(X.T, model.mean, model.cov, allow_singular=True))


def GSM_log_likelihood(X, model):
    """
    Given image patches and a GSM model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A GSM_Model object.
    :return: The log likelihood of all the patches combined.
    """
    K = model.mix.shape[0]
    N = X.shape[1]
    logsumexp_params = np.zeros((K,N))
    for y in range(K):
        logsumexp_params[y] = np.log(model.mix[y]) + multivariate_normal.logpdf(X.T,cov = model.cov[y], allow_singular=True)
    return np.sum(logsumexp(logsumexp_params,axis=0))

def ICA_log_likelihood(X, model):
    """
    Given image patches and an ICA model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: An ICA_Model object.
    :return: The log likelihood of all the patches combined.
    """
#    P_inv = np.linalg.pinv(model.P)
    S = np.dot(model.P.T, X)
    log_likelihood = 0
    D, N = X.shape
    for i in range(D):
        gsm_model = GSM_Model(cov = model.vars[i], mix = model.mix[i])
        log_likelihood += GSM_log_likelihood(S[i].reshape((1,N)), gsm_model)
    return log_likelihood

def learn_MVN(X):
    """
    Learn a multivariate normal model, given a matrix of image patches.
    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :return: A trained MVN_Model object.
    """
    #using section 1.6 of the Maximum Likelihood Estimation recitation
    mu = np.mean(X, axis = 1)
    sigma = np.cov(X)
    return MVN_Model(mu, sigma)


def calculate_log_matrix(X, k, sigmas,p_y):
    N = X.shape[1]
    log_matrix = np.zeros((k,N))
    for y in range(k):
        log_matrix[y] = np.log(p_y[y]) + multivariate_normal.logpdf(X.T, cov = sigmas[y], allow_singular=True)
    return log_matrix

def calculate_c_matrix(X, k, sigmas,p_y):
    log_matrix = calculate_log_matrix(X, k, sigmas,p_y)
    return np.exp(normalize_log_likelihoods(log_matrix))

def learn_GSM(X, k, iter_num = 50):
    """
    Learn parameters for a Gaussian Scaling Mixture model for X using EM.

    GSM components share the variance, up to a scaling factor, so we only
    need to learn scaling factors and mixture proportions.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components of the GSM model.
    :param iter_num: number of iteration to run the em 
    :return: A list of trained GSM_Model objects, a model for each iteration
    the last element would be the best model .
    """
    #initialization
    d, N = X.shape
    sigma = np.cov(X)
     
    sigma_pinv = np.linalg.pinv(sigma)
#    prev_siga = np.array(k*[0])
    r_y_power2 = ( np.random.rand(k)) ** 2 
    sigmas = np.tensordot(r_y_power2, sigma,0)
#    prev_ry = np.array(k*[0])
    p_y = np.array(k*[1/k])
#    pi_y_prev = np.array(k*[0])
    gsm_models = []
    for it in range(iter_num):
        # E-step
        
        c_matrix = calculate_c_matrix(X, k, sigmas,p_y) # c is kxN
        #M-step
        sum_c_matrix_over_axis1 = np.sum(c_matrix, axis = 1)
        p_y = sum_c_matrix_over_axis1 / N
        r_y_power2_denominator = sum_c_matrix_over_axis1 * d
        X_T_dot_sigma_pinv = np.dot(X.T, sigma_pinv)
        r_y_power2_numerator = np.dot(c_matrix, (X_T_dot_sigma_pinv * X.T).sum(-1))
        r_y_power2 = r_y_power2_numerator / r_y_power2_denominator
        sigmas = np.tensordot(r_y_power2, sigma,0)
        gsm_models.append(GSM_Model(sigmas, p_y))
    return gsm_models
        

def learn_ICA(X, k, iter_num=10):
    """
    Learn parameters for a complete invertible ICA model.

    We learn a matrix P such that X = P*S, where S are D independent sources
    And for each of the D coordinates we learn a mixture of K univariate
    0-mean gaussians using EM.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components in the source gaussian mixtures.
    :param iter_num: number of iteration to run the em
    :return: A trained ICA_Model object.
    :return: all_mix, all mix values for each iteartion. shape = (iter_num, d,k)
    :return: all_vars, all vars values for each iteration.shape = (iter_num, d,k)
    """
    sigma = np.cov(X)
    P =  np.linalg.eig(sigma)[1]
    S = np.dot(P.T, X)
    d, N = X.shape
    vars_y = np.zeros((d,k))
    p_y = np.ones((d,k)) * (1/k)
    all_mix = np.zeros((iter_num, d,k))
    all_vars = np.zeros((iter_num, d,k))
    for i in range(S.shape[0]):
        choose_k = np.random.choice(S.shape[0],k,replace=False)
        vars_y[i] = np.var(S[choose_k], axis=1)
        
        for it in range(iter_num):
            c_matrix = calculate_c_matrix(S[i].reshape((1,N)), k, vars_y[i], p_y[i])# c is kxN
            sum_c_matrix_over_axis1 = np.sum(c_matrix, axis = 1)
            p_y[i] = sum_c_matrix_over_axis1 / N
            vars_y[i] = np.dot(c_matrix, (S[i].reshape((N,1)) * S[i].reshape((N,1))).sum(-1)) / sum_c_matrix_over_axis1
            all_mix[it,i] = p_y[i]
            all_vars[it,i] = vars_y[i]
            
    return ICA_Model(P, vars_y, p_y), all_mix, all_vars

def Weiner(Y, mean, cov, noise_std):
     # simply implementation of weiner filter
    cov_inv = np.linalg.pinv(cov)
    con_inv_dot_mean = np.dot(cov_inv, mean).reshape(Y.shape[0],1)
    inv_noise_std_power2 = 1/noise_std**2
    left_side = np.linalg.pinv(cov_inv + np.eye(cov_inv.shape[0]) * inv_noise_std_power2)
#    X = np.zeros(Y.shape)
#    for i in range(Y.shape[1]):
#        X[:,i] = np.dot(left_side, con_inv_dot_mean + inv_noise_std_power2 * Y[:,i])
    X = np.dot(left_side, con_inv_dot_mean + inv_noise_std_power2 * Y)
    return X

def MVN_Denoise(Y, mvn_model, noise_std):
    """
    Denoise every column in Y, assuming an MVN model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a single
    0-mean multi-variate normal distribution.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param mvn_model: The MVN_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """
    #  apply weiner filter
    return Weiner(Y, mvn_model.mean, mvn_model.cov, noise_std)


def GSM_Denoise(Y, gsm_model, noise_std):
    """
    Denoise every column in Y, assuming a GSM model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a mixture of
    0-mean gaussian components sharing the same covariance up to a scaling factor.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param gsm_model: The GSM_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.

    """
    k = gsm_model.mix.shape[0]
    X_star = np.zeros(Y.shape)
    c_matrix = calculate_c_matrix(Y, k, gsm_model.cov + (noise_std ** 2) * np.eye(Y.shape[0]) ,gsm_model.mix) #c is kxN
    gsm_mean = np.zeros(gsm_model.cov.shape[1])
#    print(gsm_mean)
    for i in range(k):
        X_star += c_matrix[i] * Weiner(Y, gsm_mean, gsm_model.cov[i], noise_std)
    return X_star


def ICA_Denoise(Y, ica_model, noise_std):
    """
    Denoise every column in Y, assuming an ICA model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by an ICA 0-mean
    mixture model.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param ica_model: The ICA_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """
    d,N = Y.shape
    S = np.dot(ica_model.P.T, Y)
    S_denoised = np.zeros(S.shape)
    k = ica_model.mix.shape[1]
    for i in range(S.shape[0]):
        gsm_model = GSM_Model(cov = ica_model.vars[i].reshape((k,1,1)), mix = ica_model.mix[i])
        S_denoised[i] = GSM_Denoise(S[i].reshape((1,N)), gsm_model, noise_std)
    return np.dot(ica_model.P,S_denoised)

if __name__ == '__main__':

    patch_size = (8, 8)
    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)
    
    train_patches = sample_patches(train_pictures, psize=patch_size, n=10000)
    
    with open('test_images.pickle', 'rb') as f:
        test_pictures = pickle.load(f)
    
    test_patches = sample_patches(test_pictures, psize=patch_size, n=10000)
    
    # each model is tried with different k's
    start = time.time()
    model_mvn= learn_MVN(train_patches)
    end = time.time()
    print("learn time for mvn = ", end-start)
    print("log likelihood on train data for mvn = ", MVN_log_likelihood(train_patches, model_mvn))
    print("log likelihood on test data for mvn = ", MVN_log_likelihood(test_patches, model_mvn))
    images = grayscale_and_standardize(test_pictures)
    noise_range=(0.01, 0.05, 0.1, 0.2)
    # mvn denoise
    denoise_function = MVN_Denoise
    sum_time_mvn = 0
    mse_denoised_array_avg_mvn = np.zeros(len(noise_range))
    for image in images:

        sum_denoise_time, mse_denoised_array = test_denoising(image, model_mvn, denoise_function,
               noise_range=noise_range, patch_size=(8, 8))
        sum_time_mvn += sum_denoise_time
        mse_denoised_array_avg_mvn += mse_denoised_array
    mse_denoised_array_avg_mvn /= len(images)
    print("overall test time for MVN on 11 test images with four different noises on each image = " , sum_time_mvn)
    print("example of image denoising for mvn")
    test_denoising_with_plots(images[0], model_mvn, denoise_function,
               noise_range=noise_range, patch_size=(8, 8), k = 0, iters=0, model_name="mvn")
    
    print("average mse of denoised test data for MVN on 11 test images with four different noises on each image = " , mse_denoised_array_avg_mvn)

    
    k_s = [2,5,10] #The number of components
    iter_num = 100
    model_gsm_learn_time = []
    model_gsm_test_time = []
    model_ica_learn_time = []
    model_ica_test_time = []
    test_gsm_loglikelihood = []
    test_ica_loglikelihood = []
    for k in k_s:
        model_gsm_loglikelihood = []
        start = time.time()
        gsm_models = learn_GSM(train_patches, k, iter_num)
        end = time.time()
        model_gsm_learn_time.append(end - start)
        
        model_ica_loglikelihood = []
        start = time.time()
        ica_model, allmix, allvars = learn_ICA(train_patches, k, iter_num)
        end = time.time()
        model_ica_learn_time.append(end - start)
        
        for it in range(iter_num):
            model_gsm_loglikelihood.append( GSM_log_likelihood(train_patches, gsm_models[it]))
            ica_model_inside = ICA_Model(ica_model.P, allvars[it], allmix[it])
            model_ica_loglikelihood.append( ICA_log_likelihood(train_patches, ica_model_inside))
        #gsm
        plt.figure()
        plt.title(" iterations vs  log likelihood for gsm with k = " + str(k))
        plt.plot(range(iter_num), model_gsm_loglikelihood)
        plt.savefig(" iterations vs  log likelihood for gsm with k = " + str(k))
        #ica
        plt.figure()
        plt.title(" iterations vs  log likelihood for ica k = " + str(k))
        plt.plot(range(iter_num), model_ica_loglikelihood)
        plt.savefig(" iterations vs  log likelihood for ica with k = " + str(k))
        
        test_gsm_loglikelihood.append(GSM_log_likelihood(test_patches, gsm_models[-1]))
        test_ica_loglikelihood.append(ICA_log_likelihood(test_patches, ica_model))
        ##TESTING gsm
        denoise_function = GSM_Denoise
        sum_time_gsm = 0
        mse_denoised_array_avg_gsm = np.zeros(len(noise_range))
        for image in images:
            
            sum_denoise_time, mse_denoised_array = test_denoising(image, gsm_models[-1], denoise_function,
                           noise_range=noise_range, patch_size=(8, 8))
            sum_time_gsm += sum_denoise_time
            mse_denoised_array_avg_gsm += mse_denoised_array
        mse_denoised_array_avg_gsm /= len(images)
        model_gsm_test_time.append(sum_time_gsm)
        
        test_denoising_with_plots(images[0], gsm_models[-1], denoise_function,
               noise_range=noise_range, patch_size=(8, 8),k=k, iters=iter_num, model_name="gsm")
        
    
        print("GSM test data for k = ", str(k) + " learned with iteration = ", str(iter_num))
        print("average mse of denoised test data for GSM on 11 test images with four different noises on each image = " , mse_denoised_array_avg_gsm)
        ##TESTING ica
        
        
        denoise_function = ICA_Denoise
        sum_time_ica = 0
        mse_denoised_array_avg_ica = np.zeros(len(noise_range))
        for image in images:
            
            sum_denoise_time, mse_denoised_array= test_denoising(image, ica_model, denoise_function,
                           noise_range=noise_range, patch_size=(8, 8))
            sum_time_ica += sum_denoise_time
            mse_denoised_array_avg_ica += mse_denoised_array
        mse_denoised_array_avg_ica /= len(images)
        model_ica_test_time.append(sum_time_ica)
        print("ICA test data for k = ", str(k) + " learned with iteration = ", str(iter_num))
        print("average mse of denoised test data for ICA on 11 test images with four different noises on each image = " , mse_denoised_array_avg_ica)

        test_denoising_with_plots(images[0], ica_model, denoise_function,
               noise_range=noise_range, patch_size=(8, 8),k=k, iters=iter_num, model_name="ica")
        
        # noise  vs average mse on 11 images for four different noises
        plt.figure()
        ax = plt.subplot(111)
        ind = np.arange(len(noise_range))
        plt.title("noise vs average mse on 11 test images on the 3 models with k = "+  str(k))
        rects1 = ax.bar(ind-0.2, mse_denoised_array_avg_mvn,width=0.2,color='b',align='center')
        rects2 = ax.bar(ind, mse_denoised_array_avg_gsm,width=0.2,color='g',align='center')
        rects3 = ax.bar(ind+0.2, mse_denoised_array_avg_ica,width=0.2,color='r',align='center')
        ax.set_xticks(ind + 0.2 / 3)
        ax.set_xticklabels(noise_range)
        ax.legend((rects1[0], rects2[0], rects3[0]), ('MVN', 'GSM', 'ICA'))
        plt.savefig("noise vs average mse on 11 test images on the 3 models with k = "+  str(k))


    #k vs learn time 
    print("k vs learn time in seconds for GSM = ", model_gsm_learn_time)
    print("k vs learn time in seconds for ICA = ", model_ica_learn_time)
    plt.figure()
    ax = plt.subplot(111)
    ind = np.arange(len(k_s))
    plt.title(" k vs learn time in seconds for GSM and ICA with iteration = " + str(iter_num))
    rects1 = ax.bar(ind-0.2, model_gsm_learn_time,width=0.2,color='b',align='center')
    rects2 = ax.bar(ind, model_ica_learn_time,width=0.2,color='r',align='center')
    ax.set_xticks(ind + 0.2 / 2)
    ax.set_xticklabels(k_s)
    ax.legend((rects1[0], rects2[0]), ('GSM', 'ICA'))
    plt.savefig(" k vs learn time in seconds for GSM and ICA with iteration = " + str(iter_num))
    #k vs log likelihood on test data 
    print("k vs  log likelihood on test data for GSM = ", test_gsm_loglikelihood)
    print("k vs  log likelihood on test data for ICA = ", test_ica_loglikelihood)
    plt.figure()
    ax = plt.subplot(111)
    ind = np.arange(len(k_s))
    plt.title(" k vs  log likelihood on test data for GSM and ICA with iteration = " + str(iter_num))
    rects1 = ax.bar(ind-0.2, test_gsm_loglikelihood,width=0.2,color='b',align='center')
    rects2 = ax.bar(ind, test_ica_loglikelihood,width=0.2,color='r',align='center')
    ax.set_xticks(ind + 0.2 / 2)
    ax.set_xticklabels(k_s)
    ax.legend((rects1[0], rects2[0]), ('GSM', 'ICA'))
    plt.savefig(" k vs  log likelihood on test data for GSM and ICA with iteration = " + str(iter_num))
    #k vs test time
    print("k vs  test time in seconds for GSM = ",model_gsm_test_time)
    print("k vs  test time in seconds for ICA = ",model_ica_test_time)
    plt.figure()    
    ax = plt.subplot(111)
    ind = np.arange(len(k_s))
    plt.title("k vs  test time in seconds for GSM and ICA  with iteration = " + str(iter_num))
    rects1 = ax.bar(ind-0.2, model_gsm_test_time,width=0.2,color='b',align='center')
    rects2 = ax.bar(ind, model_ica_test_time,width=0.2,color='r',align='center')
    ax.set_xticks(ind + 0.2 / 2)
    ax.set_xticklabels(k_s)
    ax.legend((rects1[0], rects2[0]), ('GSM', 'ICA'))
    plt.savefig("k vs  test time in seconds for GSM and ICA  with iteration = " + str(iter_num))
    