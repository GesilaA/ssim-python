import numpy as np
import cv2

def calc_ssim(base_img, cur_img, win_size=11, k1=0.01, k2=0.03, L=255, split=False):
    # biased estimation 
    def __get_filter_kernel(shape=(3,3), f_type='gaussion', sigma=1.5):
        kernel = None
        if f_type == 'gaussion':
            r, c = shape
            r_k = cv2.getGaussianKernel(r, sigma)
            c_k = cv2.getGaussianKernel(c, sigma)
            kernel = np.multiply(r_k, c_k.T)
        elif f_type == 'uniform':
            pass
        return kernel

    def __conv2d(src, kernel, type='valid'):
        k_size = kernel.shape[0]
        pad_size = int(k_size/2)
        out = cv2.filter2D(src, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        if type == 'valid':
            out = out[pad_size:-pad_size, pad_size:-pad_size]
        return out
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    base_img = base_img.astype(np.float64)
    cur_img = cur_img.astype(np.float64)
    kernel = __get_filter_kernel(shape=(win_size, win_size), f_type='gaussion')
    base_mu = __conv2d(base_img, kernel)    # mu1
    cur_mu = __conv2d(cur_img, kernel)      # mu2
    base_mu_sq = base_mu ** 2               # mu1^2
    cur_mu_sq = cur_mu ** 2                 # mu2^2
    base_mu_X_cur_mu = base_mu * cur_mu     # mu1*mu2

    base_sigma_sq = __conv2d(base_img**2, kernel) - base_mu_sq  # sigma1^2
    base_sigma_sq[base_sigma_sq < 0.] = 0.
    cur_sigma_sq = __conv2d(cur_img**2, kernel) - cur_mu_sq     # sigma2^2
    cur_sigma_sq[cur_sigma_sq < 0.] = 0.

    between_sigma = __conv2d(base_img * cur_img, kernel) - base_mu_X_cur_mu  # sigma12

    # BUG: L * C * S != np.mean(ssim_map)
    if not split:
        ssim_map = ((2 * base_mu_X_cur_mu + C1) * (2 * between_sigma + C2)) \
                   / ((base_mu_sq + cur_mu_sq + C1) * (base_sigma_sq + cur_sigma_sq + C2))
        return np.mean(ssim_map)
    else:
        base_sigma = base_sigma_sq ** 0.5
        cur_sigma = cur_sigma_sq ** 0.5
        L = (2 * base_mu_X_cur_mu + C1) / (base_mu_sq + cur_mu_sq + C1)
        C = (2 * between_sigma + C2) / (base_sigma_sq + cur_sigma_sq + C2)
        S = (between_sigma + C3) / (base_sigma * cur_sigma + C3)
        return np.mean(L), np.mean(C), np.mean(S)

if __name__ == '__main__':
    import time
    
    img1 = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)
    s_t = time.time()
    print(calc_ssim(img1, img2))
    print('time: ', time.time() - s_t)