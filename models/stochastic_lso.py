# make a function to of random LSO
# function:
#   - index by LSO numbers
#   - load real parameters, normalised to [0, 1]
#   - return a random response

# required input: min max ILD, to normalise the output range
#

# future tasks:
#  - make it parallel for many LSO neurons

# known problems:
# - the stochasticity of the LSO is huge!

# %%
import numpy as np

# the Kerma 22kHz min max ild is +-30
# when scale the head, we only reduce the ILD range.
min_ild=-35
max_ild=35

# mind the human frequency range
a=[4.73, 8.93, 12.25, 3.86, 1.43, 9.26, 0.0, 4.43, 0.0, 1.46, 0.55, 0.34, 1.15, 0.0, 0.0, 0.0, 2.93, 0.0, 0.1, 0.0, 8.55, 0.0, 2.35, 0.54, 0.65, 0.0, 0.95, 0.18, 0.0, 0.0, 0.03, 0.16, 0.18, 0.0, 4.34, 6.27, 0.0, 1.7, 10.03, 1.49, 0.0, 2.62]

cf=[0.34, 0.37, 0.48, 0.57, 0.92, 1.35, 3.17, 4.8, 7.8, 8.3, 8.3, 8.36, 9.0, 9.5, 9.6, 9.87, 9.9, 10.0, 10.7, 11.0, 11.0, 11.03, 11.25, 11.7, 11.7, 11.74, 15.2, 16.0, 16.0, 16.6, 17.15, 18.0, 21.11, 21.94, 24.06, 28.5, 28.95, 29.6, 29.6, 30.0, 31.3, 34.3] # in kHZ

d=[-3.82, -4.94, -2.53, -2.43, -3.22, -3.0, -3.95, -4.12, -5.15, -5.04, -2.01, -4.32, -2.54, -4.59, -10.64, -4.57, -7.11, -10.8, -5.66, -3.43, -3.16, -4.62, -4.24, -5.72, -6.03, -10.82, -3.95, -3.5, -2.94, -4.96, -3.87, -3.82, -3.52, -6.75, -3.08, -2.55, -8.7, -3.07, -3.91, -5.23, -6.13, -3.69]

b=[27.88, 75.26, 91.89, 27.49, 8.95, 32.52, 27.36, 137.15, 42.37, 20.87, 24.75, 34.31, 42.34, 37.83, 102.8, 49.08, 41.55, 29.91, 29.83, 107.98, 26.55, 127.75, 41.83, 51.39, 212.1, 99.18, 87.27, 199.07, 216.35, 156.3, 60.48, 81.07, 50.64, 29.38, 12.28, 29.28, 38.4, 120.28, 66.31, 122.02, 117.52, 90.39]

c=[-7.5, 1.84, -3.3, -13.68, -2.92, 0.24, 18.61, 2.41, -1.11, -3.82, 11.3, 11.99, -8.69, 10.49, -4.66, 0.52, -7.67, -5.93, -5.47, -13.0, -0.87, 17.28, 20.11, -7.93, 0.69, -9.44, -2.49, -5.5, -17.28, -7.94, -11.25, -20.14, -16.87, -7.91, 1.98, 1.35, 0.36, 0.68, 3.93, -7.67, -6.11, -17.83]


#for var
va=[1.76, 4.41, 4.75, 3.47, 3.1, 0.21, 3.57, 1.85, 2.51, 1.8, 2.76, 2.43, 3.24, 2.74, 2.61, 2.24, 3.05, 5.69, 3.45, 2.71, 1.5, 30.27, 2.02, 2.4, 10.16, 3.43, 2.41, 2.87, 2.52, 2.82, 1.72, 2.75, 2.57, 2.59, 5.28, 41.14, 2.96, 3.48, 8.56, 3.64, 27.63, 4.13]

vb=[1.24, 0.81, 0.89, 0.96, 0.71, 1.77, 0.71, 0.99, 0.87, 1.16, 0.9, 0.79, 0.75, 0.91, 0.96, 0.81, 0.97, 0.89, 0.87, 1.04, 1.24, 0.71, 0.83, 0.91, 0.5, 0.56, 1.09, 0.86, 0.77, 0.74, 1.02, 0.84, 0.92, 0.89, 0.65, -0.06, 0.93, 0.73, 0.6, 0.65, 0.15, 0.75]

def miu_lso(ILD, a, b, c, d, e, f):
    # take the data from suppliment material
    # CF = 3.18 kHz 
    # a = 0
    # b = 27.36
    # c = 18.61
    # d = -3.95
    rate = a + b/(1+np.exp((c-ILD)/d))

    # # here we use diff notions: e, f for a, b in the function
    # e = 3.57
    # f = 0.71
    var = e * (rate**f) 

    return rate, var
    # return rate

# ild = 0
# rate, var = miu_lso(ild, a[m], b[m], c[m], d[m], va[m], vb[m])

# max_rate, max_var = miu_lso(max_ild, a[m], b[m], c[m], d[m], va[m], vb[m])
# min_rate, min_var = miu_lso(min_ild, a[m], b[m], c[m], d[m], va[m], vb[m])

# sample_rate = np.random.normal(rate, var)
# normalised_sample_rate = (sample_rate - min_rate) / (max_rate - min_rate)

def sample_normalised_lso(m, ild, valid_mode):
    """
    m is the index of LSO neurons
    """
    rate, var = miu_lso(ild, a[m], b[m], c[m], d[m], va[m], vb[m])
    max_rate, max_var = miu_lso(min_ild, a[m], b[m], c[m], d[m], va[m], vb[m]) # note that min ild gives higher rate
    min_rate, min_var = miu_lso(max_ild, a[m], b[m], c[m], d[m], va[m], vb[m])
    if valid_mode:
        # no stochasticity
        sample_rate = rate
    else:
        sample_rate = np.random.normal(rate, np.sqrt(var))
    # TODO the range is actually not always in [0, 1], but the NN should be able to handle it anyway
    # here we make it in [0, 1] just to make it more rigorous
    normalised_sample_rate = (sample_rate - min_rate) / (max_rate - min_rate)
    if normalised_sample_rate > 1:
        normalised_sample_rate = 1
    if normalised_sample_rate < 0:
        normalised_sample_rate = 0

    return sample_rate, normalised_sample_rate

# %%


def verify_human_range():
    """
    29 16.6
    30 17.15
    31 18.0
    """
    for m in range(32):
        print(m, cf[m])

# %%

def test(m, n, ild):
    for i in range(n):
        print(i, ':', sample_normalised_lso(m, ild))

# test(15, 10, 0)
# %%

def interploate_cf_ild(freqs, level_diffs, frequency_target):
    """
        see https://github.com/DrMarc/slab/blob/master/slab/binaural.py#L234C1-L234C59
        def azimuth_to_ild(azimuth, frequency=2000, ils=None)
    """
    interpolate_ild_f = np.interp(frequency_target, freqs, level_diffs) # do not use -1 to change the sign
    return interpolate_ild_f


def sample_lso_array(freqs, level_diffs, valid_mode=False, idx_list=range(32)):
    """
        for each lso in the human hearing range 
        1. interploate ILD at channel frequency
        2. sample a random normalised LSO response


        Note: "Humans can detect sounds in a frequency range from about 20 Hz to 20 kHz. (Human infants can actually hear frequencies slightly higher than 20 kHz, but lose some high-frequency sensitivity as they mature; the upper limit in average adults is often closer to 15–17 kHz.)
        Most small mammals are sensitive to very high frequencies, but not to low frequencies. For instance, some species of bats are sensitive to tones as high as 200 kHz, but their lower limit is around 20 kHz—the upper limit for young people with normal hearing. One reason for these differences is that small objects, including the auditory structures of these small mammals, are better resonators for high frequencies, whereas large objects are better for low frequencies (which also explains why the violin has a higher pitch than the cello)."
        https://www.ncbi.nlm.nih.gov/books/NBK10924/

        input freqs and level_diffs are from the slab library
        Default 24 channels: 
        array([    0.        ,    52.90941812,   114.63707259,   185.18296341,
         282.18356329,   388.00239952,   529.09418116,   696.64067187,
         899.46010798,  1146.37072585,  1446.19076185,  1816.55668866,
        2266.28674265,  2813.01739652,  3483.20335933,  4303.29934013,
        5299.76004799,  6516.67666467,  8006.95860828,  9814.69706059,
       12028.07438512, 14726.45470906, 18024.47510498, 22045.59088182])
    """

    # max_m=32 # indx of LSO neurons for human range
    #max_m=1 # indx of LSO neurons for human range
    lso_array_sample=np.zeros((len(idx_list), 2))
    for k, m in enumerate(idx_list):
        frequency_target = cf[m]*1000
        ild = interploate_cf_ild(freqs, level_diffs, frequency_target)
        rate, normalised_sample_rate = sample_normalised_lso(m, ild, valid_mode)
        lso_array_sample[k, 0] = rate
        lso_array_sample[k, 1] = normalised_sample_rate
    return lso_array_sample

# for each single LSO make a differentiable network
# or effectively set the output to be a vector of 32 neurons

def fixed_lso_array(freqs, level_diffs):
    """
        output: -1 or 1
            NOTE this is different from the 0/1 output of the LSO neurons
            or the left/right 

        majority vote by taking the mean
        fix the decision boundary of LSO neurons to 0.5, based on the normalised response rate

        if level_diff is right-left
        then output right=0, left=1
        left 1, right -1

        mean acc is about 0.92?
        too accurate aready
    """
    lso_array_sample=sample_lso_array(freqs, level_diffs)
    majority=np.mean(lso_array_sample[:, 1]>0.5)>0.5
    lr_sign=2*int(majority)-1
    return lr_sign