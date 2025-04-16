import numpy as np
import torch

def rotation_x_axis(theta, azim, elev):
    """
    given the input (azim, elev)
    rotate the input by theta around x-axis

    The difference AER and spherical coordinate system:
    the angle of AER is from the azimuth plane, while the angle of spherical coordinate system is from the polar, which is the z-axis, so the sign is different.

    An azimuth-elevation-range (AER) system uses the spherical coordinates (az,elev,range) to represent position relative to a local origin.
    https://en.wikipedia.org/wiki/Horizontal_coordinate_system

    "The azimuth elevation coordinate system is defined similarly to spherical coordinates but is slightly different in that the azimuth and elevation are measured in degrees between the r-axis (i.e z axis) and the projection on the x-z and y-z planes, respectively. Range, or r, is the distance from the origin." --- this is WRONG!

    https://en.wikipedia.org/wiki/Spherical_coordinate_system
    In mathematics, a spherical coordinate system is a coordinate system for three-dimensional space where the position of a given point in space is specified by three real numbers: the radial distance r along the radial line connecting the point to the fixed point of origin; the polar angle θ between the radial line and a polar axis; and the azimuthal angle φ as the angle of rotation of the radial line around the polar axis.[a] (See graphic re the "physics convention".) Once the radius is fixed, the three coordinates (r, θ, φ), known as a 3-tuple, provide a coordinate system on a sphere, typically called the spherical polar coordinates. 
    """

    # convert to radian
    theta = np.deg2rad(theta)
    azim = np.deg2rad(azim)
    elev = np.deg2rad(elev)

    # conver to cartesian
    x = np.cos(azim) * np.cos(elev)
    y = np.sin(azim) * np.cos(elev)
    z = np.sin(elev)

    # rotation matrix around x-axis
    rm = np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])
    
    # rotate the input
    x_rot = np.dot(rm, np.array([x, y, z]))

    # convert back to AER
    azim_rot = np.arctan2(x_rot[1], x_rot[0]) # The result is always in the range of -pi to pi radians.
    elev_rot = np.arcsin(x_rot[2]) # The result is always in the range of -pi/2 to pi/2 radians.

    # convert azimuth to [0, 2pi]
    azim_rot = np.mod(azim_rot, 2*np.pi)

    # convert back to degree
    azim_rot = np.rad2deg(azim_rot)
    elev_rot = np.rad2deg(elev_rot)

    return azim_rot, elev_rot

def degree_difference(azim1, elev1, azim2, elev2):
    """
    calculate the difference between two AER


    known issue:
    
    (Pdb) !(target_azim[11], target_elev[11], output_azim[11], output_elev[11])
    (128.6, -40.0, 129.4813995361328, -40.81143569946289)
    (Pdb) !azim1, elev1, azim2, elev2 = target_azim[11], target_elev[11], output_azim[11], output_elev[11]

    >>> print(dot_product)
    1.0002057554489532
    >>> np.arccos(dot_product)
    /rds/general/user/yc1716/home/project/hinge-net/ginvae/models/androidssl046v5D3_validater.py:1: RuntimeWarning: invalid value encountered in arccos
    from typing import OrderedDict
    nan
    >>>

    >>> print(x1, y1, z1)
    -0.4778 0.599 -0.643
    >>> print(x2, y2, z2)
    -0.48123548327755666 0.5841716088918837 -0.6535716800774684

    >>> print(x2*x2+y2*y2+z2*z2)
    1.0
    >>> print(x1*x1+y1*y1+z1*z1)
    1.0
    >>>
    """
    # convert to radian
    azim1 = np.deg2rad(azim1)
    elev1 = np.deg2rad(elev1)
    azim2 = np.deg2rad(azim2)
    elev2 = np.deg2rad(elev2)

    # conver to cartesian
    x1 = np.cos(azim1) * np.cos(elev1)
    y1 = np.sin(azim1) * np.cos(elev1)
    z1 = np.sin(elev1)

    x2 = np.cos(azim2) * np.cos(elev2)
    y2 = np.sin(azim2) * np.cos(elev2)
    z2 = np.sin(elev2)

    # calculate the angle between two vectors
    dot_product = x1*x2 + y1*y2 + z1*z2
    # the result is in the range of -pi to pi
    # TODO better way to do conversion?
    dot_product = np.clip(dot_product, -1, 1) # avoid nan, but not sure if it is the right way
    angle = np.arccos(dot_product)

    # convert back to degree
    angle = np.rad2deg(angle)

    return angle

def convert_pred_to_angle(y_pred):
    """
    convert the predicted singmoid output to angle
    used both by the trainer and the evaluator

    also change from origin point to the peak point, NO!
    we only shift the azimuth,
    this is a big shift
    """
    with torch.no_grad():
        # # calculate the predicted angle by converting the output sin and cos to angle in degree
        # # y_pred_angle = torch.atan2(y_pred[1], y_pred[0])*180/np.pi
        # # clip the cos and sin to [-1, 1]
        # cos_a=torch.clamp(y_pred[1], -1, 1)
        # y_a = torch.acos(cos_a) # [0, pi], -1->pi, 1->0
        # # convert to [0, 2pi]
        # y_a = y_a*180/np.pi
        # # change the range to [0, 360], given the sin value
        # # if y_pred[0]>0.5:
        # if y_pred[0]>=0:
        #     y_a = y_a
        # else:
        #     y_a = 360-y_a

        y_a = torch.atan2(y_pred[0], y_pred[1])*180/np.pi
        # convert from -180 to 180 to 0 to 360
        y_a = y_a if y_a >= 0 else y_a + 360

        # linear coding
        # elev_pred in [-90, 90]
        elev_pred = y_pred[2]*90 # y_pred scaled to [-2, 2], sigmoid output should be around [0.25, 0.75]

        # if elev_pred>90:
        #     elev_pred=90
        # if elev_pred<-90:
        #     elev_pred=-90
    
    # return y_a, elev_pred
    # center of peak is 
    # peak_azim = 115
    # peak_elev = -10
    # in the init state, the output is 0, 0
    # return y_a+115, elev_pred-10 --- NO
    # we only need to shift the azimuth ---- NO
    # the peak has to be able to cover the whole sound range
        a2 = y_a+115
        # e2 = elev_pred-10
        # convert to [0, 360] and [-90, 90]
        a2 = np.mod(a2, 360)

        e2=elev_pred-10
        # clip the elevation to [-90, 90]
        e2 = np.clip(e2, -90, 90)
        return a2, e2
# test


azim_rot, elev_rot = rotation_x_axis(90, 90, 0)
# compare with the expected result
#assert np.isclose(azim_rot, 0)
print(azim_rot, elev_rot)
# x_rot = array([6.123234e-17, 6.123234e-17, 1.000000e+00])
# x2=0, x1=0, x3=1
# so the azimuth is not meaningful
# in this case, how do we measure the direction difference?
# covnert to the cartesian coordinate system, and then measure the angle between the two vectors
# TODO better comparison

# use function compare the angle difference
angle_diff = degree_difference(0, 90, azim_rot, elev_rot)
assert np.isclose(angle_diff, 0)

# assert np.isclose(azim_rot, -90) # note the case where the azimuth is not meaningful
# assert np.isclose(elev_rot, 90)


azim_rot, elev_rot = rotation_x_axis(90, 0, 0)
# compare with the expected result
assert np.isclose(azim_rot, 0)
assert np.isclose(elev_rot, 0)

azim_rot, elev_rot = rotation_x_axis(90, 0, 90)
# compare with the expected result
# TODO: better print for test
assert np.isclose(azim_rot, 360-90)
assert np.isclose(elev_rot, 0)



azim_rot, elev_rot = rotation_x_axis(90, 90, 90)
# compare with the expected result
assert np.isclose(azim_rot, 360-90)
assert np.isclose(elev_rot, 0)

# azim_rot, elev_rot = rotation_x_axis(90, 45, 45)
# # compare with the expected result
# print(azim_rot, elev_rot)
# assert np.isclose(azim_rot, 360-45)
# assert np.isclose(elev_rot, 45)

# azim_rot, elev_rot = rotation_x_axis(90, 45, 90)
# # compare with the expected result
# assert np.isclose(azim_rot, 90)
# assert np.isclose(elev_rot, 0)

# test with cycle 360
azim_rot, elev_rot = rotation_x_axis(360, 45, 45)
# compare with the expected result
assert np.isclose(azim_rot, 45)
assert np.isclose(elev_rot, 45)

# # test with different sign, output range is [-180, 180] or [0, 360]?
# azim_rot, elev_rot = rotation_x_axis(90, -45, -45)
# # compare with the expected result
# assert np.isclose(azim_rot, -45)
# assert np.isclose(elev_rot, 45)
