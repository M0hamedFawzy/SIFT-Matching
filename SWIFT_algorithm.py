import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro
import os


def filter_matches(image1, image2):
    d_lows_contant = 0.85
    s_threshold = 0.002



    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)
    bf = cv2.BFMatcher()
    all_matches = bf.match(des1, des2)



    # Technique 1 Manual crossCheck
    cross_checked_matches = manual_crossCheck(des1, des2, bf)

    #Technique 2 D.lowe's ratio
    lowes_filterd_matches = lows_ratio(des1, des2, bf, d_lows_contant,cross_checked_matches)

    # Technique 3 Ransak
    inlier_matches = _ransak(kp1, kp2, all_matches, lowes_filterd_matches)

    #Technique 4 distance Threshold
    threshold_filterd_matches, distances = distance_threshold(all_matches, inlier_matches)






    #Display the matches keypoints & normal distribution
    _display(image1, kp1, image2, kp2, threshold_filterd_matches)
    _draw_normal_distribution(distances)


   #Calculate Similarity Score
    similarity_score = len(threshold_filterd_matches) / min(len(kp1), len(kp2))

    if similarity_score < s_threshold:
        result = 'Are not Similar'
    else:
        result = 'Are Similar'

    return similarity_score, result

def lows_ratio(des1, des2, bf, d_lows_contant, cross_checked_matches):
    knn_matches = bf.knnMatch(des1, des2, 2)
    matches = []
    for list in cross_checked_matches:
        for match in knn_matches:
            if list.queryIdx == match[0].queryIdx:
                matches.append(match)

    lowes_filterd_matches = []
    for i in range(len(matches)):
        if (matches[i][0].distance < matches[i][1].distance * d_lows_contant):
            lowes_filterd_matches.append(matches[i][0])

    return lowes_filterd_matches

def distance_threshold(all_matches, inlier_matches):
    distances = [match.distance for match in all_matches]
    # _, p_value = shapiro(distances)
    # if p_value > 0.05:
    #     print("The distances are normally distributed, p_value =", p_value)
    # else:
    #     print("The distances are not normally distributed,p_value =", p_value)

    log_distances = np.log1p(distances)
    mu, std = norm.fit(log_distances)
    range_threshold = 2 * std
    lower_bound_dis = np.exp(mu - range_threshold) - 1
    upper_bound_dis = np.exp(mu + range_threshold) - 1

    threshold_filterd_matches = []
    for j in range(len(inlier_matches)):
        if (inlier_matches[j].distance > lower_bound_dis and inlier_matches[j].distance < upper_bound_dis):
            threshold_filterd_matches.append(inlier_matches[j])

    return threshold_filterd_matches, log_distances

def manual_crossCheck(des1, des2, bf):
    one_way_matches = bf.match(des1, des2)
    other_way_matches = bf.match(des2, des1)
    cross_checked_matches = []
    for m1 in range(len(one_way_matches)):
        found = False
        for m2 in range(len(other_way_matches)):
            if ((one_way_matches[m1].queryIdx == other_way_matches[m2].trainIdx) and (one_way_matches[m1].trainIdx == other_way_matches[m2].queryIdx)):
                cross_checked_matches.append(one_way_matches[m1])
                found = True
                break
        if found == True:
            continue

    return cross_checked_matches

def _ransak(kp1, kp2, all_matches, lowes_filterd_matches):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in lowes_filterd_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in lowes_filterd_matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 20)

    inlier_matches = []
    for i in range(len(lowes_filterd_matches)):
        match = lowes_filterd_matches[i]
        is_inlier = mask.flatten()[i].astype(bool)
        if is_inlier:
            inlier_matches.append(match)
    return inlier_matches

def _display(gray1, kp1, gray2, kp2,list_of_matches):
    matches = sorted(list_of_matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(gray1, kp1,
                           gray2, kp2,
                           matches,
                           # matchColor=(0,255,0),
                           flags=2,
                           outImg=None)
    plt.imshow(img3), plt.show()

def _draw_normal_distribution(distances):
    plt.hist(distances, bins=50, density=True, alpha=0.75, label='Histogram')
    mu, std = norm.fit(distances)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label='Fit result (mean = {:.2f}, std = {:.2f})'.format(mu, std))
    plt.legend()
    plt.title('Fit results: Normal Distribution of Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.show()



#------------------Main-------------------------


images =  [
          ['Test Images/image1a.jpeg','Test Images/image1b.jpeg'],
          ['Test Images/image2a.jpeg','Test Images/image2b.jpeg'],
          ['Test Images/image3a.jpeg','Test Images/image3b.jpeg'],
          ['Test Images/image4a.jpeg','Test Images/image4b.jpeg'],
          ['Test Images/image4b.jpeg','Test Images/image4c.png' ],
          ['Test Images/image4a.jpeg','Test Images/image4c.png' ],
          ['Test Images/image5a.jpeg','Test Images/image5b.jpeg'],
          ['Test Images/image6a.jpeg','Test Images/image6b.jpeg'],
          ['Test Images/image7a.jpeg','Test Images/image7b.jpeg'],
          ['Test Images/img1.jpg','Test Images/img2.jpg'],
          ['Test Images/box.png','Test Images/box_in_scene.png']
          ]

for i in range(len(images)):
    image1_name = images[i][0]
    image2_name = images[i][1]
    img1 = cv2.imread(image1_name,0)
    img2 = cv2.imread(image2_name,0)
    score, result = filter_matches(img1, img2)
    print(image1_name,'&',image2_name,'-->',result)
    print('Similarity Score is: ',score)
    print('------------------------------')




