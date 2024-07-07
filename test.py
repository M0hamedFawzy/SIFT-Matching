import numpy as np
import cv2
import matplotlib.pyplot as plt


def Cross_check_Man(query_train, train_query):
    Manual_check=[]
    for inn in query_train:
        for out in train_query:
            if inn.queryIdx == out.trainIdx and out.queryIdx == inn.trainIdx:
                Manual_check.append(inn)
                break
    return Manual_check


def img_comp(img1,img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    match1to2=bf.match(des1,des2)
    match2to1=bf.match(des2,des1)
    CCNlist=Cross_check_Man(match1to2,match2to1)

#David

    matches = bf.knnMatch(des1, des2, k=2)
    Newlist=[]
    for M in CCNlist:
        for N in matches:
            if M.queryIdx == N[0].queryIdx:
                Newlist.append(N)

    new_match = apply_ratio(Newlist)
#RANSAC
    stored_matches = sorted(new_match, key=lambda x: x.distance)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in stored_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in stored_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 20)
    matchesMask = mask.ravel().tolist()
    Final_list = []
    for i, obj in enumerate(matchesMask):
        if obj == 1:
            Final_list.append(stored_matches[i])


    sr = len(Final_list)/min(len(kp1),len(kp2))
    print(sr)
    print('-------------------------------------------------')
def apply_ratio(match,ratio=0.75):
    newlist=[]
    for n,m in match:
        if n.distance < ratio * m.distance:
            newlist.append(n)
    return newlist

imgList1 = ['image1a.jpeg', 'image2a.jpeg', 'image3a.jpeg', 'image4a.jpeg', 'image4a.jpeg', 'image4b.jpeg','image5a.jpeg', 'image6a.jpeg', 'image7a.jpeg']
imgList2 = ['image1b.jpeg', 'image2b.jpeg', 'image3b.jpeg', 'image4b.jpeg', 'image4c.png',  'image4c.png','image5b.jpeg', 'image6b.jpeg', 'image7b.jpeg']

for i in range(len(imgList1)):
    img1 = cv2.imread(imgList1[i],0)
    img2 = cv2.imread(imgList2[i],0)
    img_comp(img1,img2)