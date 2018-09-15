import cv2
import numpy as np
from scipy.spatial import distance as dist
import imutils
from imutils import contours


# function for computing midpoint
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0])*.5, (ptA[1] + ptB[1])*.5)


# pts is an np array
def orderPoints(pts):
    # sorts based on x-coordinates
    # argsort sorts the array
    # pts[:,0] implies all rows with column 0
    # NOTE: argsort() returns indices
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # gets left and right most pts (using numpy indices)
    leftMost = xSorted[:2, :] # 2 not included -> (0,1)
    rightMost = xSorted[2:, :] # 2 included -> (2,3)

    # get bottom left and top left by y values
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]  # [:,1} access y-values of all rows
    # tl = top left
    # bl = bottom left
    (tl, bl) = leftMost

    # using top left to find right most b/c hypotenuse will be bottom right
    # D contains the two distances #newaxis adds extra dimension
    # the [0] makes sure 1D array (i.e. no [[]])
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    # ::-1 goes backwards
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # returns clockwise coordinates (starting with top left)
    return np.array([tl, tr, br, bl], dtype="float32")

# this gets side image


cap = cv2.VideoCapture(2)
ret, sideimg = cap.read()

grayimg = cv2.cvtColor(sideimg, cv2.COLOR_BGR2GRAY)
grayimg = cv2.GaussianBlur(grayimg, (7,7), 0)


# sideimg = cv2.pyrUp(sideimg)
# grayimg = cv2.pyrUp(grayimg)

# edge detection using canny and then dilation and erosion
edges = cv2.Canny(grayimg, 50, 100)
# dilation then erosion removes small holes aka closing

kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

edges = cv2.filter2D(edges, -1, kernel)

edges = cv2.dilate(edges, None, iterations=1)
edges = cv2.erode(edges, None, iterations=1)

# find contours in the edge map
# change retr_external to retr_tree (or something similar) to get inside contours
cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# sort contours from left to right and initializes bounding voz
(cnts, _) = contours.sort_contours(cnts)
colors = ((0,0,255),(240,0,159),(255,0,0),(255,255,0))
# real-life variables here
pixelsPerMetric = None
rupeeHeight = .158    # 2.193 cms for top camera


computedHeightOfCoin = False
computedHeightOfPart = 0

# loops over contours
for (i, c) in enumerate(cnts):
    # contour needs to be large enough
    if cv2.contourArea(c) < 100:
        continue

    # this computes rotated bounding box
    box = cv2.minAreaRect(c)
    # handles whether OpenCV 2 or 3
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    # draws contours
    cv2.drawContours(sideimg, [box], -1, (0,255,0),1)

    rect = orderPoints(box)

    # -------------------------------------------------------------
    # get midpoints of top and bottom
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # get midpoints of left and right
    (tlblX, tlblY) = midpoint(tl, bl)    # left
    (trbrX, trbrY) = midpoint(tr, br)    # right

    # distance between midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # this works b/c starts from left, so first obj encountered is coin
    if pixelsPerMetric is None:
        pixelsPerMetric = min(dA, dB)/rupeeHeight

    dimA = dA/pixelsPerMetric # Vertical!
    dimB = dB/pixelsPerMetric

    if computedHeightOfCoin is True and i == 1:
        if dimA < dimB:
            computedHeightOfPart = dimA  # ensures last height taken is part's
        else:
            computedHeightOfPart = dimB  # ensures last height taken is part's
        cv2.putText(sideimg, str('{:.3f}'.format(computedHeightOfPart)), (int(tlblX), int(tlblY + 10)), cv2.FONT_HERSHEY_SIMPLEX, .35,
                    (0, 101, 219), 1)

    computedHeightOfCoin = True

# fixes imshow size
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1300, 1100)
cv2.imshow('image', sideimg)
cv2.waitKey(0)


# ______________________________________________________________________________________________________________________
# Top View


cap = cv2.VideoCapture(1)
ret, topimg = cap.read()

orig = topimg

grayimg = cv2.cvtColor(topimg, cv2.COLOR_BGR2GRAY)
grayimg = cv2.GaussianBlur(grayimg, (7,7), 0)

# may help with zooming for interior parts
topimg = cv2.pyrUp(topimg)
grayimg = cv2.pyrUp(grayimg)

# same size image but blank in order to put just bounding box on
blankimg = grayimg.copy()

for i in range(len(grayimg)):
    for j in range(len(grayimg[0])):
        blankimg[i][j] = 0


# edge detection using canny and then dilation and erosion
edges = cv2.Canny(grayimg, 50, 100)
# dilation then erosion removes small holes aka closing

kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

edges = cv2.filter2D(edges, -1, kernel)
edges = cv2.dilate(edges, None, iterations=2)
edges = cv2.erode(edges, None, iterations=1)

# find contours in the edge map
# retr_list to get inside contours
cnts = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# sort contours from left to right and initializes bounding box
(cnts, _) = contours.sort_contours(cnts)
colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))


# real-life variables here
pixelsPerMetric = None
rupeeDiameter = 2.193    # 2.193 cms for top camera
distanceToSurface = 45.8  # constant value -> measured beforehand
coinDistance = distanceToSurface - rupeeHeight
objectDistance = distanceToSurface - computedHeightOfPart

count = 0

# loops over contours
for (i, c) in enumerate(cnts[0::2]):
    # contour needs to be large enough
    if cv2.contourArea(c) < 100:
        # this computes rotated bounding box
        edgebox = cv2.minAreaRect(c)
        # handles whether OpenCV 2 or 3
        edgebox = cv2.cv.BoxPoints(edgebox) if imutils.is_cv2() else cv2.boxPoints(edgebox)
        edgebox = np.array(edgebox, dtype="int")
        (tl, tr, br, bl) = edgebox
        for a in range(tr[1], bl[1]):
            for b in range(tr[0], bl[0]):
                edges[a][b] = 0

    # this computes rotated bounding box
    box = cv2.minAreaRect(c)
    # handles whether OpenCV 2 or 3
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    # draws contours
    cv2.drawContours(topimg, [box], -1, (0,255,0),2)

    (tl, tr, br, bl) = box

    # blankimg has only bounding boxes now

    # (only external contours)
    if i < 2:
        cv2.rectangle(blankimg, tuple(tl), tuple(br), (255, 255, 255), 1)
    else:
        for a in range(tr[1], bl[1]):
            for b in range(tr[0], bl[0]):
                edges[a][b] = 0


    rect = orderPoints(box)

    # -------------------------------------------------------------
    # get midpoints of top and bottom
    (tl,tr,br,bl) = box
    (tltrX, tltrY) = midpoint(tl,tr)
    (blbrX, blbrY) = midpoint(bl,br)

    # get midpoints of left and right
    (tlblX, tlblY) = midpoint(tl,bl)    # left
    (trbrX, trbrY) = midpoint(tr,br)    # right

    # distance between midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))  # i.e. vertical
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # this works b/c starts from left, so first obj encountered is coin
    if pixelsPerMetric is None:
        pixelsPerMetric = dB/rupeeDiameter

    dimA = dA/pixelsPerMetric
    dimB = dB/pixelsPerMetric

    # Proportions code -> Only if all subsections of part are on same plane (WATCH OUT!!!)

    if count > 0:
        actualA = dimA * (objectDistance / coinDistance)
        actualB = dimB * (objectDistance / coinDistance)

    else:
        actualA = dimA
        actualB = dimB

    cv2.putText(topimg, str('{:.3f}'.format(actualA)), (int(tlblX), int(tlblY + 20)), cv2.FONT_HERSHEY_SIMPLEX, .50,
                (0, 101, 219), 1)
    cv2.putText(topimg, str('{:.3f}'.format(actualB)), (int(tltrX) - 50, int(tltrY)), cv2.FONT_HERSHEY_SIMPLEX, .50,
                (0, 101, 219), 1)

# fixes imshow size
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1300, 1100)
cv2.imshow('image', topimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite('Results/Dimensioned_black_part_top4.jpg', topimg)

# check to see if part has rounded edges
rounded = raw_input("Calculate radius of rounded edges?")
rounded = rounded.lower()

if rounded == 'y' or rounded == 'yes':

    orig = cv2.pyrUp(orig)

    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    topim = cv2.cvtColor(topimg, cv2.COLOR_BGR2GRAY)

    origedges = cv2.Canny(orig, 50, 100)
    origedges = cv2.dilate(origedges, None, iterations=1)
    origedges = cv2.erode(origedges, None, iterations=1)

    topedges = cv2.Canny(topim, 50, 100)

    # decrease thickness
    edges = cv2.erode(edges, None, iterations=1)

    newimage = edges.copy() + blankimg.copy()

    rows = len(edges)
    columns = len(edges[0])

    # Rounded edges computation

    for i in range(rows):
        for j in range(columns):
            if i < rows - 1 and j < columns-1:
                if blankimg[i][j] == edges[i][j]:
                    newimage[i][j] = 0

    # find contours in the edge map
    # retr_external to get only outside contours
    cnts = cv2.findContours(newimage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # sort contours from left to right and initializes bounding box
    (cnts, _) = contours.sort_contours(cnts)

    # don't need new pixelpermetric b/c same image used

    # loops over contours
    for (i, c) in enumerate(cnts[0::2]):
        # contour needs to be large enough
        if cv2.contourArea(c) < 100:
            continue

        # this computes rotated bounding box
        box = cv2.minAreaRect(c)
        # handles whether OpenCV 2 or 3
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        # draws contours
        cv2.drawContours(newimage, [box], -1, (0, 255, 0), 2)

        # get midpoints of top and bottom
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # get midpoints of left and right
        (tlblX, tlblY) = midpoint(tl, bl)  # left
        (trbrX, trbrY) = midpoint(tr, br)  # right

        # distance between midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))  # i.e. vertical
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        cv2.line(newimage, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255,0,255),2)
        cv2.line(newimage, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255,0,255),2)

        # dimA and dimB should be equivalent for a circle radius
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        if i == 0:
            cv2.putText(topimg, "r="+str('{:.3f}'.format(min(dimA, dimB)/2)), (int(tlblX), int(tlblY + 30)), cv2.FONT_HERSHEY_SIMPLEX, .50,
                    (0, 101, 219), 1)

        if i < 3 and i!=0:
            cv2.putText(topimg, "r="+str('{:.3f}'.format(min(dimA, dimB))), (int(tlblX), int(tlblY + 30)), cv2.FONT_HERSHEY_SIMPLEX, .50,
                    (0, 101, 219), 1)

    cv2.imshow("Top View with Rounded Edges", topimg)
    cv2.waitKey(0)
