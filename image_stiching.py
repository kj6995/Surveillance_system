from __future__ import print_function
from threading import Thread
import numpy as np
import cv2
import time
import datetime

class WebcamVideoStream:

	def __init__(self, src=0):
		# initialize the video camera stream and read the first frame from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()

		# initialize the variable used to indicate if the thread should be stopped
		self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return

			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

class Stitcher:
	
	def __init__(self):
		self.cachedH = None

	def stitch(self, images, ratio=0.75, reprojThresh=4.0,showMatches=False):
		# unpack the images, then detect keypoints and extract local invariant descriptors from them
		(imageB, imageA) = images

		# if the cached homography matrix is None, then we need to apply keypoint matching to construct it
		if self.cachedH is None:
			
			#detect keypoints and extract
			(kpsA, featuresA) = self.detectAndDescribe(imageA)
			(kpsB, featuresB) = self.detectAndDescribe(imageB)
 
			# match features between the two images
			M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
 
			# if the match is None, then there aren't enough matched
			# keypoints to create a panorama
			if M is None:
				return None

			#cache the homography matrix
			self.cachedH = M[1]

		# otherwise, apply a perspective warp to stitch the images together
		#(matches, H, status) = M
		result = cv2.warpPerspective(imageA, self.cachedH,(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
		result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
 
		# # check to see if the keypoint matches should be visualized
		# if showMatches:
		# 	vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
 
		# 	# return a tuple of the stitched image and the
		# 	# visualization
		# 	return (result, vis)
 
		# return the stitched image
		return result

	def detectAndDescribe(self, image):
		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
		descriptor = cv2.xfeatures2d.SIFT_create()
		(kps, features) = descriptor.detectAndCompute(image, None)
 
		# convert the keypoints from KeyPoint objects to NumPy arrays
		kps = np.float32([kp.pt for kp in kps])
 
		# return a tuple of keypoints and features
		return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
		# compute the raw matches and initialize the list of actual matches
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []
 
		# loop over the raw matches
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		# computing a homography requires at least 4 matches
		if len(matches) > 4:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])
 
			# compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
 
			# return the matches along with the homograpy matrix and status of each matched point
			return (matches, H, status)
 
		# otherwise, no homograpy could be computed
		return None

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB
 
		# loop over the matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully matched
			if s == 1:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
 
		# return the visualization
		return vis

class BasicMotionDetector:

	def __init__(self, accumWeight=0.5, deltaThresh=5, minArea=5000):
		# determine by storing the frame accumulation weight, the fixed threshold for
		# the delta image, and finally the minimum area required for "motion" to be reported
		self.accumWeight = accumWeight
		self.deltaThresh = deltaThresh
		self.minArea = minArea
 
		# initialize the average image for motion detection
		self.avg = None

	def update(self, image):
		# initialize the list of locations containing motion
		locs = []
 
		# if the average image is None, initialize it
		if self.avg is None:
			self.avg = image.astype("float")
			return locs
 
		# otherwise, accumulate the weighted average between  the current frame and the previous frames, then compute
		# the pixel-wise differences between the current frame and running average
		cv2.accumulateWeighted(image, self.avg, self.accumWeight)
		frameDelta = cv2.absdiff(image, cv2.convertScaleAbs(self.avg))

		# threshold the delta image and apply a series of dilations to help fill in holes
		thresh = cv2.threshold(frameDelta, self.deltaThresh, 255,cv2.THRESH_BINARY)[1]
		thresh = cv2.dilate(thresh, None, iterations=2)
 
		# find contours in the thresholded image, taking care to use the appropriate version of OpenCV
		cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
 
		# loop over the contours
		for  c in cnts:
			# only add the contour to the locations list if it exceeds the minimum area
			if cv2.contourArea(c) > self.minArea:
				locs.append(c)
		
		# return the set of locations
		return locs





# initialize the video streams and allow them to warmup
leftStream = WebcamVideoStream(src=0).start()
rightStream = WebcamVideoStream(src=1).start()
#time.sleep(2.0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out0 = cv2.VideoWriter('output_left.avi', fourcc, 20.0, (640,480))
out1 = cv2.VideoWriter('output_right.avi',fourcc, 20.0, (640,480))
out2 = cv2.VideoWriter('output_combine.avi',fourcc, 20.0, (1280,480))
# initialize the image stitcher, motion detector, and total number of frames read
stitcher = Stitcher()
motion = BasicMotionDetector()
total = 0

while(1):
	# grab the frames from their respective video streams
	left = leftStream.read()
	right = rightStream.read()
 
	# resize the frames
	left = cv2.resize(left, (640, 480))
	right = cv2.resize(right, (640, 480))
 
	# stitch the frames together to form the panorama
	# IMPORTANT: you might have to change this line of code depending on how your cameras are oriented; frames
	# should be supplied in left-to-right order
	result = stitcher.stitch([left, right])
 
	# no homograpy could be computed
	if result is None:
		print("[INFO] homography could not be computed")
		break
 
	# convert the panorama to grayscale, blur it slightly, update the motion detector
	gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
	locs = motion.update(gray)
	
	# only process the panorama for motion if a nice average has been built up
	if total > 32 and len(locs) > 0:
		# initialize the minimum and maximum (x, y)-coordinates, respectively
		(minX, minY) = (np.inf, np.inf)
		(maxX, maxY) = (-np.inf, -np.inf)
 
		# loop over the locations of motion and accumulate the minimum and maximum locations of the bounding boxes
		for l in locs:
			(x, y, w, h) = cv2.boundingRect(l)
			(minX, maxX) = (min(minX, x), max(maxX, x + w))
			(minY, maxY) = (min(minY, y), max(maxY, y + h))
 
		# draw the bounding box
		cv2.rectangle(result, (minX, minY), (maxX, maxY),(0, 0, 255), 3)
	
	# increment the total number of frames read and draw the timestamp on the image
	total += 1
	timestamp = datetime.datetime.now()
	ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
	cv2.putText(result, ts, (10, result.shape[0] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
 	
	out0.write(left)
    #out1.write(right)
 	out2.write(result)

	# show the output images
	cv2.imshow("Result", result)
	cv2.imshow("Left Frame", left)
	cv2.imshow("Right Frame", right)

	if cv2.waitKey(10) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		leftStream.stop()
		rightStream.stop()
		break
