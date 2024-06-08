import numpy as np
import cv2

# Load the video

# cap = cv2.VideoCapture('video1.webm')
cap = cv2.VideoCapture('video2.webm')
# cap = cv2.VideoCapture('drone_vid.webm')

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Read the first frame and find initial points
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing
mask = np.zeros_like(old_frame)

# Colors for drawing different tracks
colors = np.random.randint(0, 255, (100, 3))

# Process each frame in the video
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = np.int32(new.ravel())
                c, d = np.int32(old.ravel())
                color = colors[i % 100].tolist()
                print("a, b:", a, b)
                print("c, d:", c, d)
                mask = cv2.line(mask, (a, b), (c, d), color, 2)
                frame = cv2.circle(frame, (a, b), 5, color, -1)

            img = cv2.add(frame, mask)
            cv2.imshow('frame', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

            # Update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        else:
            # Re-detect points if no points are tracked
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

except Exception as e:
    print(f"Error: {e}")

cap.release()
cv2.destroyAllWindows()
