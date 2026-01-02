# ============================================================
#  IMPORTS
# ============================================================
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
#  LOAD IMAGES
# ============================================================
fn_L1 = "/content/Pose_0-CALIB/RECT_INP_LEFT_THETA_0.jpg"
fn_L2 = "/content/Pose_+1/RECT_INP_LEFT_THETA_1.jpg"
fn_R1 = "/content/Pose_0-CALIB/RECT_INP_RIGHT_THETA_0.jpg"
fn_R2 = "/content/Pose_+1/RECT_INP_RIGHT_THETA_1.jpg"

L1 = cv2.imread(fn_L1)
R1 = cv2.imread(fn_R1)
L2 = cv2.imread(fn_L2)
R2 = cv2.imread(fn_R2)

assert L1 is not None, "Image load failed"

gL1 = cv2.cvtColor(L1, cv2.COLOR_BGR2GRAY)
gR1 = cv2.cvtColor(R1, cv2.COLOR_BGR2GRAY)
gL2 = cv2.cvtColor(L2, cv2.COLOR_BGR2GRAY)
gR2 = cv2.cvtColor(R2, cv2.COLOR_BGR2GRAY)

# ============================================================
#  WHEEL CENTER DETECTION (UNCHANGED)
# ============================================================
def detect_wheel_center(gray):
    _, th = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(c)
    return (int(x), int(y)), int(r)

def detect_wheel_center_brightness(gray):
    clahe = cv2.createCLAHE(3.0, (8,8))
    e = clahe.apply(gray)
    _, b = cv2.threshold(e, 100, 255, cv2.THRESH_BINARY)
    M = cv2.moments(b)
    if M["m00"] == 0:
        return None, None
    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])
    r = int(np.sqrt(M["m00"]/np.pi))
    return (cx, cy), r

def detect_wheel(gray):
    c1, r1 = detect_wheel_center(gray)
    c2, r2 = detect_wheel_center_brightness(gray)
    if c1 and r1 > 50:
        return c1, r1
    if c2 and r2 > 50:
        return c2, r2
    h, w = gray.shape
    return (w//2, h//2), min(h,w)//3

center_L1, rL1 = detect_wheel(gL1)
center_R1, rR1 = detect_wheel(gR1)
center_L2, rL2 = detect_wheel(gL2)
center_R2, rR2 = detect_wheel(gR2)

print("Wheel centers:")
print("L1", center_L1)
print("R1", center_R1)
print("L2", center_L2)
print("R2", center_R2)

# ============================================================
#  VISUALIZE WHEEL CENTERS
# ============================================================
plt.figure(figsize=(20,5))
for i,(img,c,n) in enumerate(zip(
        [L1,R1,L2,R2],
        [center_L1,center_R1,center_L2,center_R2],
        ["L1","R1","L2","R2"])):
    im = img.copy()
    cv2.circle(im, c, 8, (0,255,0), 3)
    plt.subplot(1,4,i+1)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.title(n)
    plt.axis("off")
plt.show()

# ============================================================
#  COMPUTE CORRECT SHIFTS (CRITICAL FIX)
# ============================================================
# Stereo (L->R)
stereo_shifts = [
    (center_R1[0]-center_L1[0], center_R1[1]-center_L1[1]),
    (center_R2[0]-center_L2[0], center_R2[1]-center_L2[1])
]
STEREO_SHIFT = (int(np.median([s[0] for s in stereo_shifts])),
                int(np.median([s[1] for s in stereo_shifts])))

# Temporal (pose 0 -> pose +1)
temporal_shifts = [
    (center_L2[0]-center_L1[0], center_L2[1]-center_L1[1]),
    (center_R2[0]-center_R1[0], center_R2[1]-center_R1[1])
]
TEMPORAL_SHIFT = (int(np.median([s[0] for s in temporal_shifts])),
                  int(np.median([s[1] for s in temporal_shifts])))

print("STEREO_SHIFT:", STEREO_SHIFT)
print("TEMPORAL_SHIFT:", TEMPORAL_SHIFT)

# ============================================================
#  HARRIS + ORB FEATURES
# ============================================================
def harris_pts(gray, n=200):
    h = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    pts = np.argwhere(h > 0.01*h.max())[:,::-1]
    if len(pts) > n:
        r = h[pts[:,1], pts[:,0]]
        pts = pts[np.argsort(r)[::-1][:n]]
    return pts.astype(np.int32)

orb = cv2.ORB_create(5000)

def orb_from_pts(gray, pts):
    kp = [cv2.KeyPoint(float(x), float(y), 10) for x,y in pts]
    return orb.compute(gray, kp)

kpL1, desL1 = orb_from_pts(gL1, harris_pts(gL1))
kpR1, desR1 = orb_from_pts(gR1, harris_pts(gR1))
kpL2, desL2 = orb_from_pts(gL2, harris_pts(gL2))
kpR2, desR2 = orb_from_pts(gR2, harris_pts(gR2))

# ============================================================
#  GUIDED MATCHING (CORRECT SHIFTS)
# ============================================================
def guided_match(kpA, desA, kpB, desB, shift, win):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    m = bf.match(desA, desB)
    wx, wy = win
    good = []
    for x in m:
        xa, ya = kpA[x.queryIdx].pt
        xb, yb = kpB[x.trainIdx].pt
        if abs(xb-(xa+shift[0]))<wx and abs(yb-(ya+shift[1]))<wy:
            good.append(x)
    return good

mL1R1 = guided_match(kpL1, desL1, kpR1, desR1, STEREO_SHIFT, (40,15))
mL2R2 = guided_match(kpL2, desL2, kpR2, desR2, STEREO_SHIFT, (40,15))
mL1L2 = guided_match(kpL1, desL1, kpL2, desL2, TEMPORAL_SHIFT, (40,40))
mR1R2 = guided_match(kpR1, desR1, kpR2, desR2, TEMPORAL_SHIFT, (40,40))

print("Matches:",
      len(mL1R1), len(mL2R2), len(mL1L2), len(mR1R2))

# ============================================================
#  BUILD QUADS
# ============================================================
LR1 = {m.queryIdx:m.trainIdx for m in mL1R1}
LR2 = {m.queryIdx:m.trainIdx for m in mL2R2}
LL  = {m.queryIdx:m.trainIdx for m in mL1L2}
RR  = {m.queryIdx:m.trainIdx for m in mR1R2}

quads = []
for iL1 in LR1:
    if iL1 in LL:
        iR1, iL2 = LR1[iL1], LL[iL1]
        if iL2 in LR2 and iR1 in RR:
            quads.append((iL1,iR1,iL2,LR2[iL2]))

print("TOTAL QUADS:", len(quads))

# ============================================================
#  SAVE TO EXCEL
# ============================================================
rows = [[*kpL1[a].pt,*kpR1[b].pt,*kpL2[c].pt,*kpR2[d].pt]
        for a,b,c,d in quads]
pd.DataFrame(rows,columns=[
    "L1_x","L1_y","R1_x","R1_y",
    "L2_x","L2_y","R2_x","R2_y"
]).to_excel("quad_matches.xlsx",index=False)

# ============================================================
#  VISUALIZE QUADS
# ============================================================
vis = np.hstack((L1,R1,L2,R2))
W = L1.shape[1]

for a,b,c,d in quads[:10]:
    p1 = kpL1[a].pt
    p2 = (kpR1[b].pt[0]+W, kpR1[b].pt[1])
    p3 = (kpL2[c].pt[0]+2*W, kpL2[c].pt[1])
    p4 = (kpR2[d].pt[0]+3*W, kpR2[d].pt[1])

    cv2.line(vis,tuple(map(int,p1)),tuple(map(int,p2)),(0,255,0),3)
    cv2.line(vis,tuple(map(int,p2)),tuple(map(int,p3)),(255,0,0),3)
    cv2.line(vis,tuple(map(int,p3)),tuple(map(int,p4)),(0,0,255),3)

plt.figure(figsize=(22,6))
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
plt.title(f"{len(quads)} VALID QUADS")
plt.axis("off")
plt.show()