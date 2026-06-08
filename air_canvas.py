"""
Air Canvas main code yahan hai
this file will use our trained model + webcam

Gestures:
  ☝️  D-shape (index up)  means DRAW
  ✌️  V-shape (two fingers) means MOVE (no drawing)
  ✊  A-shape (fist)       means STOP / lift pen
  🖐️  B-shape (open palm)  means ERASE

Keyboard mapping below:
  C means Clear canvas
  S means Save canvas as PNG
  Q or ESC means Quit
  1,2,3,4 means Change brush size

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #hides unsual tf ki warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from collections import deque
import copy, itertools, datetime

#Load model & scaler
print("Loading model...")
model       = tf.keras.models.load_model("gesture_model.h5")
scaler_mean = np.load("scaler_mean.npy")
scaler_scl  = np.load("scaler_scale.npy")

#0=fist, 1=open_palm, 2=index_up, 3=two_fingers
CLASSES = ["fist", "open_palm", "index_up", "two_fingers"]

#canvas colours setting
COLORS = {
    "Red"   : (0,   0,   255),
    "Green" : (0,   200, 0  ),
    "Blue"  : (255, 80,  0  ),
    "Yellow": (0,   220, 220),
    "White" : (255, 255, 255),
}
COLOR_NAMES  = list(COLORS.keys())
BRUSH_SIZES  = [3, 6, 10, 16]
ERASER_SIZE  = 40
SMOOTH_FRAMES = 10  # increased for much higher stability
ALPHA = 0.4         # Smoothing factor for the pen tip (0 to 1)

#features extract and normalizer wrt origin
def get_features(hand_lm):
    coords = [[lm.x, lm.y] for lm in hand_lm.landmark]
    origin = copy.deepcopy(coords[0])
    coords = [[c[0]-origin[0], c[1]-origin[1]] for c in coords]
    flat   = list(itertools.chain.from_iterable(coords))
    maxv   = max(abs(v) for v in flat) or 1.0
    flat   = [v/maxv for v in flat]
    return np.array(flat, dtype=np.float32)
#predict on basis of extracted features
def predict(features, buf):
    scaled = (features - scaler_mean) / scaler_scl
    # Direct call is significantly faster than .predict()
    probs = model(scaled.reshape(1, -1), training=False).numpy()[0]
    idx    = int(np.argmax(probs))
    conf   = float(probs[idx])
    buf.append(idx)
    smooth = int(np.argmax(np.bincount(list(buf), minlength=4)))
    return CLASSES[smooth], conf

def fingertip(hand_lm, shape):
    h, w = shape[:2]
    lm = hand_lm.landmark[8]   #index fingertip
    return int(lm.x * w), int(lm.y * h)
#Returns a list of booleans [thumb, index, middle, ring, pinky]
def get_extended_fingers(hand_lm): #helper function
    
    #Landmarks: Index(8), Middle(12), Ring(16), Pinky(20) vs PIPs (6, 10, 14, 18)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    extended = []
    
    #Special check for Thumb (uses X-axis distance from index base)
    thumb_tip = hand_lm.landmark[4]
    index_mcp = hand_lm.landmark[5]
    extended.append(abs(thumb_tip.x - index_mcp.x) > 0.05) 

    for tip, pip in zip(tips, pips):
        #Y-axis check: Tip higher than PIP (MP Y is 0 at top, 1 at bottom)
        extended.append(hand_lm.landmark[tip].y < hand_lm.landmark[pip].y)
    
    return extended

def is_finger_gun_robust(hl):
    #detects L shape of hand mathematically
    def d(p1, p2): return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
    # Index is straight if TIP is far from MCP
    idx_ext = d(hl.landmark[8], hl.landmark[5]) > d(hl.landmark[6], hl.landmark[5]) * 1.6
    # Thumb is 'up' if it's far from the middle finger base
    thumb_up = d(hl.landmark[4], hl.landmark[9]) > d(hl.landmark[5], hl.landmark[9]) * 1.2
    # Middle/Ring/Pinky must be curled (TIP near MCP)
    others_curled = d(hl.landmark[12], hl.landmark[9]) < d(hl.landmark[10], hl.landmark[9]) * 1.1
    return idx_ext and thumb_up and others_curled

#ui work
BTN_W = 80 #width
BTN_H = 40 #height

def draw_toolbar(frame, color_name, brush_idx, gesture, conf):
    w, h = frame.shape[1], frame.shape[0]
    
    #Create semi-transparent Navy Blue overlay (BGR: 100, 0, 0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w, 75), (100, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    #Color buttons (LEFT EDGE, VERTICAL)
    for i, name in enumerate(COLOR_NAMES):
        y1 = 100 + i*(BTN_H + 30)
        y2 = y1 + BTN_H
        x1, x2 = 10, 10 + BTN_W
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[name], -1)
        if name == color_name:
            cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (255,255,255), 2)
        cv2.putText(frame, name, (x1+5, y2+15),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    #Brush size dots (RIGHT EDGE, VERTICAL)
    rx = w - 50
    for i, sz in enumerate(BRUSH_SIZES):
        cy = 120 + i*80
        col = (255,255,255) if i==brush_idx else (160,160,160)
        cv2.circle(frame, (rx, cy), sz+2, col, -1)
        if i==brush_idx:
            cv2.circle(frame, (rx, cy), sz+6, (255,255,255), 1)
        cv2.putText(frame, f"S{i+1}", (rx-15, cy+35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    #Dashboard (CENTER ALIGNED)
    ACTION = {"index_up":"DRAWING","two_fingers":"MOVING",
               "fist":"STOPPED","open_palm":"ERASING"}
    label  = ACTION.get(gesture, "WAITING...")
    
    #Combined status text
    status_text = f"GESTURE: {label}  |  AI SEES: {(gesture or '...').upper()}"
    (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
    cv2.putText(frame, status_text, (int((w-tw)/2), 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    info_text = "C:Clear S:Save Q:Quit | 1-4:Size"
    (iw, ih), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)
    cv2.putText(frame, info_text, (int((w-iw)/2), 62),
                cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
#Returns color name if click is on a color button
def toolbar_click(x, y):
   
    x1, x2 = 10, 10 + BTN_W
    for i, name in enumerate(COLOR_NAMES):
        y1 = 100 + i*(BTN_H + 30)
        y2 = y1 + BTN_H
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name
    return None
#Returns brush index if click is on brush dot.
def brush_click(x, y, w):
    
    rx = w - 50
    for i in range(len(BRUSH_SIZES)):
        cy = 120 + i*80
        if abs(x-rx) < 25 and abs(y-cy) < 25:
            return i
    return None

#main implementation

def main():
    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(
        static_image_mode=False, #Tracking mode is MUCH more stable for video
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7)
    mp_draw  = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0) #launches webcam
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam!"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280) #window dimensions
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()
    frame  = cv2.flip(frame, 1)
    canvas = np.zeros_like(frame)
    #initialize colour jiss par canvas set ho shuuru mai
    color_name = "Red"
    brush_idx  = 1
    prev_pt    = None
    gesture    = None
    conf       = 0.0
    buf        = deque(maxlen=SMOOTH_FRAMES)
    smooth_tip = None   # for jitter-free drawing
    gun_active = 0      # counter for stabilizing finger gun
    smooth_aim = None   # smoothed projection point
    #mouse behaviour
    def on_mouse(event, x, y, flags, param):
        nonlocal color_name, brush_idx
        if event == cv2.EVENT_LBUTTONDOWN:
            c = toolbar_click(x, y)
            if c: color_name = c
            b = brush_click(x, y, frame.shape[1])
            if b is not None: brush_idx = b
    #text UI of windows
    cv2.namedWindow("Air Canvas")
    cv2.setMouseCallback("Air Canvas", on_mouse)

    print("\n  Air Canvas is running!")
    print("  Point your INDEX FINGER UP to draw.\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = hands.process(rgb)
        rgb.flags.writeable = True

        gesture = None
        conf    = 0.0
        tip     = None

        if result.multi_hand_landmarks:
            hl = result.multi_hand_landmarks[0]
            #Get handedness (MediaPipe handedness is flipped relative to view)
            #Label 'Left' usually refers to the hand on the left side of the screen if flipped.
            
            hand_info = result.multi_handedness[0].classification[0]
            hand_label = hand_info.label # "Left" or "Right"

            #Draw skeleton with the help of landmarks and this function
            mp_draw.draw_landmarks(
                frame, hl, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=COLORS[color_name], thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(180,180,180), thickness=1))

            features        = get_features(hl)
            gesture, conf   = predict(features, buf)
            
            #continued implementation of finger gun
            ext = get_extended_fingers(hl)
            is_fist = not any(ext[1:])
            #detect gesture
            if is_fist:
                gesture = "fist"
            elif ext[1] and not any(ext[2:]):
                gesture = "index_up"
            elif ext[1] and ext[2] and not any(ext[3:]):
                gesture = "two_fingers"
            elif all(ext[1:]):
                gesture = "open_palm"
            
            #finger fun continued part 2
            aim_pt = None
            if is_finger_gun_robust(hl):
                gun_active = min(10, gun_active + 1)
            else:
                gun_active = max(0, gun_active - 1)
            
            #FINGER GUN detect honay kay baad: part 3 Laser activates only if held for 3+ frames
            if gun_active >= 3:
                gesture = "finger_gun"
                mcp, tip_idx = hl.landmark[5], hl.landmark[8]
                vx, vy = (tip_idx.x - mcp.x), (tip_idx.y - mcp.y)
                proj_x = int((mcp.x + vx * 4.5) * frame.shape[1])
                proj_y = int((mcp.y + vy * 4.5) * frame.shape[0])
                raw_aim = (max(0, min(frame.shape[1]-1, proj_x)), 
                           max(0, min(frame.shape[0]-1, proj_y)))
                
                #Smooth the laser dot movement
                if smooth_aim is None: smooth_aim = list(raw_aim)
                else:
                    smooth_aim[0] = int(smooth_aim[0]*0.7 + raw_aim[0]*0.3)
                    smooth_aim[1] = int(smooth_aim[1]*0.7 + raw_aim[1]*0.3)
                aim_pt = tuple(smooth_aim)
            else:
                smooth_aim = None
            

            #Get raw fingertip. LIVE UPDATE HAMARI FINGERTIP
            raw_tip = fingertip(hl, frame.shape)
            
            #Apply Exponential Smoothing to kill jitter. Alpha value is used here
            if smooth_tip is None:
                smooth_tip = list(raw_tip)
            else:
                smooth_tip[0] = int(smooth_tip[0] * (1 - ALPHA) + raw_tip[0] * ALPHA)
                smooth_tip[1] = int(smooth_tip[1] * (1 - ALPHA) + raw_tip[1] * ALPHA)
            
            tip = tuple(smooth_tip)
            color           = COLORS[color_name]
            brush           = BRUSH_SIZES[brush_idx]

            if gesture == "index_up" and tip[1] > 80:
                #DRAW IF FINGER TIP IS DETECTED
                if prev_pt is not None:
                    cv2.line(canvas, prev_pt, tip, color, brush*2)
                prev_pt = tip
                cv2.circle(frame, tip, brush+6, color, 2)   # cursor ring
            #LASERLINE DRAW CALL PART 4
            elif gesture == "finger_gun" and aim_pt:
                prev_pt = None
                #Laser Line visual feedback
                cv2.line(frame, tip, aim_pt, (0, 255, 255), 2)
                cv2.circle(frame, aim_pt, 8, (255, 255, 255), -1) # "Laser dot"
                cv2.circle(frame, aim_pt, 12, (0, 255, 255), 2)
                #right and left hand implementation lasergun
                #Right hand (points left) selects colors
                if hand_label == "Right":
                    c = toolbar_click(aim_pt[0], aim_pt[1])
                    if c: color_name = c
                #Left hand (points right) selects brush sizes
                elif hand_label == "Left":
                    b = brush_click(aim_pt[0], aim_pt[1], frame.shape[1])
                    if b is not None: brush_idx = b

            elif gesture == "two_fingers":
                # MOVE — cursor visible but no drawing
                prev_pt = None
                cv2.circle(frame, tip, 10, (200,200,200), 2)

            elif gesture == "fist":
                # STOP — lift pen
                prev_pt = None

            elif gesture == "open_palm":
                # ERASE — clear circle around fingertip
                prev_pt = None
                cv2.circle(canvas, tip, ERASER_SIZE, (0,0,0), -1)
                cv2.circle(frame,  tip, ERASER_SIZE, (80,80,80), 2)
        else:
            prev_pt = None
            cv2.putText(frame, "Show Hand to Begin", (480, 380),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # Merge canvas onto frame
        out  = frame.copy()
        mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        out[mask > 0] = canvas[mask > 0]

        draw_toolbar(out, color_name, brush_idx, gesture, conf)

        cv2.imshow("Air Canvas", out)

        # Allow closing via 'X' button or Q/ESC keys
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27) or cv2.getWindowProperty("Air Canvas", cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key == ord('c'):
            canvas = np.zeros_like(frame)
            print("  Canvas cleared!")
        elif key == ord('s'):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"canvas_{ts}.png"
            cv2.imwrite(name, canvas)
            print(f"  Saved → {name}")
        elif key in [ord('1'),ord('2'),ord('3'),ord('4')]:
            brush_idx = int(chr(key)) - 1

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Goodbye!")

if __name__ == "__main__":
    main()