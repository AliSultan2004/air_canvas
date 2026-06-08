"""
FIRST PHASE!!!!!!!!
Main Funcationality:
extracts all the hand landmarks from images 
and write them into hand_landmarks.csv 
"""

import os, cv2, csv, copy, itertools
import mediapipe as mp
from pathlib import Path

OUTPUT_CSV = "hand_landmarks.csv"
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp"}

#ASL letters we want to extract here
GESTURE_MAP = {
    "A": (0, "fist"),
    "B": (1, "open_palm"),
    "D": (2, "index_up"),
    "V": (3, "two_fingers"),
}

def get_landmarks(hand_lm):
    coords = [[lm.x, lm.y] for lm in hand_lm.landmark]
    origin = copy.deepcopy(coords[0])
    coords = [[c[0]-origin[0], c[1]-origin[1]] for c in coords]
    #We set wrist as the origin here
    flat   = list(itertools.chain.from_iterable(coords))
    maxv   = max(abs(v) for v in flat) or 1.0
    flat   = [round(v/maxv, 6) for v in flat]
    #Normalising marks, allows for better recognision of Gestures.
    return flat

def find_dataset_root():
    """
    This is automatic path finder of where the dataset images actually are.
    Searches common folder patterns under 'dataset/'.
    Uses standard os library. No need to hardcode paths.
    """
    base = Path("dataset")
    if not base.exists():
        print("ERROR!!! 'dataset' folder not found!")
        print("   Must run from inside the projcet folder.")
        return None

    #Output for tracking/progress purposes..
    print("\n  Scanning dataset folder structure...")
    for root, dirs, files in os.walk("dataset"):
        imgs = [f for f in files if Path(f).suffix.lower() in IMG_EXTS]
        if imgs:
            print(f"    Found {len(imgs)} images in: {root}")

    #Find folder containing destination subfolders
    for root, dirs, files in os.walk("dataset"):
        dirnames = [d.upper() for d in dirs]
        if "A" in dirnames and "B" in dirnames:
            print(f"\n  Dataset root found: {root}")
            return Path(root)

    print("\n  ERROR!! Could not find folders A, B, D, V inside dataset/ directory")
    return None

def process_folder(folder, label_idx, gesture_name, writer, hands, limit=800):
    if not folder.exists():
        print(f"    SKIP, Not found: {folder}")
        return 0

    files = [f for f in sorted(folder.iterdir())
             if f.suffix.lower() in IMG_EXTS]

    print(f"    Found {len(files)} image files", end="", flush=True)
    files = files[:limit]

    count = 0
    for fp in files:
        #Reading images
        img = cv2.imread(str(fp))
        if img is None:
            continue

        h, w = img.shape[:2]
        if max(h, w) > 640: #Resize for efficiency.
            s = 640 / max(h, w)
            img = cv2.resize(img, (int(w*s), int(h*s)))
            
        #IMP, Mandatory color conversion
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        
        """If hand detected, pass it into get_landmark() 
            appends the gestures class index(e.g 0) and name(e.g fist)
            and writes the entire row to CSV file.
        """
        if res.multi_hand_landmarks:
            flat = get_landmarks(res.multi_hand_landmarks[0])
            writer.writerow(flat + [label_idx, gesture_name])
            count += 1

    return count

def main():
    print("\n========================================")
    print("  Air Canvas - Phase 1 - Landmarks Extracton")
    print("=========================================")

    #Find dataset
    dataset_root = find_dataset_root()
    if dataset_root is None:
        return

    #Show what letter folders exist
    print("\n  Letter folders found inside dataset root:")
    for item in sorted(dataset_root.iterdir()):
        if item.is_dir():
            imgs = list(item.glob("*"))
            img_count = sum(1 for f in imgs if f.suffix.lower() in IMG_EXTS)
            print(f"    {item.name:<10} {img_count} images")

    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.4,
    )

    header = [f"x{i}" for i in range(21)] + \
             [f"y{i}" for i in range(21)] + \
             ["label", "gesture_name"]

    print(f"\n  Extracting landmarks for A, B, D, V...")
    print(f"  Output → {OUTPUT_CSV}\n")

    total = 0
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for letter, (idx, name) in GESTURE_MAP.items():
            print(f"  [{letter}] {name:<15}", end=" ", flush=True)

            # Try different possible folder name casings
            folder = None
            for candidate in [letter, letter.upper(), letter.lower()]:
                p = dataset_root / candidate
                if p.exists():
                    folder = p
                    break

            if folder is None:
                print(f"→ folder not found, skipping")
                continue

            n = process_folder(folder, idx, name, writer, hands, limit=800)
            total += n
            bar = "█" * (n // 50) #Progress bar
            print(f" → {n} extracted  {bar}")

    hands.close()

    print(f"\n========================================")
    if total == 0:
        print("  ERROR 0 samples extracted!")
        print("  Check that MediaPipe can detect hands in your images.")
        print("  Try lowering min_detection_confidence to 0.3")
    else:
        print(f"  HOGAYA! Total samples: {total}")
        print(f"  Saved → {OUTPUT_CSV}")
        print(f"\n  PHASE 1 COMPLETE! phase 2 kay liay ye dabaien -> train_model.py")
    print(f"========================================\n")

if __name__ == "__main__":
    main()