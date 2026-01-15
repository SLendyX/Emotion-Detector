import os
import random
import shutil

# --- CONFIGURATION ---
BASE_DIR = "data/raw"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
EXTERNAL_DIR = "kdef" # Folder where you put your new KDEF/CK+ images

CATEGORIES = ["angry", "disgust", "fear", "happy", "neutral", "sadness", "surprise"]

# TARGETS
TOTAL_TARGET = 4000  # We want (FER + KDEF) to equal this
TEST_TARGET = 800    # We want Test set to reach this (0.2 of 4000)

def count_files(directory, category):
    path = os.path.join(directory, category)
    if not os.path.exists(path):
        return 0
    return len(os.listdir(path))

def get_files(directory, category):
    path = os.path.join(directory, category)
    if not os.path.exists(path):
        return []
    files = [os.path.join(path, f) for f in os.listdir(path)]
    random.shuffle(files) # Randomize!
    return files

def merge_and_balance():
    print("📊 ANALYZING DATA SOURCES...\n")
    
    print("-" * 95)
    print(f"{'CAT':<10} | {'KDEF':<6} | {'FER TRAIN':<9} | {'FER TEST':<8} | {'ACTION PLAN'}")
    print("-" * 95)

    plan = {}

    for cat in CATEGORIES:
        # 1. Get Counts
        n_kdef = count_files(EXTERNAL_DIR, cat)
        n_train = count_files(TRAIN_DIR, cat)
        n_test = count_files(TEST_DIR, cat)
        
        # 2. Calculate Goals
        # We want: n_train_final + n_kdef = TOTAL_TARGET
        target_fer_train = TOTAL_TARGET - n_kdef
        if target_fer_train < 0: target_fer_train = 0 # If KDEF has > 4000, we keep 0 FER
        
        # 3. Logic
        move_to_test = 0
        delete_train = 0
        
        # Available FER images we can play with
        current_fer_train = n_train
        
        # A. Fill Test Set first?
        test_gap = TEST_TARGET - n_test
        if test_gap > 0 and current_fer_train > 0:
            # We can only move what we have, but we shouldn't drop below our training target 
            # unless we have a MASSIVE surplus.
            # Simple logic: Take from surplus first.
            surplus = current_fer_train - target_fer_train
            
            # If we have surplus, use it to fill test
            can_move = min(surplus, test_gap)
            
            # If can_move is negative (meaning we are short on training data too),
            # we prioritize Test data slightly or just move 0. 
            # Let's simple-move only positive surplus to be safe.
            if can_move > 0:
                move_to_test = can_move
                current_fer_train -= move_to_test
        
        # B. Delete Excess Train?
        # Recalculate surplus after potential move
        final_surplus = current_fer_train - target_fer_train
        if final_surplus > 0:
            delete_train = final_surplus
            
        final_total = (n_train - move_to_test - delete_train) + n_kdef
        
        plan[cat] = {
            "move": move_to_test,
            "delete": delete_train
        }

        # Status Message
        msg = []
        if move_to_test > 0: msg.append(f"Move {move_to_test}->Test")
        if delete_train > 0: msg.append(f"Del {delete_train} FER")
        if not msg: msg.append("Keep All")
        
        print(f"{cat:<10} | {n_kdef:<6} | {n_train:<9} | {n_test:<8} | {', '.join(msg)}")

    print("-" * 95)
    print(f"\n🎯 GOAL: Total Training (FER+KDEF) = {TOTAL_TARGET} | Test = {TEST_TARGET}")
    
    # --- CONFIRMATION ---
    confirm = input("\n⚠️  Ready to modify FER folders? (Type 'yes' to proceed): ")
    if confirm.lower() != "yes":
        print("❌ Cancelled.")
        return

    # --- EXECUTION ---
    print("\n🚀 Executing...")
    for cat in CATEGORIES:
        p = plan[cat]
        
        if p['move'] == 0 and p['delete'] == 0:
            continue
            
        files = get_files(TRAIN_DIR, cat)
        
        # 1. Move to Test
        if p['move'] > 0:
            dest = os.path.join(TEST_DIR, cat)
            if not os.path.exists(dest): os.makedirs(dest)
            
            to_move = files[:p['move']]
            files = files[p['move']:] # Remove from list
            
            for f in to_move:
                shutil.move(f, os.path.join(dest, os.path.basename(f)))
        
        # 2. Delete Excess
        if p['delete'] > 0:
            to_delete = files[:p['delete']]
            for f in to_delete:
                os.remove(f)
                
    print("✅ Balancing Complete!")
    print("👉 NOTE: You must now manually move your KDEF images into 'data/raw/train' to finish the merge.")

if __name__ == "__main__":
    merge_and_balance()