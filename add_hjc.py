import sys
import os
import argparse

sys.path.insert(0, r"D:\AFO_Codes")
sys.path.insert(0, r"D:\AFO_Codes\TreadMetrix")

from TreadMetrix.hip_joint_computation import compute_hip_joints

def main():
    parser = argparse.ArgumentParser(description="Add virtual HJC markers to a TRC file")
    parser.add_argument("input_trc", help="Path to the input TRC file")
    parser.add_argument("--output_trc", help="Path to save the updated TRC file (optional)")
    
    args = parser.parse_args()
    
    input_path = args.input_trc
    
    if args.output_trc:
        output_path = args.output_trc
    else:
        # Default: append '_addedHJ' to the filename
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_addedHJ{ext}"
        
    print(f"Processing: {input_path}")
    try:
        updated_trc = compute_hip_joints(input_path, output_path)
        print(f"Success! HJC markers added.")
        print(f"Output saved to: {output_path}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == '__main__':
    main()
