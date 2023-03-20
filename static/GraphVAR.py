from static import Variable
import os

# Combination of result to draw graph
# DO NOT FORGET TO ADD EXTENSION (.txt)

# Nas mount options "dataset", "Nas"
DATASET_MOUNT = "Nas"
# DATASET_MOUNT = "dataset"

DATASET = 'office1_manolis'
REFINE_OPTION = 1

# Pose Accuracy ECDF: Draw All Lines in One CDF Figure
POSE_TEXT = [f"PC_{DATASET}_NA_sp1.0_n0.0_sw0.txt",
             f"OLC_{DATASET}_NA_sp1.0_n0.0_sw0.txt",
             f"OLC_{DATASET}_NA_sp0.5_n0.0_sw0.txt", 
             f"PPL_{DATASET}_NA_sp1.0_n0.0_sw0.txt",
             f"PPLplus_{DATASET}_NA_sp1.0_n0.0_sw0.txt"]

# Point Reconstruction ECDF: Draw All Lines in One CDF Figure
# POINT_TEXT = ["OLC_gerrard_hall_SPF_sp0.3_n0.0_sw0.txt", "PPL_gerrard_hall_TPF_sp0.2_n0.0_sw1.0.txt"]
POINT_TEXT = [f"OLC_{DATASET}_SPF_sp1.0_n0.0_sw0.txt",
              f"PPL_{DATASET}_SPF_sp1.0_n0.0_sw0.txt",
              f"PPLplus_{DATASET}_SPF_sp1.0_n0.0_sw0.txt",
              f"PPL_{DATASET}_TPF_sp1.0_n0.0_sw0.txt",
              f"PPLplus_{DATASET}_TPF_sp1.0_n0.0_sw0.txt"]

# Recovered Image Quality x Pose Accuracy Curve: Draw Curve Using Combination from Each List
# Note that Text Name Should Contain "SPARSITY" in place of "sp##"
# !!!!!!!!!!!! Order Matters !!!!!!!!!!!!!!
# QUALITY_TEXT = [f"PC_{DATASET}_SPF_SPARSITY_n0.0_SWAP.txt", 
                # f"OLC_{DATASET}_SPF_SPARSITY_n0.0_SWAP.txt",
                # f"PPL_{DATASET}_SPF_SPARSITY_n0.0_SWAP.txt"]

QUALITY_TEXT = [f"PC_{DATASET}_noest_SPARSITY_n0.0_SWAP.txt",
                f"OLC_{DATASET}_SPF_SPARSITY_n0.0_SWAP.txt",
                f"PPL_{DATASET}_SPF_SPARSITY_n0.0_SWAP.txt",
                f"PPLplus_{DATASET}_SPF_SPARSITY_n0.0_SWAP.txt",
                f"PPL_{DATASET}_TPF_SPARSITY_n0.0_SWAP.txt",
                f"PPLplus_{DATASET}_TPF_SPARSITY_n0.0_SWAP.txt"]

QUALITY_SPARSITY = ["1.0", "0.5","0.25", "0.1"]
QUALITY_METIC = ["SSIM","PSNR","MAE"]

# Graph configuraiton
LINE_COLOR = ['#d62728','#1f77b4','#ff7f0e','#2ca02c','#9467bd','#bcbd22']
ERROR_TYPE = ["Rotation", "Translation"]
DOT_SIZE = [1.0, 0.5 ,0.25, 0.1, 0.05]


def VERIFY_TEXT_FILE(filename, result_type):
    split_filename = filename.split("_")

    if result_type == "Quality":
        line_type = split_filename[1]
        dataset_name = "_".join(split_filename[2:-4])
    
    else:
        line_type = split_filename[0]
        dataset_name = "_".join(split_filename[1:-4])

    est_type = split_filename[-4]

    sparsity_level = float(split_filename[-3][2:])
    noise_level = float(split_filename[-2][1:])
    swap_level = float(split_filename[-1][2:-4])

    base_dir = os.path.abspath(os.curdir)
    dataset_dir = Variable.getDatasetName(dataset_name)
    
    filepath = os.path.join(base_dir, 'output', dataset_dir, dataset_name, result_type, filename)

    if not os.path.isfile(filepath):
        raise Exception("No file exists", filepath)

    # Pose ECDF를 그릴 때는 swap이 모두 0이어야함. 
    if result_type == "PoseAccuracy" and swap_level != 0:
        raise Exception("Pose Accuracy must have swap level 0. Provided ", swap_level)

    # L2Precon ECDF 그릴 때는 swap이 모두 0이어야함. 
    if result_type == "L2Precon" and swap_level != 0:
        raise Exception("L2Precon must have swap level 0. Provided ", swap_level)
    
    if result_type == "Quality":
        # OLC = swap이 0일 수 밖에 없음. 
        if line_type == "OLC" and swap_level != 0:
            raise Exception("Quality: OLC must have swap level 0. Provided ", swap_level)

        # Quality를 그릴 때 PPL, PPLPLUS는 swap이 무조건 0.5이어야한다. 
        if line_type in ["PPL", "PPLplus"] and swap_level != 0.5:
            raise Exception("Quality: ", line_type, "must have swap level 0.5. Provided", swap_level)



def getPoseQualityText(text_file, sparsity_level,q_type):
    if "SPARSITY" not in text_file:
        raise Exception("Curve Graph: Check QUALITY_TEXT list. Wrong syntax. No SPARSITY present.")
                
    if "SWAP" not in text_file:
        raise Exception("Curve Graph: Check QUALITY_TEXT list. Wrong syntax. No SWAP present.")

    if "SPF" not in text_file and "TPF" not in text_file and "noest" not in text_file:
        raise Exception("Curve Graph: Check QUALITY_TEXT list. Wrong syntax. Invalid est type.", est_type)

    split_filename = text_file.split("_")
    line_type = split_filename[0]

    text_file = text_file.replace("SPARSITY", "sp" + str(sparsity_level))

    pose_text_file = text_file.replace("SWAP", "sw"+str(0))
    est_type = ""
    if "SPF" in pose_text_file:
        pose_text_file = pose_text_file.replace("SPF", "NA")
        est_type = "SPF"
    elif "TPF" in pose_text_file:
        pose_text_file = pose_text_file.replace("TPF", "NA")
        est_type = "TPF"
    elif "noest" in pose_text_file:
        pose_text_file = pose_text_file.replace("noest", "NA")
        est_type = "noest"
    
    swap_ratio = None
    if line_type in ["PC", "OLC"]:
        swap_ratio = str(0)

    elif line_type in ["PPL", "PPLplus"]:
        swap_ratio = str(0.5)

    else:
        raise Exception("Wrong Line Type", line_type)
    
    quality_text_file = text_file.replace("SWAP", "sw"+swap_ratio)
    quality_text_file = f"{q_type}_{quality_text_file}"

    return pose_text_file, quality_text_file, est_type