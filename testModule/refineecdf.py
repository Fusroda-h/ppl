import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt

cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath("."))

from static import graph_var
from static import variable
from domain.loader.result_obj import ResultObject

def rfcheck_drawPoseAccuracyCDF(pose_result_obj_lst, output_path):
    for errtype in graph_var.ERROR_TYPE:
        fig, ax = plt.subplots(figsize=(8,4))
        sparsity_level_history = []
        noise_level_history = []
        for i, pose_obj in enumerate(pose_result_obj_lst):
            rf_option = 'rf_' if i//3==0 else 'x_rf_'
            print(f"Drawing Pose Estimation Graph: {errtype} || {pose_obj.dataset_name} - {pose_obj.line_type} | Sparsity: {pose_obj.sparsity_level}  Noise : {pose_obj.noise_level}")
            sparsity_level_history.append(str(pose_obj.sparsity_level))
            noise_level_history.append(str(pose_obj.noise_level))
            if errtype.lower() == "rotation":
                unit = "deg"
                errs = pose_obj.r_error
            else:
                unit = "m"
                errs = pose_obj.t_error

            if not errs:
                raise Exception(f"No result is available {pose_obj.dataset_name} - {pose_obj.line_type} | Sparsity: {pose_obj.sparsity_level}  Noise : {pose_obj.noise_level}")
            
            bins = np.arange(0, max(errs), 0.00001)
            hist_e, bins = np.histogram(errs, bins, density=True)
            cum_e = np.cumsum(hist_e)*(bins[1]-bins[0])

            legend = rf_option+pose_obj.line_type + "-" + str(pose_obj.sparsity_level)

            ax.plot(bins[:-1], cum_e, label=legend, c=graph_var.LINE_COLOR[i])
            ax.set_xlim([0, max(errs)/2])
            # ax.set_xlim([0, 1.5])
            ax.set_ylim([0,1.0])
            ax.set_yticks(np.linspace(0,1.0,11))
            ax.set_yticklabels([f'{n*10}%' for n in range(11)])
            axisfontlabel = {"fontsize":16, "color":"black"}
            plt.xlabel(f'{errtype} errors ({unit})', fontdict = axisfontlabel)
            plt.ylabel('ECDF', fontdict = axisfontlabel)
        ax.legend(fontsize=11,loc='lower right')#,'lower left'][numd])#, bbox_to_anchor=(1, 0.5))
        fig.tight_layout()
        os.makedirs(os.path.join(cur_dir,'output'),exist_ok=True)
        fig.savefig(os.path.join(cur_dir,'output',f'test_{pose_obj.dataset_name}_{errtype}.png'))
           
def findTextPath(text_name, result_type, ro):
    base_dir = os.path.abspath(os.curdir)
    dataset_dir = variable.getDatasetName(ro.dataset_name)
    dataset_name = ro.dataset_name
    filename = text_name
    
    filepath = os.path.join(base_dir,'output', dataset_dir, dataset_name, result_type, text_name)
    if os.path.isfile(filepath):
        print(f"Processing {dataset_dir} : {dataset_name} : {filename} \n")
    
    else:
        raise Exception("No file exists", filepath)
    
    return base_dir, dataset_dir, dataset_name, text_name
 
def rfcheck_load_result_obj(_pose, _recover, _quality):
    result_obj = []

    if _pose:
        for i in range(2):
            for pose_text in graph_var.POSE_TEXT:
                ro = ResultObject(pose_text)
                base_dir, dataset_dir, dataset_name, filename = findTextPath(pose_text, "PoseAccuracy", ro)
                
                if i==0:
                    ro.loadPoseResult(os.path.join(base_dir, "output", dataset_dir, dataset_name, "PoseAccuracy","test","refined", filename))
                if i==1:
                    ro.loadPoseResult(os.path.join(base_dir, "output", dataset_dir, dataset_name, "PoseAccuracy","test","notrefined", filename))

                result_obj.append(ro)
        
        return result_obj
            

result_obj = rfcheck_load_result_obj(True, False, False)
rfcheck_drawPoseAccuracyCDF(result_obj, cur_dir)
