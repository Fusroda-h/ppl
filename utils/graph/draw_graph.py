import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter,NullFormatter 
import numpy as np
import sys
import os

sys.path.append(os.path.abspath("."))

from static import GraphVAR
from static import Variable


def drawPoseAccuracyCDF(pose_result_obj_lst, output_path):
    for errtype in GraphVAR.ERROR_TYPE:
        fig, ax = plt.subplots(figsize=(8,4))
        sparsity_level_history = []
        noise_level_history = []
        for i, pose_obj in enumerate(pose_result_obj_lst):
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

            legend = pose_obj.line_type + "-" + str(pose_obj.sparsity_level)

            ax.plot(bins[:-1], cum_e, label=legend, c=GraphVAR.LINE_COLOR[i])
            ax.set_xlim([0, max(errs)/2])
            ax.set_ylim([0,1.0])
            ax.set_yticks(np.linspace(0,1.0,11))
            ax.set_yticklabels([f'{n*10}%' for n in range(11)])
            axisfontlabel = {"fontsize":16, "color":"black"}
            plt.xlabel(f'{errtype} errors ({unit})', fontdict = axisfontlabel)
            plt.ylabel('ECDF', fontdict = axisfontlabel)

        ax.legend(fontsize=11,loc='lower right')#,'lower left'][numd])#, bbox_to_anchor=(1, 0.5))
        os.makedirs(os.path.join(output_path, "ECDF", "PoseCDF"), exist_ok=True)
        fig.tight_layout()
        if GraphVAR.REFINE_OPTION:
            filename = f'{pose_obj.dataset_name}_rf_{errtype}_sp{"_".join(sparsity_level_history)}_n{"_".join(noise_level_history)}.png'
        else:
            filename = f'{pose_obj.dataset_name}_norf_{errtype}_sp{"_".join(sparsity_level_history)}_n{"_".join(noise_level_history)}.png'
        fig.savefig(os.path.join(output_path, "ECDF", "PoseCDF", filename))

        print("Pose ECDF Saved \n")
        # plt.show()


def drawPoints3DCDF(points_result_obj_lst, output_path):
    fig, ax = plt.subplots(figsize=(7,4))
    dataset_name_history = []
    sparsity_level_history = []
    noise_level_history = []
    swap_level_history = []
    line_type_history = []
    est_type_history = []

    for i, pts_obj in enumerate(points_result_obj_lst):
        print(f"Drawing Points ECDF: {pts_obj.dataset_name} - {pts_obj.line_type} | Sparsity: {pts_obj.sparsity_level}  Noise : {pts_obj.noise_level}  Estimator : {pts_obj.est_type}")
        if Variable.getDatasetName(pts_obj.dataset_name) == "cambridge":
            numd = 0
        else:
            numd = 1

        dataset_name_history.append(pts_obj.dataset_name)
        sparsity_level_history.append(str(pts_obj.sparsity_level))
        noise_level_history.append(str(pts_obj.noise_level))
        swap_level_history.append(str(pts_obj.swap_level))
        line_type_history.append(pts_obj.line_type)
        est_type_history.append(pts_obj.est_type)

        scale = Variable.getScale(pts_obj.dataset_name)
        scaled_errs = pts_obj.recover_error * scale
        bins = np.linspace(0, max(scaled_errs), 20000)
        hist_e, bins = np.histogram(scaled_errs, bins, density=True)
        cum_e = np.cumsum(hist_e)*(bins[1]-bins[0])

        ax.plot(bins[:-1],cum_e,label=f"{pts_obj.line_type}({pts_obj.est_type})", c=GraphVAR.LINE_COLOR[i])
        ax.set_xlim([[-0.05,-0.005][numd],[3.0,0.3][numd]])
        ax.set_ylim([[-0.05,-0.005][numd],1.0])
        ax.set_xticks(np.linspace(0,[3.0,0.3][numd],7))
        ax.set_yticks(np.linspace(0,1.0,6))
        axisfontlabel = {"fontsize":16, "color":"black"}
        plt.xlabel('3D Recon. errors (m)', fontdict = axisfontlabel)
        plt.ylabel('CDF', fontdict = axisfontlabel)
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax.grid(True)
        ax.legend(fontsize=11,loc='lower right')#,'lower left'][numd])#, bbox_to_anchor=(1, 0.5))

    os.makedirs(os.path.join(output_path, "ECDF", "L2Precon"), exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, "ECDF", "L2Precon", f'{"_".join(dataset_name_history)}|sp{"_".join(sparsity_level_history)}|n{"_".join(noise_level_history)}|sw{"_".join(swap_level_history)}|{"_".join(line_type_history)}|{"_".join(est_type_history)}.png'))

    print("Reconstructed 3D Point ECDF Saved \n")
    # plt.show()



def drawQualityCurve(quality_result_obj_lst, output_path):
    for j,q_type in enumerate(GraphVAR.QUALITY_METIC):
        for err_type in GraphVAR.ERROR_TYPE:
            fig, ax = plt.subplots(figsize=(6.5,3))
            line_type_history = []
            est_type_history = []
            lines = []
            for i, result_lst in enumerate(quality_result_obj_lst[j]):
                markers = ['o','^','s','*','P','D']
                x = []
                y = []
                line_type_history.append(result_lst[0].line_type)
                est_type_history.append(result_lst[0].est_type)
                print(f'{err_type} x {q_type} Dataset: {result_lst[0].dataset_name}({result_lst[0].line_type}) Sparsity: {result_lst[0].sparsity_level} Noise: {result_lst[0].noise_level} Swap: {result_lst[0].swap_level}')
                for quality_obj in result_lst:
                    if err_type == "Rotation":
                        y.append(quality_obj.r_error_mean)
                        ylabels = 'Rot. error(log)[deg]'

                    elif err_type == "Translation":
                        # Scale is already applied when saving the pose accuracy text file
                        y.append(quality_obj.t_error_mean)
                        ylabels = 'Trans. error(log)[m]'
                
                    x.append(quality_obj.img_quality_mean)
                
                label_name = f'{quality_obj.line_type}({quality_obj.est_type})'
                if quality_obj.line_type == 'PPLplus':
                    label_name = f'PPL+({quality_obj.est_type})'
                if quality_obj.est_type=='noest':
                    label_name = f'{quality_obj.line_type}'
                    
                ax.plot(x, y, color=GraphVAR.LINE_COLOR[i], label=label_name)
                tmpline = Line2D([],[], color=GraphVAR.LINE_COLOR[i], label=label_name, marker=markers[i])
                lines.append(tmpline)
                for m in range(len(GraphVAR.QUALITY_SPARSITY)):
                    ax.scatter(x[m], y[m], s = GraphVAR.DOT_SIZE[m]*200, c=GraphVAR.LINE_COLOR[i], alpha=0.3, marker=markers[i])

            axisfontlabel = {"fontsize":18,"color":"black"}
            
            ##############################
            manolis_yticks = {'Rotation':[0.04,0.06,0.1,0.2,0.3],'Translation':[0.001,0.002,0.003,0.006]}
            manolis_xticks = {'MAE':[30,40,50,60],'PSNR':[9,10,12,15,16],'SSIM':[0.35,0.4,0.5,0.55]}
            ##############################
            
            ax.set_xlabel(q_type, fontdict = axisfontlabel)
            ax.set_ylabel(ylabels, fontdict = axisfontlabel)
            ax.set_yscale('log')
            ax.set_xscale('log')
            
            ##############################
            ax.set_xticks(manolis_xticks[q_type])
            ax.set_yticks(manolis_yticks[err_type])
            ax.set_xticklabels(manolis_xticks[q_type])
            ax.set_yticklabels(manolis_yticks[err_type])
            ax.xaxis.set_minor_locator(plt.NullLocator())
            ax.yaxis.set_minor_locator(plt.NullLocator())
            ##############################
            
            legend_properties = {'weight':'light'}
            ax.legend(handles=lines,fontsize=10,prop=legend_properties)

            fig.tight_layout()
            title = f'{err_type}_{q_type}'
            
            dirname = os.path.join(output_path, "Curve", "Quality",q_type, quality_obj.dataset_name)
            os.makedirs(dirname, exist_ok=True)
            filename = '_'.join([q_type,quality_obj.dataset_name,err_type,
                                f'n{quality_obj.noise_level}',
                                f'sw{quality_obj.swap_level}',
                                "|",
                                f'sp{"_".join(GraphVAR.QUALITY_SPARSITY)}',
                                "|",
                                "_".join(line_type_history),
                                "|",
                                "_".join(est_type_history)])+'.png'
            fig.savefig(os.path.join(dirname,filename))
            print("Image Quality x Pose Accuracy Curve Graph Saved: ", err_type, "\n")

            
