import os 
from static.Variable import getDatasetName
import utils.read_write_model as model
import numpy as np

class ResultObject():
    def __init__(self, filename):
        self.filename = filename.strip()

        split_filename = filename.split("_")
        self.line_type = split_filename[0]
        self.dataset_name = "_".join(split_filename[1:-4])
        self.est_type = split_filename[-4]

        self.sparsity_level = float(split_filename[-3][2:])
        self.noise_level = float(split_filename[-2][1:])
        self.swap_level = float(split_filename[-1][2:-4])

        self.dataset_dir = getDatasetName(self.dataset_name)


    def loadPoseResult(self, filename):
        self.r_error_mean = 0
        self.r_error_median = 0
        self.t_error_mean = 0
        self.t_error_median = 0

        self.effective_query_imgs = 0
        self.total_query_imgs = 0

        self.r_error = list()
        self.t_error = list()
        self.qvec = list()
        self.tvec = list()
        self.timesec = list()

        with open(filename, "r") as file_data:
            stats = file_data.readline().split(",")
            self.r_error_mean = float(stats[0].rsplit(":")[-1])
            self.t_error_mean = float(stats[1].rsplit(":")[-1])
            self.r_error_median = float(stats[2].rsplit(":")[-1])
            self.t_error_median = float(stats[3].rsplit(":")[-1])

            query_info = file_data.readline().rstrip().rsplit("/")
            self.effective_query_imgs = int(query_info[0][-1])
            self.total_query_imgs = int(query_info[-1])

            file_data.readline()
            for line in file_data:
                accuracy_result = line.rstrip().split(" ")
                _qvec = list(map(float, accuracy_result[:4]))
                _tvec = list(map(float, accuracy_result[4:7]))
                _r_error = float(accuracy_result[7])
                _t_error = float(accuracy_result[8])
                _timesec = float(accuracy_result[9])

                self.qvec.append(_qvec)
                self.tvec.append(_tvec)
                self.r_error.append(_r_error)
                self.t_error.append(_t_error)
                self.timesec.append(_timesec)


    def loadReconResult(self, filename, gt_img_path):
        self.recover_error = list()
        self.est_pts = model.read_points3D_text(filename)
        self.gt_pts = model.read_points3D_text(gt_img_path)

        for k, v in self.est_pts.items():
            self.recover_error.append(np.linalg.norm(np.subtract(self.gt_pts[k].xyz, self.est_pts[k].xyz)))

        self.recover_error = np.array(self.recover_error)
        
        print("Recovered 3D Points Error :", np.mean(self.recover_error), "\n")


    def loadQualityResult(self, filename):
        self.img_quality = list()
        self.img_quality_mean = None
        self.img_quality_median = None

        with open(filename, "r") as file_data:
            f1 = file_data.readline().strip()
            f2 = file_data.readline().strip()

            self.img_quality_mean = float(f1.split(": ")[-1])
            self.img_quality_median = float(f2.split(": ")[-1])

            for line in file_data:
                self.img_quality.append(float(line.strip()))

        self.img_quality = np.array(self.img_quality)
        