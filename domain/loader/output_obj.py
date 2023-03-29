import os 

class PoseAccuracyLoader():
    def __init__(self, base_dir, dataset_dir, dataset_name, filename):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.filename = filename

        self.line_type = filename.split("_")[0]
        self.sparsity_level = float(filename.split("_")[-2])
        self.noise_level = float(filename.split("_")[-1][:-4])

        self.r_error = list()
        self.t_error = list()
        self.qvec = list()
        self.tvec = list()
        self.timesec = list()
        
        self.r_error_mean = 0
        self.r_error_median = 0
        self.t_error_mean = 0
        self.t_error_median = 0

        self.effective_query_imgs = 0
        self.total_query_imgs = 0

        with open(os.path.join(base_dir, "output", dataset_dir, dataset_name, "PoseAccuracy", filename), "r") as file_data:
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
