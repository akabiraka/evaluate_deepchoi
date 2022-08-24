import sys
sys.path.append("../evaluate_deepchoi")
import torch

class Config(object):
    def __new__(cls):
        # Singleton class instance
        if not hasattr(cls, "instance"):
            cls.instance = super(Config, cls).__new__(cls)
        return cls.instance

    def __init__(self, 
                 species="yeast", 
                 GO="BP", 
                 ) -> None:
        super(Config, self).__init__()

        self.species = species #"yeast"
        self.GO = GO #["BP", "CC", "MF"]
        self.data_generation_process = "time_series_no_knowledge" # time_series_no_knowledge, time_delay_no_knowledge, random_split_leakage
        
    def get_model_name(self, task="DeepChoi") -> str:
        return f"{task}_{self.species}_{self.GO}_{self.data_generation_process}"


# config = Config(max_len=2708)
# print(config.get_model_name())


# separate decoder config 
# Decoder related configs
# self.n_classes = n_classes

