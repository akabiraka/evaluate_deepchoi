import sys
sys.path.append("../evaluate_deepchoi")

import eval_metrics_1 as eval_metrics
from GO import Ontology
import pickle_utils
from config import Config

config = Config()

species = config.species #"yeast"
GOname= config.GO #"BP"
data_generation_process = config.data_generation_process #"time_delay_no_knowledge"

# for evaluation purposes
go_rels_val = Ontology('ak_data/go.obo', with_rels=True)
term_to_idx_dict = terms_dict = pickle_utils.load_pickle(f"ak_data/{data_generation_process}/{GOname}/studied_terms.pkl")
idx_to_term_dict_val = {i:term for term, i in term_to_idx_dict.items()}
terms_set_val = set(term_to_idx_dict.keys())

train_dataset_val = pickle_utils.load_pickle(f"ak_data/{data_generation_process}/{GOname}/train.pkl") # list of uniprot_id, set([terms])
print(f"Length of train set: {len(train_dataset_val)}")

val_set = pickle_utils.load_pickle(f"ak_data/{data_generation_process}/{GOname}/val.pkl")
print(f"Length of eval set: {len(val_set)}")


val_annotations = [annots for uniprot_id, annots in val_set]
train_annotations_val = [annots for uniprot_id, annots in train_dataset_val]
go_rels_val.calculate_ic(train_annotations_val + val_annotations)


print("Log: finished computing ic")

def run_val(pred_scores):
    tmax, fmax, smin, aupr = eval_metrics.Fmax_Smin_AUPR(pred_scores, val_set, idx_to_term_dict_val, go_rels_val, terms_set_val, val_annotations)
