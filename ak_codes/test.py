import sys
sys.path.append("../deepchoi")

import eval_metrics_1 as eval_metrics
from GO import Ontology
import pickle_utils
from config import Config

config = Config()

species = config.species #"yeast"
GOname= config.GO #"BP"
data_generation_process = config.data_generation_process #"time_delay_no_knowledge"


# for evaluation purposes
go_rels = Ontology('ak_data/go.obo', with_rels=True)
term_to_idx_dict = terms_dict = pickle_utils.load_pickle(f"ak_data/{data_generation_process}/{GOname}/studied_terms.pkl")
idx_to_term_dict = {i:term for term, i in term_to_idx_dict.items()}
terms_set = set(term_to_idx_dict.keys())

train_dataset = pickle_utils.load_pickle(f"ak_data/{data_generation_process}/{GOname}/train.pkl") # list of uniprot_id, set([terms])
print(f"Length of train set: {len(train_dataset)}")

test_set = pickle_utils.load_pickle(f"ak_data/{data_generation_process}/{GOname}/test.pkl")
print(f"Length of eval set: {len(test_set)}")


test_annotations = [annots for uniprot_id, annots in test_set]
train_annotations = [annots for uniprot_id, annots in train_dataset]
go_rels.calculate_ic(train_annotations + test_annotations)

print("Log: finished computing ic")



def run_test(pred_scores):
    tmax, fmax, smin, aupr = eval_metrics.Fmax_Smin_AUPR(pred_scores, test_set, idx_to_term_dict, go_rels, terms_set, test_annotations)
