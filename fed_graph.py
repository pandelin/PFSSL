import copy
import math
import os.path
import random
import pandas as pd
from fedgraph.config import get_args
from fedgraph.util import *
# from model import simplecnn, textcnn
from models.deeplabv3 import UNet
from models.unet34 import Model
# from prepare_data import get_dataloader
from fedgraph.dataset import get_dataloader
from fedgraph.attack import *
# from fedgraph.train_fl_sup import local_train
from fedgraph.train_fl import local_train
# from mt.train_mt import local_train
import hashlib
import importlib
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dynamic_import(module_name, function_name):
    module = importlib.import_module(module_name)
    return getattr(module, function_name)

def main():
    args, cfg = get_args()
    print(args)
    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]  # node list
    party_list_rounds = []  # [[],[]]  node list round
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    benign_client_list = random.sample(party_list, int(args.n_parties * (1 - args.attack_ratio)))  # random choose node
    benign_client_list.sort()
    print(f'>> -------- Benign clients: {benign_client_list} --------')

    # get dataset
    local_dls, test_dl = get_dataloader(args.dataset_type, args.data_path1, args.data_path2, args.image_size,
                                        args.batch_size, args.data_parties, args.labeled_ratio)

    model = None
    if args.dataset == 'cifar10':
        model = None
    elif args.dataset == 'liver':
        model = UNet(in_channels=3, out_channels=2)
        # model = Model(n_channels=3, n_classes=2)

    global_model = model
    global_parameters = global_model.state_dict()  # get model params
    local_models = []
    local_train_models = []
    best_test_dice_list, best_test_iou_list = [], []
    best_test_dice_avg_list, best_test_iou_avg_list = [], []
    dw = []  # local model's param init none
    for i in range(cfg['client_num']):
        local_model = model
        local_models.append(local_model)
        local_train_models.append(local_model)
        dw.append({key: torch.zeros_like(value) for key, value in local_models[i].named_parameters()})
        # best_test_dice_list.append(0)
        # best_test_iou_list.append(0)

    # get ssl method
    ssl = args.ssl
    module_choice = ""
    if ssl == "mt":
        module_choice = "mt.train_mt"
    elif ssl == "bcp":
        module_choice = "mt.train_bcp"
    elif ssl == "sup":
        module_choice = "mt.train_sup"
    elif ssl == "ours":
        module_choice = "fedgraph.train_fl"
    elif ssl == "dhc":
        module_choice = "mt.train_dhc_new"
    local_train = dynamic_import(module_choice, "local_train")

    # graph
    graph_matrix = torch.ones(len(local_models), len(local_models)) / (len(local_models) - 1)  # Collaboration Graph
    graph_matrix[range(len(local_models)), range(len(local_models))] = 0  # init

    for net in local_models:
        net.load_state_dict(global_parameters)  # local global model

    cluster_model_vectors = {}  # aggregate model by GCN
    args.project = args.project + '_{}_{}_{}/'.format(args.backbone, args.ssl, str(args.seed))
    graph_csv_path = os.path.join(args.project, 'graph_data')
    os.makedirs(graph_csv_path, exist_ok=True)
    graph_matrix_list = []
    for round in range(cfg["comm_round"]):
        party_list_this_round = party_list_rounds[round]  # get node list
        if args.sample_fraction < 1.0:
            print(f'>> Clients in this round : {party_list_this_round}')
        nets_this_round = {k: local_models[k] for k in party_list_this_round}  # this round model list
        nets_param_start = {k: copy.deepcopy(local_models[k]) for k in party_list_this_round}  # this round none-model list
        # start training
        best_test_dice_list, best_test_iou_list = local_train(args, round, nets_this_round, local_train_models,
                             cluster_model_vectors, local_dls)

        total_data_points = sum([len(label)+len(un_label) for label, un_label, test in local_dls])  # all node list
        fed_avg_freqs = {k: (len(label)+len(un_label)) / total_data_points for k, (label, un_label, test)  in
                         enumerate(local_dls)}  # average data

        # manipulate_gradient(args, None, nets_this_round, benign_client_list, nets_param_start)

        # cal graph matrix
        # Graph Matrix is not normalized yet  tensor(2,2)
        graph_matrix = update_graph_matrix_neighbor(graph_matrix, nets_this_round, global_parameters, dw, fed_avg_freqs,
                                                    best_test_dice_list, args.alpha, args.difference_measure)
        graph_matrix_list = [graph_matrix] + graph_matrix_list
        # aggregate historical information
        graph_matrix_time = aggregate_adj_matrices(graph_matrix_list, args.time_alpha)
        # Aggregation weight is normalized here
        cluster_model_vectors = aggregation_by_graph(cfg, graph_matrix_time, nets_this_round, global_parameters, fed_avg_freqs)
        best_test_dice_list_avg = np.array(best_test_dice_list).mean()
        best_test_iou_list_avg = np.array(best_test_iou_list).mean()
        best_test_dice_avg_list.append(best_test_dice_list_avg)
        best_test_iou_avg_list.append(best_test_iou_list_avg)

        print(f'>> (Current) Round {round} | best_val_iou_list_avg: {best_test_iou_list_avg:.5f}, '+
              f' best_val_dice_list_avg: {best_test_dice_list_avg:.5f}, ')
        print('-' * 80)

        df = pd.DataFrame(graph_matrix.numpy())
        # save DataFrame to CSV
        df.to_csv(os.path.join(graph_csv_path, f'graph_matrix_round_{round}.csv'), index=False)

    avg_list = {'dice':best_test_dice_avg_list, 'iou':best_test_iou_avg_list}
    avg_df = pd.DataFrame(avg_list)
    # save DataFrame to CSV
    avg_df.to_csv(os.path.join(graph_csv_path, f'best_avg.csv'), index=False)



if __name__ == "__main__":
    main()
