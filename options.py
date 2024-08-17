import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()
    # Dataset and model
    parser.add_argument('--dataset', type=str, default='IOT', help='Dataset name')
    parser.add_argument('--model', type=str, default='simple_nn', help='Model type')
    parser.add_argument('--input_size', type=int, default=20, help='Input feature size')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of output classes')

    # Neural network training hyperparameters
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size when trained on client')
    parser.add_argument('--num_communication', type=int, default=1, help='Number of communication rounds with the cloud server')
    parser.add_argument('--num_local_update', type=int, default=1, help='Number of local updates (tau_1)')
    parser.add_argument('--num_edge_aggregation', type=int, default=1, help='Number of edge aggregation (tau_2)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of the SGD when trained on client')
    parser.add_argument('--lr_decay', type=float, default=1, help='LR decay rate')
    parser.add_argument('--lr_decay_epoch', type=int, default=1, help='LR decay epoch')
    parser.add_argument('--momentum', type=float, default=0, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay rate')
    parser.add_argument('--verbose', type=int, default=0, help='Verbose for print progress bar')

    # Federated learning settings
    parser.add_argument('--iid', type=int, default=0, help='Distribution of the data, 1 (IID), 0 (non-IID), -2 (one-class)')
    parser.add_argument('--edgeiid', type=int, default=0, help='Distribution of the data under edges, 1 (edgeiid), 0 (edgeniid) (used only when iid = -2)')
    parser.add_argument('--frac', type=float, default=1, help='Fraction of participated clients')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of all available clients')
    parser.add_argument('--num_edges', type=int, default=1, help='Number of edges')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--dataset_root', type=str, default='data', help='Dataset root folder')
    parser.add_argument('--show_dis', type=int, default=0, help='Whether to show distribution')
    parser.add_argument('--classes_per_client', type=int, default=2, help='Under artificial non-iid distribution, the classes per client')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to be selected, 0, 1, 2, 3')

    parser.add_argument('--mtl_model', default=0, type=int)
    parser.add_argument('--local_model', default=0, type=int)

    parser.add_argument('--num_fogs', type=int, default=2, help="Number of fog nodes")
    parser.add_argument('--num_fog_aggregation', type=int, default=2, help="Fog aggregation times")
    parser.add_argument('--cuda', action='store_true', help="Use CUDA")
    
    # Flags to enable or disable fog and edge nodes
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--global_model', action='store_true', help="Use global model")
    parser.add_argument('--enable_edge', action='store_true', help="Enable edge servers")
    parser.add_argument('--enable_fog', action='store_true', help="Enable fog nodes")

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args
