import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier
import time
import torch
import copy
from tensorboardX import SummaryWriter
from client import Client
from edge import Edge
from cloud import Cloud
from datasets.get_data import get_dataloaders, show_distribution
from options import args_parser
from tqdm import tqdm

def load_iot_data():
    start = time.time()

    # Load the data from the CSV file
    df = pd.read_csv('IOT_TEST.csv')

    print("Original 'type' column values:")
    print(df['type'].value_counts())

    # Check if 'type' column contains string values
    if df['type'].dtype == 'object':
        labelencoder = LabelEncoder()
        df['type'] = labelencoder.fit_transform(df['type'])
    else:
        print("'type' column is already numeric")

    print("\nEncoded 'type' column values:")
    print(df['type'].value_counts())

    X = df.drop(['ts', 'label', 'type'], axis=1).values
    y = df['type'].values

    print("\nUnique values in y:")
    print(np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1337, stratify=y)

    transfer = StandardScaler()
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.transform(X_test)

    # Decision tree training and prediction
    model = DecisionTreeClassifier(random_state=1337)

    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    scores = cross_validate(model, X_train, y_train, scoring=scoring, cv=4, n_jobs=-1)

    print('\nscores:', scores)
    print("fit_time: %0.3f " % (scores['fit_time'].mean()))
    print("score_time: %0.3f " % (scores['score_time'].mean()))

    print("Accuracy (Testing): %0.4f " % (scores['test_accuracy'].mean()))
    print("Precision (Testing): %0.4f " % (scores['test_precision_macro'].mean()))
    print("Recall (Testing): %0.4f " % (scores['test_recall_macro'].mean()))
    print("F1-Score (Testing): %0.4f " % (scores['test_f1_macro'].mean()))

    end = time.time()
    print("Time taken {}".format(end - start))

def initialize_global_model():
    # Initialize global model for DecisionTreeClassifier
    return DecisionTreeClassifier(random_state=1337)


def HierFAVG(args):
    # Make experiments repeatable
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cuda_to_use = torch.device(f'cuda:{args.gpu}')
    device = cuda_to_use if torch.cuda.is_available() else "cpu"
    print(f'Using device {device}')
    
    FILEOUT = f"{args.dataset}_clients{args.num_clients}_edges{args.num_edges}_" \
              f"t1-{args.num_local_update}_t2-{args.num_edge_aggregation}" \
              f"_model_{args.model}iid{args.iid}edgeiid{args.edgeiid}epoch{args.num_communication}" \
              f"bs{args.batch_size}lr{args.lr}lr_decay_rate{args.lr_decay}" \
              f"lr_decay_epoch{args.lr_decay_epoch}momentum{args.momentum}"
    writer = SummaryWriter(comment=FILEOUT)
    
    # Load and preprocess data
    load_iot_data()
    
    # Create data loaders
    train_loader, test_loader = get_dataloaders(args)
    
    # Initialize clients and server
    clients = []
    for i in range(args.num_clients):
        client_train_loader = train_loader[i]  # Replace with the correct method to get client-specific loaders
        client_test_loader = test_loader[i]    # Replace with the correct method to get client-specific loaders
        clients.append(Client(id=i, args=args, device=device, 
                              train_loader=client_train_loader, 
                              test_loader=client_test_loader))
    
    # Initialize edge servers and assign clients
    edges = []
    cids = np.arange(args.num_clients)
    clients_per_edge = int(args.num_clients / args.num_edges)
    p_clients = [0.0] * args.num_edges
    
    if args.iid == -2:
        if args.edgeiid == 1:
            client_class_dis = get_client_class(args, clients)
            edges, p_clients = initialize_edges_iid(num_edges=args.num_edges,
                                                    clients=clients,
                                                    args=args,
                                                    client_class_dis=client_class_dis)
        elif args.edgeiid == 0:
            client_class_dis = get_client_class(args, clients)
            edges, p_clients = initialize_edges_niid(num_edges=args.num_edges,
                                                     clients=clients,
                                                     args=args,
                                                     client_class_dis=client_class_dis)
    else:
        for i in range(args.num_edges):
            selected_cids = np.random.choice(cids, clients_per_edge, replace=False)
            cids = list(set(cids) - set(selected_cids))
            edges.append(Edge(id=i, cids=selected_cids))
            [edges[i].client_register(clients[cid]) for cid in selected_cids]
            edges[i].refresh_edgeserver()

    # Initialize cloud server
    cloud = Cloud()
    [cloud.edge_register(edge=edge) for edge in edges]
    
    global_model = initialize_global_model()

    # Begin training
    for num_comm in tqdm(range(args.num_communication)):
        # Refresh cloud server and edge servers
        cloud.refresh_cloudserver()
        [cloud.edge_register(edge=edge) for edge in edges]
        
        for num_edgeagg in range(args.num_edge_aggregation):
            edge_loss = [0.0] * args.num_edges
            edge_sample = [0] * args.num_edges
            correct_all = 0.0
            total_all = 0.0
            
            for i, edge in enumerate(edges):
                edge.refresh_edgeserver()
                selected_cnum = max(int(clients_per_edge * args.frac), 1)
                selected_cids = np.random.choice(edge.cids, selected_cnum, replace=False, p=p_clients[i])
                for selected_cid in selected_cids:
                    edge.client_register(clients[selected_cid])
                for selected_cid in selected_cids:
                    edge.send_to_client(clients[selected_cid])
                    clients[selected_cid].sync_with_edgeserver()
                    client_loss = clients[selected_cid].local_update()
                    edge_loss[i] += client_loss
                    clients[selected_cid].send_to_edgeserver(edge)
                edge.aggregate(args)
                correct, total = all_clients_test(edge, clients, edge.cids, device)
                correct_all += correct
                total_all += total
            
            all_loss = sum([e_loss * e_sample for e_loss, e_sample in zip(edge_loss, edge_sample)]) / sum(edge_sample)
            avg_acc = correct_all / total_all
            writer.add_scalar(f'Partial_Avg_Train_loss',
                              all_loss,
                              num_comm * args.num_edge_aggregation + num_edgeagg + 1)
            writer.add_scalar(f'All_Avg_Test_Acc_edgeagg',
                              avg_acc,
                              num_comm * args.num_edge_aggregation + num_edgeagg + 1)
        
        # Cloud aggregation
        for edge in edges:
            edge.send_to_cloudserver(cloud)
        cloud.aggregate(args)
        for edge in edges:
            cloud.send_to_edge(edge)

        # Update global model
        global_model = cloud.get_global_model()

        # Test the global model
        v_test_loader = None  # Replace with appropriate validation/test data handling
        correct_all_v, total_all_v = fast_all_clients_test(v_test_loader, global_model, device)
        avg_acc_v = correct_all_v / total_all_v
        writer.add_scalar(f'All_Avg_Test_Acc_cloudagg_Vtest',
                          avg_acc_v,
                          num_comm + 1)
    
    writer.close()
    print(f"The final virtual acc is {avg_acc_v}")



def main():
    args = args_parser()
    HierFAVG(args)

if __name__ == '__main__':
    main()
