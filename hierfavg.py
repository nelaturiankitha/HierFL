import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter
from client import Client
from edge import Edge
from fog import Fog
from cloud import Cloud
from options import args_parser
from tqdm import tqdm
from models.initialize_model import initialize_model

def load_iot_data():
    start = time.time()

    df = pd.read_csv('IOT_TEST.csv')

    if df['type'].dtype == 'object':
        labelencoder = LabelEncoder()
        df['type'] = labelencoder.fit_transform(df['type'])

    X = df.drop(['ts', 'label', 'type'], axis=1).values
    y = df['type'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1337, stratify=y)

    transfer = StandardScaler()
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.transform(X_test)

    end = time.time()
    print("Time taken {}".format(end - start))

    return X_train, X_test, y_train, y_test

def sample_edges(fog, edges_per_fog, frac):
    num_samples = max(int(edges_per_fog * frac), 1)
    selected_eids = np.random.choice(fog.eids, num_samples, replace=False)
    return selected_eids

def create_dataloaders(X_train, X_test, y_train, y_test, batch_size=32):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def HierFAVG(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}') if args.cuda and torch.cuda.is_available() else "cpu"
    print(f'Using device {device}')
    
    FILEOUT = f"{args.dataset}_clients{args.num_clients}_edges{args.num_edges}_fog{args.num_fogs}_model_{args.model}"
    writer = SummaryWriter(comment=FILEOUT)
    
    X_train, X_test, y_train, y_test = load_iot_data()
    train_loader, test_loader = create_dataloaders(X_train, X_test, y_train, y_test, batch_size=args.batch_size)
    
    clients = []
    for i in range(args.num_clients):
        clients.append(Client(id=i, args=args, device=device, 
                              train_loader=train_loader, 
                              test_loader=test_loader))

    edges = []
    if args.enable_edge:
        cids = np.arange(args.num_clients)
        clients_per_edge = int(args.num_clients / args.num_edges)
        for i in range(args.num_edges):
            selected_cids = np.random.choice(cids, clients_per_edge, replace=False)
            cids = list(set(cids) - set(selected_cids))
            edges.append(Edge(id=i, cids=selected_cids))
            for cid in selected_cids:
                edges[i].client_register(clients[cid])
            edges[i].refresh_edgeserver()

    fogs = []
    if args.enable_fog:
        eids = np.arange(args.num_edges)
        edges_per_fog = int(args.num_edges / args.num_fogs)
        for i in range(args.num_fogs):
            selected_eids = np.random.choice(eids, edges_per_fog, replace=False)
            eids = list(set(eids) - set(selected_eids))
            fogs.append(Fog(id=i, eids=selected_eids))
            for eid in selected_eids:
                fogs[i].edge_register(edges[eid])
            fogs[i].refresh_fognode()

    cloud = Cloud()
    if args.enable_fog:
        for fog in fogs:
            cloud.fog_register(fog=fog)
    
    global_model = initialize_model(args, device)

    for num_comm in tqdm(range(args.num_communication)):
        if args.enable_fog:
            cloud.refresh_cloudserver()
            for fog in fogs:
                cloud.fog_register(fog=fog)
        
        for num_fogagg in range(args.num_fog_aggregation if args.enable_fog else 1):
            if args.enable_fog:
                fog_loss = [0.0] * args.num_fogs
                fog_sample = [0] * args.num_fogs
                correct_all = 0.0
                total_all = 0.0
                
                for i, fog in enumerate(fogs):
                    fog.refresh_fognode()
                    selected_eids = sample_edges(fog, edges_per_fog, args.frac)
                    for selected_eid in selected_eids:
                        fog.edge_register(edges[selected_eid])
                        edges[selected_eid].send_to_fog(fog)
                        # Call local_update with required arguments
                        edge_loss = edges[selected_eid].local_update(num_iter=args.num_local_update, device=device)
                        fog_loss[i] += edge_loss
                        edges[selected_eid].send_to_fog(fog)
                    fog.aggregate(args)
                    correct, total = all_edges_test(fog, edges, fog.eids, device)
                    correct_all += correct
                    total_all += total
                
                all_loss = sum([f_loss * f_sample for f_loss, f_sample in zip(fog_loss, fog_sample)]) / sum(fog_sample)
                avg_acc = correct_all / total_all
                writer.add_scalar(f'Partial_Avg_Train_loss', all_loss, num_comm * args.num_fog_aggregation + num_fogagg + 1)
                writer.add_scalar(f'All_Avg_Test_Acc_fogagg', avg_acc, num_comm * args.num_fog_aggregation + num_fogagg + 1)
        
        if args.enable_fog:
            for fog in fogs:
                fog.send_to_cloudserver(cloud)
            cloud.aggregate(args)
            for fog in fogs:
                cloud.send_to_fog(fog)
        
        for epoch in range(args.num_epochs):
            global_loss = 0
            for input_batch, label_batch in train_loader:
                loss = global_model.train(input_batch, label_batch)
                global_loss += loss
            
            avg_loss = global_loss / len(train_loader)
            writer.add_scalar('Global_Model_Train_Loss', avg_loss, num_comm * args.num_epochs + epoch)

            correct = 0
            total = 0
            for input_batch, label_batch in test_loader:
                outputs = global_model.evaluate(input_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += label_batch.size(0)
                correct += (predicted == label_batch).sum().item()
            
            accuracy = 100 * correct / total
            writer.add_scalar('Global_Model_Test_Acc', accuracy, num_comm * args.num_epochs + epoch)

    writer.close()
    print(f"The final global model accuracy is {accuracy}")

def main():
    args = args_parser()
    HierFAVG(args)

if __name__ == '__main__':
    main()
