class Edge:
    def __init__(self, id, cids):
        self.id = id
        self.cids = cids
        self.clients = []
        self.edge_model = None
    
    def client_register(self, client):
        self.clients.append(client)

    def refresh_edgeserver(self):
        # Reinitialize or refresh the edge server, if necessary
        pass

    def aggregate(self, args):
        # Aggregate the models or data from the clients
        models = [client.get_model() for client in self.clients]
        self.edge_model = self.average_models(models)
    
    def average_models(self, models):
        # Implement model averaging (or any other aggregation technique)
        return sum(models) / len(models)
    
    def send_to_fog(self, fog):
        # Send the aggregated model or data to the fog
        fog.receive_from_edge(self.edge_model)
    
    def send_to_client(self, client):
        # Send data/model from edge to client
        client.receive_from_edge(self.edge_model)
    
    # def local_update(self, num_iter, device):
    #     # Perform local updates at the edge
    #     # Example: call local training on all clients
    #     for client in self.clients:
    #         client.local_update(num_iter,device)

    def local_update(self, num_iter, device):
        total_loss = 0
        for client in self.clients:
            client_loss = client.local_update(num_iter, device)
            total_loss += client_loss
        return total_loss / len(self.clients)

    def receive_from_fog(self, fog_model):
        # Receive updated model or data from the fog
        self.edge_model = fog_model

    def receive_model_update_from_fog(self, fog):
        # Get the model update from the fog
        self.receive_from_fog(fog.fog_model)
