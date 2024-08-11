import torch

class Fog:
    def __init__(self, id, eids):
        self.id = id
        self.eids = eids
        self.edges = []
        self.fog_model = None
        self.edge_models = []  # To store models received from edges
    
    def edge_register(self, edge):
        self.edges.append(edge)

    def refresh_fognode(self):
        # Reinitialize or refresh the fog node, if necessary
        pass

    def receive_from_edge(self, edge_model):
        # Store the model received from the edge if it's not None
        if edge_model is not None:
            self.edge_models.append(edge_model)
        else:
            print(f"Warning: Received None model from an edge at Fog {self.id}")

    def aggregate(self, args):
        # Aggregate the models or data from the edge servers
        if self.edge_models:
            self.fog_model = self.average_models(self.edge_models)
        else:
            print(f"Warning: No models to aggregate for Fog {self.id}")
        self.edge_models = []  # Clear the list after aggregation
    
    def average_models(self, models):
        if not models:
            print("Warning: No models passed for aggregation")
            return None
        
        # Ensure all models are on the same device
        device = next(models[0].parameters()).device
        
        # Initialize a dictionary to store the sum of parameters
        sum_params = {}
        
        for model in models:
            for name, param in model.named_parameters():
                if name not in sum_params:
                    sum_params[name] = torch.zeros_like(param.data)
                sum_params[name] += param.data
        
        # Create a new model with the same structure as the input models
        avg_model = type(models[0])(*models[0].__init_args__, **models[0].__init_kwargs__)
        avg_model = avg_model.to(device)
        
        # Set the averaged parameters
        with torch.no_grad():
            for name, param in avg_model.named_parameters():
                param.data = sum_params[name] / len(models)
        
        return avg_model
    
    def send_to_cloudserver(self, cloud):
        # Send the aggregated model or data to the cloud
        if self.fog_model is not None:
            cloud.receive_from_fog(self.fog_model)
        else:
            print(f"Warning: No model to send from Fog {self.id} to Cloud")

    def send_to_edge(self, edge):
        # Send data/model from fog to edge
        if self.fog_model is not None:
            edge.receive_from_fog(self.fog_model)
        else:
            print(f"Warning: No model to send from Fog {self.id} to Edge {edge.id}")
