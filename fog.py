import torch
import copy

class Fog:
    def __init__(self, id, eids, model_class, *model_args, **model_kwargs):
        self.id = id
        self.eids = eids
        self.edges = []
        self.fog_model = model_class(*model_args, **model_kwargs)
        self.edge_models = []
    
    # ... (other methods remain the same)

    def receive_from_edge(self, edge_model):
        if edge_model is not None:
            self.edge_models.append(edge_model)
        else:
            print(f"Warning: Received None model from an edge at Fog {self.id}")

    def aggregate(self, args):
        if self.edge_models:
            self.fog_model = self.average_models(self.edge_models)
        else:
            print(f"Warning: No models to aggregate for Fog {self.id}")
        self.edge_models = []
    
    def average_models(self, models):
        if not models:
            print("Warning: No models passed for aggregation")
            return None
        
        avg_model = copy.deepcopy(self.fog_model)
        avg_dict = avg_model.state_dict()
        for key in avg_dict.keys():
            avg_dict[key] = torch.stack([model.state_dict()[key] for model in models]).mean(0)
        avg_model.load_state_dict(avg_dict)
        return avg_model
    
    def send_to_cloudserver(self, cloud):
        if self.fog_model is not None:
            cloud.receive_from_fog(self.fog_model)
        else:
            print(f"Warning: No model to send from Fog {self.id} to Cloud")

    def send_to_edge(self, edge):
        if self.fog_model is not None:
            edge.receive_from_fog(self.fog_model)
        else:
            print(f"Warning: No model to send from Fog {self.id} to Edge {edge.id}")

    # ... (other methods remain the same)