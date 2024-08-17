import torch
import copy

class Edge:
    def __init__(self, id, cids, model_class, *model_args, **model_kwargs):
        self.id = id
        self.cids = cids
        self.clients = []
        self.edge_model = model_class(*model_args, **model_kwargs)
    
    # ... (other methods remain the same)

    def average_models(self, models):
        if not models:
            return None

        avg_model = copy.deepcopy(self.edge_model)
        avg_dict = avg_model.state_dict()
        for key in avg_dict.keys():
            avg_dict[key] = torch.stack([model[key] for model in models]).mean(0)
        avg_model.load_state_dict(avg_dict)
        return avg_model
    
    def send_to_fog(self, fog):
        fog.receive_from_edge(self.edge_model)
    
    def receive_from_fog(self, fog_model):
        self.edge_model.load_state_dict(fog_model.state_dict())


    def receive_model_update_from_fog(self, fog):
        # Get the model update from the fog
        self.receive_from_fog(fog.fog_model)
