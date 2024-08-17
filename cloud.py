import torch
import copy

class Cloud:
    def __init__(self, model_class, *model_args, **model_kwargs):
        self.fogs = []
        self.global_model = model_class(*model_args, **model_kwargs)
    
    def fog_register(self, fog):
        self.fogs.append(fog)

    def refresh_cloudserver(self):
        # Reinitialize or refresh the cloud server, if necessary
        pass

    def aggregate(self, args):
        # Aggregate the models from the fog nodes
        models = [fog.fog_model for fog in self.fogs if fog.fog_model is not None]
        if models:
            self.global_model = self.average_models(models)
        else:
            print("Warning: No models to aggregate in Cloud")
    
    def average_models(self, models):
        if not models:
            return None
        
        avg_model = copy.deepcopy(self.global_model)
        avg_dict = avg_model.state_dict()
        for key in avg_dict.keys():
            avg_dict[key] = torch.stack([model.state_dict()[key] for model in models]).mean(0)
        avg_model.load_state_dict(avg_dict)
        return avg_model
    
    def send_to_fog(self, fog):
        # Send the updated global model to the fog
        fog.receive_from_cloud(self.global_model)
    
    def get_global_model(self):
        # Return the current global model
        return self.global_model
    
    def receive_from_fog(self, model):
        # Receive updated model from the fog
        if self.global_model is None:
            self.global_model = copy.deepcopy(model)
        else:
            self.global_model = self.average_models([self.global_model, model])