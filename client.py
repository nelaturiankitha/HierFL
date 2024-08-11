import torch
from models.initialize_model import initialize_model

class Client:
    def __init__(self, id, args, device, train_loader, test_loader):
        self.id = id
        self.args = args
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = initialize_model(args, device)  # Initialize the model for the client
    
    def local_update(self, num_iter, device):
        self.model.to(device)
        total_loss = 0
        for epoch in range(num_iter):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                loss = self.model.train(inputs, labels)
                total_loss += loss
        return total_loss / (num_iter * len(self.train_loader)) 

    # def local_update(self, num_iter, device):
    #     # Perform local training
    #     self.model.to(device)
    #     self.model.train()
    #     optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)  # Example optimizer
    #     criterion = torch.nn.CrossEntropyLoss()  # Example loss function
        
    #     for epoch in range(num_iter):
    #         for inputs, labels in self.train_loader:
    #             inputs, labels = inputs.to(device), labels.to(device)
    #             optimizer.zero_grad()
    #             outputs = self.model(inputs)
    #             loss = criterion(outputs, labels)
    #             loss.backward()
    #             optimizer.step()
        
    #     # Return some loss value for tracking
    #     return loss.item()

    def get_model(self):
        return self.model

    def receive_from_edge(self, model):
        self.model = model

    def evaluate(self, inputs):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs
