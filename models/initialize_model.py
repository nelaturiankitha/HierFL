import torch
import torch.nn as nn
import torch.optim as optim
from models.iot_model import SimpleNN  # Import the new model

class MTL_Model(object):
    def __init__(self, shared_layers, specific_layers, learning_rate, lr_decay, lr_decay_epoch, momentum, weight_decay):
        self.shared_layers = shared_layers
        self.specific_layers = specific_layers
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.lr_decay_epoch = lr_decay_epoch
        self.momentum = momentum
        self.weight_decay = weight_decay
        param_dict = [{"params": self.shared_layers.parameters()}]
        if self.specific_layers:
            param_dict += [{"params": self.specific_layers.parameters()}]
        self.optimizer = optim.SGD(params=param_dict,
                                  lr=learning_rate,
                                  momentum=momentum,
                                  weight_decay=weight_decay)
        self.optimizer_state_dict = self.optimizer.state_dict()
        self.criterion = nn.CrossEntropyLoss()

    def exp_lr_sheduler(self, epoch):
        if (epoch + 1) % self.lr_decay_epoch == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.lr_decay

    def step_lr_scheduler(self, epoch):
        if epoch < 150:
            lr = 0.1
        elif epoch < 250:
            lr = 0.01
        else:
            lr = 0.001
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def print_current_lr(self):
        for param_group in self.optimizer.param_groups:
            print(param_group['lr'])

    def optimize_model(self, input_batch, label_batch):
        self.shared_layers.train(True)
        if self.specific_layers:
            self.specific_layers.train(True)
        if self.specific_layers:
            output_batch = self.specific_layers(self.shared_layers(input_batch))
        else:
            output_batch = self.shared_layers(input_batch)
        self.optimizer.zero_grad()
        batch_loss = self.criterion(output_batch, label_batch)
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss.item()

    def test_model(self, input_batch):
        self.shared_layers.train(False)
        with torch.no_grad():
            if self.specific_layers:
                output_batch = self.specific_layers(self.shared_layers(input_batch))
            else:
                output_batch = self.shared_layers(input_batch)
        self.shared_layers.train(True)
        return output_batch

    def update_model(self, new_shared_layers):
        self.shared_layers.load_state_dict(new_shared_layers)

def initialize_model(args, device):
    if args.mtl_model:
        print('Using different task-specific layer for each user')
        if args.dataset == 'IOT':
            if args.model == 'simple_nn':
                shared_layers = SimpleNN(input_size=args.input_size, num_classes=args.num_classes)
                specific_layers = None
            else:
                raise ValueError('Model not implemented for IoT dataset')
        else:
            raise ValueError('The dataset is not implemented for MTL yet')
        if args.cuda:
            shared_layers = shared_layers.to(device)
            if specific_layers:
                specific_layers = specific_layers.to(device)
    elif args.global_model:
        print('Using the same global model for all users')
        if args.dataset == 'IOT':
            if args.model == 'simple_nn':
                shared_layers = SimpleNN(input_size=args.input_size, num_classes=args.num_classes)
                specific_layers = None
            else:
                raise ValueError('Model not implemented for IoT dataset')
        else:
            raise ValueError('The dataset is not implemented for global model yet')
        if args.cuda:
            shared_layers = shared_layers.to(device)
    else:
        raise ValueError('Wrong input for the --mtl_model and --global_model, only one is valid')
    
    model = MTL_Model(shared_layers=shared_layers,
                      specific_layers=specific_layers,
                      learning_rate=args.lr,
                      lr_decay=args.lr_decay,
                      lr_decay_epoch=args.lr_decay_epoch,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)
    return model
