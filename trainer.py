import torch
import numpy as np
from tqdm.notebook import tqdm

from torch.utils.tensorboard import SummaryWriter

class Trainer:
    
    node_dim = 0
    
    def __init__(self, model, optimizer, data, tboard_log_dir=''):
        self.model = model
        self.optimizer = optimizer
        self.node_features, self.adjacency_matrix, self.node_labels = data[0], data[1], data[2]
        self.train_indices, self.val_indices, self.test_indices = data[3], data[4], data[5]
        
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = self.model.to(self.device)
    
        self.global_step = 0
        self.tboard_log_dir = tboard_log_dir
        if self.tboard_log_dir:
            self.log_writer = SummaryWriter(log_dir=tboard_log_dir)
    
    def train(self, num_epochs=1):
        
        model = self.model
        optimizer = self.optimizer
        
        train_labels = self.node_labels.index_select(self.node_dim, self.train_indices)
        val_labels = self.node_labels.index_select(self.node_dim, self.val_indices)
        test_labels = self.node_labels.index_select(self.node_dim, self.test_indices)
        
        graph_data = (self.node_features, self.adjacency_matrix)
        
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        
        for i in tqdm(range(num_epochs)):
            model.train()
            
            train_output = model.forward(graph_data)[0].index_select(self.node_dim, self.train_indices)
            train_loss = cross_entropy_loss(train_output, train_labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            print('----------------------')
            
            train_class_predictions = torch.argmax(train_output, dim=-1)
            accuracy = torch.sum(torch.eq(train_class_predictions, train_labels).long()).item() / len(train_labels)
            print('train', accuracy)
            
            model.eval()
            val_output = model.forward(graph_data)[0].index_select(self.node_dim, self.val_indices)
            val_class_predictions = torch.argmax(val_output, dim=-1)
            val_accuracy = torch.sum(torch.eq(val_class_predictions, val_labels).long()).item() / len(val_labels)
            val_loss = cross_entropy_loss(val_output, val_labels)
            print('val', val_accuracy)
            
            if self.tboard_log_dir:
                self.log_writer.add_scalar('train/', train_loss, global_step=self.global_step)
                self.log_writer.add_scalar('val/', val_loss, global_step=self.global_step)
                self.global_step += 1
            
        test_output = model.forward(graph_data)[0].index_select(self.node_dim, self.test_indices)
        class_predictions = torch.argmax(test_output, dim=-1)
        test_accuracy = torch.sum(torch.eq(class_predictions, test_labels).long()).item() / len(test_labels)
        print('test', test_accuracy)