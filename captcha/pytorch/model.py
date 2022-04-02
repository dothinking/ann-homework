import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Model:
    def __init__(self, 
            model:nn.Module, 
            optimizer=None, 
            loss=None, 
            gpu:bool=True,
            stored_model_filename:str=None) -> None:
        '''Initialize network, optimizer and loss function.

        Args:
            model (nn.Module): Network instance.
            optimizer (_type_, optional): Optimizer. Defaults to None.
            loss (_type_, optional): Loss function. Defaults to None.
            gpu (bool, optional): use GPU or not. Defaults to True.
            stored_model_filename (str, optional): Restore model parameters if provided.
        '''
        # gpu supported
        use_gpu = gpu and torch.cuda.is_available()
        self.device = torch.device("cuda") if use_gpu else torch.device('cpu')

        # model
        if stored_model_filename and os.path.exists(stored_model_filename):
            model.load_state_dict(torch.load(stored_model_filename))
        self.model = model.to(device=self.device)

        # default optimizer
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=0.02)

        # default loss function
        self.fun_loss = loss or nn.CrossEntropyLoss().to(self.device)

        # training metrics
        self.metrics = {
            'time': [],
            'train_accuracy': [],
            'validation_accuracy': [],
            'loss': []
        }


    def fit(self, epochs:int, 
                    train_data:DataLoader, 
                    validate_data:DataLoader, 
                    fun_evaluate=None,
                    output_step:int=50,
                    store_model_filename:str=None) -> None:
        '''Training over epochs.

        Args:
            epochs (int): Epoch count.
            train_data (DataLoader): Train dataset.
            validate_data (DataLoader): Test dataset.
            fun_evaluate (function): Validation function, taking two parameters: 
                batch predicted outputs, batch labels.
            output_step (int): step intervals to display loss and accracy evaluation.
            store_model_filename (str, optional): save model parameters if provided.
        '''
        self.__t0 = time.time()
        for epoch in range(epochs):
            # train over epoch
            self.__train_over_epoch(epoch, train_data, validate_data, fun_evaluate, output_step)
            
            # store model parameters at the end of each epoch
            if store_model_filename:
                torch.save(self.model.state_dict(), store_model_filename)


    def predict(self, input):
        x = Variable(input).to(self.device)
        return self.model(x)


    def validate_accuracy(self, validate_data:DataLoader, fun_evaluate) -> float:
        '''Calculate accuracy on validation dataset.'''
        acc_history = []
        self.model.eval()
        for x, y in validate_data:
            output = self.predict(x)
            b_y = Variable(y).to(self.device)
            acc = fun_evaluate(output, b_y)
            acc_history.append(acc)        
        return torch.mean(torch.Tensor(acc_history)).item()
    

    def __train_over_epoch(self, epoch:int,
                    train_data:DataLoader, 
                    validate_data:DataLoader,
                    fun_evaluate=None,
                    output_step:int=50) -> None:
        '''Training process.

        Args:
            epoch (int): Epoch index.
            train_data (DataLoader): Train dataset.
            validate_data (DataLoader): Test dataset.
            fun_evaluate (function): Validation function, taking two parameters: 
                batch predicted outputs, batch labels.
            output_step (int): step intervals to display loss and accracy evaluation.
        '''
        loss_history = []
        acc_history = []

        # training process
        self.model.train()
        for step, (x, y) in enumerate(train_data): 
            # batch x, y variables
            b_x = Variable(x).to(self.device)
            b_y = Variable(y).to(self.device)

            # training
            output = self.model(b_x)
            loss = self.fun_loss(output, b_y)
            self.optimizer.zero_grad() # clear gradients for this training step
            loss.backward() # backpropagation, compute gradients
            self.optimizer.step() # apply gradients

            # validate training accuracy
            if fun_evaluate:
                acc = fun_evaluate(output, b_y)
                acc_history.append(acc)
            loss_history.append(loss.cpu().detach().item())

            # step summary
            if step%output_step == 0:
                # store loss metric
                avg_loss = torch.mean(torch.Tensor(loss_history)).item()
                self.metrics['time'].append(time.time()-self.__t0)
                self.metrics['loss'].append(avg_loss)
                
                info = f'epoch: {epoch}, step: {step}, train loss: {round(avg_loss, 4)}'
                if fun_evaluate:
                    train_acc = torch.mean(torch.Tensor(acc_history)).item()
                    validate_acc = self.validate_accuracy(validate_data, fun_evaluate)
                    # store accuracy metric
                    self.metrics['train_accuracy'].append(train_acc)
                    self.metrics['validation_accuracy'].append(validate_acc)

                    info += f', train accuracy: {round(train_acc, 2)}, validation accuracy: {round(validate_acc, 2)}'
                
                print(info)

                
                