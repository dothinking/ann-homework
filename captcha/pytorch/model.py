import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(# (N, 3, 60, 120)                
            nn.Conv2d(3, 16, 3, padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(), # (N, 16, 30, 60)
            
            nn.Conv2d(16, 64, 3, padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(), # (N, 64, 15, 30)
            
            nn.Conv2d(64, 512, 3, padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(), # (N, 512, 7, 15)
            
            nn.Conv2d(512, 512, 3, padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(), # (N, 512, 3, 7)
            )
        self.fc = nn.Linear(512*3*7, 26*4)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)   # (N, 512*3*7)
        x = self.fc(x)
        return x


class VGG(nn.Module):
    def __init__(self, 
        input_size:list, 
        num_classes:int, 
        conv_pattern:list=None, 
        hidden_pattern:list=None,
        dropout:float=0.0):
        '''VGG net.

        Args:
            input_size (list): input image size: (channels, height, width).
            num_classes (int): the count of final classes.
            conv_pattern (list, optional): convolution layers pattern: [(num_layers, channels), (), ...].
            hidden_pattern (list, optional): full connection layers pattern: [num_layers, hidden_size].
            dropout (float, optional): full connection layers drop out value. Defaults to 0.5.
        '''
        super().__init__()

        # input size
        c, h, w = input_size
        self.input_size = input_size

        # convolution layers
        if not conv_pattern:
            conv_pattern = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)] # (num_convs, channels)
        self.features = self._make_conv_layers(c, conv_pattern)

        out_h = int(h / 2**len(conv_pattern))
        out_w = int(w / 2**len(conv_pattern))
        out_channels = conv_pattern[-1][-1]
        conv_out_size = out_channels * out_h * out_w

        # full connection layers
        if not hidden_pattern:
            hidden_pattern = [3, 1024]
        self.classifier =  self._make_fc_layers(conv_out_size, num_classes, hidden_pattern, dropout)


    def forward(self, x): 
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


    @staticmethod
    def _make_conv_layers(in_channels:int, pattern:list) -> nn.Sequential:
        '''Combine convolution blocks based on specified pattern.

        Args:
            in_channels (int): initial input channels.
            pattern (list): convolution layers pattern: [(num_layers, channels), (), ...].
        '''
        layers = []
        x = in_channels
        for (num, channels) in pattern:
            layers.extend(VGG.vgg_block(num, x, channels))
            x = channels
        return nn.Sequential(*layers)
    

    @staticmethod
    def _make_fc_layers(in_size:int, out_size:int, pattern:list, dropout:float) -> nn.Sequential:
        '''Create connection layers.

        Args:
            in_size (int): output size from convolution layers.
            out_size (int): count of final classes.
            pattern (list): full connection layers pattern: [num_layers, hidden_size].
        '''
        num_layers, hidden_size = pattern
        if num_layers==1: return nn.Sequential(nn.Linear(in_size, out_size))

        # first layer
        layers = []
        layers.extend([
            nn.Linear(in_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
            ])

        # hidden layers
        for i in range(num_layers-2):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])

        # output layer
        layers.append(nn.Linear(hidden_size, out_size))

        return nn.Sequential(*layers)


    @staticmethod
    def vgg_block(num_convs, in_channels, out_channels):
        blocks = []
        # first layer
        blocks.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])

        # the rest layers
        for i in range(num_convs-1):
            blocks.extend([
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        
        # pool layer
        blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return blocks


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
            'loss': []
        }

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters())


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
        best_acc, acc = 0.0, 0.0
    
        for epoch in range(epochs):
            # train over epoch
            self.__train_over_epoch(epoch, train_data, fun_evaluate, output_step)
            
            # evaluate and store model at the end of each epoch
            if fun_evaluate:
                t = round(time.time()-self.__t0, 2)
                acc = self.evaluate(validate_data, fun_evaluate)
                info = f'====== epoch: {epoch}, time: {t}, validation accuracy: {acc:6.4f} ======'
                print(info)

            if store_model_filename and acc>best_acc:
                best_acc = acc
                torch.save(self.model.state_dict(), store_model_filename)


    def predict(self, input):
        self.model.eval() # validation mode
        x = Variable(input).to(self.device)
        return self.model(x)


    def evaluate(self, validate_data:DataLoader, fun_evaluate) -> float:
        '''Calculate accuracy on validation dataset.'''
        acc_history = []
        with torch.no_grad():
            for x, y in validate_data:
                output = self.predict(x)
                b_y = Variable(y).to(self.device)
                acc = fun_evaluate(output, b_y)
                acc_history.append(acc)        
        return torch.mean(torch.Tensor(acc_history)).item()
    

    def __train_over_epoch(self, epoch:int,
                    train_data:DataLoader, 
                    fun_evaluate=None,
                    output_step:int=50) -> None:
        '''Training process.

        Args:
            epoch (int): Epoch index.
            train_data (DataLoader): Train dataset.
            fun_evaluate (function): Validation function, taking two parameters: 
                batch predicted outputs, batch labels.
            output_step (int): step intervals to display loss and accracy evaluation.
        '''
        loss_history = []
        acc_history = []

        # training process
        self.model.train() # switch to training mode
        
        for step, (x, y) in enumerate(train_data, start=1): 
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
            loss_history.append(loss.detach().item())

            # step summary
            if step%output_step == 1:
                # store loss metric
                avg_loss = torch.mean(torch.Tensor(loss_history)).item()
                train_acc = round(torch.mean(torch.Tensor(acc_history)).item(), 4) if acc_history else 'n.a.'
                t = round(time.time()-self.__t0, 2)
                self.metrics['time'].append(t)
                self.metrics['loss'].append(avg_loss)
                self.metrics['train_accuracy'].append(train_acc)
                
                info = f'epoch: {epoch:<3} step: {step:<8} time: {t:<10} loss: {avg_loss:<8.4f} accuracy: {train_acc:<6}'
                print(info)