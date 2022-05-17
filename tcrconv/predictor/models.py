
import torch
import torch.nn as nn

class CNN3(nn.Module):
    """TCRconv predictor for single TCR chain. """
    def __init__(self,input_size,output_size,filters=[120,100,80,60],kernel_sizes=[5,9,15,21,3],dos=[0.1,0.2],pool='max'):
        super(CNN3, self).__init__()

        n_f = sum(filters)
        self.dos=dos

        self.output_size=output_size
        self.cnn1 = nn.Conv1d(input_size, filters[0], kernel_size=kernel_sizes[0], padding=(kernel_sizes[0]-1)//2)
        self.cnn2 = nn.Conv1d(input_size, filters[1], kernel_size=kernel_sizes[1], padding=(kernel_sizes[1]-1)//2)
        self.cnn3 = nn.Conv1d(input_size, filters[2], kernel_size=kernel_sizes[2], padding=(kernel_sizes[2]-1)//2)
        self.cnn4 = nn.Conv1d(input_size, filters[3], kernel_size=kernel_sizes[3], padding=(kernel_sizes[3]-1)//2)

        self.bn= nn.BatchNorm1d(n_f)
        self.relu = nn.ReLU()
        if pool=='max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif pool=='sum':
            self.pool = lambda x: nn.AdaptiveAvgPool1d(1)(x) * x.shape[2]
        else: # pool=='avg'
            self.pool = nn.AdaptiveAvgPool1d(1)

        self.cnn5 = nn.Conv1d(n_f, 100, kernel_size=kernel_sizes[4], padding=(kernel_sizes[4]-1)//2)
        self.bn5 = nn.BatchNorm1d(100)

        self.dense_i = nn.Linear(input_size,256)
        self.dense = nn.Linear(100+256,output_size)
        self.dropout1 = nn.Dropout(self.dos[0])
        self.dropout2 = nn.Dropout(self.dos[1])

    def forward(self,x):
        # CNN part
        xc = torch.cat((self.cnn1(x),self.cnn2(x),self.cnn3(x),self.cnn4(x)),dim=1)
        xc = self.bn(xc)
        xc = self.relu(xc)
        xc = self.dropout1(xc)
        xc = self.cnn5(xc)
        xc = self.bn5(xc)
        xc = self.relu(xc)
        xc = self.pool(xc)
        xc = torch.squeeze(xc,dim=2)

        # PARALLEL LNN part
        x = self.pool(x)
        x = torch.squeeze(x,dim=2)
        x = self.dense_i(x)
        x = self.relu(x)

        # COMBINED part
        x = torch.cat((xc,x),dim=1)
        x = self.dropout2(x)
        x = self.dense(x)

        return x

class CNN3AB(nn.Module):
    """TCRconv predictor for two paired TCR chains."""
    def __init__(self,input_size,output_size,filters=[120,100,80,60],kernel_sizes=[5,9,15,21,3],dos=[0.1,0.2],pool='max'):
        super(CNN3AB, self).__init__()

        n_f = sum(filters)
        self.dos=dos
        self.output_size=output_size

        self.relu = nn.ReLU()
        if pool=='max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif pool=='sum':
            self.pool = lambda x: nn.AdaptiveAvgPool1d(1)(x) * x.shape[2]
        else: # pool=='avg'
            self.pool = nn.AdaptiveAvgPool1d(1)

        # CNN A part
        self.cnn1a = nn.Conv1d(input_size,filters[0], kernel_size=kernel_sizes[0], padding=(kernel_sizes[0]-1)//2)
        self.cnn2a = nn.Conv1d(input_size,filters[1], kernel_size=kernel_sizes[1], padding=(kernel_sizes[1]-1)//2)
        self.cnn3a = nn.Conv1d(input_size,filters[2], kernel_size=kernel_sizes[2], padding=(kernel_sizes[2]-1)//2)
        self.cnn4a = nn.Conv1d(input_size,filters[3], kernel_size=kernel_sizes[3], padding=(kernel_sizes[3]-1)//2)

        self.bna= nn.BatchNorm1d(n_f)
        self.dropout1a = nn.Dropout(self.dos[0])

        self.cnn5a = nn.Conv1d(n_f,100, kernel_size=kernel_sizes[4], padding=(kernel_sizes[4]-1)//2)
        self.bn5a = nn.BatchNorm1d(100)

        # PARALLEL LNN A part
        self.dense_ia = nn.Linear(input_size,256)

        # CNN B part
        self.cnn1b = nn.Conv1d(input_size,filters[0], kernel_size=kernel_sizes[0], padding=(kernel_sizes[0]-1)//2)
        self.cnn2b = nn.Conv1d(input_size,filters[1], kernel_size=kernel_sizes[1], padding=(kernel_sizes[1]-1)//2)
        self.cnn3b = nn.Conv1d(input_size,filters[2], kernel_size=kernel_sizes[2], padding=(kernel_sizes[2]-1)//2)
        self.cnn4b = nn.Conv1d(input_size,filters[3], kernel_size=kernel_sizes[3], padding=(kernel_sizes[3]-1)//2)

        self.bnb= nn.BatchNorm1d(n_f)
        self.dropout1b = nn.Dropout(self.dos[0])

        self.cnn5b = nn.Conv1d(n_f,100,kernel_size=kernel_sizes[4],padding=(kernel_sizes[4]-1)//2)
        self.bn5b = nn.BatchNorm1d(100)

        # PARALLEL LNN B part
        self.dense_ib = nn.Linear(input_size,256)

        # COMBINED part
        self.dense = nn.Linear((100+256)*2,output_size)
        self.dropout2 = nn.Dropout(self.dos[1])

    def forward(self,x,y):
        # CNN A part
        xc = torch.cat((self.cnn1a(x),self.cnn2a(x),self.cnn3a(x),self.cnn4a(x)),dim=1)
        xc = self.bna(xc)
        xc = self.relu(xc)
        xc = self.dropout1a(xc)
        xc = self.cnn5a(xc)
        xc = self.bn5a(xc)
        xc = self.relu(xc)
        xc = self.pool(xc)
        xc = torch.squeeze(xc,dim=2)

        # PARALLEL LNN A part
        x = self.pool(y)
        x = torch.squeeze(x,dim=2)
        x = self.dense_ia(x)
        x = self.relu(x)

        # CNN B part
        yc = torch.cat((self.cnn1b(y),self.cnn2b(y),self.cnn3b(y),self.cnn4b(y)),dim=1)
        yc = self.bnb(yc)
        yc = self.relu(yc)
        yc = self.dropout1b(yc)
        yc = self.cnn5b(yc)
        yc = self.bn5b(yc)
        yc = self.relu(yc)
        yc = self.pool(yc)
        yc = torch.squeeze(yc,dim=2)

        # PARALLEL LNN B part
        y = self.pool(y)
        y = torch.squeeze(y,dim=2)
        y = self.dense_ib(y)
        y = self.relu(y)

        # COMBINED part
        x = torch.cat((xc,x,yc,y),dim=1)
        x = self.dropout2(x)
        x = self.dense(x)
        return x
