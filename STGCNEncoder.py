# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:21:48 2024

@author: konodioda
"""
import torch
from torch import nn
from torch_geometric.nn import GCNConv 


class st_gcn(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1,
                 dropout=0.6, residual=True):
        super().__init__()

        padding = ((kernel_size - 1) // 2, 0)
        self.out_channels = out_channels

        self.att1 = nn.Conv2d(in_channels, 1, 1)
        self.att2 = nn.Conv2d(in_channels, 1, 1)

        self.leaky_relu = nn.LeakyReLU()
        
        self.gcn = GCNConv(in_channels, out_channels)
        # self.tcn() dose not change the shape of x
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size, 1),
                (stride, 1),
                padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, X, edges):
        
        z = torch.tensor([]).to(X.device)
        
        att1 = self.att1(X.permute(0, 1, 3, 2)).permute(0, 2, 1, 3)
        att2 = self.att2(X.permute(0, 1, 3, 2))

        att = nn.functional.softmax(self.leaky_relu(att1 + att2), dim=1).permute(0, 3, 1, 2)
        X = torch.matmul(att, X.permute(0, 2, 3, 1))
        X = X.permute(0, 3, 1, 2)
        
        res = self.residual(X)
        
        for i in range(len(X)):
            z = torch.cat((z, self.gcn(X[i].permute(1, 2, 0), edges[i]).unsqueeze(0)), 0)

        # z = torch.matmul(att, z)
                
        z = z.permute(0, 3, 1, 2)
        z = self.tcn(z) + res
        
        return self.relu(z)