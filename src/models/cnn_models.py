import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class HighAccuracy3DCNN(nn.Module):
    def __init__(self):
        super(HighAccuracy3DCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.feature_dim = 64 * 6 * 7 * 6 
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.feature_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Base3DCNN(nn.Module):
    def __init__(self):
        super(Base3DCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 2, kernel_size=7, stride=4, padding=0),
            nn.BatchNorm3d(2),
            nn.ReLU(inplace=True)
        )
        self.feature_dim = 2*23*28*23
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.feature_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x = x1.view(x1.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class Base3DCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1,  2, kernel_size=7, stride=2, padding=0),
            nn.BatchNorm3d(2), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(2,  4, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(4), nn.ReLU(inplace=True)
        )
        self.gap  = nn.AdaptiveAvgPool3d(1) 
        self.drop = nn.Dropout(0.5)
        self.fc   = nn.Linear(4, 1)       

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gap(x).flatten(1)           
        x = self.drop(x)
        return self.fc(x)

class Base3DCNN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1,  2, kernel_size=7, stride=2, padding=0),
            nn.BatchNorm3d(2), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(2,  4, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(4), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(4,  8, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(8), nn.ReLU(inplace=True)
        )
        self.gap  = nn.AdaptiveAvgPool3d(1)  
        self.drop = nn.Dropout(0.5)
        self.fc   = nn.Linear(8, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x).flatten(1)        
        x = self.drop(x)
        return self.fc(x)

class Base3DCNN4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1,   2, kernel_size=7, stride=2, padding=0),
            nn.BatchNorm3d(2), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(2,   4, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(4), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(4,   8, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(8), nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(8,  16, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True)
        )
        self.gap  = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout(0.5)
        self.fc   = nn.Linear(16, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x).flatten(1)        
        x = self.drop(x)
        return self.fc(x)

class Base3DCNN5(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8,  kernel_size=7, stride=2, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

        self.gap  = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout(0.5)
        self.fc   = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x).flatten(1)   
        x = self.drop(x)
        return self.fc(x)

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm3d(out_channels)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm3d(out_channels)
            )
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return self.relu(out)

class ResNetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        
        self.res_block1 = ResidualBlock(16, 32, stride=2)
        self.res_block2 = ResidualBlock(32, 64, stride=2)
        self.res_block3 = ResidualBlock(64, 128, stride=2)
        
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)



# class CNNGRUClassifier(nn.Module):
#     def __init__(self,
#                  cnn_hidden_dim=128,  # Changed default
#                  gru_hidden_dim=128,  # Changed default
#                  gru_num_layers=1,    # New parameter
#                  # gru_bidirectional=False, # New parameter
#                  dropout_rate=0.5,
#                  # Potentially pass CNN filter configs if you want to tune those more dynamically
#                  ):
#         super().__init__()

#         # --- More Robust CNN Backbone Example ---
#         # (Using the ModuleList example from above, ensure cnn_channels[-1] is known)
#         # For simplicity, let's assume the last CNN layer outputs, say, 128 channels
#         # This means the Linear layer input should be 128
#         # This part needs careful design based on your chosen CNN structure
#         # Simplified for now, assuming your current CNN structure but with more output channels from conv4:
#         # Suppose self.conv4 now outputs `final_cnn_channels` (e.g., 64 or 128)
#         final_cnn_channels = 128 # EXAMPLE: This should be the output channels of your last conv layer before GAP

#         self.conv1 = nn.Sequential( # Example: 1 -> 32
#             nn.Conv3d(1,   32, kernel_size=5, stride=1, padding=2), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
#             nn.MaxPool3d(2,2))
#         self.conv2 = nn.Sequential( # Example: 32 -> 64
#             nn.Conv3d(32,  64, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
#             nn.MaxPool3d(2,2))
#         self.conv3 = nn.Sequential( # Example: 64 -> 128
#             nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(128), nn.ReLU(inplace=True),
#             nn.MaxPool3d(2,2))
#         # Removed conv4 for this example, self.conv3 is now the last main conv block
        
#         self.gap = nn.AdaptiveAvgPool3d(1)
#         self.cnn_fc = nn.Linear(final_cnn_channels, cnn_hidden_dim) # Input matches output of last conv block

#         self.gru = nn.GRU(
#             input_size=cnn_hidden_dim,
#             hidden_size=gru_hidden_dim,
#             num_layers=gru_num_layers,
#             batch_first=True,
#             # bidirectional=gru_bidirectional,
#             dropout=dropout_rate if gru_num_layers > 1 else 0 # Apply dropout between GRU layers
#         )

#         # Adjust final_fc input if GRU is bidirectional
#         # gru_output_features = gru_hidden_dim * 2 if gru_bidirectional else gru_hidden_dim
#         gru_output_features = gru_hidden_dim
#         self.drop = nn.Dropout(dropout_rate)
#         self.final_fc = nn.Linear(gru_output_features, 1)

#     # ... (forward method remains largely the same, just ensure cnn_out is processed by the new CNN structure)
#     def forward(self, x):
#         B, S, D_vol, H_vol, W_vol = x.shape
#         cnn_in = x.view(B * S, 1, D_vol, H_vol, W_vol)
        
#         # Assuming the simplified 3-block CNN from __init__ example:
#         cnn_out = self.conv1(cnn_in)
#         cnn_out = self.conv2(cnn_out)
#         cnn_out = self.conv3(cnn_out) 
#         cnn_out = self.gap(cnn_out)
        
#         cnn_out_flat = cnn_out.flatten(start_dim=1)
#         cnn_features = self.cnn_fc(cnn_out_flat)
        
#         gru_in = cnn_features.view(B, S, self.cnn_fc.out_features)
        
#         gru_out, _ = self.gru(gru_in)
        
#         # gru_output_features_actual = gru_out.shape[-1] # To handle bidirectional case if output dim changes
#         sequence_embedding = gru_out[:, -1, :]
        
#         out = self.drop(sequence_embedding)
#         out = self.final_fc(out)
        
#         return out
    
# Bryan
class CNNGRUClassifier(nn.Module):
    def __init__(self, cnn_hidden_dim=16, gru_hidden_dim=16, dropout_rate=0.1):
        """
        Args:
            cnn_hidden_dim (int): The dimension of the features extracted by the CNN for each sequence element.
                                  This will be the input size for the GRU.
            gru_hidden_dim (int): The hidden dimension of the GRU layer.
            dropout_rate (float): Dropout probability.
        """
        super().__init__()
        
        # # --- CNN Backbone (processes each 3D volume in the sequence) ---
        # self.conv1 = nn.Sequential(
        #     nn.Conv3d(1, 32, kernel_size=7, stride=2, padding=0), # Output channels: 2
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv3d(32, 128, kernel_size=5, stride=2, padding=0), # Output channels: 4
        #     nn.BatchNorm3d(128),
        #     nn.ReLU(inplace=True)
        # )
        # self.gap = nn.AdaptiveAvgPool3d(1) # Global Average Pooling

        self.conv1 = nn.Sequential(
            nn.Conv3d(1,   8, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(8), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(8,   16, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True)
        )

        self.gap  = nn.AdaptiveAvgPool3d(1)
        
        # --- FC layer to project CNN features to cnn_hidden_dim ---
        # The output of conv2 + gap will have 4 features (the number of out_channels of conv2)
        self.cnn_fc = nn.Linear(16, cnn_hidden_dim) 
        
        # --- GRU Layer (processes sequence of features) ---
        self.gru = nn.GRU(
            input_size=cnn_hidden_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,       # Using a single GRU layer
            batch_first=True,   # Expects input as (Batch, Sequence, Features)
            bidirectional=False # Using a unidirectional GRU
        )
        
        # --- Classifier Head ---
        self.drop = nn.Dropout(dropout_rate)
        self.final_fc = nn.Linear(gru_hidden_dim, 1) # Binary classification (1 output node)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, S, D_vol, H_vol, W_vol)
                              B: Batch size
                              S: Sequence length
                              D_vol, H_vol, W_vol: Depth, Height, Width of the 3D volume.
                                                   These should match the expected input order for nn.Conv3d.
                                                   If your H, W, D are in a different order,
                                                   you might need to x.permute() them first.
                                                   For example, if input is (B, S, H, W, D_depth), you might need
                                                   x = x.permute(0, 1, 4, 2, 3) # to get (B, S, D_depth, H, W)
                                                   before the view operation.
        """
        B, S, D_vol, H_vol, W_vol = x.shape
        
        # 1. Process with CNN
        # Reshape to (B*S, 1, D_vol, H_vol, W_vol) to process each 3D volume in the sequence independently
        # The '1' is for the input channel to the first Conv3d layer
        cnn_in = x.view(B * S, 1, D_vol, H_vol, W_vol)
        
        cnn_out = self.conv1(cnn_in)     # Shape: (B*S, 2, D_out1, H_out1, W_out1)
        cnn_out = self.conv2(cnn_out)   # Shape: (B*S, 4, D_out2, H_out2, W_out2)
        cnn_out = self.gap(cnn_out)     # Shape: (B*S, 4, 1, 1, 1)
        
        # Flatten the output from GAP
        cnn_out_flat = cnn_out.flatten(start_dim=1) # Shape: (B*S, 4)
        
        # Project CNN features to the desired dimension for GRU input
        # Output shape: (B*S, cnn_hidden_dim)
        cnn_features = self.cnn_fc(cnn_out_flat)
        
        # 2. Reshape for GRU
        # Reshape to (B, S, cnn_hidden_dim)
        gru_in = cnn_features.view(B, S, self.cnn_fc.out_features) # self.cnn_fc.out_features is cnn_hidden_dim
        
        # 3. Process with GRU
        # gru_out shape: (B, S, gru_hidden_dim)
        # last_hidden_state shape: (num_layers*num_directions, B, gru_hidden_dim)
        gru_out, _ = self.gru(gru_in) # We don't need the last hidden state separately here
        
        # Take the output of the last time step for classification
        # Shape: (B, gru_hidden_dim)
        sequence_embedding = gru_out[:, -1, :] 
        
        # 4. Classify
        out = self.drop(sequence_embedding)
        out = self.final_fc(out) # Shape: (B, 1)
        
        return out
