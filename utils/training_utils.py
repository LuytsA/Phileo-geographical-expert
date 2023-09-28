import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys; sys.path.append("./")

import matplotlib.pyplot as plt
import buteo as beo
import tqdm
import config_geography
pos_feature_pred = config_geography.feature_positions_predictions
pos_feature_label = config_geography.feature_positions_label

class TiledMSE(nn.Module):
    """
    Calculates the MSE at full image level and at the pixel level and weights the two.
    result = (sum_mse * (1 - bias)) + (mse * bias)
    """
    def __init__(self, bias=0.2):
        super(TiledMSE, self).__init__()
        self.bias = bias

    def forward(self, y_pred, y_true):
        y_pred_sum = torch.sum(y_pred, dim=(1, 2, 3)) / (y_pred.shape[1] * y_pred.shape[2] * y_pred.shape[3])
        y_true_sum = torch.sum(y_true, dim=(1, 2, 3)) / (y_true.shape[1] * y_true.shape[2] * y_true.shape[3])

        sum_mse = ((y_pred_sum - y_true_sum) ** 2).mean()
        mse = ((y_pred - y_true) ** 2).mean()

        weighted = (sum_mse * (1 - self.bias)) + (mse * self.bias)
        
        return weighted 

class MvMFLoss(nn.Module):
    ''' credits to https://www.ecmlpkdd2019.org/downloads/paper/63.pdf '''
    
    def __init__(self, n_center, direction_inits, density_inits, directions_learnable=False, safe_exp_shift=-3):
        super(MvMFLoss, self).__init__()
        
        self.safe_exp_shift = safe_exp_shift
        self.n_center = n_center
        self.directions = torch.FloatTensor(direction_inits).cuda()
        self.densities =  torch.nn.Parameter(torch.FloatTensor(np.array([density_inits])).cuda())
        if directions_learnable:
            self.directions = torch.nn.parameter(self.directions)


    def dist_encoded(self, coord_true, directions, radius=1):

        # print(coord_true.shape, directions.shape)
        lat1_enc,long1_sin, long1_cos = coord_true[:,0], (2*coord_true[:,1]-1), (2*coord_true[:,2]-1)
        lat2_enc,long2_sin, long2_cos = directions[:,0],(2*directions[:,1]-1), (2*directions[:,2]-1)

        lat1_r = lat1_enc*np.pi
        lat2_r = lat2_enc*np.pi

        #cos_dif = torch.zeros((len(long1_cos),len(long2_cos)))
        cos_dif = torch.outer(long1_cos,long2_cos) + torch.outer(long1_sin,long2_sin)
        
        return radius*radius*(torch.outer(torch.sin(lat1_r),torch.sin(lat2_r))*cos_dif +torch.outer(torch.cos(lat1_r),torch.cos(lat2_r)))

    # def MvMF(coordinate,weights,r, density,mu):
    #     weights = weights/np.sum(weights)

    #     #return density/np.sinh(density)*np.exp(density*dist_encoded(coordinate, ref=mu,r=r))
    #     return np.average(density/np.sinh(density)*np.exp(density*dist_encoded(coordinate, ref=mu,r=r)),weights=weights, axis=0)


    def forward(self, y_pred, y_true):

        weights = y_pred
        encoded_coordinate = y_true[:,pos_feature_label['coords']]

        # # naive implementation
        # dens = torch.exp(self.densities)
        # normalisation = dens/torch.sinh(dens)
        # dist = self.dist_encoded(coord_true=encoded_coordinate, directions=self.directions)
        # exp_factor = torch.exp(torch.mul(dens, dist))
        # weighted_sum = torch.sum(weights * torch.mul(normalisation, exp_factor), dim=-1)
        # loss_naive = -torch.log(weighted_sum)

        # weighted_sum2 = torch.sum(weights * (2*torch.exp(self.densities + dens*(dist-1))), dim=-1)
        # loss_naive2 = -torch.log(weighted_sum2)

        # safer version 
        dist = self.dist_encoded(coord_true=encoded_coordinate, directions=self.directions)
        gamma = torch.log(2*weights) + self.densities + torch.mul(torch.exp(self.densities), dist-1)
        #print(gamma.shape)
        gamma_max = torch.max(gamma, dim=-1, keepdim=True)[0]
        #print(gamma_max.shape)
        deltas = gamma - gamma_max
        loss = -(gamma_max + torch.log(torch.sum(torch.exp(deltas),dim=-1)))

        # print(gamma)
        # print()
        # print(gamma_max)



        # log_likelihood_component = 0.69314718056 + self.densities + torch.mul(torch.exp(self.densities), dist-1)
        # top_components = torch.topk(log_likelihood_component, k=600, dim=-1)[1]
        # print(log_likelihood_component.shape)
        # print(top_components.shape)

        # print(self.densities.shape)

        # densities_best = self.densities.take_along_dim(top_components, dim=-1)
        # weights_best = weights.take_along_dim(top_components, dim=-1)
        # dist_best = dist.take_along_dim(top_components, dim=-1)
        # print(dist_best.shape, weights_best.shape,  densities_best.shape)


        # safe_exp_shift = torch.mean(torch.exp(densities_best) * (dist_best-1)) + torch.mean(densities_best)
        # print(safe_exp_shift)
        # #print(dist_best)
        # print(densities_best)

        # exp = densities_best+ torch.exp(densities_best) * (dist_best-1) - safe_exp_shift
        # exp_factors_shifted = torch.exp(exp)
        # # torch.mul(torch.exp(self.densities), dist-1) - self.safe_exp_shift
        # # elements = torch.exp(self.densities) * torch.exp(exp_factors_shifted)
        # print(exp)
        # # print(exp_factors_shifted)
        # loss =   -0.69314718056 - safe_exp_shift - torch.log(torch.sum(weights_best * exp_factors_shifted , dim=-1))


        # log_likelihood_component = torch.log(2*torch.exp(self.densities)) + torch.mul(torch.exp(self.densities), dist-1)
        # log_likelihood_component = 0.69314718056 + self.densities + torch.mul(torch.exp(self.densities), dist-1)
        
        # print(torch.mul(torch.exp(self.densities), dist-1).shape)
        # print(self.densities.shape)
        # print(weights.shape)
        # print(log_likelihood_component.shape)

        # loss = torch.sum(torch.multiply(-log_likelihood_component, weights), axis=-1)
        # print(weighted_likelihood.shape)
        # print(loss.shape)
        loss = torch.mean(loss)

        return loss, loss, loss, loss, False

# CENTERS = np.load('centers_all_more_enc.npy')
# criterion = MvMFLoss(n_center=len(CENTERS), direction_inits=CENTERS, density_inits=[20 for i in range(len(CENTERS))], safe_exp_shift=-23000) #GeographicalLoss(n_classes=len(config_geography.regions.keys()))#nn.MSELoss() #nn.CrossEntropyLoss() # vit_mse_losses(n_patches=4)
# y_pred = torch.rand((4,len(CENTERS))).cuda()
# y_true = torch.rand((4,35)).cuda()
# print(criterion(y_pred=y_pred, y_true=y_true))

class GeographicalLoss(nn.Module):
    """
    Combination of coordinate prediction (regression) and region/climate prediction (classification)
    """
    def __init__(self, n_classes, coord_loss=nn.MSELoss()):
        super(GeographicalLoss, self).__init__()
        self.crossentropy_loss = nn.CrossEntropyLoss()
        self.coord_loss = coord_loss
        self.mse_loss = nn.MSELoss()
        self.weights = np.array([10,1,1])
        self.weights = self.weights/self.weights.sum()
        self.climate_classes = {}

        kg_map = config_geography.kg_map
        for k in kg_map.keys():
            climate_class = kg_map[k]['climate_class']
            if climate_class in self.climate_classes:
                self.climate_classes[climate_class] = self.climate_classes[climate_class]+[k]
            else: self.climate_classes[climate_class] = [k]


    def kg_loss(self,kg_pred_logits,kg_true, weight=0.5):
        superclass_pred_logits = torch.zeros((kg_pred_logits.shape[0],len(self.climate_classes)))
        superclass_true = torch.zeros((kg_true.shape[0],len(self.climate_classes)))

        for k in self.climate_classes.keys():
            superclass_pred_logits[:,k] = torch.mean(kg_pred_logits[:,self.climate_classes[k]], axis=1)
            superclass_true[:,k] = torch.mean(kg_true[:,self.climate_classes[k]], axis=1)
        
        superclass_loss = self.crossentropy_loss(superclass_pred_logits,superclass_true)
        class_loss = self.crossentropy_loss(kg_pred_logits,kg_true)

        return weight*class_loss + (1-weight)*superclass_loss


    def forward(self, y_pred, y_true):

        coord_pred, coord_true = y_pred[:,pos_feature_pred['coords']], y_true[:,pos_feature_label['coords']]
        kg_pred_logits,kg_true = y_pred[:,pos_feature_pred['kg']], y_true[:,pos_feature_label['kg']]
        time_pred, time_true = y_pred[:,pos_feature_pred['time']], y_true[:,pos_feature_label['time']]

        non_sea_patches = torch.where(kg_true[:,0]<0.9)
        only_sea =(len(non_sea_patches[0])<1)
        # if len(non_sea_patches[0])<1:
        #     print(len(non_sea_patches[0]))
        #     print('only sea patches')
        #     kg_loss = self.kg_loss(kg_pred_logits,kg_true)
        #     return self.weights[1]*kg_loss, kg_loss, kg_loss, kg_loss
        # sea_patches = torch.where(kg_true[:,0]>0.8)
        # if len(kg_true[sea_patches])>0:
        #     r = np.random.rand()
        #     np.save(f"sea_{r}",kg_true[sea_patches].detach().cpu().numpy())
        #     np.save(f"c_{r}",coord_true[sea_patches].detach().cpu().numpy())


        coordinate_loss = self.coord_loss(coord_pred[non_sea_patches],coord_true[non_sea_patches])
        kg_loss = self.kg_loss(kg_pred_logits,kg_true)
        time_loss = self.mse_loss(time_pred[non_sea_patches], time_true[non_sea_patches])

        
        return self.weights[0]*coordinate_loss + self.weights[1]*kg_loss + self.weights[2]*time_loss, coordinate_loss, kg_loss, time_loss, only_sea
    
           
class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 

        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]

            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer """
    def __init__(self, dim, channel_first=False):
        super().__init__()
        self.channel_first = channel_first
        if self.channel_first:
            self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        else:
            self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        if self.channel_first:
            Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
            Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        else:
            Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
            Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)

        return self.gamma * (x * Nx) + self.beta + x


def cosine_scheduler(base_value, final_value, epochs, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    # print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs

    return schedule


class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, channels, reduction=16, activation="relu"):
        super().__init__()
        self.reduction = reduction
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // self.reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)

        return x * y.expand_as(x)


class SE_BlockV2(nn.Module):
    # The is a custom implementation of the ideas presented in the paper:
    # https://www.sciencedirect.com/science/article/abs/pii/S0031320321003460
    def __init__(self, channels, reduction=16, activation="relu"):
        super(SE_BlockV2, self).__init__()

        self.channels = channels
        self.reduction = reduction
        self.activation = get_activation(activation)
   
        self.fc_spatial = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(channels, channels, kernel_size=2, stride=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.fc_reduction = nn.Linear(in_features=channels * (4 * 4), out_features=channels // self.reduction)
        self.fc_extention = nn.Linear(in_features=channels // self.reduction , out_features=channels)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        identity = x
        x = self.fc_spatial(identity)
        x = self.activation(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_reduction(x)
        x = self.activation(x)
        x = self.fc_extention(x)
        x = self.sigmoid(x)
        x = x.reshape(x.size(0), x.size(1), 1, 1)

        return x


class SE_BlockV3(nn.Module):
    """ Squeeze and Excitation block with spatial and channel attention. """
    def __init__(self, channels, reduction_c=2, reduction_s=8, activation="relu", norm="batch", first_layer=False):
        super(SE_BlockV3, self).__init__()

        self.channels = channels
        self.first_layer = first_layer
        self.reduction_c = reduction_c if not first_layer else 1
        self.reduction_s = reduction_s
        self.activation = get_activation(activation)
   
        self.fc_pool = nn.AdaptiveAvgPool2d(reduction_s)
        self.fc_conv = nn.Conv2d(self.channels, self.channels, kernel_size=2, stride=2, groups=self.channels, bias=False)
        self.fc_norm = get_normalization(norm, self.channels)

        self.linear1 = nn.Linear(in_features=self.channels * (reduction_s // 2 * reduction_s // 2), out_features=self.channels // self.reduction_c)
        self.linear2 = nn.Linear(in_features=self.channels // self.reduction_c, out_features=self.channels)

        self.activation_output = nn.Softmax(dim=1) if first_layer else nn.Sigmoid()


    def forward(self, x):
        identity = x

        x = self.fc_pool(x)
        x = self.fc_conv(x)
        x = self.fc_norm(x)
        x = self.activation(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        if self.first_layer:
            x = self.activation_output(x) * x.size(1)
        else:
            x = self.activation_output(x)
            
        x = identity * x.reshape(x.size(0), x.size(1), 1, 1)

        return x


class MvFM_layer(nn.Module):
    ''' credits to https://www.ecmlpkdd2019.org/downloads/paper/63.pdf '''
    
    def __init__(self, n_center, feature_dim):
        super(MvFM_layer, self).__init__()
        
        self.n_center = n_center
        self.feature_dim = feature_dim
        self.linear = nn.Linear(in_features=self.feature_dim, out_features=n_center, bias=False)
        self.softmax = nn.Softmax(dim=0)
        self.directions = None
        self.kappas = None

    def forward(self,x):
        x = self.linear(x)
        x = self.softmax(x)

        return x 


def get_activation(activation_name):
    if activation_name == "relu":
        return nn.ReLU6(inplace=True)
    elif isinstance(activation_name, torch.nn.modules.activation.ReLU6):
        return activation_name

    elif activation_name == "gelu":
        return nn.GELU()
    elif isinstance(activation_name, torch.nn.modules.activation.GELU):
        return activation_name

    elif activation_name == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif isinstance(activation_name, torch.nn.modules.activation.LeakyReLU):
        return activation_name

    elif activation_name == "prelu":
        return nn.PReLU()
    elif isinstance(activation_name, torch.nn.modules.activation.PReLU):
        return activation_name

    elif activation_name == "selu":
        return nn.SELU(inplace=True)
    elif isinstance(activation_name, torch.nn.modules.activation.SELU):
        return activation_name

    elif activation_name == "sigmoid":
        return nn.Sigmoid()
    elif isinstance(activation_name, torch.nn.modules.activation.Sigmoid):
        return activation_name

    elif activation_name == "tanh":
        return nn.Tanh()
    elif isinstance(activation_name, torch.nn.modules.activation.Tanh):
        return activation_name

    elif activation_name == "mish":
        return nn.Mish()
    elif isinstance(activation_name, torch.nn.modules.activation.Mish):
        return activation_name
    else:
        raise ValueError(f"activation must be one of leaky_relu, prelu, selu, gelu, sigmoid, tanh, relu. Got: {activation_name}")


def get_normalization(normalization_name, num_channels, num_groups=32, dims=2):
    if normalization_name == "batch":
        if dims == 1:
            return nn.BatchNorm1d(num_channels)
        elif dims == 2:
            return nn.BatchNorm2d(num_channels)
        elif dims == 3:
            return nn.BatchNorm3d(num_channels)
    elif normalization_name == "instance":
        if dims == 1:
            return nn.InstanceNorm1d(num_channels)
        elif dims == 2:
            return nn.InstanceNorm2d(num_channels)
        elif dims == 3:
            return nn.InstanceNorm3d(num_channels)
    elif normalization_name == "layer":
        return LayerNorm(num_channels)
    elif normalization_name == "group":
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif normalization_name == "bcn":
        if dims == 1:
            return nn.Sequential(
                nn.BatchNorm1d(num_channels),
                nn.GroupNorm(1, num_channels)
            )
        elif dims == 2:
            return nn.Sequential(
                nn.BatchNorm2d(num_channels),
                nn.GroupNorm(1, num_channels)
            )
        elif dims == 3:
            return nn.Sequential(
                nn.BatchNorm3d(num_channels),
                nn.GroupNorm(1, num_channels)
            )    
    elif normalization_name == "none":
        return nn.Identity()
    else:
        raise ValueError(f"normalization must be one of batch, instance, layer, group, none. Got: {normalization_name}")


def convert_torch_to_float(tensor):
    if torch.is_tensor(tensor):
        return float(tensor.detach().cpu().numpy().astype(np.float32))
    elif isinstance(tensor, np.ndarray) and tensor.size == 1:
        return float(tensor.astype(np.float32))
    elif isinstance(tensor, float):
        return tensor
    elif isinstance(tensor, int):
        return float(tensor)
    else:
        raise ValueError("Cannot convert tensor to float")



def render_s2_as_rgb(arr, channel_first=False):
    # If there are nodata values, lets cast them to zero.
    if np.ma.isMaskedArray(arr):
        arr = np.ma.getdata(arr.filled(0))

    if channel_first:
        arr = beo.channel_first_to_last(arr)
    # Select only Blue, green, and red. Then invert the order to have R-G-B
    rgb_slice = arr[:, :, 0:3][:, :, ::-1]

    # Clip the data to the quantiles, so the RGB render is not stretched to outliers,
    # Which produces dark images.
    rgb_slice = np.clip(
        rgb_slice,
        np.quantile(rgb_slice, 0.02),
        np.quantile(rgb_slice, 0.98),
    )

    # The current slice is uint16, but we want an uint8 RGB render.
    # We normalise the layer by dividing with the maximum value in the image.
    # Then we multiply it by 255 (the max of uint8) to be in the normal RGB range.
    rgb_slice = (rgb_slice / rgb_slice.max()) * 255.0

    # We then round to the nearest integer and cast it to uint8.
    rgb_slice = np.rint(rgb_slice).astype(np.uint8)

    return rgb_slice

def decode_time(encoded_time):
    doy_sin,doy_cos = encoded_time
    doy = np.arctan2((2*doy_sin-1),(2*doy_cos-1))*365/(2*np.pi)
    if doy<1:
        doy+=365
    return np.array([np.round(doy)])


def decode_coordinates(encoded_coords):
    lat_enc,long_sin,long_cos = encoded_coords
    lat = -lat_enc*180+90
    long = np.arctan2((2*long_sin-1),(2*long_cos-1))*360/(2*np.pi)
    return np.array([lat,long])

def encode_coordinates(coords):
    lat,long = coords
    lat = (-lat + 90)/180
    long_sin = (np.sin(long*2*np.pi/360)+1)/2
    long_cos = (np.cos(long*2*np.pi/360)+1)/2

    return np.array([lat,long_sin,long_cos], dtype=np.float32)

import config_geography
regions = config_geography.regions
regions_inv = config_geography.region_inv
def visualise(x, y, y_pred=None, images=5, channel_first=False, vmin=0, vmax=1, save_path=None, centers=None):
    rows = images
    if y_pred is None:
        columns = 1
    else:
        columns = 1
    i = 0
    fig = plt.figure(figsize=(10 * columns, 10 * rows))

    for idx in range(0, images):
        arr = x[idx]
        rgb_image = render_s2_as_rgb(arr, channel_first)

        i = i + 1
        fig.add_subplot(rows, columns, i)

        # coord_pred, coord_true = y_pred[idx,pos_feature_pred['coords']], y[idx,pos_feature_label['coords']]
        # kg_pred_logits,kg_true = y_pred[idx,pos_feature_pred['kg']], y[idx,pos_feature_label['kg']]
        # time_pred,time_true = y_pred[idx,pos_feature_pred['time']], y[idx,pos_feature_label['time']]
        coord_true = y[idx,pos_feature_label['coords']]

        # print(y_pred[idx])
        # print(y_pred[idx].shape)

        nearest_center = centers[np.argmax(y_pred[idx])]

        lat_pred,long_pred = decode_coordinates(nearest_center)#coord_pred)
        lat,long = decode_coordinates(coord_true)
        doy_pred, doy = 0,0#decode_time(time_pred), decode_time(time_true)
        climate_pred = 0 #config_geography.kg_map[int(np.argmax([kg_pred_logits]))]['climate_class_str']
        climate = 0 #config_geography.kg_map[int(np.argmax([kg_true]))]['climate_class_str']
        s1 = f"pred  : lat-long = {np.round(lat_pred,1),np.round(long_pred,1)} \n climate - {climate_pred} \n DoY - {doy_pred}"
        s2 = f"target: lat-long = {np.round(lat,1),np.round(long,1)} \n climate - {climate} \n DoY - {doy}"

        plt.text(25, 25, s1,fontsize=18, bbox=dict(fill=True))
        plt.text(25, 45, s2,fontsize=18, bbox=dict(fill=True))
        plt.imshow(rgb_image)
        plt.axis('on')
        plt.grid()

        # i = i + 1
        # fig.add_subplot(rows, columns, i)
        # plt.imshow(y[idx], vmin=vmin, vmax=vmax, cmap='magma')
        # plt.axis('on')
        # plt.grid()

        # if y_pred is not None:
        #     i = i + 1
        #     fig.add_subplot(rows, columns, i)
        #     plt.imshow(y_pred[idx], vmin=vmin, vmax=vmax, cmap='magma')
        #     plt.axis('on')
        #     plt.grid()

    fig.tight_layout()

    del x
    del y
    del y_pred

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()



