import numpy as np
from utils.encoding_utils import encode_coordinates, decode_coordinates
from matplotlib import pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
from tqdm import tqdm 
import torch
import torch.nn as nn


class MvMFLoss(nn.Module):
    '''
    Loss function that exploits spherical geometry of the Earth.
    The loss is based on the von Mises-Fisher (vMF) distribution which is the spherical analogue of a Gaussian.
    vMF(y, center, density) = density / sinh(density) * exp( density * < center, y > )
    The mixture distribution (MvMF) for a point y on the sphere is a convex combination of vMFs where each vMF is characterised by its own center and density. 
    MvMF(y, centers, densities, weights) = \sum_i Gamma_i * vMF(y, center_i, density_i).
    These mixture weights Gamma are the outputs of the model and can be interpreted as logits to which cluster (class) it belongs.
    The loss itself is the negative logarithm of the distribution: loss = - log(MvMF)

    Initialisation parameters:
    center_inits, array:           array of shape (n_clusters, 2) containing the lat-long coordinates of all the centers
    density_exponent_inits, array: array of shape (n_clusters) containing the exponents that characterise the density ( density = exp(density_exponent_inits) )
    centers_learnable, bool:       wrap the centers in a parameter so they can be updated during training
    density_learnable, bool:       wrap the density exponents in a parameter so they can be updated during training
    softmax_input, bool:           Gamma_i should sum to one, so if the output of the model are raw logits, they must be softmaxed first
    device:                        Put density and centers on cpu or gpu

    Credits to Izbicki, Michael et al. “Exploiting the Earth's Spherical Geometry to Geolocate Images.” ECML/PKDD (2019). 
    
    '''
    
    def __init__(self, center_inits, density_inits,device, centers_learnable=False, density_learnable=True, softmax_input=True):
        super(MvMFLoss, self).__init__()
        
        centers = torch.FloatTensor(center_inits).to(device)
        densities =  torch.FloatTensor(np.array([density_inits])).to(device)

        self.centers = nn.Parameter(centers) if centers_learnable else centers
        self.densities = nn.Parameter(densities) if density_learnable else densities

        self.prep_weights = nn.Softmax(dim=1) if softmax_input else nn.Identity()


    def encode_coordinates_torch(self, coords):
        '''
        inputs coords in lat-long of shape (batch_size, 2)
        output: encoded coords of shape (batch_size, 3)
        '''
        encoded_coords = torch.zeros((coords.shape[0],3),dtype=coords.dtype)
        # lat,long = coords.T
        # lat = (-lat + 90)/180
        # long_sin = (torch.sin(long*2*np.pi/360)+1)/2
        # long_cos = (torch.cos(long*2*np.pi/360)+1)/2
        
        encoded_coords[:,0] = (-coords[:,0] + 90)/180 
        encoded_coords[:,1] = (torch.sin(coords[:,1]*2*torch.pi/360) + 1)/2
        encoded_coords[:,2] = (torch.cos(coords[:,1]*2*torch.pi/360) + 1)/2

        return encoded_coords

    def dist_encoded(self, coord_true, centers):
        '''
        Calculates cosine of angle between points on the sphere (result between -1 and 1)
        coord_true: encoded coordinates
        centers: encoded coordinates of the centers defining the MvMF distrubution
        if coord_true.shape = (batch_size, 3) and centers.shape = (n_clusters,3) the output will have shape (batch_size, n_clusters)
        '''

        lat1_enc,long1_sin, long1_cos = coord_true[:,0], (2*coord_true[:,1]-1), (2*coord_true[:,2]-1)
        lat2_enc,long2_sin, long2_cos = centers[:,0],(2*centers[:,1]-1), (2*centers[:,2]-1)
        lat1_r = lat1_enc*np.pi
        lat2_r = lat2_enc*np.pi

        cos_dif = torch.outer(long1_cos,long2_cos) + torch.outer(long1_sin,long2_sin)
        
        return torch.outer(torch.sin(lat1_r),torch.sin(lat2_r))*cos_dif +torch.outer(torch.cos(lat1_r),torch.cos(lat2_r))


    def forward(self, y_pred, y_true):
        '''
        forward pass where y_true are the encoded coordinates of the samples in the batch and y_pred are the predicted logits for each center.
        If input_softmax==True, y_pred will be softmaxed first.
        The safer version of the forwards pass uses the identity log( \sum_i exp(c_i) ) = max(c) + log( \sum_i exp( c_i - max(c) ) )
        '''

        weights = self.prep_weights(y_pred)

        encoded_coordinate = y_true #[:,pos_feature_label['coords']]

        # # naive implementation
        # dens = torch.exp(self.densities)
        # normalisation = dens/torch.sinh(dens)
        # dist = self.dist_encoded(coord_true=encoded_coordinate, directions=self.directions)
        # exp_factor = torch.exp(torch.mul(dens, dist))
        # weighted_sum = torch.sum(weights * torch.mul(normalisation, exp_factor), dim=-1)
        # loss_naive = -torch.log(weighted_sum)

        # safer version based on evaluation of log-sum-exp
        centers_encoded = self.encode_coordinates_torch(self.centers)
        dist = self.dist_encoded(coord_true=encoded_coordinate, centers=centers_encoded)
        
        gamma = torch.log(2*weights) + self.densities + torch.mul(torch.exp(self.densities), dist-1)
        gamma_max = torch.max(gamma, dim=-1, keepdim=True)[0]
        deltas = gamma - gamma_max
        loss = -(gamma_max + torch.log(torch.sum(torch.exp(deltas),dim=-1)))

        loss = torch.mean(loss)

        return loss


# centers = np.load('centers_encoded.npy')
# criterion = MvMFLoss(center_inits=CENTERS, density_inits=[8 for i in range(len(centers))], device='cpu')
# y_true = torch.rand((4,3))#.cuda()
# print(criterion(y_pred=y_pred, y_true=y_true))


def get_coordinate_array(coord_bbox, shape):
    '''
    coord_bbox in format [lat_min,lat_max,lng_min,lng_max]
    shape =(n_pixels in latitude, n_pixels in longitude)
    
    returns coordinate meshgrid determined by given boundingbox and reference shape
    '''
    stepsize_lat=(coord_bbox[1]-coord_bbox[0])/shape[0]
    stepsize_lng=(coord_bbox[3]-coord_bbox[2])/shape[1]
    start_lat = coord_bbox[0]+stepsize_lat/2
    stop_lat = coord_bbox[1]+stepsize_lat/2
    start_lng = coord_bbox[2]+stepsize_lng/2
    stop_lng = coord_bbox[3]+stepsize_lng/2

    xy = np.mgrid[start_lat:stop_lat:stepsize_lat,start_lng:stop_lng:stepsize_lng]

    return xy


class MvMF_visuals():
    '''
    Class for visualising the output probability map for a model trained with the MvMF loss

    '''
    
    def __init__(self, centers, densities):
        self.centers = centers 
        self.densities = densities 

        self.coords_global = get_coordinate_array([-89,89,-178.5,178.5], shape=(600,600)).reshape((2,-1))
        self.coords_global_enc = np.array([encode_coordinates(self.coords_global[:,i]) for i in range(self.coords_global.shape[1])]).swapaxes(0,1)


    def dist_encoded(self, coords, centers):
        '''
        Calculates cosine of angle between 2 points on the sphere (result between -1 and 1)
        coords: encoded coordinates
        centers: encoded coordinates of the centers defining the MvMF distrubution
        if coords.shape = (N, 3) and centers.shape = (n_clusters,3) the output will have shape (N, n_clusters) 
        '''

        lat1_enc,long1_sin, long1_cos = coords[0,:], (2*coords[1,:]-1), (2*coords[2,:]-1)
        lat2_enc,long2_sin, long2_cos = centers[0,:],(2*centers[1,:]-1), (2*centers[2,:]-1)

        lat1_r = lat1_enc*np.pi
        lat2_r = lat2_enc*np.pi

        cos_dif = np.zeros((len(long2_cos),len(long1_cos)))
        cos_dif = np.outer(long2_cos,long1_cos) + np.outer(long2_sin,long1_sin)
        
        return np.outer(np.sin(lat2_r),np.sin(lat1_r))*cos_dif +np.outer(np.cos(lat2_r),np.cos(lat1_r))


    def MvMF_llh(self, coordinate,weights, density,centers):
        '''
        coordinates: encoded coordinates
        weights: output of the model, raw_logits of shape (N, n_clusters). Will be softmaxed
        centers, array: array of shape (n_clusters, 3) containing the encoded coordinates of all the centers
        density_exponents, array of shape (n_clusters) containing the exponents that characterise the density ( density = exp(density_exponent_inits) ) 

        output: array of length len(coordinates) assigning to each coordinate a 'probability'. When coordinates are global (spanning [-90,90,-180,180]) the result is a global probability map.
        '''
        
        exp_weights = np.exp(weights)
        soft_maxed = exp_weights/np.sum(exp_weights, axis=-1, keepdims=True)

        center_activations = np.mean(soft_maxed, axis=0)
        dist = self.dist_encoded(coordinate, centers).T 


        llh_components = np.exp(density.squeeze())* (dist-1) 

        return np.matmul(np.exp(llh_components), center_activations.T)



    # def MvMF_loss_best_torch(self, coordinate,weights, density,centers, batch_size=64):

    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     final_probs = np.zeros(coordinate.shape[1])-10000

    #     distances = self.dist_encoded(coordinate,centers).T

    #     closest_center_dist = np.max(distances, axis=-1)
    #     closest_points = np.where(closest_center_dist>0.90)
    #     relevant_points = distances[closest_points]
        
    #     batches = distances.shape[0]//batch_size +1
        
    #     ds = torch.zeros(relevant_points.shape[0]).to(device)
    #     relevant_points = torch.tensor(relevant_points).to(device)

    #     exp_weights = np.exp(weights)
    #     weights = exp_weights/np.sum(exp_weights, axis=-1, keepdims=True)

    #     density = torch.tensor(density).to(device)
    #     weights = torch.tensor(weights).to(device)

    #     for i in tqdm(range(batches)):
            
    #         dist = relevant_points[batch_size*i:batch_size*(i+1)]
    #         gamma_points = torch.exp(density.squeeze())* (dist-1)
    #         gamma_weights = torch.log(2*weights) + density

    #         gamma = gamma_points[:,None,:] + gamma_weights[None,:,:]
    #         gamma_max = torch.max(gamma, dim=-1, keepdims=True)[0]
    #         deltas = gamma - gamma_max
    #         loss = gamma_max + torch.log(torch.sum(torch.exp(deltas),axis=-1, keepdims=True))
    #         loss = torch.mean(loss.squeeze(), axis=-1)
    #         ds[batch_size*i:batch_size*(i+1)] = loss #.cpu().numpy()

    #     final_probs[closest_points] = ds.cpu().numpy()
    #     final_probs = np.exp(final_probs)

    #     import pdb; pdb.set_trace()

    #     return final_probs


    def distribution_global_pred(self, weights):
        '''
        weights:    Output of the model, raw_logits of shape (n_patches, n_clusters). All weights should come from patches of the same S2 tile.
                    Global probability map is computed by taking into account all predictions on the smaller (e.g. 128x128) patches. Output weights will be softmaxed first.

        outputs
        global_activations:  global probability map based on all weights
        pred_coords:         coordinates (in lat-long) of the most likely centers for each patch
        counts:              The amount of times a center was the most likely
        '''

        global_activations = self.MvMF_llh(self.coords_global_enc, weights = weights, density=self.densities, centers=self.centers.T )
        # global_activations = self.MvMF_loss_best_torch(self.coords_global_enc, weights = weights, density=self.densities, centers=self.centers.T )
        activated_centers, counts = np.unique(np.argmax(weights, axis=1), return_counts=True)
        centers_dec = np.array([decode_coordinates(c) for c in self.centers])
        pred_coord = centers_dec[activated_centers]

        return global_activations, pred_coord, counts

    def plot_globe_dist(self, ax, dss, pred_coord, coord_true, counts):
        '''
        ax:  axis on which to draw probability map
        dss: global probility map for the coordinates
        pred_coords: coordinates (in lat-long) of the most likely centers for each patch
        coord_true: ground truth (longitude, latitude) for the S2 image 
        counts: the amount of times a center was the most likely

        Draws global probabilty map with activated centers
        '''



        #ax = fig.add_subplot(1,1,1, projection=crs.Robinson())
        ax.set_global()
        #ax.set_extent([-20,20,-20,20])
        ax.add_feature(cfeature.COASTLINE, zorder=2, color='lightgrey', alpha=0.5)
        ax.add_feature(cfeature.BORDERS, zorder=2,color='lightgrey', alpha=0.2)
        ax.add_feature(cfeature.OCEAN, zorder=3, alpha=0.2, color='black')

        # ax.set_extent([20,55,-50,0])
        # ax.gridlines()
        a = plt.scatter(x=self.coords_global[1,:], y=self.coords_global[0,:],  #x=long, y=lat
                    #color="dodgerblue",
                    c=dss,
                    s=1,
                    # cmap='hot',
                    alpha=0.5,
                    transform=crs.PlateCarree(),
                    zorder=1
                    #norm=matplotlib.colors.LogNorm() ## Important
            )

        plt.scatter(x=pred_coord[:,1], y=pred_coord[:,0],  #x=long, y=lat
                    color="white",
                    #c=ds/np.max(ds),
                    s=counts/10,
                    alpha=0.8,
                    transform=crs.PlateCarree(),
                    zorder=4) ## Important

        plt.scatter(x=coord_true[1], y=coord_true[0],  #x=long, y=lat
                    color="red",
                    #c=ds/np.max(ds),
                    s=40,
                    marker='*',
                    alpha=1,
                    transform=crs.PlateCarree(),
                    zorder=5) ## Important
        plt.colorbar()
        plt.title('coordinate distribution over patches')
    

        
