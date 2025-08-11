import torch
import torch.nn.functional as F
from .data_losses import LpLoss, H1Loss


def difference_x(U):
    core = U[..., 1:] - U[..., :-1]
    return F.pad(core, (0, 1, 0, 0), mode='replicate')

def difference_y(U):
    core = U[:, :, 1:, :] - U[:, :, :-1, :]
    return F.pad(core, (0, 0, 0, 1), mode='replicate')


class BCLoss(object):

    def __call__(self, y_pred, **kwargs):
        loss = 0.0
        loss += F.mse_loss(y_pred[:, :, 0, :], torch.zeros_like(y_pred[:, :, 0, :]))     # top edge (y=0)
        loss += F.mse_loss(y_pred[:, :, -1, :], torch.zeros_like(y_pred[:, :, -1, :]))   # bottom edge (y=1)
        loss += F.mse_loss(y_pred[:, :, :, 0], torch.zeros_like(y_pred[:, :, :, 0]))     # left edge (x=0)
        loss += F.mse_loss(y_pred[:, :, :, -1], torch.zeros_like(y_pred[:, :, :, -1]))   # right edge (x=1)
        return loss
    


class VinoDarcyLoss(object):
    
    def __init__(self, d=2, p=2):
        super().__init__()
        # self.bc_loss_fn = BCLoss()
        # self.data_loss_fn = H1Loss(d=d)

    def __call__(self, y_pred, **kwargs):
        data_processor = kwargs['data_processor']
        a = kwargs['x'] if data_processor.in_normalizer is None else data_processor.in_normalizer.inverse_transform(kwargs['x'])
        y_pred = y_pred if data_processor.out_normalizer is None else data_processor.out_normalizer.inverse_transform(y_pred)

        a_00 = a[:, :, 0:-1, 0:-1]
        a_10 = a[:, :, 1:,   0:-1]
        a_01 = a[:, :, 0:-1, 1:]
        a_11 = a[:, :, 1:,   1:]

        dudx = difference_x(y_pred)
        dudy = difference_y(y_pred)
        dudx_0 = dudx[:, :, 0:-1, 0:-1]
        dudx_1 = dudx[:, :, 1:,   0:-1]
        dudy_0 = dudy[:, :, 0:-1, 0:-1]


        loss_grad = 0.5 * torch.sum(
            a_00 * ((dudx_0**2)/6 + (dudx_1**2)/12 + (dudy_0**2)/4 - (dudx_0*dudy_0)/6 + (dudx_1*dudy_0)/6)
            + a_01 * ((dudx_0**2)/4 + (dudx_1**2)/6 + (dudy_0**2)/4 - (dudx_0*dudx_1)/6 - (dudx_0*dudy_0)/3 + (dudx_1*dudy_0)/3)
            + a_10 * ((dudx_0**2)/12 + (dudx_1**2)/6 + (dudy_0**2)/4 - (dudx_0*dudy_0)/6 + (dudx_1*dudy_0)/6)
            + a_11 * ((dudx_0**2)/6 + (dudx_1**2)/4 + (dudy_0**2)/4 - (dudx_0*dudx_1)/6 - (dudx_0*dudy_0)/3 + (dudx_1*dudy_0)/3)
        )

        loss_int = torch.sum(y_pred) / ((a.shape[2] - 1) * (a.shape[3] - 1))
        return loss_grad - loss_int

        # loss_bc = self.bc_loss_fn(y_pred, **kwargs)
        # loss_data = self.data_loss_fn(y_pred, **kwargs)
        # return loss_grad - loss_in + loss_bc + loss_data