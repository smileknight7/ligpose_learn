import torch
import torch.nn.functional as F
from torch_scatter import scatter_min, scatter_mean, scatter_add
from einops import rearrange, repeat

from utils.focal_loss import FocalLoss
from utils.data_utils import batch_index_select


class StructLoss(torch.nn.Module):
    def __init__(self, args):
        super(StructLoss, self).__init__()
        self.coor_scale = args.coor_scale
        self.focal_loss = FocalLoss()

        self.gamma_1 = args.gamma_1
        self.gamma_2 = args.gamma_2

        self.gamma_p_x_1 = args.gamma_p_x_1
        self.gamma_p_x_2 = args.gamma_p_x_2
        self.gamma_l_x = args.gamma_l_x
        self.gamma_edge = args.gamma_edge
        self.gamma_noise = args.gamma_noise

    def forward(self, tup_pred, complex_graph, epoch=1e+5):
        coor_hidden, aff_pred, p_x_pred_1, p_x_pred_2, l_x_pred, edge_pred = tup_pred

        ################################################################################################################
        # for coor
        ################################################################################################################
        # for coordinate prediction
        coor_pred = coor_hidden
        coor_true_cycle = batch_index_select(complex_graph.coor_true,
                                             complex_graph.node_sampling_loc[complex_graph.cycle_i])
        coor_true = coor_true_cycle
        coor_pred = rearrange(coor_pred, 'h b n c -> (b n) h c')[
            complex_graph.ligand_node_loc_after_sampling_flat]  # to (batch*n_atom, n_hidden_state, 3)
        coor_true = rearrange(coor_true, 'b n c -> (b n) c')[
            complex_graph.ligand_node_loc_after_sampling_flat]  # to (batch*n_atom, 3)
        coor_pred = coor_pred[complex_graph.ligand_match]  # to (batch*n_atom*match, n_hidden_state, 3)
        coor_true = coor_true[complex_graph.ligand_nomatch]  # to (batch*n_atom*match, 3)

        coor_loss = torch.norm(coor_pred - coor_true.unsqueeze(dim=1), dim=-1, p=2)  # to (batch*n_atom*match, n_hidden_state)
        coor_loss = scatter_mean(coor_loss, complex_graph.scatter_ligand_1, dim=0)  # to (batch*match, n_hidden_state)
        coor_loss = scatter_min(coor_loss, complex_graph.scatter_ligand_2, dim=0)[0]  # to (batch, n_hidden_state)
        coor_grad_loss = coor_loss[:, -1] + coor_loss[:, 1:-1].mean(dim=-1)  # to (batch,)
        coor_grad_loss = (coor_grad_loss * complex_graph.coor_mask).mean()
        coor_eval_loss = coor_loss[:, -1].mean()

        # for RMSD
        with torch.no_grad():
            coor_pred = coor_hidden[-1] * self.coor_scale
            coor_true = coor_true_cycle * self.coor_scale
            coor_pred = rearrange(coor_pred, 'b n c -> (b n) c')[
                complex_graph.ligand_node_loc_after_sampling_flat]  # to (batch*n_atom, n_hidden_state, 3)
            coor_true = rearrange(coor_true, 'b n c -> (b n) c')[
                complex_graph.ligand_node_loc_after_sampling_flat]  # to (batch*n_atom, 3)
            coor_pred = coor_pred[complex_graph.ligand_match]  # to (batch*n_atom*match, 3)
            coor_true = coor_true[complex_graph.ligand_nomatch]  # to (batch*n_atom*match, 3)

            coor_loss = ((coor_pred - coor_true) ** 2).sum(dim=-1)  # to (batch*n_atom*match,)
            coor_loss = scatter_add(coor_loss, complex_graph.scatter_ligand_1, dim=0)  # to (batch*match,)
            coor_loss = scatter_min(coor_loss, complex_graph.scatter_ligand_2, dim=0)[0]  # to (batch,)
            rmsd_loss = (coor_loss / complex_graph.len_ligand) ** 0.5
            rmsd_value = rmsd_loss.mean()
            rmsd_rate = (rmsd_loss < 2.0).to(rmsd_loss.dtype).mean()


        ################################################################################################################
        # for affinity
        ################################################################################################################
        # for affinity prediction
        aff_true = complex_graph.aff_true
        aff_loss = (torch.pow(aff_true - aff_pred, 2) * complex_graph.aff_mask).mean()


        ################################################################################################################
        # for mask
        ################################################################################################################
        cycle_i = complex_graph.cycle_i

        p_x_mask_scatter = complex_graph.x_batch_info_cycle[cycle_i].reshape(-1)[
            complex_graph.p_x_mask_bool_cycle[cycle_i].reshape(-1)]
        l_x_mask_scatter = complex_graph.x_batch_info_cycle[cycle_i].reshape(-1)[
            complex_graph.l_x_mask_bool_cycle[cycle_i].reshape(-1)]
        edge_mask_scatter = complex_graph.edge_batch_info_cycle[cycle_i].reshape(-1)[
            complex_graph.edge_mask_bool_cycle[cycle_i].reshape(-1)]

        p_x_true_1 = complex_graph.p_x_mask_label_1_cycle[cycle_i].reshape(-1)[
            complex_graph.p_x_mask_bool_cycle[cycle_i].reshape(-1)]
        p_x_true_2 = complex_graph.p_x_mask_label_2_cycle[cycle_i].reshape(-1)[
            complex_graph.p_x_mask_bool_cycle[cycle_i].reshape(-1)]
        l_x_true = complex_graph.l_x_mask_label_cycle[cycle_i].reshape(-1)[
            complex_graph.l_x_mask_bool_cycle[cycle_i].reshape(-1)]
        edge_true = complex_graph.edge_mask_label_cycle[cycle_i].reshape(-1)[
            complex_graph.edge_mask_bool_cycle[cycle_i].reshape(-1)]

        p_x_mask_loss_1 = self.focal_loss(p_x_pred_1, p_x_true_1)
        p_x_mask_loss_2 = self.focal_loss(p_x_pred_2, p_x_true_2)
        l_x_mask_loss = self.focal_loss(l_x_pred, l_x_true)
        edge_mask_loss = self.focal_loss(edge_pred, edge_true)

        p_x_mask_loss_1 = scatter_mean(p_x_mask_loss_1, p_x_mask_scatter, dim=0).mean()
        p_x_mask_loss_2 = scatter_mean(p_x_mask_loss_2, p_x_mask_scatter, dim=0).mean()
        l_x_mask_loss = scatter_mean(l_x_mask_loss, l_x_mask_scatter, dim=0).mean()
        edge_mask_loss = scatter_mean(edge_mask_loss, edge_mask_scatter, dim=0).mean()


        ################################################################################################################
        # for coor noise
        ################################################################################################################
        coor_pred = tup_pred[0]
        coor_noise_pred = rearrange(coor_pred, 'h b n c -> (b n) h c')[
            complex_graph.coor_noise_bool_cycle[cycle_i].reshape(-1)]
        coor_noise_true = rearrange(complex_graph.coor_noise_true_cycle[cycle_i], 'b n c -> (b n) () c')[
            complex_graph.coor_noise_bool_cycle[cycle_i].reshape(-1)]
        coor_noise_scatter = complex_graph.x_batch_info_cycle[cycle_i].reshape(-1)[
            complex_graph.coor_noise_bool_cycle[cycle_i].reshape(-1)]

        # for coordinate prediction
        noise_loss = (coor_noise_pred - coor_noise_true).norm(p=2, dim=-1)
        noise_loss = scatter_mean(noise_loss, coor_noise_scatter, dim=0)
        noise_grad_loss = (noise_loss[:, -1] + noise_loss[:, 1:-1].mean(dim=-1)).mean()


        ################################################################################################################
        # for output
        ################################################################################################################
        grad_loss = self.gamma_1 * coor_grad_loss + self.gamma_2 * aff_loss + \
                    self.gamma_p_x_1 * p_x_mask_loss_1 + \
                    self.gamma_p_x_2 * p_x_mask_loss_2 + \
                    self.gamma_l_x * l_x_mask_loss + \
                    self.gamma_edge * edge_mask_loss + \
                    self.gamma_noise * noise_grad_loss

        eval_loss = dict(
            grad_loss=grad_loss,
            coor_loss=coor_grad_loss,

            coor_metric=coor_eval_loss,
            rmsd_value=rmsd_value,
            rmsd_rate=rmsd_rate,

            aff_loss=aff_loss,
            aff_metric=aff_loss,

            p_x_mask_1_loss=p_x_mask_loss_1,
            p_x_mask_2_loss=p_x_mask_loss_2,
            l_x_mask_loss=l_x_mask_loss,
            edge_mask_loss=edge_mask_loss,
            noise_loss=noise_grad_loss,
            p_x_mask_1_metric=p_x_mask_loss_1,
            p_x_mask_2_metric=p_x_mask_loss_2,
            l_x_mask_metric=l_x_mask_loss,
            edge_mask_metric=edge_mask_loss,
            noise_metric=noise_grad_loss,
        )
        eval_loss = {k: v.detach().cpu().numpy() for k, v in eval_loss.items()}

        return grad_loss, eval_loss


class ScreenLoss(torch.nn.Module):
    def __init__(self, args):
        super(ScreenLoss, self).__init__()
        self.coor_scale = args.coor_scale
        self.focal_loss = FocalLoss()

        self.gamma_1 = args.gamma_1
        self.gamma_2 = args.gamma_2
        self.gamma_3 = args.gamma_3

    def forward(self, tup_pred, complex_graph, epoch=1e+5):
        coor_hidden, aff_pred, scr_pred = tup_pred

        ################################################################################################################
        # for coor
        ################################################################################################################
        # for coordinate prediction
        coor_pred = coor_hidden
        coor_true_cycle = batch_index_select(complex_graph.coor_true,
                                             complex_graph.node_sampling_loc[complex_graph.cycle_i])
        coor_true = coor_true_cycle
        coor_pred = rearrange(coor_pred, 'h b n c -> (b n) h c')[
            complex_graph.ligand_node_loc_after_sampling_flat]  # to (batch*n_atom, n_hidden_state, 3)
        coor_true = rearrange(coor_true, 'b n c -> (b n) c')[
            complex_graph.ligand_node_loc_after_sampling_flat]  # to (batch*n_atom, 3)
        coor_pred = coor_pred[complex_graph.ligand_match]  # to (batch*n_atom*match, n_hidden_state, 3)
        coor_true = coor_true[complex_graph.ligand_nomatch]  # to (batch*n_atom*match, 3)

        coor_loss = torch.norm(coor_pred - coor_true.unsqueeze(dim=1), dim=-1,
                               p=2)  # to (batch*n_atom*match, n_hidden_state)
        coor_loss = scatter_mean(coor_loss, complex_graph.scatter_ligand_1, dim=0)  # to (batch*match, n_hidden_state)
        coor_loss = scatter_min(coor_loss, complex_graph.scatter_ligand_2, dim=0)[0]  # to (batch, n_hidden_state)
        coor_grad_loss = coor_loss[:, -1] + coor_loss[:, 1:-1].mean(dim=-1)  # to (batch,)
        coor_grad_loss = (coor_grad_loss * complex_graph.coor_mask).mean()
        coor_eval_loss = (coor_loss[:, -1] * complex_graph.coor_mask).mean()


        ################################################################################################################
        # for affinity
        ################################################################################################################
        # for affinity prediction
        aff_true = complex_graph.aff_true
        aff_loss = (torch.pow(aff_true - aff_pred, 2) * complex_graph.aff_mask).mean()


        ################################################################################################################
        # for screening
        ################################################################################################################
        scr_true = complex_graph.screening_label
        scr_loss = self.focal_loss(scr_pred, scr_true, bi=True).mean()


        ################################################################################################################
        # for output
        ################################################################################################################
        grad_loss = self.gamma_1 * coor_grad_loss + self.gamma_2 * aff_loss + self.gamma_3 * scr_loss

        eval_loss = dict(
            grad_loss=grad_loss,
            coor_loss=coor_grad_loss,
            coor_metric=coor_eval_loss,

            aff_loss=aff_loss,
            aff_metric=aff_loss,

            scr_loss=scr_loss,
            scr_metric=scr_loss,
        )
        eval_loss = {k: v.detach().cpu().numpy() for k, v in eval_loss.items()}

        return grad_loss, eval_loss


































