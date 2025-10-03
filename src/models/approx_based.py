"""Topolologically regularized autoencoder using approximation."""
import pickle
import numpy as np
import torch
import torch.nn as nn

from src.topology import PersistentHomologyCalculation, AlephPersistenHomologyCalculation 
from src.models import submodules
from src.models.base import AutoencoderModel
from totalpersistence import utilsTorch as tp

import csv

DEBUG = False
LOG_INTERVAL = 1500


def log(*args, **kwargs):
    """
    Log messages if DEBUG is True.
    """
    if DEBUG:
        print(*args, **kwargs)

class DebugLogger:
    def __init__(self, filename="debug_log.pkl"):
        self.filename = filename
        # self._initialize_file()
        self.items = []

    # def _initialize_file(self):
    #     """Crea el archivo CSV con cabeceras si no existe."""
    #     try:
    #         with open(self.filename, 'x', newline='') as file:  # 'x' falla si el archivo existe
    #             writer = csv.writer(file)
    #             writer.writerow(['Epoch', 'Timestamp', 'Params'])  # Cabecera básica
    #     except FileExistsError:
    #         # Si el archivo ya existe, leemos el último epoch
    #         with open(self.filename, 'r') as file:
    #             last_line = list(csv.reader(file))[-1]
    #             self.epoch = int(last_line[0]) if last_line[0].isdigit() else 0
    
    
    def log(self, dict_params, step):
        """Append to list and save pickle"""
        for k,v in dict_params.items():
            self.items.append({"step": step, "param": k, "value": v})
        with open(self.filename, 'wb') as file:
            pickle.dump(self.items, file)

    # def _log(self, **params):
    #     """Agrega una fila al CSV con los parámetros."""
    #     self.forward += 1
        
    #     if self.forward % 1000 == 0: 
    #         with open(self.filename, 'a', newline='') as file:
    #             writer = csv.writer(file)
    #             writer.writerow([self.forward, *params.values()])
            
    #         print(f"Log forward {self.forward} registrado.")


logger = DebugLogger()

class TopologicallyRegularizedAutoencoder(AutoencoderModel):
    """Topologically regularized autoencoder."""

    def __init__(self, lam=1., autoencoder_model='ConvolutionalAutoencoder',
                 ae_kwargs=None, toposig_kwargs=None):
        """Topologically Regularized Autoencoder.

        Args:
            lam: Regularization strength
            ae_kwargs: Kewords to pass to `ConvolutionalAutoencoder` class
            toposig_kwargs: Keywords to pass to `TopologicalSignature` class
        """
        super().__init__()
        self.lam = lam
        ae_kwargs = ae_kwargs if ae_kwargs else {}
        toposig_kwargs = toposig_kwargs if toposig_kwargs else {}
        self.topo_sig = TopologicalSignatureDistance(**toposig_kwargs)
        self.autoencoder = getattr(submodules, autoencoder_model)(**ae_kwargs)
        self.latent_norm = torch.nn.Parameter(data=torch.ones(1),
                                              requires_grad=True)
    
        self.count_forward = 0
        

    @staticmethod
    def _compute_distance_matrix(x, p=2):
        x_flat = x.view(x.size(0), -1)
        distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)
        return distances

    def forward(self, x):
        """Compute the loss of the Topologically regularized autoencoder.

        Args:
            x: Input data

        Returns:
            Tuple of final_loss, (...loss components...)

        """
        latent = self.autoencoder.encode(x)
        self.count_forward += 1
        if DEBUG ==True:
            if self.count_forward % LOG_INTERVAL == 0:
                logger.log({
                    "x": x.to("cpu").detach().numpy(),
                    "latent": latent.to("cpu").detach().numpy()
                }, step=self.count_forward)

        x_distances = self._compute_distance_matrix(x)

        dimensions = x.size()
        if len(dimensions) == 4:
            # If we have an image dataset, normalize using theoretical maximum
            batch_size, ch, b, w = dimensions
            # Compute the maximum distance we could get in the data space (this
            # is only valid for images wich are normalized between -1 and 1)
            max_distance = (2**2 * ch * b * w) ** 0.5
            x_distances = x_distances / max_distance
        else:
            # Else just take the max distance we got in the batch
            x_distances = x_distances / x_distances.max()

        latent_distances = self._compute_distance_matrix(latent)
        latent_distances = latent_distances / self.latent_norm

        # Use reconstruction loss of autoencoder
        ae_loss, ae_loss_comp = self.autoencoder(x)

        topo_error, topo_error_components = self.topo_sig(
            x_distances, latent_distances)

        # normalize topo_error according to batch_size
        batch_size = dimensions[0]
        topo_error = topo_error / float(batch_size) 
        loss = ae_loss + self.lam * topo_error
        loss_components = {
            'loss.autoencoder': ae_loss,
            'loss.topo_error': topo_error
        }
        loss_components.update(topo_error_components)
        loss_components.update(ae_loss_comp)
        return (
            loss,
            loss_components
        )

    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)


class TopologicalSignatureDistance(nn.Module):
    """Topological signature."""

    def __init__(self, sort_selected=False, mode=False,
                 match_edges='None',with_cycles= False ):
        """Topological signature computation.

        Args:
            p: Order of norm used for distance computation
            use_cycles: Flag to indicate whether cycles should be used
                or not.
        """
        super().__init__()
        self.mode = mode
        self.with_cycles = with_cycles
        self.match_edges = match_edges
        self.count_forward = 0
        use_aleph = False # tuve que agregar esto para el caso que with_cycles sea false, ya que sino sale error de que no esta declarada. 
        if self.with_cycles:
            use_aleph = True

        if use_aleph:
            print('Using aleph to compute signatures')
            assert not sort_selected
            self.signature_calculator = AlephPersistenHomologyCalculation(compute_cycles=True, sort_selected=sort_selected)
        else:
            print('Using python to compute signatures')
            #self.signature_calculator = AlephPersistenHomologyCalculation(compute_cycles=False, sort_selected=sort_selected)
            self.signature_calculator = PersistentHomologyCalculation()

    def _get_pairings(self, distances):
        pairs_0, pairs_1 = self.signature_calculator(
            distances.detach().cpu().numpy())

        return pairs_0, pairs_1

    def _select_distances_from_pairs(self, distance_matrix, pairs):
        # Split 0th order and 1st order features (edges and cycles)
        pairs_0, pairs_1 = pairs
        selected_distances = distance_matrix[(pairs_0[:, 0], pairs_0[:, 1])]                             

        if self.with_cycles:
            edges_1 = distance_matrix[(pairs_1[:, 0], pairs_1[:, 1])]
            edges_2 = distance_matrix[(pairs_1[:, 2], pairs_1[:, 3])]

            # tiempo de vida
            edge_differences = edges_2 - edges_1

            selected_distances = torch.cat(
                (selected_distances, edge_differences))

        return selected_distances

    @staticmethod
    def sig_error(signature1, signature2):
        """Compute distance between two topological signatures."""
        return ((signature1 - signature2)**2).sum(dim=-1)

    @staticmethod
    def _count_matching_pairs(pairs1, pairs2):
        def to_set(array):
            return set(tuple(elements) for elements in array)
        return float(len(to_set(pairs1).intersection(to_set(pairs2))))

    @staticmethod
    def _get_nonzero_cycles(pairs):
        all_indices_equal = np.sum(pairs[:, [0]] == pairs[:, 1:], axis=-1) == 3
        return np.sum(np.logical_not(all_indices_equal))

    # pylint: disable=W0221
    def forward(self, distances1, distances2):
        """Return topological distance of two pairwise distance matrices.

        Args:
            distances1: Distance matrix in space 1
            distances2: Distance matrix in space 2

        Returns:
            distance, dict(additional outputs)
        """

        #from totalpersistence.src.totalpersistence.utils.py import conematrix
        # Para forward the cone trick, 
        # distance1, 2 y f -> cone matrix
        # cone matrix -> pairs
        # pairs -> selected_distances
        # loss = max(selected_distances)

        distance_components = {} 
      
        loss = 0
        if self.mode == 'cone':
            # -TODO- cambiar interfaz , nueva variable, persistance type
            
            ##############################################
            
            distances2 = tp.general_position_distance_matrix_torch(distances2)
            distances1 = tp.general_position_distance_matrix_torch(distances1) #esto es tal cual esta en los test.
            
            ###############################################
            cone_distances, L = tp.conematrix_torch(distances1, distances2,cone_eps=0.0)        

            pairs = self._get_pairings(cone_distances)
            #
            selected_distances = self._select_distances_from_pairs(cone_distances, pairs)

            loss = torch.max(selected_distances[~torch.isnan(selected_distances)]) #el maximo de los que no son nan


            if DEBUG==True:
                self.count_forward += 1
                if self.count_forward % LOG_INTERVAL == 0:
                    logger.log({
                        "mode": self.mode,
                        "with_cycles": self.with_cycles,
                        "match_edges": self.match_edges,
                        "selected_distances": selected_distances.to("cpu").detach().numpy(),
                                "pairs": pairs,
                                "cone_distances": cone_distances.to("cpu").detach().numpy(),
                                "loss": loss.item(),
                                "L": L.item(),
                                "distances1": distances1.to("cpu").detach().numpy(),
                                "distances2": distances2.to("cpu").detach().numpy()}, step=self.count_forward
                                )

            distance_components['metrics.loss'] = loss
            distance_components['metrics.Lips'] = L   
            distance_components['metrics.MaxDx'] = torch.max(distances1) 
            distance_components['metrics.MaxDy'] = torch.max(distances2) 
            
        ## si modo es homology , pairings con conematrix y loss maximo
        # si modo es use_cycles, pairis comun , puede o no usar aleph, no ser full matrix , si none, matchedes o symetric 
        # o si ciclos, sin o homoology o modo fullmatrx , random excluyente
        #ciclos/sin ciclos con simetric o none
        #hacer bash con los 5 experimentos
        
        # Also count matched cycles if present
        elif self.mode == 'xy':

            
            pairs1 = self._get_pairings(distances1)
            pairs2 = self._get_pairings(distances2)
            
            distance_components['metrics.matched_pairs_0D'] = self._count_matching_pairs(
                    pairs1[0], pairs2[0])
        
            distance_components['metrics.matched_pairs_1D'] = \
                self._count_matching_pairs(pairs1[1], pairs2[1])
            nonzero_cycles_1 = self._get_nonzero_cycles(pairs1[1])
            nonzero_cycles_2 = self._get_nonzero_cycles(pairs2[1])
            distance_components['metrics.non_zero_cycles_1'] = nonzero_cycles_1
            distance_components['metrics.non_zero_cycles_2'] = nonzero_cycles_2

            if self.match_edges == 'default':
                sig1 = self._select_distances_from_pairs(distances1, pairs1)  #None no se puede usar con homology no ? debido a que los pairs es solo con la cone matrix y no con dinstance1 o 2 
                sig2 = self._select_distances_from_pairs(distances2, pairs2)
                loss = self.sig_error(sig1, sig2)

            elif self.match_edges == 'fullmatrix': # que todas las distancias entre puntos se preserven
                loss = ((distances1 - distances2)**2).sum()

            elif self.match_edges == 'symmetric':
                sig1 = self._select_distances_from_pairs(distances1, pairs1)
                sig2 = self._select_distances_from_pairs(distances2, pairs2)
                # Selected pairs of 1 on distances of 2 and vice versa
                sig1_2 = self._select_distances_from_pairs(distances2, pairs1)
                sig2_1 = self._select_distances_from_pairs(distances1, pairs2)

                distance1_2 = self.sig_error(sig1, sig1_2)
                distance2_1 = self.sig_error(sig2, sig2_1)

                distance_components['metrics.distance1-2'] = distance1_2
                distance_components['metrics.distance2-1'] = distance2_1

                loss = distance1_2 + distance2_1

            elif self.mode == 'random':
                # Create random selection in oder to verify if what we are seeing
                # is the topological constraint or an implicit latent space prior
                # for compactness
                n_instances = len(pairs1[0])
                pairs1 = torch.cat([
                    torch.randperm(n_instances)[:, None],
                    torch.randperm(n_instances)[:, None]
                ], dim=1)
                pairs2 = torch.cat([
                    torch.randperm(n_instances)[:, None],
                    torch.randperm(n_instances)[:, None]
                ], dim=1)

                sig1_1 = self._select_distances_from_pairs(
                    distances1, (pairs1, None))
                sig1_2 = self._select_distances_from_pairs(
                    distances2, (pairs1, None))

                sig2_2 = self._select_distances_from_pairs(
                    distances2, (pairs2, None))
                sig2_1 = self._select_distances_from_pairs(
                    distances1, (pairs2, None))

                distance1_2 = self.sig_error(sig1_1, sig1_2)
                distance2_1 = self.sig_error(sig2_1, sig2_2)
                distance_components['metrics.distance1-2'] = distance1_2
                distance_components['metrics.distance2-1'] = distance2_1

                loss = distance1_2 + distance2_1

        if 'loss' not in locals():
            raise RuntimeError
        return loss, distance_components

