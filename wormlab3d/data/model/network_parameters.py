import datetime
from abc import abstractmethod
from typing import Union

from mongoengine import *
from wormlab3d.midlines3d.rotae_net import RotAENet
from wormlab3d.nn.models.aenet import AENet
from wormlab3d.nn.models.basenet import BaseNet
from wormlab3d.nn.models.densenet import DenseNet
from wormlab3d.nn.models.fcnet import FCNet
from wormlab3d.nn.models.msrdn import MSRDN
from wormlab3d.nn.models.nunet import NuNet
from wormlab3d.nn.models.pyramidnet import PyramidNet
from wormlab3d.nn.models.rdn import RDN
from wormlab3d.nn.models.rednet import RedNet
from wormlab3d.nn.models.resnet import ResNet, RES_SHORTCUT_OPTIONS
from wormlab3d.nn.models.resnet1d import ResNet1d

NETWORK_TYPES = ['densenet', 'fcnet', 'resnet', 'resnet1d', 'pyramidnet', 'aenet', 'nunet', 'rdn', 'red', 'rotae']


class NetworkParameters(Document):
    network_type = StringField(required=True, choices=NETWORK_TYPES)
    created = DateTimeField(required=True, default=datetime.datetime.utcnow)

    # Common parameters
    input_shape = ListField(required=True)
    output_shape = ListField(required=True)
    dropout_prob = FloatField(default=0)

    meta = {
        'allow_inheritance': True,
        'ordering': ['-created']
    }

    @abstractmethod
    def instantiate_network(self, build_model: bool = True) -> BaseNet:
        pass


class NetworkParametersFC(NetworkParameters):
    layers_config = ListField(IntField(), required=True)

    def instantiate_network(self, build_model: bool = True) -> FCNet:
        return FCNet(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            layers_config=self.layers_config,
            dropout_prob=self.dropout_prob,
            build_model=build_model
        )


class NetworkParametersAE(NetworkParameters):
    n_init_channels = IntField(required=True)
    growth_rate = IntField(required=True)
    compression_factor = IntField(required=True)
    block_config_enc = ListField(IntField(), required=True)
    block_config_dec = ListField(IntField(), required=True)
    latent_size = IntField(required=True)

    def instantiate_network(self, build_model: bool = True) -> AENet:
        assert self.input_shape == self.output_shape
        return AENet(
            data_shape=self.input_shape,
            latent_size=self.latent_size,
            n_init_channels=self.n_init_channels,
            growth_rate=self.growth_rate,
            block_config_enc=self.block_config_enc,
            block_config_dec=self.block_config_dec,
            compression_factor=self.compression_factor,
            dropout_prob=self.dropout_prob,
            build_model=build_model
        )


class NetworkParametersResNet(NetworkParameters):
    n_init_channels = IntField(required=True)
    blocks_config = ListField(IntField(), required=True)
    shortcut_type = StringField(choices=RES_SHORTCUT_OPTIONS, required=True)
    use_bottlenecks = BooleanField(required=True)

    def instantiate_network(self, build_model: bool = True) -> ResNet:
        return ResNet(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            n_init_channels=self.n_init_channels,
            block_config=self.blocks_config,
            shortcut_type=self.shortcut_type,
            use_bottlenecks=self.use_bottlenecks,
            dropout_prob=self.dropout_prob,
            build_model=build_model
        )


class NetworkParametersResNet1d(NetworkParameters):
    n_init_channels = IntField(required=True)
    blocks_config = ListField(IntField(), required=True)
    shortcut_type = StringField(choices=RES_SHORTCUT_OPTIONS, required=True)
    use_bottlenecks = BooleanField(required=True)

    def instantiate_network(self, build_model: bool = True) -> ResNet1d:
        return ResNet1d(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            n_init_channels=self.n_init_channels,
            block_config=self.blocks_config,
            shortcut_type=self.shortcut_type,
            use_bottlenecks=self.use_bottlenecks,
            dropout_prob=self.dropout_prob,
            build_model=build_model
        )


class NetworkParametersDenseNet(NetworkParameters):
    n_init_channels = IntField(required=True)
    growth_rate = IntField(required=True)
    compression_factor = IntField(required=True)
    blocks_config = ListField(IntField(), required=True)
    shortcut_type = StringField(choices=RES_SHORTCUT_OPTIONS, required=True)
    use_bottlenecks = BooleanField(required=True)

    def instantiate_network(self, build_model: bool = True) -> DenseNet:
        return DenseNet(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            n_init_channels=self.n_init_channels,
            growth_rate=self.growth_rate,
            block_config=self.blocks_config,
            compression_factor=self.compression_factor,
            dropout_prob=self.dropout_prob,
            build_model=build_model
        )


class NetworkParametersPyramidNet(NetworkParameters):
    n_init_channels = IntField(required=True)
    blocks_config = ListField(IntField(), required=True)
    alpha = IntField(required=True)
    shortcut_type = StringField(choices=RES_SHORTCUT_OPTIONS, required=True)
    use_bottlenecks = BooleanField(required=True)

    def instantiate_network(self, build_model: bool = True) -> PyramidNet:
        return PyramidNet(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            n_init_channels=self.n_init_channels,
            block_config=self.blocks_config,
            shortcut_type=self.shortcut_type,
            use_bottlenecks=self.use_bottlenecks,
            alpha=self.alpha,
            dropout_prob=self.dropout_prob,
            build_model=build_model
        )


class NetworkParametersNuNet(NetworkParameters):
    n_init_channels = IntField(required=True)
    depth = IntField(required=True)
    up_mode = StringField(required=True)

    def instantiate_network(self, build_model: bool = True) -> NuNet:
        assert self.input_shape == self.output_shape
        return NuNet(
            data_shape=self.input_shape,
            n_init_channels=self.n_init_channels,
            depth=self.depth,
            up_mode=self.up_mode,
            dropout_prob=self.dropout_prob,
            build_model=build_model
        )


class NetworkParametersRDN(NetworkParameters):
    D = IntField(required=True)
    K = IntField(required=True)
    M = IntField(required=True)
    N = IntField(required=True)
    G = IntField(required=True)
    C_out = IntField(required=False)
    kernel_size = IntField(required=True)
    activation = StringField(required=True)
    batch_norm = BooleanField(required=True)
    act_out = StringField()

    def instantiate_network(self, build_model: bool = True) -> Union[RDN, MSRDN]:
        model_params = {
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'K': self.K,
            'M': self.M,
            'N': self.N,
            'G': self.G,
            'kernel_size': self.kernel_size,
            'activation': self.activation,
            'batch_norm': self.batch_norm,
            'act_out': self.act_out,
        }
        if self.D > 1:
            model_params['D'] = self.D
            return MSRDN(**model_params)
        else:
            return RDN(**model_params)


class NetworkParametersRED(NetworkParameters):
    latent_size = IntField(required=True)
    K = IntField(required=True)
    M = IntField(required=True)
    N = IntField(required=True)
    G = IntField(required=True)
    discriminator_layers = ListField(IntField(), required=True)
    kernel_size = IntField(required=True)
    activation = StringField(required=True)
    batch_norm = BooleanField(required=True)
    act_out = StringField()

    def instantiate_network(self, build_model: bool = True) -> RedNet:
        model_params = {
            'input_shape': self.input_shape,
            'latent_size': self.latent_size,
            'parameters_size': self.output_shape[0],
            'K': self.K,
            'M': self.M,
            'N': self.N,
            'G': self.G,
            'discriminator_layers': self.discriminator_layers,
            'kernel_size': self.kernel_size,
            'activation': self.activation,
            'batch_norm': self.batch_norm,
            'act_out': self.act_out,
        }
        return RedNet(**model_params)


class NetworkParametersRotAE(NetworkParameters):
    c2d_net = ReferenceField(NetworkParameters, required=True)
    c3d_net = ReferenceField(NetworkParameters, required=True)
    d0_net = ReferenceField(NetworkParameters)
    d2d_net = ReferenceField(NetworkParameters)
    d3d_net = ReferenceField(NetworkParameters)

    def instantiate_network(self, build_model: bool = True) -> RotAENet:
        pass
