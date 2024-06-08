from .encoder import SpatialEncoder, ImageEncoder
from .resnetfc import ResnetFC
from .backbone import NetWrapper

def make_mlp(conf, d_in, d_latent, d_out, **kwargs):

    mlp_type = conf.get_string("type", "transformer") 
    if mlp_type == "resnet":
        net = ResnetFC.from_conf(conf, d_in=d_in, d_latent=d_latent, d_out=d_out, **kwargs)
    elif mlp_type == 'transformer':
        net = NetWrapper.from_conf(conf, d_in=d_in, d_latent=d_latent, d_out=d_out, **kwargs)
    else:
        raise NotImplementedError("Unsupported MLP type")
    return net


def make_encoder(conf, **kwargs):
    enc_type = conf.get_string("type", "spatial")  # spatial | global
    if enc_type == "spatial":
        net = SpatialEncoder.from_conf(conf, **kwargs)
    elif enc_type == "global":
        net = ImageEncoder.from_conf(conf, **kwargs)
    else:
        raise NotImplementedError("Unsupported encoder type")
    return net
