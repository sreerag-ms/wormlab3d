import torch
import torch.nn.functional as F

VOL_SIZE = 11


def make_gaussian_kernel(sigma):
    ks = int(sigma * 5)
    if ks % 2 == 0:
        ks += 1
    ts = torch.linspace(-ks // 2, ks // 2 + 1, ks)
    gauss = torch.exp((-(ts / sigma)**2 / 2))
    kernel = gauss / gauss.sum()

    return kernel


def test_3d_gaussian_blur(blur_sigma=2):
    # Make a test volume
    vol = torch.zeros([VOL_SIZE] * 3)
    vol[VOL_SIZE // 2, VOL_SIZE // 2, VOL_SIZE // 2] = 1

    # 3D convolution
    vol_in = vol.reshape(1, 1, *vol.shape)
    k = make_gaussian_kernel(blur_sigma)
    k3d = torch.einsum('i,j,k->ijk', k, k, k)
    k3d = k3d / k3d.sum()
    vol_3d = F.conv3d(vol_in, k3d.reshape(1, 1, *k3d.shape), stride=1, padding=len(k) // 2)
    vol_3d = vol_3d.squeeze(1)

    # Separable 1D convolutions
    vol_in = vol[None, None, ...]
    k1d = k[None, None, :, None, None]
    for i in range(3):
        vol_in = vol_in.permute(0, 1, 4, 2, 3)
        vol_in = F.conv3d(vol_in, k1d, stride=1, padding=(len(k) // 2, 0, 0))
    vol_3d_sep = vol_in.squeeze(1)

    assert torch.allclose(vol_3d, vol_3d_sep)


if __name__ == '__main__':
    test_3d_gaussian_blur()
