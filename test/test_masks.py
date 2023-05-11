import sys
sys.path.append('masks/mask_class')

from mask_class import FlatCam_Mask, PhlatCam_Mask, FZA_Mask


def test_masks():
    print('\nTesting...')
    mask1 = FlatCam_Mask(None, None, None, None, 'MURA', n_bits=25)
    assert mask1.shape() == (101, 101)
    mask2 = FlatCam_Mask(None, None, None, None, 'MLS', n_bits=5)
    assert mask2.shape() == (62, 62)
    mask3 = PhlatCam_Mask((256, 256), None, None, None, (8,8))
    assert mask3.shape() == (256, 256)
    mask4 = FZA_Mask((512, 512), None, None, None, 30)
    assert mask4.shape() == (512, 512)
    print('All good!\n')

test_masks()