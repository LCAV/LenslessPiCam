from lensless.mask import CodedAperture, PhaseContour, FresnelZoneAperture


def test_masks():

    mask1 = CodedAperture(None, None, None, None, 'MURA', n_bits=25)
    assert mask1.shape() == (101, 101)
    mask2 = CodedAperture(None, None, None, None, 'MLS', n_bits=5)
    assert mask2.shape() == (62, 62)
    mask3 = PhaseContour((256, 256), None, None, None, (8,8))
    assert mask3.shape() == (256, 256)
    mask4 = FresnelZoneAperture((512, 512), None, None, None, 30)
    assert mask4.shape() == (512, 512)

test_masks()