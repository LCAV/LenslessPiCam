from lensless.mask import CodedAperture, PhaseContour, FresnelZoneAperture


def test_masks():

    lambd, sensor_size, nb_px, dz = 532e-9, 5e-3, 256, 0.5e-3
    d1 = sensor_size / nb_px

    mask1 = CodedAperture((101, 101), None, d1, dz, lambd, 'MURA', n_bits=25)
    assert mask1.shape() == (101, 101)
    mask1.phase_retrieval(lambd, d1, dz)
    assert mask1.phase_mask.shape == mask1.shape()

    mask2 = CodedAperture((62, 62), None, d1, dz, lambd, 'MLS', n_bits=5)
    assert mask2.shape() == (62, 62)
    mask2.phase_retrieval(lambd, d1, dz)
    assert mask2.phase_mask.shape == mask2.shape()
    
    mask3 = PhaseContour((256, 256), None, d1, dz, lambd, (8,8))
    assert mask3.shape() == (256, 256)
    mask3.phase_retrieval(lambd, d1, dz)
    assert mask3.phase_mask.shape == mask3.shape()
    
    mask4 = FresnelZoneAperture((512, 512), None, d1, dz, lambd, 30)
    assert mask4.shape() == (512, 512)
    mask4.phase_retrieval(lambd, d1, dz)
    assert mask4.phase_mask.shape == mask4.shape()

test_masks()