.. automodule:: lensless.recon.recon


   Reconstruction API
   ------------------


   Abstract Class
   ~~~~~~~~~~~~~~
   

   .. autoclass:: lensless.ReconstructionAlgorithm
      :members:
      :private-members: _update, _form_image
      :special-members: __init__, set_data, apply, reset, get_image_est


   Gradient Descent
   ~~~~~~~~~~~~~~~~

   .. autoclass:: lensless.GradientDescent
      :special-members: __init__

   .. autoclass:: lensless.NesterovGradientDescent
      :special-members: __init__

   .. autoclass:: lensless.FISTA
      :special-members: __init__

   .. autofunction:: lensless.recon.gd.non_neg


   ADMM
   ~~~~

   .. autoclass:: lensless.ADMM
      :special-members: __init__

   .. autofunction:: lensless.recon.admm.soft_thresh

   .. autofunction:: lensless.recon.admm.finite_diff

   .. autofunction:: lensless.recon.admm.finite_diff_adj

   .. autofunction:: lensless.recon.admm.finite_diff_gram


   Tikhonov (Ridge Regression)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. autoclass:: lensless.CodedApertureReconstruction
      :special-members: __init__, apply


   Accelerated Proximal Gradient Descent (APGD)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. autoclass:: lensless.recon.apgd.APGD
      :special-members: __init__



   Trainable reconstruction API
   ----------------------------


   Abstract Class (Trainable)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~
   

   .. autoclass:: lensless.TrainableReconstructionAlgorithm
      :members: forward, apply, reset, set_data
      :special-members: __init__
      :show-inheritance:


   Unrolled FISTA
   ~~~~~~~~~~~~~~

   .. autoclass:: lensless.UnrolledFISTA
      :members: forward
      :special-members: __init__
      :show-inheritance:

   Unrolled ADMM
   ~~~~~~~~~~~~~

   .. autoclass:: lensless.UnrolledADMM
      :members: forward
      :special-members: __init__
      :show-inheritance:

   Trainable Inversion
   ~~~~~~~~~~~~~~~~~~~

   .. autoclass:: lensless.TrainableInversion
      :members: forward
      :special-members: __init__
      :show-inheritance:

   Multi-Wiener Deconvolution Network
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. autoclass:: lensless.MultiWiener
      :members: forward
      :special-members: __init__
      :show-inheritance:


   Reconstruction Utilities
   ------------------------

   .. autoclass:: lensless.recon.utils.Trainer
      :members: 
      :special-members: __init__

   .. autofunction:: lensless.recon.utils.load_drunet

   .. autofunction:: lensless.recon.utils.apply_denoiser

   .. autofunction:: lensless.recon.utils.get_drunet_function

   .. autofunction:: lensless.recon.utils.measure_gradient

   .. autofunction:: lensless.recon.utils.create_process_network
