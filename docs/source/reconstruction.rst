.. automodule:: lensless.recon


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

   Accelerated Proximal Gradient Descent (APGD)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   .. autoclass:: lensless.APGD
      :special-members: __init__



   Trainable reconstruction API
   ----------------------------


   Abstract Class (Trainable)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~
   

   .. autoclass:: lensless.TrainableReconstructionAlgorithm
      :members: batch_call, apply, reset, set_data
      :special-members: __init__
      :show-inheritance:


   Unrolled FISTA
   ~~~~~~~~~~~~~~

   .. autoclass:: lensless.UnrolledFISTA
      :members: batch_call
      :special-members: __init__
      :show-inheritance:

   Unrolled ADMM
   ~~~~~~~~~~~~~

   .. autoclass:: lensless.UnrolledADMM
      :members: batch_call
      :special-members: __init__
      :show-inheritance: