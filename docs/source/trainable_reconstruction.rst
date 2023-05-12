.. automodule:: lensless.trainable_recon


   Trainable reconstruction API
   ----------------------------


   Abstract Class
   ~~~~~~~~~~~~~~
   

   .. autoclass:: lensless.TrainableReconstructionAlgorithm
      :members: batch_call, apply, get_image_est, reset, set_data
      :special-members: __init__
      :show-inheritance:


   Gradient Descent
   ~~~~~~~~~~~~~~~~

   .. autoclass:: lensless.UnrolledFISTA
      :members: batch_call
      :special-members: __init__
      :show-inheritance:

   ADMM
   ~~~~

   .. autoclass:: lensless.UnrolledADMM
      :members: batch_call
      :special-members: __init__
      :show-inheritance:

