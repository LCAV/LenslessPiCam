.. automodule:: lensless.recon


   Abstract Class API
   ------------------

   .. autoclass:: lensless.ReconstructionAlgorithm
      :members:
      :special-members: __init__, set_data, apply, _update, reset, _form_image, get_image_est


   Gradient Descent API
   --------------------

   .. autoclass:: lensless.GradientDescent
      :special-members: __init__

   .. autoclass:: lensless.NesterovGradientDescent
      :special-members: __init__

   .. autoclass:: lensless.FISTA
      :special-members: __init__

   .. autofunction:: lensless.gradient_descent.non_neg


   ADMM API
   --------

   .. autoclass:: lensless.ADMM
      :special-members: __init__

   .. autofunction:: lensless.admm.soft_thresh

   .. autofunction:: lensless.admm.finite_diff

   .. autofunction:: lensless.admm.finite_diff_adj

   .. autofunction:: lensless.admm.finite_diff_gram

   Accelerated Proximal Gradient Descent (APGD) API
   ------------------------------------------------

   .. autoclass:: lensless.APGD
      :special-members: __init__
