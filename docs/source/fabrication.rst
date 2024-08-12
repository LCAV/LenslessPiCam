.. automodule:: lensless.hardware.fabrication

   This masks are meant to be used with a mount for the Raspberry Pi HQ sensor (shown below).
   The design files can be found `here <https://drive.switch.ch/index.php/s/nDe50iC7zn52r07#/>`_.

   .. image:: mount_components.png
      :alt: Mount components.
      :align: center

 
   Note that the most recent version of the mount looks like this, with the addition of stoppers, 
   to prevent the mask from scratching the Pi Camera.
   This new version of the mount can be found `here <https://drive.switch.ch/index.php/s/QYH9vSqPd21DIHQ>`_

   .. image:: mount_V4.png
      :alt: New Inner Mount w/ Stopper.
      :align: center

   Mask3DModel
   ~~~~~~~~~~~

   Below is a screenshot of a Fresnel Zone Aperture mask that can be designed with the above notebook
   (using ``simplify=True``).

   .. image:: fza.png
    :alt: Fresnel Zone Aperture.
    :align: center

   .. autoclass:: lensless.hardware.fabrication.Mask3DModel
      :members:
      :special-members: __init__

   Because newer versions of the masks' size are smaller, the following adapter enables
   them to be used with the current mounts design.

   .. image:: mask_adapter.png
    :alt: Mask Adapter.
    :align: center


   MultiLensMold
   ~~~~~~~~~~~~~

   Below is a screenshot of a mold that can be designed for a multi-lens array with the above notebook.

   *Note: We were not successful in our attempts to remove the mask from the mold
   (we poured epoxy and it was impossible to remove the mask from the mold).
   Perhaps the mold needs to be coated with a non-stick material.*

   .. image:: mla_mold.png
    :alt: Multi-lens array mold.
    :align: center

   .. autoclass:: lensless.hardware.fabrication.MultiLensMold
      :members:
      :special-members: __init__