.. automodule:: lensless.hardware.fabrication

   This masks are meant to be used with a mount for the Raspberry Pi HQ sensor (shown below).
   The design files can be found `here <https://drive.switch.ch/index.php/s/nDe50iC7zn52r07#/>`_.

   .. image:: mount_components.png
      :alt: Mount components.
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