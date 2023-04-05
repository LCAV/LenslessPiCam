import numpy as np

try:
    import bpy

except:
    print("\nError : This script needs to be run from Blender, not from the Lensless environment."
          "\nRead the instructions inside the file to continue.\n")
    quit()


"""
This script allows to export a rgb image and a depth map from a blender scene.
It is meant to be run directly in blender, not from the lensless environment

The exported scenes can then be used in the simulator to generate lensless data

Credits:
    The contents of these file are inspired from "Generate Depth and Normal Maps with Blender", Saif Khan, 26.12.2021,
    under the Creative Commons Attribution 4.0 International License : https://creativecommons.org/licenses/by/4.0/ 
    Link to the original work : https://www.saifkhichi.com/blog/blender-depth-map-surface-normals
    
Instructions :
 - Load or create any scene of your choice in blender (https://docs.blender.org/)
    - Lots of scenes can be downloaded freely from websites such as https://www.blendswap.com/
    - You may want to set the background color of the scene to black ; otherwise, it will be 
      considered by the simulator as a physical plane which will be placed at the maximum depth
      of the scene. To do so, the "Layout" tab, search the "World" menu on the right and, in the
      "Surface" sub-menu, change the "Color" field to black.
      If you forgot this step, you can still manually edit the exported image later to the image
      editor of your choice in order to change the background pixels to black. 
 
 - In the "Layout" tab, search the "View Layer Properties" menu on the right    and mark the "Combined" and "Z" boxes
 
 - In the "Compositing" tab, add the following nodes :
    - Tick "Use Nodes" to create two nodes : "Render Layer" and "Composite"
    - Select "Add" -> "Output" -> "Viewer" to create a new node of the same name.
    - Select "Add" -> "Vector" -> "Normalize" to create a new node of the same name.
 
 - Still in the "Compositing" tab, connect the nodes in the following way :
    - Render Layer's field "Image" should already be connected to Composite's field Image. If not, connect it now.
    - Connect Render Layer's field "Depth" to Normalize's input, which is the "Value" field at the bottom left.
    - Connect Normalize's output, which is the "Value" field at the top right, to Viewer's field "Image"
    - In Composite node, "Use Alpha" should already be ticked. If not, tick it now.
    - In Viewer node, "Use Alpha" should already be ticked. If not, tick it now.
 
 - Still in the "Compositing" tab, in the Render Layer node, click on the Render button on bottom right
 
 - In the "Scripting" tab, go in the Text Editor field. It should be in the middle by default ; if not the case, open it
    with the shortcut Shift+F11. Open the current field in it. Set the output path to your liking, then run the script.

 - Your data should now be properly exported at the specified path !

"""

output_path = "/your/custom/path/"

bpy.context.scene.render.filepath = output_path + "scene.png"
bpy.ops.render.render(False, animation=False, write_still=True)

data = bpy.data.images['Viewer Node']
w, h = data.size
depths = np.fliplr(np.rot90(np.reshape(np.array(data.pixels[:], dtype=np.float32), (h, w, 4))[:,:,0], k=2))

np.save(output_path + "scene-normals.npy", depths)
