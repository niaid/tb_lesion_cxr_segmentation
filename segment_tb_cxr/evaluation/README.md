## Results:

###UNet:
#### Without any augmentation:
Dice	| IoU	| hausdorff_distance |	mean_surface_distance |	median_surface_distance
---  | --- | --- | --- | --- |
0.9323	|0.8897	|62.0825	|9.3472	|0.0|

#### With augmentation:
Dice	| IoU	| hausdorff_distance |	mean_surface_distance |	median_surface_distance
---  | --- | --- | --- | --- |
0.8179	|0.7117	|	284.45|38.0240	|21.1166|

###ResNet18-UNet:
#### Without any augmentation:
Dice	| IoU	| hausdorff_distance |	mean_surface_distance |	median_surface_distance
---  | --- | --- | --- | --- |
0.9445	|0.9002	|62.9599	|9.2118	|0.0|

#### With augmentation:
Dice	| IoU	| hausdorff_distance |	mean_surface_distance |	median_surface_distance
---  | --- | --- | --- | --- |
0.9395	|0.8949	|	60.8487|1.8332	|0.0|

Augmentation results have shown to be performing lesser than when augmentation is not done during training. One reason could be that the augmentation effects might have been "too heavy" for the model to perform on "regular" Chest X Rays in the test set. Maybe lowering the rotation range or translation range a little bit could probably show the augmentation results higher than when no-augmentation is done.