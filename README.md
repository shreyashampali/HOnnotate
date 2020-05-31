**Under construction**: Currently this repo contains inference scripts for hand+object
segmentation and keypoint detections. Optimization scripts for full annotation of hand and 
object 3D poses will be released soon!

# Installation
- This code has been tested with Tensorflow 1.12.

# Setup
`HOnnotate_ROOT` is the directory where you download this repo.
- Clone the *deeplab* repository and checkout the commit on which our code is tested
```
            git clone https://github.com/tensorflow/models.git
            git checkout 834902277b8d9d38ef9982180aadcdaa0e5d24d3
```
- Copy **research/deeplab** and **research/slim** folders to **models** folder in ``HOnnotate_ROOT`` repo
- Download the checkpoint files from [here](https://files.icg.tugraz.at/f/f23053e075a140ca8756/?dl=1) and extract in ``HOnnotate_ROOT``
- Download the objects 3D corner files [here](https://files.icg.tugraz.at/f/b400540c6c81425e8978/?dl=1) and extract in `HOnnotate_ROOT` 
- Download and extract the test sequence from [here]() and update `HO3D_MULTI_CAMERA_DIR` variable in `HOdatasets/mypaths.py` with its location
- Finally, your folder structure should look like this:
```
            - checkpoints
                - CPM_Hand
                - CPM_Object
                - Deeplab_seg
            - eval
            - HOdatasets
            - models
                - CPM
                - deeplab
                - slim
            - objCorners
                - 003_cracker_box
                - 004_sugar_box
                ....
            - onlineAug
            - utils
```

# Run
## 1. Hand+Object Segmentations
```
        python inference_seg.py --seq 'test'
```
The segmentations are saved in *segmentation* directory of the `test` sequence

## 2. Hand 2D keypoints
This requires the segmentation script to be run beforehand
```.env
        python inference_hand.py --seq 'test'
```
The 2D keypoints are saved in *CPMHand* directory of the `test` sequence

## 3. Object 2D/3D keypoints
This requires the segmentation script to be run beforehand
```.env
        python inference_obj.py --seq 'test'
```
The 2D/3D keypoints are saved in *CPMObj* directory of the `test` sequence
