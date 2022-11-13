In order to test the final model attempted by Team 16, open the following Colaboratory notebook and connect to a GPU-acceleration-enabled runtime (ideally with high RAM):

https://colab.research.google.com/drive/1rGDgCCjdV3McRFNhnAIeUGX4Tc-4f_94?usp=sharing.

- Run all the cells under the sections:
	- Clone Repository and Install Dependencies
	- Build OpenPose (this step could take several minutes, depending upon the available RAM)

- Upload all test images into the directory /content/cs4243_miniproject_team16/test_images

- Run all the cells under the sections:
	- Run Model
	- Generate Confusion Matrix

Note that in order for the confusion matrix to be generated correctly, the correct label for each test image should be contained as a substring within its filename, eg. A0170549Y_20220904_threat.jpg

Team 16's baseline model and training code can be found at:
https://colab.research.google.com/github/pratyushghosh/CS4243_Mini_Project/blob/main/Baseline_Image_Classification.ipynb. It can also be seen inside the code submission, in the folder training_code_and_baseline.

Team 16's pose-analysis model and training code can be found at:
https://colab.research.google.com/drive/1069awpfZSgsOh6v0tSdHPgrmZPnC61zA?usp=sharing
This is referred to as Model B in the final two-step model. It can also be seen inside the code submission, in the folder training_code_and_baseline.

Team 16's gun-detection model is a trained model based on an open-source project. Its training code can be found at https://github.com/RunzeXU/AI-detection-weapons.