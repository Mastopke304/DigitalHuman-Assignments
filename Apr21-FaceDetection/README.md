# Usage
## Evaluation
1. Download the dataset [from this link](http://shuoyang1213.me/WIDERFACE/)(WIDER Face Validation Images and Face annotations) and put it in ./images/, the path should be like this: `./images/WIDER_val` and `./images/wider_face_split`
2. Run `python main.py`, the plots will be saved at `./plots/`

## Fake faces
1. Run `python cnn_face_detection.py -i images/1000202882.jpg` or `python hog_face_detection.py -i images/1000202882.jpg` to use CNN or HOG model to detect.

## Show examples
1. Run `python show_example.py` to show the predictions on example images if you don't want to dowload the dataset.

**FOR COURSE ONLY**
