# Phone Detection via CNN

### Code Files

* train_phone_finder.py: trains model to predict phone position
* find_phone.py: predict phone position in a test image
* /find_phone: contains 129 images each with a phone
* /find_phone/labels.txt: contains position of phone in each image

### Run it yourself

To train model or compute accuracy, run the following in the terminal
* python train_phone_finder.py ~/find_phone

To predict accuracy of model, run the following in the terminal with own test directory
* python find_phone.py ~/find_phone_test_images/51.jpg

### Project created with

* Python 3.7.0
NVIDIA GeoForce RTX 2060
CUDA 11.3
cuDNN 8.2.1.32

* **Modules**
opencv-python==4.5.2
tensorflow==2.5.0

### Experiment

* Tested if predicted position is within a 0.05 radius around the ground truth phone position

### Result

* Model trained with sgd optimizer
    * 116/130 correct = 89.23%
* Model trained with adam optimizer
    * 124/130 correct = 95.38%
* Adam optimizer model predicts more accurately

### Future Work
* Try deeper architecture to decrease loss
* Try adding dropout layers to prevent overfitting
* Decrease step size for greater accuracy, but slower run-time

### Notes to Customer

* Coordinates are normalized and the origin is at the top-left of image (x,y)=(0,0), left-bottom is (x,y)=(0,1), right-top is (x,y)=(1,0), and right-bottom corner is (x,y)=(1,1).
* In train_phone_finder.py, can comment out train code once model trained and run accuracy test separately.
  ```
  # train model
  train_data, test_data, train_label, test_label = prepare_data(path, os.path.join(path, 'labels.txt'))
  create_model(train_data, test_data, train_label, test_label)
  ```
