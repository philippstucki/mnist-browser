# Digit Recognition using Tensorflow and Tensorflow.js

The model is trained with Tensorflow using Python and then saved as using simple_save. The saved model is then converted to a web format using tensorflowjs_converter.

Using tensorflowjs_converter the saved and converted model is loaded in JS and then used to recognize digits drawn in an image canvas. Images are preprocessed in the same way as was done for the images in the MNIST dataset:

1.  crop image from white background
1.  center in square canvas respecting aspect ratio
1.  resize to 20x20
1.  place on 28x28 canvas respecting center of mass

# Model

The model used consists of one input layer and one output layer which are fully connected: A\*x + b

# Todo

*   display prediction scores for all digits
*   add link to show example digits from mnist dataset
*   add more models with more layers and option to switch model in the ui

# Convert Saved Model to Web Format

*   https://github.com/tensorflow/tfjs-converter

`tensorflowjs_converter --input_format=tf_saved_model --output_node_names=y --saved_model_tags=serve ./saved ./saved_web`

# Other examples of Digit Recognition in the Browser using models trained with MNIST

*   http://www.denseinl2.com/webcnn/digitdemo.html
*   http://myselph.de/neuralNet.html
