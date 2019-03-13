# Style_Transfer
It is a simple neural style transfer project using tensorflow.  
You gotta download vgg19 pretrainned weights at first. http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat  
Train the model on AWS.  
Basic logic of project: 1)extract weights from pre trainned vgg19. 2) Feed weights into style img, content img and noise img. 3) Compute loss and update.
