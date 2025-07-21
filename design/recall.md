# Recognizing an object

Recall is the ability to recognize the same object across multiple viewings.  To do this, we'll leverage our short and long-term storage for specific images and associated tags such as category and individual designator.  Then, we use ResNet50 (https://keras.io/api/applications/resnet/#resnet50-function) to execute image recognition and determine a similarity value.

Recall is performed asynchronously - feature extraction, classification, and depth estimation should be returned and then the recall results available after the fact (and in 3 waves - short term recall, long-term recall, and archival recall).  This is because it may take some time for these results to come back, and we need to tell the decision-making cortex what might smack into it ASAP.