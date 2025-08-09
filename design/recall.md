# Recognizing an object

Recall is the ability to recognize the same object across multiple viewings.  To do this, we'll leverage our short and long-term storage for specific images and associated tags such as category and individual designator.  Then, we use ResNet50 (https://keras.io/api/applications/resnet/#resnet50-function) to execute image recognition and determine a similarity value.

Recall is performed asynchronously - feature extraction, classification, and depth estimation should be returned and then the recall results available after the fact (and in 3 waves - short term recall, long-term recall, and archival recall).  This is because it may take some time for these results to come back, and we need to tell the decision-making cortex what might smack into it ASAP.

## Updating ML Models to predict the new Object

ML Operates by building statistical regression models, and then optimizing constants within the model to produce the next prediction.  This makes true recall really complex - it means we don't just need to store things in a database to "learn" them.  But we also need to be consistantly re-training models to allow that to happen.  Basically, we take the things we've learned throughout the day and use it to generate new training sets automatically, and then re-train the models periodically (sort of like going to sleep).

Easier said than done though.  Up until this point, Rose has largely relied on pre-trained models to operate.  Trying to do this changes *everything*, and makes the problem a million times harder.

## Custom vs Pre-Trained Models



## Generating Training Data & Correctness



## Periodic Re-Training
