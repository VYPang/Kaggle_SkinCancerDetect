# ISIC 2024 - Skin Cancer Detection with 3D-TBP
**Competition**

In this [Kaggle competition](https://www.kaggle.com/competitions/isic-2024-challenge), you'll develop image-based algorithms to identify histologically confirmed skin cancer cases with single-lesion crops from 3D total body photos (TBP). The image quality resembles close-up smartphone photos regularly submitted for telehealth purposes. Your binary classification algorithm could be used in settings without access to specialized care to improve triage for early skin cancer detection.

**Solution Overview**

Our solution is based on ensembling various implementations of models along with image models. This repository only contains our image model and corresponding training, validation, and testing workflow. For the image models, We used EfficientNet as the architecture, accompanied by pre-trained weighting 'tf_efficientnet_b0_aa-827b6e33.pth' from [timm/tf-efficientnet](https://www.kaggle.com/models/timm/tf-efficientnet).

The greatest challenge would be handling an imbalanced dataset, in which images labeled as malignant only occupy a very small proportion of the whole dataset. During the project, most effort is spent on experimenting with data augmentations and modifying the training configuration.

**Model Training**
- To address the significant class imbalance, the examples from different classes in the training batches were extracted in a ratio that can be modified in config.yaml. Through experiments, the model performance would be the best when the ratio is 1:1.
