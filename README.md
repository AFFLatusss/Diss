# Diss
Master Project


```bash
├── Data
│   ├── Augmentation
│   ├── Originals/Apple
├── ML_Models
│   ├── iOS models
├── Models
│   ├── efficient_bO models
│   ├── mobile_v2 models
├── Code
│   ├── CNN code
├── iOS_App
├── README.MD
├── RongJianYee_PoP.pdf
└── .gitignore
```

iOS Application is under the folder iOS_App/plane_disease_identification

To test the application, open plant_disease_identification folder with XCode and build.
Once the build is successful, then the app can run on the simulator. To test images with the app, drag the images into the simulator so that the images will be shown in the simulator's photo library.

`Data`: Use to store part of the images used in development, since full data are too big to store on github

`Ml_Models`: Contain `.mlmodel` files used in iOS application development for classification

`Models`: Contain pytorch models for `mobile_v2` models and `efficient_b0` models

`Code`: Contain Jupyter notebooks for CNN and python files for utility functions. 

`iOS_App`: Contain the XCode workspace for the plant disease identification application
