import torch
import coremltools as ct
from torchvision import transforms



def convert_to_mlmodel(model, class_labels, name, test_img):
    scale = 1/(0.226*255.0)
    bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]

    image_input = ct.ImageType(name="x_1",
                            shape=test_img.shape,
                            scale=scale, bias=bias)
    
    traced_model = torch.jit.trace(model, test_img)

    ios_model = ct.convert(
        traced_model,
        inputs=[image_input],
        classifier_config = ct.ClassifierConfig(class_labels),
    )

    ios_model.save("ML_Models/"+ name +".mlmodel")
    
