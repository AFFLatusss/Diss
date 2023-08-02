import torch
import coremltools as ct
from torchvision import transforms
from PIL import Image
import PIL



def convert_to_mlmodel(model, class_labels, name, test_img_path):
    img = Image.open(test_img_path)

    transform = transforms.Compose([
                        transforms.Resize((224, 224)), # 1. Reshape all images to 224x224 (though some models may require different sizes)
                        transforms.ToTensor(), # 2. Turn image values to between 0 & 1 
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                                            std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
    ])

    img1 = transform(img).unsqueeze(dim=0)

    scale = 1/(0.226*255.0)
    bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]

    image_input = ct.ImageType(name="x_1",
                            shape=img1.shape,
                            scale=scale, bias=bias)
    
    traced_model = torch.jit.trace(model, img1)

    ios_model = ct.convert(
        traced_model,
        inputs=[image_input],
        classifier_config = ct.ClassifierConfig(class_labels),
    )

    ios_model.save("ML_Models/"+ name +".mlmodel")


    img = PIL.Image.open(test_img_path)
    img = img.resize([224, 224])
    ios_model.predict({"x_1":img})['classLabel']
    
