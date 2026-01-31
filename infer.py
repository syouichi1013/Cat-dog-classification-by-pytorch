import torch
from models import Resnet18Model
from torchvision import transforms
from PIL import Image
import cv2

SAVED_MODEL_NAME="./model.pth"

#load saved model and parameter
model = Resnet18Model()
model.load_state_dict(torch.load(SAVED_MODEL_NAME))
model.eval()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    img_paths = ['./infer/infer1.jpg', './infer/infer2.jpg']
    for idx, img_path in enumerate(img_paths):
        img1 = img_to_tensor(img_path).unsqueeze(0)
        output1 = model(img1)
        softmax = torch.nn.Softmax(dim=1)
        output1 = softmax(output1)
        pred = torch.argmax(output1, 1) .item() # Get predicted class index (max value along dim 1)
        conf=output1[0][pred].item()#Get raw score of the predicted class(still tensor

        image=cv2.imread(img_path)
        font=cv2.FONT_HERSHEY_SIMPLEX
        color=(255,255,255)
        org=(20,30)
        fontScale=1
        thickness=2
        x=round(float(conf)*100,2)

        if pred==0:
            image=cv2.putText(image,f'dog{x}%',org,font,fontScale,color,thickness)
        else:
            image=cv2.putText(image,f'cat{x}%',org,font,fontScale,color,thickness)


        cv2.imwrite(f'./infer/infer{idx+1}{idx+1}.jpg',image)
        cv2.imshow(f'image{idx+1}',image)
    while True:
        k=cv2.waitKey(1)& 0xFF
        if k==ord('q'):
            break
    cv2.destroyAllWindows()

def img_to_tensor(img):
    img = Image.open(img)
    resized_img=img.resize((224,224))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    img = transform(resized_img)
    return img

if __name__ == "__main__":
    main()

