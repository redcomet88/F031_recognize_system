# F031_recognize_system  Vue+Flask深度学习+机器学习多功能识别系统
> 完整项目收费，可联系QQ: 81040295 微信: mmdsj186011 注明从github来的，谢谢！
也可以关注我的B站： 麦麦大数据 https://space.bilibili.com/1583208775
> 

关注B站，有好处！
编号:  F031
## 视频

[video(video-a8qjXFjq-1760775902019)(type-bilibili)(url-https://player.bilibili.com/player.html?aid=603147486)(image-https://i-blog.csdnimg.cn/img_convert/804206fca1da0b72f286d1655caaf4de.jpeg)(title-vue+flask Python 多种AI识别系统 人脸检测 口罩检测 动物识别 菜品识别 宠物识别 中药识别 植物识别)]

## 1 系统简介
系统简介：本系统是一个基于Vue.js前端框架和Flask后端框架构建的多功能识别系统。系统的核心功能包括用户管理和多种识别功能。主要功能模块包括：登录与注册模块，提供用户的基本身份认证功能；身份证OCR识别模块，支持用户上传身份证图片并自动提取关键信息；人脸识别模块，能够对上传的图片进行人脸检测并在照片中标注识别框；口罩识别模块，能够识别图片中人物是否佩戴口罩并标注识别框；动物识别模块，利用ResNet模型对上传的动物图片进行识别并返回识别结果。此外，系统还集成了百度Paddle的接口和模型，以提升识别的准确性和效率，为用户提供更加智能化的服务。
## 2 功能设计
该系统采用典型的B/S（浏览器/服务器）架构模式，前端使用Vue.js框架构建，集成了Vue Router用于路由管理和Vuex用于状态管理，后端则使用Flask框架提供API接口。系统通过百度Paddle的接口和模型实现多种识别功能，包括身份证OCR识别、人脸识别、口罩识别和动物识别。为了处理这些识别功能，系统设计了专门的数据处理模块，能够接收用户上传的图片或文件，并通过百度Paddle的API进行分析，最后将结果返回给前端展示。同时，系统还包含用户管理模块，支持用户的登录、注册和基本信息管理，确保系统的安全性和用户体验。
### 2.1系统架构图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9a354bf8bd7443188be2652e305f5cb0.png)
### 2.2 功能模块图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a1070f9cb17f415c8d35b6c748d1189f.png)
## 3 功能展示
### 3.1 登录 & 注册
登录注册做的是一个可以切换的登录注册界面，点击去登录后者去注册可以切换，背景是一个视频，循环播放。
登录需要验证用户名和密码是否正确，如果**不正确会有错误提示**。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/63405e69ea114a488476bcab7a5758c9.png)
注册需要**验证用户名是否存在**，如果错误会有提示。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/420fdbbc3c1b4554b1b6a11b822b96ae.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d30ed435915a4f0c8dd909e454bb2dbb.png)
### 3.2 主页
主页的布局采用了左侧是菜单，右侧是操作面板的布局方法，右侧的上方还有用户的头像和退出按钮，如果是新注册用户，没有头像，这边则不显示，需要在个人设置中上传了头像之后就会显示。
### 3.3 身份证识别
上传身份证识别出姓名、身份证号等要素：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c164c20e3ba84f42ad70af57510e84d6.png)
### 3.4 人脸识别
通过上传照片，返回带框的识别结果：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7e1f5de0488b4c72830a9aed9172b7b5.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5dc267f2965c44e98c42f938c8e96b80.png)
### 3.5 口罩识别
通过上传照片，返回带框的识别结果：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bcdcb41dc9fe4b1882373fce00c2029e.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ab4e7a7304654260b68715adb14c61a4.png)
### 3.6 动物识别
通过上传照片，返回最可能识别结果和概率：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9387753ef20448e2981e4ff3ab7dd14d.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/72e931c61476454a8a1e60b88adb734f.png)
## 4程序代码
### 4.1 代码说明
代码介绍：该算法利用PaddlePaddle框架中的ResNet模型，对输入的动物图片进行分类识别。通过加载预训练的ResNet模型并在动物数据集上进行微调，可以实现高效的动物识别任务。
数据加载与预处理：加载动物图片数据集，进行归一化、调整大小和数据增强。
模型定义：加载PaddlePaddle的ResNet预训练模型，并根据需要进行微调。
模型训练：定义损失函数和优化器，进行模型训练。
模型评估：在验证集或测试集上评估模型性能。
模型部署：将训练好的模型用于新的动物图片分类。
### 4.2 流程图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/dce46f08abb242069e7ce4ac81659607.png)
### 4.3 代码实例
```python
import paddle
from paddle import nn
from paddle.vision import transforms
from paddle.vision import datasets
import os

# 数据加载与预处理
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 假设有以下目录结构：
# data/
#   train/
#       class1/
#           img1.jpg
#           ...
#       class2/
#           ...
#   val/
#       class1/
#           ...
#       ...

train_dataset = datasets.ImageFolder(root='data/train', transform=train_transforms)
val_dataset = datasets.ImageFolder(root='data/val', transform=val_transforms)

train_loader = paddle.io.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = paddle.io.DataLoader(val_dataset, batch_size=32, shuffle=False)

# 模型定义
model = paddle.vision.models.resnet50(num_classes=len(train_dataset.classes))

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

# 模型训练
def train(model, loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            inputs, labels = batch
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 模型评估
def evaluate(model, loader):
    model.eval()
    total = 0
    correct = 0
    with paddle.no_grad():
        for batch in loader:
            inputs, labels = batch
            outputs = model(inputs)
            _, predicted = paddle.max(outputs, axis=1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Validation Accuracy: {accuracy:.4f}')

# 模型部署
def predict(model, image_path):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = datasets.ImageFolder.root_load(image_path, transform)
    image = paddle.to_tensor(image.unsqueeze(0))
    output = model(image)
    _, predicted = paddle.max(output, axis=1)
    class_name = train_dataset.classes[predicted.item()]
    print(f'Predicted class: {class_name}')

# 主函数
def main():
    # 训练模型
    train(model, train_loader, criterion, optimizer)
    
    # 评估模型
    evaluate(model, val_loader)
    
    # 使用模型进行预测
    image_path = 'path/to/test/image.jpg'
    predict(model, image_path)

if __name__ == "__main__":
    main()


```
