修改自[调用pytorch的resnet，训练出准确率高达96%的猫12类分类模型。 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/662512540)

## 代码结构

```bash
-data                  //数据集
   .class1              //分类1
   .class2              //分类2
   ...
-tempaltes
   .index.html          //前端
-app.py                //Flask
-train.py              //模型训练
-pachong.py            //爬虫
-README.md             //使用说明
-best_model_train92.81.pth   //训练好的模型文件
```



## 使用

```bash
git clone https://github.com/yty666zsy/Flask_Cat_7classify.git
cd Flask_Cat_7classify
python pachong.py   //然后将自己爬取的图片分好类，放在data文件夹中
python train.py     //自行修改数据集跟模型保存位置，运行后，得到一个模型文件，
python app.py       //自行修改模型路径
访问127.0.0.1:5000/index.html
```

