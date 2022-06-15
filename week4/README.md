#WEEK4


今週はYoLov3を実装してもらいます。


GPUがない人もいるので二人一組でやってもらいます。


実装するステップは以下のようになっています


①必要なパッケージをインストール


　pip install -r requirements.txt


②学習に用いる初期重みをインストール


　./weights/download_weights.sh
 
 
③cocoデータセットのインストール


　./data/get_coco_dataset.sh



④学習(ここまで今週)


　python3 train_coco.py


⑤推論(学習で得た重みから、好きな画像で物体検出)


　python3 detect_image.py --input data/sazae.jpeg
　
 
![sazae](https://user-images.githubusercontent.com/85509359/173790445-a59e5751-096d-4491-b0bd-12cec7cea5bb.jpeg)
　
 
 波平とカツオは人間と認識されませんでした。
