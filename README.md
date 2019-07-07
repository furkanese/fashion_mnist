# FashionMnist

`Fashion-MNIST` veriseti [Zalando Research](https://www.kaggle.com/zalando-research/fashionmnist) tarafından MNIST verisetine alternatif olarak oluşturulmuştur. 
Toplamda 60.000 tane eğitim, 10.000 tane test verisi bulunmaktadır. 28x28 gri seviyesinde 10 farklı sınıftan oluşmaktadır.

### Sonuçlar
<img src="images/fashion-mnist-sprite.png" width="50%">

Sınıflar:

| Label | Class 	|
| ----- |:-------------:|
| 0 	| T-shirt/top 	|
| 1 	| Trouser 	|
| 2 	| Pullover 	|
| 3 	| Dress 	|
| 4 	| Coat 		|
| 5 	| Sandal 	|
| 6 	| Shirt 	|
| 7 	| Sneaker 	|
| 8 	| Bag 		|
| 9 	| Ankle boot 	|

## Kurulum - Çalıştırma
### Docker kurulumu
sudo apt-get update<br>
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common<br>
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - <br>
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"<br>

sudo apt-get update <br>
sudo apt-get install docker-ce <br>

Aşağıdaki komut çalışıyorsa sistem hazırdır: <br>

sudo docker run hello-world <br><br>

### Keraslı image oluşturulması
cd docker <br>
docker build --no-cache=true -t docker_cpu:latest . <br><br>

### Keraslı containerın oluşturulması
docker run -it -p 8888:8888 -v ~/Desktop/Fashion_Mnist:/home/code docker_cpu:latest <br>

-v komutu ile local makinede çalışacağımız konumu belirtiyoruz <br>

Container çalışırken konsola token değerini basacaktır, localhost:8888 adresine giderek basılan token değerini yapıştırarak girebiliriz<br> <br>

### Eğitim

Veriseti indirildikten sonra,<br>
normal eğitim için training.ipynb<br>
Early stopping, data augmentation ve ReduceLROnPlateau ile yapılan eğitim için training_with_improv.ipynb<br>
Transfer learning için pre_trained_tuning.ipynb<br>
çalıştırılabilir.<br><br>

### Sonuçlar
<img src="images/results.png" height="50%">
