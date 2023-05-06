```
sudo docker build -t qai .

sudo docker run -d --name qai \
-p 8080:8080 \
-v ./files:/home/myrustappuser/files \
-v ./storage:/home/myrustappuser/storage \
qai
```