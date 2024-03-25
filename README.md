# Introduction to neural networks - Windesheim University of Applied Sciences, 2024

Although it is possible to use Windows as well, we strongly encourage the participants to use Linux during the lab session.

### Manual installation of Python and the following packages (with pip):
- scikit-learn
- matplotlib
- tensorflow (it should install keras as well)

### Using Docker (https://www.docker.com/):
- Install docker
- Pull image (~ 1 GB):
`docker pull zalanbodo/ubuntu_w_python`
- Run image:
```docker run -it -v "[path to local files]:/home" ubuntu_w_python /bin/sh```
- or create + run:
```docker create -it --name some_name -v "path_to_local_files":"/home" ubuntu_w_python
docker start some_name
docker attach some_name```
