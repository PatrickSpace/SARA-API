# SARA
MVP de un sistema que responde preguntas en base a una tesis

## Requisitos Back End
El sistema requiere la preinstalación de distintos softwares y herramientas.

### Anaconda
Instalar anaconda desde su [página oficial](https://www.anaconda.com/).
La version que soporta anaconda por defecto es la 3.8 actualmente y *posiblemente cambie en un futuro*, por lo que es necesario actualizar las versiones de python.
Para usar Pytorch es necesario tener las version 3.6 en su defecto, debido a que las siguientes no trabajan bien con las librerias y herramientas que usaremos. En el buscador abran Anaconda Prompt y ejecuten lo siguiente:
```
conda install python=3.6
```
Una vez dentro de anaconda se crea un nuevo entorno y la sección Home instalan el CMD o el PowerShell de su entorno creado para instalar las librerias necesarias para este proyecto. En este caso las librerias son:

#### 1 Pytorch
```
pip install torch
```
para una instalación más exacta pueden ejecutar el codigo que brinda la [página oficial de Pytorch](https://pytorch.org/)

#### 2 Transformers
```
pip install transformers
```

#### 3 Flask
```
pip install flask
```

#### 4 Flask-cors
```
pip install flask-cors
```
