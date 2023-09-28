
# Herbal-AI

## Overview
This personal open-source project is designed to be a helpful tool for plant enthusiasts like myself. It features a deep learning model for classifying over 40 different Herbal Indian Plants. Additionally, the project utilizes an external API to fetch detailed medicinal properties and information for each classified plant. By combining the capabilities of deep learning with my passion for botanical knowledge, I aim to create a bridge between traditional herbal wisdom and modern technology.

## Features
- **Plant Classification:** This project employs the MobileNetV2 deep learning architecture for accurate classification of Herbal Indian Plants.

- **Medicinal Properties API:** I have integrated an external API to retrieve comprehensive data about each plant's medicinal properties, traditional uses, and related scientific research.


## Optimizations

- **1st RUN :** I used VGG16 model but soon ran out of RAM and VRAM because of large image size.

- **2nd RUN :** I used MobileNetV2 but with same IMAGE SIZE but ran out of RAM and VRAM again.

- **3rd RUN :** Made a custom 5 layer CNN, ran with test 74.34% accuracy.

- **4th RUN :** Reduced image size and used MobileNetV2 again without trainable parameters, ran with 84.65% test accuracy.

- **5th RUN :** Ran with MobileNetV2 with trainable parameters, ran with 92.50% test accuracy.


## Acknowledgements

 - [Kaggle Reference](https://www.kaggle.com/code/codefantasy/identifying-plants-and-it-s-medicinal-properties)
 - [Plants Dataset](https://data.mendeley.com/datasets/748f8jkphb/3)


## Run Locally

If you have a GPU then make sure to
- Have latest drivers installed
- Have CUDA ToolKit installed
- Have cuDNN installed
Need help? \
Follow [this](https://www.tensorflow.org/install/pip#windows-native)
```bash
  git clone https://github.com/puranjayb/Herbal-AI
  cd Herbal-AI
```

Open terminal to install required files
```bash
  conda create --name <new_environment_name> --file requirements.txt
```

Now just run all the cells in the notebook and let the GPU do its work :)


    
## Authors

- [@puranjayb](https://www.github.com/puranjayb)

