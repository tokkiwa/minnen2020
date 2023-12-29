# minnen2020
Unofficial Pytorch Implementation of Channel-wise Autoregressive Entropy Models for Learned Image Compression(ICIP 2020)

# Despcription
This is UNOFFICIAL Pytorch Implementation of Channel-wise Autoregressive Entropy Models for Learned Image Compression(ICIP 2020).
The codes are based on CompressAI(https://github.com/InterDigitalInc/CompressAI/tree/master/compressai). 

# Models
`minnen20.py` contains two models: `Minnen2020` and `Minnen2020LRP`.
The former one emits the LRP(Latent Residual Prediction) Module to performance comparison. The latter is full model.
Also, this model does not include the Straight-Forward Estimater(STE) based quantization method shown in the paper. 

# Usage
Please use example code from CompressAI to train and test a model. 

# Citation
```
@article{minnen2020channelwise,
      title={Channel-wise Autoregressive Entropy Models for Learned Image Compression}, 
      author={David Minnen and Saurabh Singh},
      year={2020},
      journal={2020 IEEE International Conference on Image Processing (ICIP)},
      year={2020},
      pages={3339-3343},
}
