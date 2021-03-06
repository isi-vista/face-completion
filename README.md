# Does Generative Face Completion Help Face Recognition?

### Encoder-Decoder for Face Completion based on Convolutions with a Gating Mechanism

This page contains the encoder-decoder model definition for learning the face completion task, aka face inpainting, following the method mentioned in _J. Mathai\*, I. Masi\*, W. AbdAlmageed, "[Does Generative Face Completion help Face Recognition?](#) ", in Proc. of IAPR International Conference on Biometrics (ICB) 2019 [1]_.


![Teaser](https://i.imgur.com/Pv0W9mb.png)

<sub>A realistically occluded face and its completed version by our face inpainter. We show the effect of face inpainting on the performance of a deep face recognition pipeline with occlusions. Note that we assume _no_ access to the recognition pipe. Our study aims to quantitatively assess (a) the impact of different synthetic yet realistic occlusions on recognition and  (b) how much face perception is restored via face inpainting.</sub>


## Features
* **Encoder-Decoder** for face completion. Multiple architecture are tested: 
  - [Simple U-Net](models/conv_unet.py) with standard Convolution 
  - U-Net with [partial Convolution](models/partialconv_unet.py)
  - U-Net with soft, [gated Convolution](models/gated_unet.py)
* Single forward pass to get the completed face
* The user can input the part to be completed, the model uses this mask with a gating mechanism

## Dependencies

* [Pytorch](https://pytorch.org)
* [Numpy](http://www.numpy.org/)
* [Python3.6](https://www.python.org/download/releases/3.6/)

The code has been tested on Linux only. On Linux you can rely on the default version of python, installing all the packages needed from the package manager or on Anaconda Python and install required packages through `conda`. 

## Current Limitations
Due to time constraints we have uploaded the model definition only. In the future, we could upload the trained model along with a decoding, testing script.


## Citation

Please cite our paper with the following bibtex if you use our face inpainter:

``` latex
@inproceedings{mathai2019doesgenerative,
  title={{D}oes {G}enerative {F}ace {C}ompletion {H}elp {F}ace {R}ecognition?},
  author={Mathai, Joe and Masi, Iacopo and Abd-Almageed, Wael},
  booktitle={IAPR International Conference on Biometrics (ICB)},
  year={2019},
}
```

## License and Disclaimer
Please, see [the LICENSE here](LICENSE.txt)

## References

[1] J. Mathai*, I. Masi*, W. AbdAlmageed, "Does Generative Face Completion help Face Recognition? ", in Proc. of IAPR International Conference on Biometrics (ICB) 2019

<sub>\* denotes equal contribution</sub>
    
## Changelog
- July 2019, Model definition updated
- April 2019, First  Release 

## Contacts

If you have any questions, drop an email to _iacopo@isi.edu_ and _jmathai@isi.edu_ or leave a message below with GitHub (log-in is needed).
