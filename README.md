# Does Generative Face Completion Help Face Recognition?

### Encoder-Decoder for Face Completion based on Gated Convolution

This page contains the code along pre-trained model for solving the face completion task aka face inpainting following the method mentioned in _J. Mathai\*, I. Masi\*, W. AbdAlmageed, "[Does Generative Face Completion help Face Recognition?](#) ", in Proc. of IAPR International Conference on Biometrics (ICB) 2019 [1]_.

![Teaser](https://i.imgur.com/Pv0W9mb.png)


## Features
* **Encoder-Decoder** for face completion.
* ...

## Dependencies

* [Pytorch](http://pytorch.net/)
* [Numpy](http://www.numpy.org/)
* [Python3.6](https://www.python.org/download/releases/3.6/)

The code has been tested on Linux only. On Linux you can rely on the default version of python, installing all the packages needed from the package manager or on Anaconda Python and install required packages through `conda`. 

**Importantly:** OpenGL or other 3D rendering libraries are **not** required to run this code.

## Usage

### Run it

```bash
$ python demo.py <image-path>
```

## Current Limitations


## Citation

Please cite our paper with the following bibtex if you use our face renderer:

``` latex
@inproceedings{mathai2019doesgenerative,
  title={Does Generative Face Completion Help Face Recognition?},
  author={Mathai, Joe and Masi, Iacopo and Abd-Almageed, Wael},
  booktitle={International Conference on Biometrics (ICB)},
  year={2019},
}
```

## References

[1] J. Mathai*, I. Masi*, W. AbdAlmageed, "Does Generative Face Completion help Face Recognition? ", in Proc. of IAPR International Conference on Biometrics (ICB) 2019

    \* denotes equal authorship
## Changelog
- April 2019, First  Release 

## Disclaimer

_The SOFTWARE PACKAGE provided in this page is provided "as is", without any guarantee made as to its suitability or fitness for any particular use. It may contain bugs, so use of this tool is at your own risk. We take no responsibility for any damage of any sort that may unintentionally be caused through its use._

## Contacts

If you have any questions, drop an email to _iacopo@isi.edu_ and _jmathai@isi.edu_ or leave a message below with GitHub (log-in is needed).
