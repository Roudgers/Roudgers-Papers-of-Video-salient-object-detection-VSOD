# VSOD CNNs-based Papers       
    
**NO.** | **Years** | **Title** | **Links** 
:-: | :-: | :-:  | :-: 
01 | **ECCV20** | TENet: Triple Excitation Network for Video Salient Object Detection | [Paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500205.pdf)/[Code](https://github.com/OliverRensu/TENet-Triple-Excitation-Network-for-Video-Salient-Object-Detection)
02 | **TIP20** | Learning Long-term Structural Dependencies for Video Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/9199537)/[Code](https://github.com/bowangscut/LSD_GCN-for-VSOD)  
03 | **TIP20** | A Novel Video Salient Object Detection Method via Semi-supervised Motion Quality Perception | [Paper](https://arxiv.org/abs/2008.02966)/[Code](https://github.com/qduOliver/MQP)
04 | **TIP20** | Exploring Rich and Efficient Spatial Temporal Interactions for Real Time Video Salient Object Detection | [Paper](https://arxiv.org/abs/2008.02973)/[Code](https://github.com/guotaowang/STVS)
05 | **AAAI20** | Pyramid Constrained Self-Attention Network for Fast Video Salient Object Detection | [Paper](http://mftp.mmcheng.net/Papers/20AAAI-PCSA.pdf)/[Code](https://github.com/guyuchao/PyramidCSA)
06 | **ICCV19** | Semi-Supervised Video Salient Object Detection Using Pseudo-Labels | [Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yan_Semi-Supervised_Video_Salient_Object_Detection_Using_Pseudo-Labels_ICCV_2019_paper.pdf)/[Code](https://github.com/Kinpzz/RCRNet-Pytorch)
07 | **ICCV19** | Motion Guided Attention for Video Salient Object Detection | [Paper](https://arxiv.org/abs/1909.07061)/[Code](https://github.com/lhaof/Motion-Guided-Attention)
08 | **CVPR19** | Shifting More Attention to Video Salient Objection Detection | [Paper](https://github.com/DengPingFan/DAVSOD/blob/master/%5B2019%5D%5BCVPR%5D%5BOral%5D【SSAV】【DAVSOD】Shifting%20More%20Attention%20to%20Video%20Salient%20Object%20Detection.pdf)/[Code](https://github.com/DengPingFan/DAVSOD)
09 | **ECCV18** | Pyramid Dilated Deeper ConvLSTM for Video Salient Object Detection | [Paper](https://github.com/shenjianbing/PDBConvLSTM/blob/master/Pyramid%20Dilated%20Deeper%20CoonvLSTM%20for%20Video%20Salient%20Object%20Detection.pdf)/[Code](https://github.com/shenjianbing/PDB-ConvLSTM)
10 | **CVPR18** | Flow Guided Recurrent Neural Encoder for Video Salient Object Detection | [Paper](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1226.pdf)/Code
11 | **TIP18** | A Benchmark Dataset and Saliency-guided Stacked Autoencoders for Video-based Salient Object Detection | [Paper](https://arxiv.org/pdf/1611.00135.pdf)/[Code](http://cvteam.net/projects/TIP18-VOS/VOS.html)

# VSOD Datasets

**NO.** | **Datasets** | **Papers** | **Years** | **Videos** | **Annotated Frames** | **Links** | **Tasks** 
:-: | :-: | :-:  | :-: | :-: | :-:  | :-: | :-: 
01 | **DAVSOD** | Shifting More Attention to Video Salient Objection Detection | CVPR2020 | 226 | 23938 | [Origin version](http://dpfan.net/DAVSOD/)/[Reorganized version (Password: oip1)](https://pan.baidu.com/s/1ll7Agqwx8Y-G-h3bNgi6Tg) | Video Salient Object Detection
02 | **DAVIS** | A benchmark dataset and evaluation methodology for video object segmentation | CVPR2016 | 50 | 3455 | [Origin version](https://davischallenge.org)/[Reorganized version (Password: oip1)](https://pan.baidu.com/s/1TkHSCxd8sLcp0A1dvZZL1g) | Video Object Segmentation
03 | **VOS** | A benchmark dataset and saliency-guided stacked autoencoders for videobased salient object detection | TIP2018 | 200 | 7650 | [Origin version](http://cvteam.net/projects/TIP18-VOS/VOS.html)/[Reorganized version (Password: oip1)](https://pan.baidu.com/s/18UsV8ns30mkeXpXjPSFvcQ) | Video Salient Object Detection
04 | **SegTrack-V2** | Video segmentation by tracking many figureground segments | ICCV2013 | 13 | 1025 | [Origin version](http://web.engr.oregonstate.edu/~lif/SegTrack2/dataset.html)/[Reorganized version (Password: oip1)](https://pan.baidu.com/s/1xoScyJVy0j-VS1Vx4_L81g) | Video Object Segmentation
05 | **ViSal** | Consistent video saliency using local gradient flow optimization and global refinement | TIP2015 | 17 | 193 | [Origin version](https://github.com/shenjianbing/ViSalDataset)/[Reorganized version (Password: oip1)](https://pan.baidu.com/s/1nAoqrscUrfOgTh7Q7oGMuQ) | Video Salient Object Detection
06 | **FBMS** | Segmentation of moving objects by long term video analysis | TPAMI2014 | 59 | 720 | [Origin version](https://lmb.informatik.uni-freiburg.de/resources/datasets/)/[Reorganized version (Password: oip1)](https://pan.baidu.com/s/1vGagLB8ENekwQLwAQbrKug) | Motion Segmentation

# VSOD Evaluation codes

To guarantee fair comparisons, we utilize the widely-used evaluation toolbox provided by [SSAV](https://github.com/DengPingFan/DAVSOD).

# Details of VSOD CNNs-based Papers       
    
**NO.** | **Years** | **Title** | **Links** | **Optical Flow** | **ConvLSTM/GRU** | **3DConv** | **GraphConv** | **Fully Supervised** | **Semi-Supervised** | **Unsupervised**
:-: | :-: | :-:  | :-: | :-: | :-:  | :-: | :-: | :-: | :-: | :-:
01 | **ECCV20** | TENet | [Paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500205.pdf)/[Code](https://github.com/OliverRensu/TENet-Triple-Excitation-Network-for-Video-Salient-Object-Detection) | **Y** | | | | **Y** | | |
02 | **TIP20** | LSD | [Paper](https://ieeexplore.ieee.org/document/9199537)/[Code](https://github.com/bowangscut/LSD_GCN-for-VSOD) | | | | **Y** | **Y** | | |
03 | **TIP20** | MQP | [Paper](https://arxiv.org/abs/2008.02966)/[Code](https://github.com/qduOliver/MQP) | | | | | | **Y** | |
04 | **TIP20** | STVS | [Paper](https://arxiv.org/abs/2008.02973)/[Code](https://github.com/guotaowang/STVS) | | | **Y** | | **Y** | | |
05 | **AAAI20** | PCSA | [Paper](http://mftp.mmcheng.net/Papers/20AAAI-PCSA.pdf)/[Code](https://github.com/guyuchao/PyramidCSA) | | | **Y** | | **Y** | | |
06 | **ICCV19** | RCR | [Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yan_Semi-Supervised_Video_Salient_Object_Detection_Using_Pseudo-Labels_ICCV_2019_paper.pdf)/[Code](https://github.com/Kinpzz/RCRNet-Pytorch) | | **Y** | | | | **Y** | |
07 | **ICCV19** | MGA | [Paper](https://arxiv.org/abs/1909.07061)/[Code](https://github.com/lhaof/Motion-Guided-Attention) | **Y** | | | | **Y** | | |
08 | **CVPR19** | SSAV | [Paper](https://github.com/DengPingFan/DAVSOD/blob/master/%5B2019%5D%5BCVPR%5D%5BOral%5D【SSAV】【DAVSOD】Shifting%20More%20Attention%20to%20Video%20Salient%20Object%20Detection.pdf)/[Code](https://github.com/DengPingFan/DAVSOD) | | **Y** | | | **Y** | | |
09 | **ECCV18** | PDB | [Paper](https://github.com/shenjianbing/PDBConvLSTM/blob/master/Pyramid%20Dilated%20Deeper%20CoonvLSTM%20for%20Video%20Salient%20Object%20Detection.pdf)/[Code](https://github.com/shenjianbing/PDB-ConvLSTM) | | **Y** | | | **Y** | | |
10 | **CVPR18** | FGRN | [Paper](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1226.pdf)/Code | **Y** | | | | **Y** | | |
11 | **TIP18** | SSA | [Paper](https://arxiv.org/pdf/1611.00135.pdf)/[Code](http://cvteam.net/projects/TIP18-VOS/VOS.html) | | | **Y** | | | | **Y** |
