# Thinking like a radiologist

Official code of ''Thinking Like a Radiologist: A Dataset for Anatomy-Guided Interleaved Vision Language Reasoning in Chest X-ray Interpretation''

<p align="center">
  📤 <a href="https://github.com/qiuzyc/thinking_like_a_radiologist" target="_self">Get Started</a> &nbsp; | &nbsp;
  📄 <a href="https://arxiv.org/abs/2602.12843" target="_blank">Preprint</a> &nbsp; | &nbsp;
  🤗 <a href="https://github.com/qiuzyc/thinking_like_a_radiologist" target="_blank">Dataset</a>
</p>

<p align="center">
<img src="./statistics.png" width="900">
</p>

## Highlights
**MMRad-IVL-22K**: the first large-scale dataset designed for natively interleaved visual language reasoning in chest X-ray interpretation. It reflects **a repeated cycle of reasoning and visual inspection workflow**
of radiologists, containing 22K high-quality and expert-verified multimodal diagnostic traces.

## Anole-RadCoT 
This repository is adapted from the [Thinking with Generated Images](https://github.com/GAIR-NLP/thinking-with-generated-images)  repository.

## Setup
Install requirements and `transformers`.
```
conda create -n anole python=3.10
cd anole
bash install.sh
```

## Training
### Download Checkpoint
Set the `HF_HOME` in `download_model.py` to the path of the base model checkpoint you want to download.

```
python download_model.py
```
Some reference checkpoints: [Anole-7b](https://huggingface.co/GAIR/Anole-7b-v0.1), [Anole-Zebra-CoT](https://huggingface.co/multimodal-reasoning-lab/Anole-Zebra-CoT) 

### Tokenization
Tokenize the input to fit the training code. The input needs to be restructured to match the Anole format.
```
cd training
python tokenization.py
```
We also provide the example initial and tokenized input data in `./training/input_reference`.


### Train Model with LoRA Adaptation
```
cd training
bash train.sh
```

## Inference
Inference consists of `inference.py` and `detokenization.py`. `combined.py` is used for unified calling.
```
cd inference
bash combined.sh
```

## TODO 
- [x] Release training and inference codes
- [ ] Release a subset of MMRad-IVL dataset
- [ ] Release full MMRad-IVL dataset

## Acknowledgements
- [GeMeX-ThinkVG](https://huggingface.co/datasets/BoKelvin/GEMeX-ThinkVG)
- [Anole-Zebra-CoT](https://huggingface.co/multimodal-reasoning-lab/Anole-Zebra-CoT)
- [Thinking with Generated Images](https://github.com/GAIR-NLP/thinking-with-generated-images)

## Citation
Please consider citing our paper if it is helpful in your research and development.
```
@article{zhao2026thinking,
  title={Thinking Like a Radiologist: A Dataset for Anatomy-Guided Interleaved Vision Language Reasoning in Chest X-ray Interpretation},
  author={Zhao, Yichen and Peng, Zelin and Yang, Piao and Yang, Xiaokang and Shen, Wei},
  journal={arXiv preprint arXiv:2602.12843},
  year={2026}
}
```
