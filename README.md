# ConfigX: Modular Configuration for Evolutionary Algorithms via Multitask Reinforcement Learning

Here we provide sourcecodes of ConfigX, which has been recently accpeted by AAAI 2025 as an Oral paper.

## Citation

The PDF version of the paper is available [here](https://arxiv.org/abs/2412.07507). If you find our ConfigX useful, please cite it in your publications or projects.

```latex
@inproceedings{guo2024configx,
  title={ConfigX: Modular Configuration for Evolutionary Algorithms via Multitask Reinforcement Learning},
  author={Guo, Hongshu and Ma, Zeyuan and Chen, Jiacheng and Ma, Yining and Cao, Zhiguang and Zhang, Xinglin and Gong, Yue-Jiao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

## Requirements
You can install all of dependencies of ConfigX via the command below.
```bash
pip install -r requirements.txt
```

## Train
The training process can be easily activated via the command below.
```bash
python main.py
```
For more adjustable settings, please refer to `main.py` and `config.py` for details.

Recording results: Log files will be saved to `./logs`, the file structure is as follow:
```
logs
|--run_name
   |--logging files
   |--...
```
The saved checkpoints will be saved to `./outputs`, the file structure is as follow:
```
outputs
|--run_name
   |--epoch-0.pt
   |--epoch-1.pt
   |--...
```

## Rollout
The rollout process can be easily activated via the command below.
```bash
python main.py --test --load_path [The checkpoint saving directory, default to be "./outputs"] --load_name [The run_name of the target ConfigX model] --load_epoch [The epoch of the model]
```
For example, for testing the model with run_name "20240704T221142" at the 50th epoch, the command is:
```bash
python main.py --test --load_name 20240704T221142 --load_epoch 50
```
