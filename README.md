# TFeat C++ Frontend/API
This repository is forked from [TFeat](https://github.com/vbalnt/tfeat).    
Additionally C++ frontend / API example (for PyTorch) is placed here.
If you want to use this, you should do the following.

## How to use

1. Export a model using `export_model.ipynb`.    
   Now you have `tfeat_model.pt`. This is loaded by a cpp-example.
 
2. Download and unzip `libtorch`. This is necessary if we use cpp-frontend of PyTroch.
   ```shell
   cd cpp_example && bash setup_libtorch.sh
   ```
   
3. Compile `tfeat_demo.cpp` using `CMakeLists.txt`.
   ```shell
   mkdir build && cd build
   cmake .. && make
   ```
4. Execute `tfeat_demo` in `build`!
   ```shell
   (e.g.) ./tfeat_demo ../../tfeat_model.pt ../../imgs/v_churchill/1.ppm ../../imgs/v_churchill/6.ppm
   ```
## Result
This figure is the result of the above.

![tfeat_cpp_example](https://github.com/cashiwamochi/tfeat/blob/master/cpp_example/result_image/matches-by-tfeat.png)


### Note
As you know, [C++ API of PyTorch is "beta" stability.](https://pytorch.org/cppdocs/)     
Now this implementation works on my environment(Ubuntu18, Pytorch1.0), but in the future this may not work.   
By the way, the result of C++ API is slightly different from the result of Python. 
I'm investigating this issue.

-------------------------------------

# TFeat shallow convolutional patch descriptor
Code for the BMVC 2016 paper [Learning local feature descriptors with triplets and shallow convolutional neural networks](http://www.bmva.org/bmvc/2016/papers/paper119/paper119.pdf)

## Pre-trained models
We provide the following pre-trained models:

| network name      | model link                                                        | training dataset   |
| -------------     | :-------------:                                                   | -----:             |
| `tfeat-liberty`   | [tfeat-liberty.params](./pretrained-models/tfeat-liberty.params)  | liberty (UBC)      |
| `tfeat-yosemite`  | [tfeat-yosemite.params](./pretrained-models/tfeat-yosemite.params) | yosemite (UBC)     |
| `tfeat-notredame` | [tfeat-notredame.params](./pretrained-models/tfeat-notredame.params) | notredame (UBC)    |
| `tfeat-ubc`       | coming soon...                                                    | all UBC            |
| `tfeat-hpatches`  | coming soon...                                                    | HPatches (split A) |
| `tfeat-all`       | coming soon...                                                    | All the above      |


## Quick start guide
To run `TFeat` on a tensor of patches:

```python
tfeat = tfeat_model.TNet()
net_name = 'tfeat-liberty'
models_path = 'pretrained-models'
net_name = 'tfeat-liberty'
tfeat.load_state_dict(torch.load(os.path.join(models_path,net_name+".params")))
tfeat.cuda()
tfeat.eval()

x = torch.rand(10,1,32,32).cuda()
descrs = tfeat(x)
print(descrs.size())

#torch.Size([10, 128])
```

Note that no normalisation is needed for the input patches, 
it is done internally inside the network. 

## Testing `TFeat`: Examples (WIP)
We provide an `ipython` notebook that shows how to load and use 
the pre-trained networks. We also provide the following examples:

- extracting descriptors from image patches
- matching two images using `openCV`
- matching two images using `vlfeat`

For the testing example code, check [tfeat-test notebook](tfeat-test.ipynb)

## Re-training `TFeat`
We provide an `ipython` notebook with examples on how to train
`TFeat`.  Training can either use the `UBC` datasets `Liberty,
Notredame, Yosemite`, the `HPatches` dataset, and combinations 
of all the datasets. 

For the training code, check [tfeat-train notebook](tfeat-train.ipynb)
