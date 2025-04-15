# **Human Intention Recognition Using Context Relationships in Complex Scenes**

This repository provides code for human intention recognition using context relationships in complex scenes. We propose a Spatial Temporal Graph Attention Informer Neural Network. The code is executed on dual RTX2080ti for both training and testing.

## **Requirements**

- python=3.8
- pytorch=1.1
- scipy=1.1.0
- cypthon
- dill
- transformer
- easydict
- h5py
- opencv
- pandas
- tqdm
- yaml
- pytorch geometric
- torch_scatter-2.0.9-cp38-cp38-linux_x86
- torch_sparse-0.6.12-cp38-cp38-linux_x86_64
- torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64

## Action Genome Setup

For the [Action Genome] (https://www.actiongenome.org), the file structure should be organized as follows. Additionally, download the file `object_bbox_and_relationship_filtersmall.pkl` and place it in the dataloader. The dataset should look like:

- action_genome 
    - annotations  
    - frames  
    - videos   
  
For the objects detection, we use the FasterRCNN (https://github.com/jwyang/faster-rcnn.pytorch).

## Train and Evaluation

-Put the Action Genome in the dataloader file.  
-Run the scene_graph_train to get the graph.    
-You can convert the object_bbox_and_relationship.pkl to a csv file to check the relationships between humans and objects.  
-Run dataloader.py to encode the graph for training.  
-The model can be trained by train.py.   
-The model can be evaluated by test.py.

## Citation

If our work is helpful for your research, please cite our publication:
```bibtex
@inproceedings{TongSTGAIN,
  title={Human Intention Recognition Using Context Relationships in Complex Scenes},
  author={Tong Tong and Rossi Setchi and Yulia Hicks},
  Journal={Expert Systems with Applications},
  pages={126147},
  year={2025}
}
