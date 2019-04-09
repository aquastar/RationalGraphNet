# Rational Neural Networks (RemezNet)
![remez net](ratnet.png)

# Demo of Rational Net in spectral domain
**func**: the target function to approximate  
**fit**: rational function by nerual networks  
![approximation by rational neural networks](demo.gif)

# Rational Neural Network for Graph Signal Prediction


## Instruction (under construction)
install required python package
```
pip install -r requirements.txt  
```

Choose on pre-processed datasets (Default is 1st below): 
- [America Revolution (unweighted)](https://github.com/corybrunson/triadic), pre-processed: [TSV file](http://konect.uni-koblenz.de/networks/brunson_revolution))
- [crime (unweighted)](http://konect.uni-koblenz.de/networks/moreno_crime), pre-processed:[TSV file](http://konect.uni-koblenz.de/networks/moreno_crime)
- [language-country (weighted)](http://www.unicode.org/cldr/charts/25/supplemental/territory_language_information.html), pre-processed:[TSV file](http://konect.uni-koblenz.de/networks/unicodelang)

data switch can be configured by pass dataset index in remez_net.py
```python
gen_data(data=1)
```
Then, run the main program:
```
python remez_net.py  
```
GCN configuration is also provided. To perform regression task and compare fairly, neural network weights are removed:
```
python pygcn/train.py  
```
MSE 
|               | RationalNet   | GCN |
| ------------- | ------------- | ------------- |
| America Revolution  | 0.0236  | 1.3641  |
| crime  | 0.26021 | 1.0605  |
| language  |0.0329  | 0.3912  |

# Related papar
Codes for the paper 
> Zhiqian Chen, Feng Chen, Rongjie Lai, Xuchao Zhang, and Chang-Tien Lu, Rational Neural Networks for Approximating Jump Discontinuities of Graph Convolution Operator, International Conference on Data Mining(ICDM), Singpore, 2018

# Citation
```
@article{ratgraphnet
  author    = {Zhiqian Chen and
               Feng Chen and
               Rongjie Lai and
               Xuchao Zhang and
               Chang{-}Tien Lu},
  title     = {Rational Neural Networks for Approximating Jump Discontinuities of Graph Convolution Operator},
  booktitle = {Proceedings of the The IEEE International Conference on Data Mining},
  year      = {2018},
}
```
