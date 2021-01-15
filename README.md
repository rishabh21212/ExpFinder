# ExpFinder: An Ensemble Expert Finding Model Integrating N-gram Vector Space Model and μCO-HITS

## Introduction
<p align="justify">
Finding an expert plays a crucial role in driving successful collaborations and speeding up high-quality research development and innovations. However, the rapid growth of scientific publications and digital expertise data makes identifying the right experts a challenging problem. Existing approaches for finding experts given a topic can be categorised into information retrieval techniques based on vector space models, document language models, and graph-based models. In this paper, we propose <i>ExpFinder</i>, a new ensemble model for expert finding, that integrates a novel <i>N</i>-gram vector space model, denoted as <i>n</i>VSM, and a graph-based model, denoted as <i>μCO-HITS</i>, that is a proposed variation of the CO-HITS algorithm. The key of <i>n</i>VSM is to exploit recent inverse document frequency weighting method for <i>N</i>-gram words, and <i>ExpFinder</i> incorporates <i>n</i>VSM into <i>μCO-HITS</i> to achieve expert finding. We comprehensively evaluate <i>ExpFinder</i> on four different datasets from the academic domains in comparison with six different expert finding models. The evaluation results show that <i>ExpFinder</i> is an highly effective model for expert finding, substantially outperforming all the compared models in 19% to 160.2%.
</p>

## Setup steps
1. Clone the repository
```
git clone https://github.com/Yongbinkang/ExpFinder.git
```
2. Install dependencies
```
pip install requirements.txt
```
3. Download the SciBert model into the `model/` folder as mentioned in [this](https://github.com/Yongbinkang/ExpFinder/tree/main/model).

## Directory structure

For more instructions on setting up the project to run the pipeline in the `experimental pipline.ipynb` file, we sketch out the directory structure with description below:

* The __`data/`__ directory contains input or output data for the entire process. Four current data files in this directory is required for the data generation process.
* The __`model/`__ directory contains the SciBert model. Due to the large size of the model, we do not upload it here. For more details on how to download the model, please refer to the instruction at [this](https://github.com/Yongbinkang/ExpFinder/blob/main/model/README.md).
* The __`src/`__ directory contains the source code for the entire process including:
  * The __`algo/`__ directory has the `expfinder.py` file which is the source code for the ExpFinder algorithm. For more details about the algorithm, please refer to our paper.
  * The __`controller`__ directory has the `generator.py` file which is used to control the data generation process.
  * The __`lib`__ directory has four different python files serving different purposes as:
    * The `np_extractor.py` file aims to extract noun phrases from documents (using the `tokenization` module below) and estimates N-gram TFIDF for each noun phrase.
    * The `semantic.py` file aims to vectorise every single phrase by using the SciBert model.
    * The `tokenization.py` file aims to extract tokens and noun phrases with their statistical information. Note that this contains the parser for the noun phrase extraction.
    * The `weight.py` file aims to calculate personalised weights for given vectors or matrices.
* The __`experimental pipeline.ipynb`__ file contains pipelines for the entire process which is shown in the __Flow__ section below.

## Flow

![Execution flow](https://github.com/Yongbinkang/ExpFinder/blob/main/images/flow.png)

1. Raw data is read and transform to a proper format like dataframes or vectors.
2. With the prepared data, we generate the necessary data for the ExpFinder algorithm such as expert-document, document-phrase, document-topic, personalised matrices and expert-document counted vectors.
3. The data is fitted into the ExpFinder algorithm the best parameters based on the empirical experiment.
4. The expected output from the algorithm contains weights of between experts and topics as well as documents and topics



## Citing
