# Entity Linking


## Step1 : Graph Embedding using the DistMult model (https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/emnlp15.pdf)



### Setup

1. Using the name of the database (<data_name>) create a folder in `$PROJECT_ROOT/data`. e.g. `$PROJECT_ROOT/data/umls`  or `$PROJECT_ROOT/data/fb15k-237`  
2. In the above create another folder `raw_text` and copy all the following data resources here.
   * train.txt
   * valid.txt
   * test.txt
3. The knowledge data is in the format : **entity1 \t entity2 \t relation \t 1**
4. Create a conf file with <data_name>.conf in the folder `$PROJECT_ROOT/config` and update the parameters as given in `config/umls.conf`


### Processing Data

`./scripts/preprocess.sh <data.conf> graph`


### Training Models

To train a model use the run script with a data config and a model config like this:  

IF running on CPU :
`./scripts/train.sh configs/umls.conf config/dist_mult.conf graph`  

IF running on GPU :
`./scripts/train.sh configs/umls.conf config/dist_mult.conf graph use_gpu`
   
## Step2 : Mention Context Embedding (http://cogcomp.org/page/publication_view/817)
### Setup

1. Using the name of the database (<data_name>) create a folder in `$PROJECT_ROOT/data`. e.g. `$PROJECT_ROOT/data/umls`  or `$PROJECT_ROOT/data/ncbi_disease_corpus`  
2. In the above create another folder `raw_text` and copy all the following data resources here.
3. The data is in the PUBTATOR format 
4. Create a conf file with <data_name>.conf in the folder `$PROJECT_ROOT/config` and update the parameters as given in `config/ncbi_disease_corpus.conf`

### Processing Data

`./scripts/preprocess.sh <data.conf> mentions`

### Training Models
To train a model use the run script with a data config and a model config like this:

IF running on CPU :
`./scripts/train.sh config/ncbi_disease_corpus.conf config/joint_context.conf mentions`  

IF running on GPU :
`./scripts/train.sh config/ncbi_disease_corpus.conf config/joint_context.conf mentions use_gpu`