# 0.Overview

This repository is related to a novel GRN (Gene Regulatory Network) inference framework named GRINCD which integrate node embedding and causal discovery. GRINCD used to pipelines of linear and non-linear, it first generates high-quality gene representation using GraphSAGE, and then using pairwise causal discovery model ANM to calculate regulatory relations. GRINCD does not generate deterministic directed networks, it only output a ranked list representing importance of each regulatory relations as GINIE3.

# 1.Directory Description

## -'exp_scripts'

This directory contains all experimental scripts in GRN inference.

### --'exp_script.py'

This script includes cross-comparison of perforamce of methods, identification of key factors for colon cancer, and so on.

### --'graph_property.py'

This script is used to caulculate property of each gold standard.

### --'input_process.py'

This script is used to process raw expression matrix for single-cell datasets.

### --'PIDC.jl'

This script is used to inference GRN using PIDC, you may need to build Julia environment containing package NetworkInference for the running of this script.

## -'input'

This directory includes all datasets and other materials necessary for all methods and framework. In DREAM5
datasets,'AUPR.mat' and 'AUROC.mat' are used to calculate confidence score, they are generated from respective
distribution. 'expression_data.tsv' as method input is gene expression matrix whose rows represent samples and columns
represent genes. 'gene_ids.tsv' is the mapping between genes' names and genes' ids. 'gold_standards.tsv' is the ground
truth used for evaluation. 'mi.tsv' is a persistent mutual information matrix used for improving efficiency.
'transcription_factors.tsv' is the gene ids of transcription factors.Actually, only 'expression_data.tsv' and '
transciption_factors.tsv' are fed into methods or frameworks, other files are only used for evaluation or auxiliary
function.

## -'output'

This directory includes all outputs for all methods and framework. The outputs of our framework are placed in separate
sub-folders named after the name of respective datasets. Each of these sub-folders include 100 predictions, and the
performance of our framework is evaluated using mean values of these 100 predictions.

### --'DREAM5_alg_output'

This directory includes the outputs of methods on DREAM5 datasets that used for comparison.

### --'sc_alg_output'

This directory includes the outputs of methods on hESC and mDC that used for comparison.

## -'probability_densities'

This directory includes AUROC values and AUPR values distribution for all datasets.

# 2.Usage

Using `python eval.py` to evaluation the results for the datasets mentioned in the paper, 
Running `main.py` or `main_v2.py` for newly added datasets, `main_v2.py` is designed for parallel computing which means that two cores of GPU are necessary, if this need is not met, please run `main.py` instead. For more introductions about methods, please refer to code annotations.

# 3.Some statement

The prediction results of 'DREAM5_alg_output' and some auxiliary files are downloaded from https://www.synapse.org/#!Synapse:syn2787211.
The directory "exp_scripts" contains some experiments scripts mentioned in the paper. You may change some absolute path in the codes to ensure normal running.
The full content of input and output can be downloaded from https://zenodo.org/record/6794599#.YsLzqXbP1PZ
