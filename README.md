# 1.Directory Description

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
Running `main.py` or `main_v2.py` for newly added datasets, `main_v2.py` is designed for parallel computing

# 3.Some statement

The prediction results of 'DREAM5_alg_output' and some auxiliary files are downloaded from https://www.synapse.org/#!Synapse:syn2787211.
The directory "exp_scripts" contains some experiments scripts mentioned in the paper. You may change some absolute path in the codes to ensure normal running.

