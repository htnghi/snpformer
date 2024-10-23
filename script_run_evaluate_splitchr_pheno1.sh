#echo "Profiling Perf: MLP_Mixer Splitchr Pheno1"
#python run_evaluate_model.py --best_param ./best_params/splitchr_pheno1_best_param_mlpmixer17_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Chromosomesplit --model_name MLPMixer --gpucuda 1 > ./data/results/prof_perf/prof_splitchr_pheno1_kmernonoverlap_mlpmixer.txt

echo "Profiling Perf: CNN_Transformer Splitchr Pheno1"
/home/di35hef/miniconda3/bin/python run_evaluate_model.py --best_param ./best_params/splitchr_pheno1_best_param_testcnntransformer50_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Chromosomesplit --model_name Test1_CNN --gpucuda 1 > ./data/results/prof_perf/prof_splitchr_pheno1_kmernonoverlap_cnntransf_1.txt

echo "Profiling Perf: Test1_Transformer Splitchr Pheno1"
/home/di35hef/miniconda3/bin/python run_evaluate_model.py --best_param ./best_params/splitchr_pheno1_best_param_transformer54_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Chromosomesplit --model_name Transformer --gpucuda 1 > ./data/results/prof_perf/prof_splitchr_pheno1_kmernonoverlap_transf_1.txt

echo "Profiling Perf: Infinite_Transformer Splitchr Pheno1"
/home/di35hef/miniconda3/bin/python run_evaluate_model.py --best_param ./best_params/splitchr_pheno1_best_param_test30_infinitetransformer_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Chromosomesplit --model_name Test2_Infinite_Transformer --gpucuda 1 > ./data/results/prof_perf/prof_splitchr_pheno1_kmernonoverlap_inftransf_1.txt

echo "Profiling Perf: HyperMixer Splitchr Pheno1"
/home/di35hef/miniconda3/bin/python run_evaluate_model.py --best_param ./best_params/splitchr_pheno1_best_param_hypermixer128_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Chromosomesplit --model_name HyperMixer --gpucuda 1 > ./data/results/prof_perf/prof_splitchr_pheno1_kmernonoverlap_hypermixer_1.txt