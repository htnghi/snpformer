# echo "Profiling Perf: MLP_Mixer Regular Pheno1"
# python run_evaluate_model.py --best_param ./best_params/regular_pheno1_best_param_mlpmixer44_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Regular --model_name MLPMixer --gpucuda 1 > ./data/results/prof_perf/prof_regular_pheno1_kmernonoverlap_mlpmixer.txt

echo "Profiling Perf: CNN_Transformer Regular Pheno1"
/home/di35hef/miniconda3/bin/python run_evaluate_model.py --best_param ./best_params/regular_pheno1_best_param_testcnntransformer58_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Regular --model_name Test1_CNN --gpucuda 1 > ./data/results/prof_perf/prof_regular_pheno1_kmernonoverlap_cnntransf_1.txt

echo "Profiling Perf: Test1_Transformer Regular Pheno1"
/home/di35hef/miniconda3/bin/python run_evaluate_model.py --best_param ./best_params/regular_pheno1_best_param_testtransformer63_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Regular --model_name Test1_Transformer --gpucuda 1 > ./data/results/prof_perf/prof_regular_pheno1_kmernonoverlap_transf_1.txt

echo "Profiling Perf: Infinite_Transformer Regular Pheno1"
/home/di35hef/miniconda3/bin/python run_evaluate_model.py --best_param ./best_params/regular_pheno1_best_param_test33_infinitetransformer_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Regular --model_name Test2_Infinite_Transformer --gpucuda 1 > ./data/results/prof_perf/prof_regular_pheno1_kmernonoverlap_inftransf_1.txt

echo "Profiling Perf: HyperMixer Regular Pheno1"
/home/di35hef/miniconda3/bin/python run_evaluate_model.py --best_param ./best_params/regular_pheno1_best_param_test78_hypermixer_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Regular --model_name Test1_HyperMixer --gpucuda 1 > ./data/results/prof_perf/prof_regular_pheno1_kmernonoverlap_hypermixer_1.txt

