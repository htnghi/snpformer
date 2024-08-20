echo "Profiling Perf: MLP_Mixer Regular Pheno1"
python run_evaluate_model.py --best_param ./best_params/regular_pheno1_best_param_mlpmixer44_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Regular --model_name MLPMixer --gpucuda 1 > ./data/results/prof_perf/prof_regular_pheno1_kmernonoverlap_mlpmixer.txt

echo "Profiling Perf: CNN_Transformer Regular Pheno1"
python run_evaluate_model.py --best_param ./best_params/regular_pheno1_best_param_testcnntransformer37_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Regular --model_name Test1_CNN --gpucuda 1 > ./data/results/prof_perf/prof_regular_pheno1_kmernonoverlap_cnntransf.txt

echo "Profiling Perf: Test1_Transformer Regular Pheno1"
python run_evaluate_model.py --best_param ./best_params/regular_pheno1_best_param_testtransformer49_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Regular --model_name Test1_Transformer --gpucuda 1 > ./data/results/prof_perf/prof_regular_pheno1_kmernonoverlap_transf.txt

echo "Profiling Perf: Infinite_Transformer Regular Pheno1"
python run_evaluate_model.py --best_param ./best_params/regular_pheno1_best_param_test24_infinitetransformer_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Regular --model_name Test2_Infinite_Transformer --gpucuda 1 > ./data/results/prof_perf/prof_regular_pheno1_kmernonoverlap_inftransf.txt

echo "Profiling Perf: HyperMixer Regular Pheno1"
python run_evaluate_model.py --best_param ./best_params/regular_pheno1_best_param_test75_hypermixer_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Regular --model_name Test1_HyperMixer --gpucuda 1 > ./data/results/prof_perf/prof_regular_pheno1_kmernonoverlap_hypermixer.txt



echo "Profiling Perf: MLP_Mixer Splitchr Pheno1"
python run_evaluate_model.py --best_param ./best_params/splitchr_pheno1_best_param_mlpmixer17_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Chromosomesplit --model_name MLPMixer --gpucuda 1 > ./data/results/prof_perf/prof_splitchr_pheno1_kmernonoverlap_mlpmixer.txt

echo "Profiling Perf: CNN_Transformer Splitchr Pheno1"
python run_evaluate_model.py --best_param ./best_params/splitchr_pheno1_best_param_testcnntransformer25_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Chromosomesplit --model_name Test1_CNN --gpucuda 1 > ./data/results/prof_perf/prof_splitchr_pheno1_kmernonoverlap_cnntransf.txt

echo "Profiling Perf: Test1_Transformer Splitchr Pheno1"
python run_evaluate_model.py --best_param ./best_params/splitchr_pheno1_best_param_transformer22_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Chromosomesplit --model_name Transformer --gpucuda 1 > ./data/results/prof_perf/prof_splitchr_pheno1_kmernonoverlap_transf.txt

echo "Profiling Perf: Infinite_Transformer Splitchr Pheno1"
python run_evaluate_model.py --best_param ./best_params/splitchr_pheno1_best_param_test15_infinitetransformer_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Chromosomesplit --model_name Test2_Infinite_Transformer --gpucuda 1 > ./data/results/prof_perf/prof_splitchr_pheno1_kmernonoverlap_inftransf.txt

echo "Profiling Perf: HyperMixer Splitchr Pheno1"
python run_evaluate_model.py --best_param ./best_params/splitchr_pheno1_best_param_hypermixer43_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Chromosomesplit --model_name HyperMixer --gpucuda 1 > ./data/results/prof_perf/prof_splitchr_pheno1_kmernonoverlap_hypermixer.txt