echo "Profiling Perf: MLP_Mixer Regular Pheno1"
python run_evaluate_model.py --best_param ./best_params/regular_pheno1_best_param_mlpmixer44_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Regular --model_name MLPMixer > ./data/results/prof_perf/prof_regular_pheno1_kmernonoverlap_mlpmixer.txt

echo "Profiling Perf: CNN_Transformer Regular Pheno1"
python run_evaluate_model.py --best_param ./best_params/regular_pheno1_best_param_testcnntransformer37_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Regular --model_name Test1_CNN > ./data/results/prof_perf/prof_regular_pheno1_kmernonoverlap_cnntransf.txt

echo "Profiling Perf: Test1_Transformer Regular Pheno1"
python run_evaluate_model.py --best_param ./best_params/regular_pheno1_best_param_testtransformer49_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Regular --model_name Test1_Transformer > ./data/results/prof_perf/prof_regular_pheno1_kmernonoverlap_transf.txt

echo "Profiling Perf: Infinite_Transformer Regular Pheno1"
python run_evaluate_model.py --best_param ./best_params/regular_pheno1_best_param_test24_infinitetransformer_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Regular --model_name Test2_Infinite_Transformer > ./data/results/prof_perf/prof_regular_pheno1_kmernonoverlap_inftransf.txt

echo "Profiling Perf: HyperMixer Regular Pheno1"
python run_evaluate_model.py --best_param ./best_params/regular_pheno1_best_param_test75_hypermixer_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Regular --model_name Test1_HyperMixer > ./data/results/prof_perf/prof_regular_pheno1_kmernonoverlap_hypermixer.txt
