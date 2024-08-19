echo "Profiling Perf: MLP_Mixer Regular Pheno1"
python run_evaluate_model.py --best_param ./best_params/regular_pheno1_best_param_mlpmixer44_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Regular --model_name MLPMixer

echo "Profiling Perf: CNN_Transformer Regular Pheno1"
python run_evaluate_model.py --best_param ./best_params/regular_pheno1_best_param_testcnntransformer37_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Regular --model_name MLPMixer

echo "Profiling Perf: Test1_Transformer Regular Pheno1"
python run_evaluate_model.py --best_param ./best_params/regular_pheno1_best_param_testtransformer49_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Regular --model_name Test1_Transformer

echo "Profiling Perf: Infinite_Transformer Regular Pheno1"
python run_evaluate_model.py --best_param ./best_params/regular_pheno1_best_param_test24_infinitetransformer_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Regular --model_name MLPMixer

echo "Profiling Perf: HyperMixer Regular Pheno1"
python run_evaluate_model.py --best_param ./best_params/regular_pheno1_best_param_test75_hypermixer_nonoverlap.json --data_dir ./ --dataset 1 --embedding_type kmer_nonoverlap --model_type Regular --model_name MLPMixer
