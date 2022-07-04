using NetworkInference
file = ARGS[1]
println(file)
input_dir = "/home/fengke/pycode/my_exp/paper/exp/tmp/"
output_dir = "/home/fengke/pycode/my_exp/paper/exp/output/sc_alg_output/PIDC/"
alg = PIDCNetworkInference()
@time genes = get_nodes(string(input_dir,file,".txt"));
@time network = InferredNetwork(alg,genes);
write_network_file(string(output_dir,file,".txt"),network)