import argparse

def ParseArgs():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=3e-3, type=float, help='learning rate')#tune source:1e-3
	parser.add_argument('--difflr', default=1e-3, type=float, help='learning rate')
	parser.add_argument('--batch', default=8, type=int, help='batch size')
	parser.add_argument('--patience',   type=int,   default=20)
	parser.add_argument('--ratio', type=int, default=[20, 40, 60])
	#retain params
	parser.add_argument('--threshold', default=0.5, type=float, help='threshold to filter users')
	parser.add_argument('--data', default='aminer', type=str, help='name of dataset')
	parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
	parser.add_argument('--task_name', default='retain7', type=str, help='specific task')
	parser.add_argument('--eval_hdfs', default='', type=str, help='eval hdfs path to save the posterior result')
	parser.add_argument('--input_hdfs1', default='', type=str, help='dataset_input')
	parser.add_argument('--input_hdfs2', default='', type=str, help='dataset_input')
	parser.add_argument('--output_model1', default='', type=str, help='output_model1')
	parser.add_argument('--tb_log_dir', default='', type=str, help='tb_log_dir')

	#gat setting
	parser.add_argument('--head', default=4, type=int, help='number of heads in attention')


	#gcn_setting
	parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
	parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')
	parser.add_argument('--decay_step', type=int,   default=1)
	parser.add_argument('--init', default=False, type=bool, help='whether initial embedding')
	parser.add_argument('--latdim', default=256, type=int, help='embedding size')
	parser.add_argument('--gcn_layer', default=5, type=int, help='number of gcn layers')#
	parser.add_argument('--uugcn_layer', default=1, type=int, help='number of gcn layers')
	parser.add_argument('--load_model', default=None, help='model name to load')
	parser.add_argument('--topk', default=20, type=int, help='K of top K')
	parser.add_argument('--dropRate', default=0.5, type=float, help='rate for dropout layer')
	parser.add_argument('--gpu', default=0, type=int, help='indicates which gpu to use')
	
	
	#diffusion setting
	parser.add_argument('--dims', type=str, default='[64]')
	parser.add_argument('--d_emb_size', type=int, default=8)
	parser.add_argument('--norm', type=bool, default=True)
	parser.add_argument('--steps', type=int, default=250)#tune 5 20 50 100 150
	parser.add_argument('--noise_scale', type=float, default=1e-5) #tune
	parser.add_argument('--noise_min', type=float, default=0.0001)
	parser.add_argument('--noise_max', type=float, default=0.001)
	parser.add_argument('--sampling_steps', type=int, default=0)

	parser.add_argument('--ssl_reg', default=1e-3, type=float, help='weight for contrative learning')
	parser.add_argument('--temp', default=0.5, type=float, help='temperature in contrastive learning')

	return parser.parse_args()
args = ParseArgs()


# DBLP
# python main.py --data DBLP --lr 1e-4 --temp 0.3 --ssl_reg 0.1 --steps 200 --gpu 2
#aminer
# python main.py --data aminer --uugcn_layer 2 --difflr 1e-1 --gpu 3