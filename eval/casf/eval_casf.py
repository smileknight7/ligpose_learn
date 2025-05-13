import os
import shutil
import sys
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-3]))
#以上代码是设置这个python文件向上三层作为根目录，方便导入其他文件
import torch
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')

from keras_progbar import Progbar

from model.layers import LigPoseScr
from utils.pdbbind_utils import ComplexScreeningDataset, collate_dummy, collate_screening
from utils.common import *

#------------------
#coreset.dat是评估数据集的文件，包含pdb代码，分辨率，年份，亲和力，蛋白质编号
#TargetInfo.dat是配置蛋白质靶点信息的重要文件，包含T，蛋白质列，L配体列
#-----------------

weight_path = '/home/dtj/work_site/final_data/LigPose_demo/suppl/LigPose_param.chk'
r_path = './TargetInfo.dat'
core_path = '/home/dtj/work_site/final_data/LigPose_demo/eval/pdbbind/core_list.txt'
ens = 1
batch_size = 12
gpu_list = '0,1,2'
world_size = 3
#这里使用到了分布式训练，这个还要再看看怎么弄下


chk = torch.load(weight_path, map_location='cpu')
args = chk['screen_args']
set_all_seed(2023)
#加载并提取模型的配置参数，设置随机种子


# ligand 285
l_list = load_idx_list(core_path)
# receptor 57
lines = open(r_path, 'r').readlines()
dic_r = {line.split('  ')[0]: line[:-1].split('  ')[1:] for line in lines if not line.startswith('#')}
r_list = list(dic_r.keys())
# pair 285-57
l_r_list = [(l, r) for l in l_list for r in r_list]
#配体受体加载



#----------------------
#一下是分布式训练的环境设置
#----------------------
def get_score(rank, world_size, port, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
#实例化LigPose模型，加载与训练权重，设置为评估模式，将模型迁移至GPU，包装为分布式模型
    my_model = LigPoseScr(args)
    my_model.load_state_dict(chk['screen_state_dict'])
    my_model.train(False)
    my_model = my_model.to(rank)
    my_model = torch.nn.parallel.DistributedDataParallel(my_model, device_ids=[rank], find_unused_parameters=True)
#数据集和数据加载器设置，创建测试数据集，创建分布式采样器，设置数据加载器，明确设置为测试模式（这里测试集不是通过训练集/测试集划分的，是专门为评估模型创建的）
    test_dataset = ComplexScreeningDataset('test', args, l_r_list, ens=ens, specific_list=l_r_list)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,
                                            num_replicas=world_size, rank=rank, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size if ens == 1 else 1,
                                              sampler=test_sampler,
                                              num_workers=args.num_workers,
                                              collate_fn=collate_screening if ens == 1 else collate_dummy)
    test_loader.dataset.training = False
#初始化结果存储（进度条设置，禁用梯度计算，批量评估；获取批次数据标识符，将数据移至GPU；模型前向传播；解析预测结果：配体姿势、亲和力、筛选分数）
    scr_result = []
    if rank == 0:
        progBar = Progbar(len(test_loader))
    with torch.no_grad():
        for i, complex_graph in enumerate(test_loader):
            idx_batch = complex_graph.idx
            complex_graph = complex_graph.to(rank)
            tup_pred = my_model(complex_graph)
            _, aff_pred_batch, scr_pred_batch = tup_pred
#集合模式，多个预测取平均。常规模式，处理批次中的每个样本
            if ens > 1:
                aff_pred = aff_pred_batch.mean().item() * 10
                scr_pred = scr_pred_batch.sigmoid().mean().item()

                idx = idx_batch[0]
                scr_result += [[idx[0], idx[1], scr_pred, aff_pred]]
            else:
                for sample_i in range(aff_pred_batch.shape[0]):
                    aff_pred = aff_pred_batch[sample_i].item() * 10
                    scr_pred = scr_pred_batch[sample_i].sigmoid().item()
# 存储结果，记录配体-蛋白质标识符，筛选分数和亲和力预测
                    idx = idx_batch[sample_i]
                    scr_result += [[idx[0], idx[1], scr_pred, aff_pred]]

            if rank == 0:
                progBar.update(i + 1)

    save_val(scr_result, rank, 'tmp_val')

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

#设置GPU设备，准备分布式参数，启动多进程并执行GET_SCORE函数在多GPU上同时评估
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    parallel_args = (world_size, args.port, args)
    torch.multiprocessing.spawn(get_score, args=parallel_args, nprocs=world_size, join=True)
#查找所有临时预测结果文件，合并多GPU产生的预测结果，并删除临时文件
    f_list = [f for f in os.listdir() if f.startswith('tmp_val')]
    scr_result = []
    for i, f_name in enumerate(f_list):
        with open(f_name, 'rb') as f:
            scr_result += pickle.load(f)
        os.remove(f_name)
    with open(f'./scr_pred_ens{ens}.pkl', 'wb') as f:
        pickle.dump(scr_result, f)
    print('Prediction done')


#----------------------
#将模型结果转换为标准评估格式，按按照蛋白质来分组，为CASF基准测试的筛选能力做准备
#----------------------
#CASF基准测试
#为评估准备数据格式，重新加载预测结果，创建评分输出目录，按蛋白质分组生成评分文件（每个蛋白质一个文件，包含与该蛋白配对的所有配体的评分）
#评分为筛选得分与亲和力预测的乘积
    with open(f'./scr_pred_ens{ens}.pkl', 'rb') as f:
        scr_result = pickle.load(f)

    score_path = './score'
    delmkdir(score_path)

    for r in r_list:
        sub_f = f'{r}_score.dat'
        with open(score_path + '/' + sub_f, 'w') as f:
            f.write('#code_ligand_num,score\n')
            for i in scr_result:
                if i[1] == r:
                    f.write(f'{i[0]}_ligand_1,{i[2] * i[3]}\n')
    cmd = f'python forward_screening_power.py -c CoreSet.dat -s {score_path} -p positive -o forward_result.out -t TargetInfo.dat'
    print(cmd)
#此处使用了CASF评估脚本














