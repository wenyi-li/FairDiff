import os,sys
os.chdir(sys.path[0])
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from data.dataset import *
from utils.misc import *
from data.data import *
from models.vae_flow_genz import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from evaluation import *

# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--model', type=str, default='flow')
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.02)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--truncate_std', type=float, default=2.0)
parser.add_argument('--latent_flow_depth', type=int, default=14)
parser.add_argument('--latent_flow_hidden_dim', type=int, default=256)
parser.add_argument('--num_samples', type=int, default=6)
parser.add_argument('--sample_num_points', type=int, default=512)
parser.add_argument('--kl_weight', type=float, default=0.0001)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/pointcloud')
parser.add_argument('--result_path', type=str, default='./result')
parser.add_argument('--categories', type=str, default='gender',
                    choices = ['ethnicity', 'gender', 'language', 'maritalstatus',
                    'race'])
parser.add_argument('--attributes', type=str, default='Female',
                    choices = ['Female', 'Male'])
parser.add_argument('--scale_mode', type=str, default='global_unit',
                    choices = ['shape_unit', 'global_unit', 'no'])
parser.add_argument('--train_batch_size', type=int, default=48)
parser.add_argument('--val_batch_size', type=int, default=48)

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=100*THOUSAND, choices=[True, 200*THOUSAND])
parser.add_argument('--sched_end_epoch', type=int, default=300*THOUSAND, choices=[True, 400*THOUSAND])

# Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./train_logs-gen')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=2500000)
parser.add_argument('--val_freq', type=int, default=250*THOUSAND)
parser.add_argument('--tag', type=str, default=None)
args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    category_log = os.path.join(args.log_root, args.categories)
    os.makedirs(category_log, exist_ok=True)
    log_dir = get_new_log_dir(category_log, prefix='GEN_' + args.attributes)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    log_hyperparams(writer, args)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
logger.info('Loading datasets...')
#shuffle to train all
train_dset = Fundus(
    path=args.dataset_path,
    categories = args.categories,
    attributes=args.attributes,
    scale_mode=args.scale_mode,
    shuffle = True
)
train_iter = get_data_iterator(DataLoader(
    train_dset,
    batch_size=args.train_batch_size,
    num_workers=0,
))
gen_z_dset = Fundus(
    path=args.dataset_path,
    categories = args.categories,
    attributes=args.attributes,
    scale_mode=args.scale_mode,
    shuffle = False
)
gen_z_iter = DataLoader(train_dset, batch_size=args.val_batch_size, shuffle = True)

# Model
logger.info('Building model...')
if args.model == 'flow':
    model = FlowVAE_Genz(args).to(args.device)
logger.info(repr(model))
if args.spectral_norm:
    add_spectral_norm(model, logger=logger)

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
)
scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)

# Train, validate and test
def train(it):
    batch = next(train_iter)
    x = batch['pointcloud'].to(args.device)
    optimizer.zero_grad()
    model.train()
    if args.spectral_norm:
        spectral_norm_power_iteration(model, n_power_iterations=1)

    # Forward
    kl_weight = args.kl_weight
    loss = model.get_loss(x, kl_weight=kl_weight, writer=writer, it=it)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f' % (
        it, loss.item(), orig_grad_norm, kl_weight
    ))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/kl_weight', kl_weight, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()

def validate_inspect_noise2pc(it):
    logger.info('Start noise generate pointcloud...')
    z = torch.randn([args.train_batch_size, args.latent_dim]).to(args.device)
    y = model.sample(z, args.sample_num_points, flexibility=args.flexibility)
    stats_dir = os.path.join(args.dataset_path, args.categories, args.attributes + '_stats')
    stats_save_path = os.path.join(stats_dir, 'stats_' + args.attributes + '.pt')
    stats = (torch.load(stats_save_path))
    shift = stats['mean'].reshape(1, 3)
    scale = stats['std'].reshape(1, 1)
    y = y.cpu()
    x = y * scale + shift

    data_dir = os.path.join(args.result_path, args.categories, args.attributes, "train_val_noise2pc")
    os.makedirs(data_dir, exist_ok=True)
    writer.add_mesh('noise2pc', x, global_step=it)
    writer.flush()

    Number = (it // args.val_freq) - 1
    for i in range(args.train_batch_size):
        filename = f'{x.shape[0] * Number + i}.xyz'
        filename = os.path.join(data_dir, filename)
        np_array = x[i].squeeze(0).numpy()
        np.savetxt(filename, np_array, fmt='%f', delimiter=' ')

def validate_inspect_pc2pc(it):
    path_txt = os.path.join(args.dataset_path, args.categories, args.attributes + '.txt')
    with open(path_txt, 'r') as file:
        lines = [line.strip() for line in file]
    data_num = len(lines)
    batch_num = math.ceil(data_num / args.val_batch_size)
    batch_index = random.randint(0, batch_num - 1)
    logger.info('Start pointcloud generate pointcloud...')
    for i, batch in enumerate(gen_z_iter):
        if i == batch_index:
            data = batch

    x = data['pointcloud'].to(args.device)
    id = data['id'].to(args.device)
    z, pcy = model.pc2z(x, args.flexibility)
    pcy = pcy.cpu()
    stats_dir = os.path.join(args.dataset_path, args.categories, args.attributes + '_stats')
    stats_save_path = os.path.join(stats_dir, 'stats_' + args.attributes + '.pt')
    stats = (torch.load(stats_save_path))
    shift = stats['mean'].reshape(1, 3)
    scale = stats['std'].reshape(1, 1)
    pcx = pcy * scale + shift
    x = x.cpu()
    x = x * scale + shift
    writer.add_mesh('original-pc', x, global_step=it)
    writer.flush()
    writer.add_mesh('pc-pc', pcx, global_step=it)
    writer.flush()
    data_dir = os.path.join(args.result_path, args.categories, args.attributes, "train_val_pc2pc")
    latent_dir = os.path.join(args.result_path, args.categories, args.attributes, "latent")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(latent_dir, exist_ok=True)
    for index, order in enumerate(id):
        filename_pc = os.path.join(data_dir, lines[order])
        z_name = lines[order].replace('.xyz', '.pt')
        filename_z = os.path.join(latent_dir, z_name)

        np_array_pc = pcx[index].squeeze(0).numpy()
        torch.save(z[index].squeeze(0), filename_z)
        np.savetxt(filename_pc, np_array_pc, fmt='%f', delimiter=' ')

# Main loop
logger.info('Start training...')
try:
    it = 1
    while it <= args.max_iters:
        train(it)
        # if it % args.val_freq == 0 or it == args.max_iters:
        #     validate_inspect_noise2pc(it)
        #     validate_inspect_pc2pc(it)
        #     opt_states = {
        #         'optimizer': optimizer.state_dict(),
        #         'scheduler': scheduler.state_dict(),
        #     }
        #     ckpt_mgr.save(model, args, 0, others=opt_states, step=it)
        if it == args.max_iters:
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            ckpt_mgr.save(model, args, 0, others=opt_states, step=it)
        it += 1

except KeyboardInterrupt:
    logger.info('Terminating...')
