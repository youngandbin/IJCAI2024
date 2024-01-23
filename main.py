import math, time, argparse, gc, torch, pickle, wandb, numpy as np
from pathlib import Path
from tqdm import tqdm
from model.tgn import TGN
from evaluation.evaluation import eval_recommendation
from utils.data import get_data, compute_time_statistics
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
torch.manual_seed(0)
np.random.seed(0)

"""
argument
"""
parser = argparse.ArgumentParser('Stock Rec')
# setting
parser.add_argument('--wandb_name', type=str, default='stock_rec', help='Name of the wandb project')
parser.add_argument('--gpu', type=int, default=1, help='Idx for the gpu to use')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--test_run', action='store_true', help='*run only first two batches')
# model
parser.add_argument('--memory_dim', type=int, default=64, help='Dimensions of the memory for each user')
parser.add_argument('--model_name', type=str, default="ours", choices=["ours", "tgn", "tgat", "jodie", "dyrep"], help='Type of model')
parser.add_argument('--loss_alpha', type=float, default=0.5, help='weight on contrastive loss')
parser.add_argument('--n_heads', type=int, default=2, help='Number of attention heads') 
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate') 
# training
parser.add_argument('--period', type=str, default='1', help='Period of data to use (1 to 7)')
parser.add_argument('--bs', type=int, default=1024, help='Batch_size')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--num_negatives', type=int, default=10, help='*part of batch items')
parser.add_argument('--num_neg_train', type=int, default=3, help='*p_pos and p_neg items')
# evaluation
parser.add_argument('--num_neg_eval', type=int, default=100, help='*neg items for evaluation')
parser.add_argument('--num_rec', type=int, default=1, help='*top k items for evaluation')
args = parser.parse_args()

# WANDB_NAME = args.wandb_name
# wandb.init(project=f"{WANDB_NAME}", config=args, name=args.prefix)

"""
global variables
"""
BATCH_SIZE = args.bs
NUM_EPOCH = 20
DATA = 'transaction' # 'Dataset name (eg. wikipedia or reddit)'
PERIOD = args.period 
USE_MEMORY = True
BACKPROP_EVERY = 1
DROP_OUT = args.drop_out
N_LAYERS = 1
N_HEADS = 2
LEARNING_RATE = 0.0001
args.memory_updater = 'gru' # choices=["gru", "rnn"]
args.embedding_module = 'graph_attention' # choices=["graph_attention", "graph_sum", "identity", "time"]
args.dyrep = False
args.use_destination_embedding_in_message = False
args.n_degree = 10 # Number of neighbors to sample
args.uniform = False # uniform: take uniform sampling from temporal neighbors (else: most recent neighbors)
if args.model_name=='jodie':
  args.memory_updater = 'rnn'
  args.embedding_module = 'time'
elif args.model_name=='dyrep':
  args.memory_updater = 'rnn'
  args.use_destination_embedding_in_message = True
  args.dyrep = True
elif args.model_name=='tgat':
  USE_MEMORY = False
  N_HEADS = args.n_heads
  LEARNING_RATE = args.lr
  args.uniform = True

"""
save paths
"""
Path(f"results/{args.prefix}").mkdir(parents=True, exist_ok=True) # save results from valid, test data
# Path(f"saved/{args.prefix}").mkdir(parents=True, exist_ok=True) # save model checkpoints 
# get_checkpoint_path = lambda epoch: f'./saved/{args.prefix}/{epoch}.pth'

"""
data
""" 
node_features, edge_features, full_data, train_data, val_data, test_data, upper_u = get_data(DATA, PERIOD)
node_features = np.random.rand(len(node_features), args.memory_dim)             # Generate new node features randomly based on the memory dimension (memory dim).
time_feature = pickle.load(open(f'data/time_feature_past.pkl', 'rb'))           # Dictionary containing historical daily prices for all stocks for each timestamp (ts).
map_item_id = pickle.load(open(f'data/period_{PERIOD}/map_item_id.pkl', 'rb'))  # Used to convert stock codes in a user portfolio to item idx.

"""
init
"""
# Initialize neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, uniform=args.uniform) 
full_ngh_finder = get_neighbor_finder(full_data, uniform=args.uniform)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

# Set device
device = torch.device('cuda:{}'.format(args.gpu))

# Initialize Model
tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
          edge_features=edge_features, device=device,
          n_layers=N_LAYERS,
          n_heads=N_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
          message_dimension=100, memory_dimension=args.memory_dim,
          memory_update_at_start=True,
          embedding_module_type=args.embedding_module,
          message_function='identity',
          aggregator_type='last',
          memory_updater_type=args.memory_updater,
          n_neighbors=args.n_degree,
          mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
          mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
          use_destination_embedding_in_message=args.use_destination_embedding_in_message,
          use_source_embedding_in_message=False,
          dyrep=args.dyrep)

optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
tgn = tgn.to(device)

# # Load pretrained model if specified
# if args.load_model is not None:
#   load_path = sorted(Path(f'saved/{args.load_model}').glob('*.pth'))[-2]
#   tgn.load_state_dict(torch.load(load_path))
#   print(f'Loaded model from {load_path}')

num_instance = len(train_data.sources)
num_batch = math.ceil(num_instance / BATCH_SIZE)

"""
epoch loop
"""

gc.collect() # These commands help you when you face CUDA OOM error
torch.cuda.empty_cache()

best_val_score = 0

for epoch in tqdm(range(NUM_EPOCH), desc="Progress: Epoch Loop" ):  

  start_epoch = time.time()
  
  """
  Train=======================================================================================================================================
  """

  # Reinitialize memory of the model at the start of each epoch
  if USE_MEMORY:
    tgn.memory.__init_memory__()

  # Train using only training graph
  tgn.set_neighbor_finder(train_ngh_finder)

  """
  batch loop
  """
  losses_batch = []
  for batch in tqdm(range(0, num_batch, BACKPROP_EVERY), total=num_batch//BACKPROP_EVERY, desc="Progress: Train Batch Loop"):

    # test run
    if args.test_run:
      if batch == 2:
        break

    loss = 0
    optimizer.zero_grad()

    # Custom loop to allow to perform backpropagation only every a certain number of batches
    for j in range(BACKPROP_EVERY):

      batch_idx = batch + j

      if batch_idx >= num_batch:
        continue

      s_idx = batch_idx * BATCH_SIZE
      e_idx = min(num_instance, s_idx + BATCH_SIZE)
      
      # batch data 뽑기: <class 'numpy.ndarray'>
      sources_batch = train_data.sources[s_idx:e_idx]           # (BATCH_SIZE,) 
      destinations_batch = train_data.destinations[s_idx:e_idx] # (BATCH_SIZE,) # item idx
      edge_idxs_batch  = train_data.edge_idxs[s_idx: e_idx]     # (BATCH_SIZE,)
      timestamps_batch = train_data.timestamps[s_idx:e_idx]     # (BATCH_SIZE,)
      portfolios_batch = train_data.portfolios[s_idx:e_idx]     # (BATCH_SIZE,) # stock code (6 digits)
      
      # calculate embeddings and loss
      if args.model_name == 'ours': 

        # negative sampling (candidates)
        train_rand_sampler = RandEdgeSampler(sources_batch, destinations_batch, portfolios_batch, upper_u, map_item_id)
        negatives_batch = train_rand_sampler.sample(size=args.num_negatives)  # (BATCH_SIZE, size) # item idx

        """
        p_negative sampling: interaction loop
        """
        # Convert item idx to stock codes 
        negatives_batch = negatives_batch - (upper_u + 1)                                           # item idx -> Index starting from 0
        negatives_batch = np.vectorize({v: k for k, v in map_item_id.items()}.get)(negatives_batch) # Index starting from 0 -> stock code (6 digits)

        """
        batch loop
        """
        p_pos_batch = []
        p_neg_batch = []
        neg_batch = []
        for idx, (stocks_p, items_c, t) in enumerate(zip(portfolios_batch, negatives_batch, timestamps_batch)):
          
          # ts conversion
          t = str(t)[:8] # Use only up to the DAY
          
          # First, random sampling for BPR loss: neg sampling from items_c
          neg_items = np.random.choice(items_c, args.num_neg_train, replace=False) # (k,)
          neg_batch.append(neg_items) 

          # 1. Load time features for the portfolio and candidates
          port_feature = np.array([time_feature[t][p] for p in stocks_p]) # shape: (n_stocks_p, 30)
          cand_feature = np.array([time_feature[t][c] for c in items_c])  # shape: (n_stocks_c, 30)

          """
          candidates loop
          """
          # 2. Create a new portfolio by appending one candidate to the portfolio at a time
          max_sharpes = {}
          for c, feature in zip(items_c, cand_feature):

            # The daily log return of the new portfolio.
            port_feature_new = np.append(port_feature, feature.reshape(1,30), axis=0)   # shape: (n_stocks_p+1, 30)
            daily_returns = np.log(port_feature_new[:, 1:] / port_feature_new[:, :-1])  # shape: (n_stocks_p+1, 29)
            daily_return = np.mean(daily_returns, axis=0)                               # shape: (29,)

            # mean of the daily log return
            mean = np.mean(daily_return) # shape: (1,)

            # std of the daily log return
            std = np.std(daily_return) # shape: (1,)

            # calculate Sharpe ratio
            Sharpe = mean / std # shape: (1,)

            max_sharpes[c] = Sharpe
          
          # 3. Set the negative with the highest max Sharpe as 'potential_pos' and the negative with the lowest max Sharpe as 'potential_neg'
          potential_pos = [x[0] for x in sorted(max_sharpes.items(), key=lambda x: x[1], reverse=True)[:args.num_neg_train]]   # Retrieve only the items from tuples (item, score).
          potential_neg = [x[0] for x in sorted(max_sharpes.items(), key=lambda x: x[1], reverse=True)[-args.num_neg_train:]]
          p_pos_batch.append(potential_pos) # e.g., [[695, 480, 460], [372, 231, 424], ...]
          p_neg_batch.append(potential_neg) # e.g., [[344, 538, 415], [247, 306, 418], ...]

        # flatten list of lists
        p_pos_batch = [x for y in p_pos_batch for x in y] # (BATCH_SIZE*k,)
        p_neg_batch = [x for y in p_neg_batch for x in y] 
        neg_batch = [x for y in neg_batch for x in y]

        # Convert stock codes to item idx
        p_pos_batch = [map_item_id[item] for item in p_pos_batch]   # stock code (6 digits) -> Index starting from 0
        p_pos_batch = np.array(p_pos_batch) + (upper_u + 1)         # Index starting from 0 -> item idx
        p_neg_batch = [map_item_id[item] for item in p_neg_batch]
        p_neg_batch = np.array(p_neg_batch) + (upper_u + 1)
        neg_batch = [map_item_id[item] for item in neg_batch]
        neg_batch = np.array(neg_batch) + (upper_u + 1)


        """
        emb calculation
        """
        tgn = tgn.train()
        source_embedding, destination_embedding, p_pos_embedding, p_neg_embedding, neg_embedding = tgn.compute_temporal_embeddings_p(sources_batch,
                                                                                                                                    destinations_batch,
                                                                                                                                    p_pos_batch,
                                                                                                                                    p_neg_batch,
                                                                                                                                    neg_batch,
                                                                                                                                    timestamps_batch,
                                                                                                                                    edge_idxs_batch,
                                                                                                                                    args.n_degree)

        """
        loss calculation
        """

        bsbs = source_embedding.shape[0]

        # reshape source and destination to (bs, 1, emb_dim) 
        source_embedding = source_embedding.view(bsbs, 1, -1)
        destination_embedding = destination_embedding.view(bsbs, 1, -1)

        # reshape p_pos and p_neg to (bs, k, emb_dim) 
        p_pos_embedding = p_pos_embedding.view(bsbs, args.num_neg_train, -1)
        p_neg_embedding = p_neg_embedding.view(bsbs, args.num_neg_train, -1)
        neg_embedding = neg_embedding.view(bsbs, args.num_neg_train, -1)

        # BPR loss
        pos_scores = torch.sum(source_embedding * destination_embedding, dim=2)                             # (bsbs, 1)
        neg_scores = torch.matmul(source_embedding, neg_embedding.transpose(1, 2)).squeeze()                # (bsbs, k)
        score_diff = pos_scores - neg_scores                                                                # (bsbs, k)
        score_diff_mean = torch.mean(score_diff, dim=1)                                                     # (bsbs, )
        log_and_sigmoid = torch.log(torch.sigmoid(score_diff_mean))                                         # (bsbs, )
        loss_BPR = -torch.mean(log_and_sigmoid)                                                             # (1, )

        # CL loss

        tau = 0.1 

        sims = torch.nn.functional.cosine_similarity(source_embedding, p_pos_embedding, dim=2) # (bsbs, k)
        exps = torch.exp(sims / tau)                                                           # (bsbs, k)
        sum_nominator = torch.sum(exps, dim=1)                                                 # (bsbs,)

        sims = torch.nn.functional.cosine_similarity(source_embedding, p_neg_embedding, dim=2) 
        exps = torch.exp(sims / tau) 
        sum_denominator = torch.sum(exps, dim=1)

        fraction = sum_nominator/sum_denominator # (bsbs,)
        loss_CL = -torch.mean(torch.log(fraction))

        # Combine loss
        alpha = args.loss_alpha
        loss_combined = loss_BPR*(1-alpha) + loss_CL*alpha
        loss += loss_combined

      else:
        
        # negative sampling
        train_rand_sampler = RandEdgeSampler(sources_batch, destinations_batch, portfolios_batch, upper_u, map_item_id)
        negatives_batch = train_rand_sampler.sample(size=args.num_neg_train)  # (BATCH_SIZE, size) # item idx
        """
        emb calculation
        """
        tgn = tgn.train()
        source_embedding, destination_embedding, negative_embedding = tgn.compute_temporal_embeddings(sources_batch,
                                                                                                      destinations_batch,
                                                                                                      negatives_batch.flatten(), # (BATCH_SIZE * size,)
                                                                                                      timestamps_batch,
                                                                                                      edge_idxs_batch,
                                                                                                      args.n_degree)

        """
        loss calculation
        """ 

        bsbs = source_embedding.shape[0]

        # reshape source and destination to (bs, 1, emb_dim) 
        source_embedding = source_embedding.view(bsbs, 1, -1)
        destination_embedding = destination_embedding.view(bsbs, 1, -1)

        # reshape p_pos and p_neg to (bs, k, emb_dim) 
        negative_embedding = negative_embedding.view(bsbs, args.num_neg_train, -1)

        # BPR loss

        pos_scores = torch.sum(source_embedding * destination_embedding, dim=2)                             # (bsbs, 1)
        neg_scores = torch.matmul(source_embedding, negative_embedding.transpose(1, 2)).squeeze()           # (bsbs, k)
        score_diff = pos_scores - neg_scores                                                                # (bsbs, k)
        score_diff_mean = torch.mean(score_diff, dim=1)                                                     # (bsbs, )
        log_and_sigmoid = torch.log(torch.sigmoid(score_diff_mean))                                         # (bsbs, )
        loss_BPR = -torch.mean(log_and_sigmoid)                                                             # (1, )

        loss += loss_BPR

    loss /= BACKPROP_EVERY
    if args.dyrep:
      loss.requires_grad_()
    loss.backward()
    optimizer.step()
    losses_batch.append(loss.item())

    # Detach memory after 'BACKPROP_EVERY' number of batches so we don't backpropagate to the start of time
    if USE_MEMORY:
      tgn.memory.detach_memory()
  
  # wandb.log({'loss': np.mean(losses_batch)}, step=epoch)
  # wandb.log({'time_train': time.time() - start_epoch}, step=epoch)

  """
  Valid=======================================================================================================================================
  """
  # Validation uses the full graph
  tgn.set_neighbor_finder(full_ngh_finder)

  eval_dict = eval_recommendation(tgn=tgn,
                                    data=val_data, 
                                    batch_size=BATCH_SIZE,
                                    n_neighbors=args.n_degree,
                                    upper_u=upper_u,
                                    NUM_NEG_EVAL = args.num_neg_eval,
                                    NUM_REC=args.num_rec,
                                    period=PERIOD,
                                    is_test_run=args.test_run,
                                    EVAL = 'valid',
                                    )
  if USE_MEMORY:
    val_memory_backup = tgn.memory.backup_memory()

  # save results
  # wandb.log(eval_dict, step=epoch)
  results_path = f'./results/{args.prefix}/{epoch}.pkl'
  pickle.dump(eval_dict, open(results_path, 'wb'))
  # wandb.log({'time_valid': time.time() - start_epoch}, step=epoch)

  """
  Test=======================================================================================================================================
  """
  tgn.embedding_module.neighbor_finder = full_ngh_finder

  eval_dict_test = eval_recommendation(tgn=tgn,
                                        data=test_data, 
                                        batch_size=BATCH_SIZE,
                                        n_neighbors=args.n_degree,
                                        upper_u=upper_u,
                                        NUM_NEG_EVAL = args.num_neg_eval,
                                        NUM_REC=args.num_rec,
                                        period=PERIOD,
                                        is_test_run=args.test_run,
                                        EVAL = 'test'
                                        )
  # save results
  # wandb.log(eval_dict_test, step=epoch)
  results_path = f'./results/{args.prefix}/{epoch}_test.pkl'
  pickle.dump(eval_dict_test, open(results_path, 'wb'))
  # wandb.log({'time_test': time.time() - start_epoch}, step=epoch)

# wandb.finish()