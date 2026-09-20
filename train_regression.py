import os
import csv
import json
import shutil
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset.vevo_dataset import create_vevo_datasets
from model.video_regression import VideoRegression

from utilities.constants import *
from utilities.device import get_device, use_cuda
from utilities.lr_scheduling import LrStepTracker, get_lr
from utilities.argument_funcs import parse_train_args, print_train_args, write_model_params

from utilities.run_model_regression import train_epoch, eval_model

CSV_HEADER = ["Epoch", "Learn rate", "Avg Train loss", "Train RMSE", "Avg Eval loss", "Eval RMSE"]
BASELINE_EPOCH = -1

version = VERSION
split_ver = SPLIT_VER
split_path = "split_" + split_ver

num_epochs = 200
VIS_MODELS_ARR = [
    "2d/clip_l14p"
]

regModel = "bigru"
# lstm
# bilstm
# gru
# bigru

# main
def main( vm = "" , isPrintArgs = True ):
    args = parse_train_args()
    args.epochs = num_epochs

    if isPrintArgs:
        print_train_args(args)
    if vm != "":
        args.vis_models = vm
    
    if args.is_video:
        vis_arr = args.vis_models.split(" ")
        vis_arr.sort()
        vis_abbr_path = ""
        for v in vis_arr:
            vis_abbr_path = vis_abbr_path + "_" + VIS_ABBR_DIC[v]
        vis_abbr_path = vis_abbr_path[1:]
    else:
        vis_abbr_path = "no_video"

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")
    else:
        use_cuda(True)

    os.makedirs( args.output_dir, exist_ok=True)
    os.makedirs( os.path.join( args.output_dir, version) ,  exist_ok=True)

    ##### Output prep #####
    params_file = os.path.join(args.output_dir, version, "model_params_regression.txt")
    write_model_params(args, params_file)

    weights_folder = os.path.join(args.output_dir, version, "weights_regression_" + regModel)
    os.makedirs(weights_folder, exist_ok=True)

    results_folder = os.path.join(args.output_dir, version)
    os.makedirs(results_folder, exist_ok=True)

    results_file = os.path.join(results_folder, "results_regression.csv")
    best_rmse_file = os.path.join(results_folder, "best_rmse_weights.pickle")
    best_text = os.path.join(results_folder, "best_epochs_regression.txt")

    ##### Tensorboard #####
    if(args.no_tensorboard):
        tensorboard_summary = None
    else:
        from torch.utils.tensorboard import SummaryWriter
        tensorboad_dir = os.path.join(args.output_dir, version, "tensorboard_regression")
        tensorboard_summary = SummaryWriter(log_dir=tensorboad_dir)
        
    train_dataset, val_dataset, _ = create_vevo_datasets(
        dataset_root = "./dataset/", 
        max_seq_chord = args.max_sequence_chord, 
        max_seq_video = args.max_sequence_video, 
        vis_models = args.vis_models,
        emo_model = args.emo_model, 
        split_ver = SPLIT_VER, 
        random_seq = True)
    
    total_vf_dim = 0
    for vf in train_dataset[0]["semanticList"]:
        total_vf_dim += vf.shape[1]

    total_vf_dim += 1 # Scene_offset
    total_vf_dim += 1 # Motion
    
    # Emotion
    if args.emo_model.startswith("6c"):
        total_vf_dim += 6
    else:
        total_vf_dim += 5

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_workers)

    # 학습셋에서 정규화 통계 계산
    print("정규화 통계 계산 중...")
    import numpy as np
    nd_all, lv_all = [], []
    for item in train_dataset:
        nd_all.extend(item["note_density"].tolist())
        lv_all.extend(item["loudness"].tolist())
    nd_mean, nd_std = float(np.mean(nd_all)), float(np.std(nd_all))
    lv_mean, lv_std = float(np.mean(lv_all)), float(np.std(lv_all))
    nd_std = max(nd_std, 1e-6)
    lv_std = max(lv_std, 1e-6)
    norm_stats = {"nd_mean": nd_mean, "nd_std": nd_std, "lv_mean": lv_mean, "lv_std": lv_std}
    norm_stats_file = os.path.join(results_folder, "norm_stats.json")
    with open(norm_stats_file, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Note Density — mean: {nd_mean:.3f}, std: {nd_std:.3f}")
    print(f"Loudness     — mean: {lv_mean:.4f}, std: {lv_std:.4f}")

    model = VideoRegression(max_sequence_video=args.max_sequence_video, total_vf_dim=total_vf_dim, regModel= regModel).to(get_device())
    
    start_epoch = BASELINE_EPOCH
    if(args.continue_weights is not None):
        if(args.continue_epoch is None):
            print("ERROR: Need epoch number to continue from (-continue_epoch) when using continue_weights")
            assert(False)
        else:
            model.load_state_dict(torch.load(args.continue_weights))
            start_epoch = args.continue_epoch
    elif(args.continue_epoch is not None):
        print("ERROR: Need continue weights (-continue_weights) when using continue_epoch")
        assert(False)

    eval_loss_func = nn.MSELoss()
    train_loss_func = nn.MSELoss()

    opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    ##### Tracking best evaluation accuracy #####
    best_eval_rmse        = float("inf")
    best_eval_rmse_epoch  = -1
    best_eval_loss       = float("inf")
    best_eval_loss_epoch = -1

    ##### Results reporting #####
    if(not os.path.isfile(results_file)):
        with open(results_file, "w", newline="") as o_stream:
            writer = csv.writer(o_stream)
            writer.writerow(CSV_HEADER)

    ##### TRAIN LOOP #####
    for epoch in range(start_epoch, args.epochs):
        if(epoch > BASELINE_EPOCH):
            print(SEPERATOR)
            print("NEW EPOCH:", epoch+1)
            print(SEPERATOR)
            print("")
            # Train
            train_epoch(epoch+1, model, train_loader, train_loss_func, opt, None, args.print_modulus, norm_stats=norm_stats)
            print(SEPERATOR)
            print("Evaluating:")
        else:
            print(SEPERATOR)
            print("Baseline model evaluation (Epoch 0):")
            
        # Eval
        train_loss, train_rmse, train_rmse_note_density, train_rmse_loudness  = eval_model(model, train_loader, train_loss_func, norm_stats=norm_stats)
        eval_loss, eval_rmse, eval_rmse_note_density, eval_rmse_loudness = eval_model(model, val_loader, eval_loss_func, norm_stats=norm_stats)

        # Learn rate
        lr = get_lr(opt)
        print("Epoch:", epoch+1)
        print("Avg train loss:", train_loss)
        print("Avg train RMSE:", train_rmse)
        print("Avg train RMSE (Note Density):", train_rmse_note_density)
        print("Avg train RMSE (Loudness):", train_rmse_loudness)
        
        print("Avg val loss:", eval_loss)
        print("Avg val RMSE:", eval_rmse)
        print("Avg val RMSE (Note Density):", eval_rmse_note_density)
        print("Avg val RMSE (Loudness):", eval_rmse_loudness)
        
        print(SEPERATOR)
        print("")

        lr_scheduler.step(eval_rmse)

        new_best = False
        if(eval_rmse < best_eval_rmse):
            best_eval_rmse = eval_rmse
            best_eval_rmse_epoch  = epoch+1
            torch.save(model.state_dict(), best_rmse_file)
            new_best = True
        
        # Writing out new bests
        if(new_best):
            with open(best_text, "w") as o_stream:
                print("Best val RMSE epoch:", best_eval_rmse_epoch, file=o_stream)
                print("Best val RMSE:", best_eval_rmse, file=o_stream)
                print("")
                print("Best val loss epoch:", best_eval_loss_epoch, file=o_stream)
                print("Best val loss:", best_eval_loss, file=o_stream)
        
        if((epoch+1) % args.weight_modulus == 0):
            epoch_str = str(epoch+1).zfill(PREPEND_ZEROS_WIDTH)
            path = os.path.join(weights_folder, "epoch_" + epoch_str + ".pickle")
            torch.save(model.state_dict(), path)
            
        with open(results_file, "a", newline="") as o_stream:
            writer = csv.writer(o_stream)
            writer.writerow([epoch+1, lr, train_loss, train_rmse, eval_loss, eval_rmse])
    return

if __name__ == "__main__":
    if len(VIS_MODELS_ARR) != 0 :
        for vm in VIS_MODELS_ARR:
            main(vm, False)
    else:
        main()
