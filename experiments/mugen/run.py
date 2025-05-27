
from collections import defaultdict
import logging
import torch
import os
import time
import datetime
from tqdm import tqdm
import torch.nn.functional as F
# from AlignModuleTorchQL import AlignModule
from AlignModule import AlignModule
from SVMugenDataset import SVMugenDataset
import argparse
import wandb

# logger = logging.getLogger("torchql.alignmodule")
# logger_distribution = logging.getLogger("torchql.symbolic")
# logger_table = logging.getLogger("torchql.table")

# logger.stats = defaultdict(float)
# logger.reset_stats = lambda : logger.stats.clear()

def contrastive_loss(logits):
    neg_ce = torch.diag(logits)
    return -neg_ce.mean()

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity)
    return (caption_loss + image_loss) / 2.0

def build_loaders(args, split):
    dataset = SVMugenDataset(args=args, split=split)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True if split == "train" else False,
        drop_last=True if split == "train" else False,
        collate_fn = dataset.collate_fn,
    )

    return dataloader


class Trainer():

    def __init__(self, args, train_loader, valid_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.save_name = args.save_name
        if args.load:
            model_path = os.path.join(args.model_save_dir, args.model_name)
            if not os.path.exists(model_path):
                print(model_path)
            assert(os.path.exists(model_path))
            self.model = AlignModule(batch_size=args.batch_size, video_enc=True, audio_enc=False, text_enc=True,
                            pretrained=args.pretrained, provenance=args.provenance, device=args.device,
                            top_k=args.top_k, trainable=args.trainable, text_embedding=768, debug=args.debug_prov,
                            video_decoder_layers=args.video_decoder_layers, text_decoder_layers=args.text_decoder_layers,
                            multi_text=args.multi_text, load_path=model_path, gt_text=args.use_text_gt).to(args.device)
        else:
            self.model = AlignModule(batch_size=args.batch_size, video_enc=True, audio_enc=False, text_enc=True,
                            pretrained=args.pretrained, provenance=args.provenance, device=args.device,
                            top_k=args.top_k, trainable=args.trainable, text_embedding=768, debug=args.debug_prov,
                            video_decoder_layers=args.video_decoder_layers, text_decoder_layers=args.text_decoder_layers,
                            multi_text=args.multi_text, gt_text=args.use_text_gt).to(args.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.constraint_violation_loss = F.binary_cross_entropy
        self.match_loss = clip_loss
        self.device = args.device
        self.constraint_weight = args.constraint_weight
        self.alternative_train = args.alternative_train
        self.early_stopping = args.early_stopping

    def accuracy(self, y_pred, y):
        batch_size = len(y)
        # pred = torch.argmax(y_pred, dim=1)
        # gt = torch.argmax(y, dim=1)

        y = torch.arange(len(y_pred)).to(y_pred.device)

        img2cap_match_idx = y_pred.argmax(dim=1)
        cap2img_match_idx = y_pred.argmax(dim=0)

        img_acc = sum(img2cap_match_idx == y)
        cap_acc = sum(cap2img_match_idx == y)

        # num_correct = len([() for i, j in zip(pred, gt) if i == j])
        return (img_acc, cap_acc, batch_size)

    def train_epoch(self, epoch):
        if self.alternative_train:
            self.model.toggle_training_model()
        else:
            self.model.train()

        total_loss = []
        total_img_correct = 0
        total_text_correct = 0
        total_count = 0

        iterator = tqdm(self.train_loader)
        t_begin_epoch = time.time()

        for i, batch in enumerate(iterator):
            self.optimizer.zero_grad()
            batch = {k:v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch_size = len(batch['text_idx'])
            pred_match, pred_constraint_violation = self.model(batch)
            ground_truth = torch.diag(torch.tensor([1.0] * batch_size)).to(self.device)
            constraint_violation = torch.zeros(batch_size, batch_size).to(self.device)

            loss = self.constraint_violation_loss(pred_match, ground_truth) + self.constraint_weight * self.constraint_violation_loss(pred_constraint_violation, constraint_violation)
            # loss = self.match_loss(pred_match) + self.constraint_weight * self.constraint_violation_loss(pred_constraint_violation, constraint_violation)
            # loss = self.loss(pred_match, ground_truth) + self.constraint_weight * self.loss(pred_constraint_violation, constraint_violation)
            loss.backward()
            self.optimizer.step()

            img_acc, cap_acc, batch_size = self.accuracy(pred_match, ground_truth)
            total_loss.append(loss.item())
            total_img_correct += img_acc
            total_text_correct += cap_acc

            total_count += batch_size
            avg_loss = sum(total_loss) / (i + 1)
            correct_img_perc = (total_img_correct / total_count) * 100.0
            correct_text_perc = (total_text_correct / total_count) * 100.0

            iterator.set_description(f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Video Accu: {total_img_correct}/{total_count} ({correct_img_perc:.2f}%), Text Accu: {total_text_correct}/{total_count} ({correct_text_perc:.2f}%)")
            wandb.log({"epoch": epoch, "train/loss": loss})
            wandb.log({"epoch": epoch, "train/avg_loss": avg_loss})
        
        t_epoch = time.time() - t_begin_epoch
        print(f"Total Epoch Time: {t_epoch}")
        # print(logger.stats)
        # print(logger_distribution.stats)
        # print(logger_table.stats)
        # logger.reset_stats()
        # logger_distribution.reset_stats()
        # logger_table.reset_stats()
        wandb.log({"epoch": epoch, "train/t_time": t_epoch})

        avg_loss = sum(total_loss) / (i + 1)
        correct_img_perc = (total_img_correct / total_count) * 100.0
        correct_text_perc = (total_text_correct / total_count) * 100.0

        return avg_loss, correct_img_perc, correct_text_perc, t_epoch

    def eval_epoch(self, epoch):
        self.model.eval()
        total_loss = []
        total_img_correct = 0
        total_text_correct = 0
        total_count = 0

        with torch.no_grad():
            iterator = tqdm(self.valid_loader)
            for i, batch in enumerate(iterator):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                batch_size = len(batch['text_idx'])
                pred_match, pred_constraint_violation = self.model(batch)
                ground_truth = torch.diag(torch.tensor([1.0] * batch_size)).to(self.device)
                constraint_violation = torch.zeros(batch_size, batch_size).to(self.device)
                # loss = self.match_loss(pred_match) + self.constraint_weight * self.constraint_violation_loss(pred_constraint_violation, constraint_violation)
                loss = self.constraint_violation_loss(pred_match, ground_truth) + self.constraint_weight * self.constraint_violation_loss(pred_constraint_violation, constraint_violation)

                # loss = self.loss(pred_match, ground_truth) + self.constraint_weight * self.loss(pred_constraint_violation, constraint_violation)
                img_acc, cap_acc, batch_size = self.accuracy(pred_match, ground_truth)
                total_loss.append(loss.item())
                total_img_correct += img_acc
                total_text_correct += cap_acc

                total_count += batch_size
                avg_loss = sum(total_loss) / (i + 1)
                correct_img_perc = (total_img_correct / total_count) * 100.0
                correct_text_perc = (total_text_correct / total_count) * 100.0

                iterator.set_description(f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Video Accu: {total_img_correct}/{total_count} ({correct_img_perc:.2f}%), Text Accu: {total_text_correct}/{total_count} ({correct_text_perc:.2f}%)")

        wandb.log(
            {
                "epoch": epoch,
                "test/avg_loss": avg_loss,
                "test/video_acc": total_img_correct / total_count,
                "test/text_acc": total_text_correct / total_count,
            }
        )

        return avg_loss, correct_img_perc, correct_text_perc

    def test_epoch(self):
        self.model.eval()
        total_loss = []
        total_img_correct = 0
        total_text_correct = 0
        total_count = 0

        with torch.no_grad():
            iterator = tqdm(self.valid_loader)
            for i, batch in enumerate(iterator):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                batch_size = len(batch['text_idx'])
                pred_match, pred_constraint_violation = self.model.predict(batch)
                ground_truth = torch.diag(torch.tensor([1.0] * batch_size)).to(self.device)
                constraint_violation = torch.zeros(batch_size, batch_size).to(self.device)
                loss = self.constraint_violation_loss(pred_match, ground_truth) + self.constraint_weight * self.constraint_violation_loss(pred_constraint_violation, constraint_violation)

                # loss = self.loss(pred_match, ground_truth) + self.constraint_weight * self.loss(pred_constraint_violation, constraint_violation)
                img_acc, cap_acc, batch_size = self.accuracy(pred_match, ground_truth)
                total_loss.append(loss.item())
                total_img_correct += img_acc
                total_text_correct += cap_acc

                total_count += batch_size
                avg_loss = sum(total_loss) / (i + 1)
                correct_img_perc = (total_img_correct / total_count) * 100.0
                correct_text_perc = (total_text_correct / total_count) * 100.0

                iterator.set_description(f"[Test] Avg Loss: {avg_loss}, Video Accu: {total_img_correct}/{total_count} ({correct_img_perc:.2f}%), Text Accu: {total_text_correct}/{total_count} ({correct_text_perc:.2f}%)")

        return avg_loss, correct_img_perc, correct_text_perc

    def train(self):

        best_loss = float('inf')
        best_video_acc = 0
        best_text_acc = 0
        total_training_time = 0
        last_updated_epoch = 0

        for epoch in range(args.epochs):
            train_avg_loss, train_correct_img_perc, train_correct_img_perc, t_epoch = self.train_epoch(epoch)
            val_avg_loss, val_correct_img_perc, val_correct_text_perc = self.eval_epoch(epoch)

            if not args.do_not_save_model:
                self.model.save(os.path.join(args.model_save_dir, f"latest_{self.save_name}.pt"))

                if val_avg_loss < best_loss:
                    best_loss = val_avg_loss
                    self.model.save(os.path.join(args.model_save_dir, f"best_{self.save_name}.pt"))
                    print("Saved Best Model!")
                if val_correct_img_perc > best_video_acc:
                    best_video_acc = val_correct_img_perc
                    last_updated_epoch = epoch
                if val_correct_text_perc > best_text_acc:
                    best_text_acc = val_correct_text_perc
                    last_updated_epoch = epoch
                    
                print(f"Best loss: {best_loss}")
                print(f"Best video acc: {best_video_acc}")
                print(f"Best text acc: {best_text_acc}")
            total_training_time += t_epoch
            print("Total Training Time: ", total_training_time, f"({total_training_time / (epoch + 1)} per epoch)")

            # check if convergence
            if self.early_stopping and epoch - last_updated_epoch >= 5 and best_text_acc > 60 and best_video_acc > 60:
                print(f"Early Stopping! Converged at epoch {epoch}")
                break
        print(f"Total Training Time: {total_training_time} ({total_training_time / args.epochs} per epoch)")

    def test(self):
        self.eval_epoch(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="best_checkpoint.pt")
    parser.add_argument('--save_name', type=str, default="checkpoint")
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--default_root_dir', type=str, default='saved_checkpoints')
    parser.add_argument('--load', action='store_true')

    parser.add_argument('--train_data_ct', type=int, default=50)
    parser.add_argument('--test_data_ct', type=int, default=50)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--constraint_weight', type=float, default=0.01)

    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--train_size', type=int, default=None)
    parser.add_argument('--alternative_train', action='store_true')

    parser.add_argument('--video_enc', action='store_true')
    parser.add_argument('--audio_enc', action='store_true')
    parser.add_argument('--text_enc', action='store_true')
    parser.add_argument('--multi_text', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--trainable', action='store_true')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--debug_prov', action='store_true')
    parser.add_argument('--use_text_gt', action='store_true')

    parser.add_argument('--do-not-save-model', action='store_true')

    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--provenance', type=str, default="damp")
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--video_decoder_layers', type=int, default=2)
    parser.add_argument('--text_decoder_layers', type=int, default=2)
    parser.add_argument('--folder_name', type=str, default=None)
    parser.add_argument('--save_video', type=bool, default=False)
    parser.add_argument('--save_video_dir', type=str, default=None)

    parser.add_argument('--alternative_train_freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--phase', type=str, default="train")
    parser.add_argument('--save_pred', type=bool, default=True)
    parser.add_argument('--early_stopping', action= 'store_true')

    args = parser.parse_args()
    args.text_enc = False
    args.video_enc = True
    args.audio_enc = False
    args.trainable = True
    args.pretrained = True
    args.get_audio = False
    args.get_text_desc = True
    args.use_manual_annotation = False
    args.use_auto_annotation = True
    args.get_game_frame = True
    # args.use_cuda = False
    args.debug = True
    args.debug_prov = False
    args.multi_text=False
    args.use_text_gt = True
    args.alternative_train=False
    args.use_text_gt = True
    if args.phase == "test":
        args.load=True
        args.batch_size=3
    if args.debug_prov:
        args.batch_size=1

    args.model_save_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/mugen"))
    args.data_dir =  os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data/mugen"))
    args.device = f"cuda:{args.gpu}" if args.use_cuda else "cpu"
    args.video_save_dir = os.path.join(args.data_dir, 'video')

    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    print(args)
    return args

if __name__ == "__main__":

    args = parse_args()
    train_loader = build_loaders(args, "train")
    valid_loader = build_loaders(args, "val")
    trainer = Trainer(args=args, train_loader=train_loader, valid_loader=valid_loader)

    # Setup wandb
    config = {
        "device": args.device,
        "provenance": args.provenance,
        "top_k": args.top_k,
        "seed": args.seed,
        "n_epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_size": args.train_size,
        "learning_rate": args.lr,
        "experiment_type": f"torchql-large-{args.train_size}",
    }

    timestamp = datetime.datetime.now()
    id = f'torchql_mugen_{args.provenance}({args.top_k})_{args.seed}_{timestamp.strftime("%Y-%m-%d %H-%M-%S")}'

    wandb.login()
    wandb.init(project="WIP", config=config, id=id)
    wandb.define_metric("epoch")
    wandb.define_metric("train/t_time", step_metric="epoch", summary="mean")
    wandb.define_metric("test/avg_loss", step_metric="epoch", summary="min")
    wandb.define_metric("test/video_acc", step_metric="epoch", summary="max")
    wandb.define_metric("test/text_acc", step_metric="epoch", summary="max")

    if args.phase == "train":
        trainer.train()
    else:
        trainer.test_epoch()
