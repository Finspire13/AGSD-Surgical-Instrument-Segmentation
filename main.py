import cv2
import argparse
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from skimage import morphology
from tqdm import tqdm
from tqdm import trange
from utils import *
from dataset import *
from model import MyUNet
from feature_extraction import FeatureExtraction


def add_border(pred, border_value=0):

    crop_i = preprocess_crop_location[0]
    crop_j = preprocess_crop_location[1]
    crop_h = preprocess_crop_size[0]
    crop_w = preprocess_crop_size[1]

    out = np.ones(original_img_size).astype(float) * border_value
    out[crop_i:crop_i+crop_h, crop_j:crop_j+crop_w] = np.array(pred)

    return out


def test(test_loader, model, 
    pos_prob_dir=None, neg_prob_dir=None, 
    pos_mask_dir=None, neg_mask_dir=None):

    results = {
        'name': [],
        'pred': [],
        'ground_truth': [],
        'pos_anchor': [],
        'neg_anchor': [],
    }

    with torch.no_grad():

        model.eval()

        for idx, data in enumerate(tqdm(test_loader)):    
            
            name = np.array(data['img_short_name']).item()
            sub_dir = name.split('/')[0]
            
            img = data['model_img'].squeeze(1).to(device) 

            logits = model(img)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            probs = cv2.resize(probs, (preprocess_crop_size[1], preprocess_crop_size[0]))
            probs = (probs * 255).astype(np.uint8)            
            
            if 'preprocess_ground_truth' in data:
                ground_truth = (data['preprocess_ground_truth'].squeeze().numpy() * 255).astype(np.uint8)
            else:
                ground_truth = None
                
            pos_anchor = (data['preprocess_pos_anchor'].squeeze().numpy() * 255).astype(np.uint8)
            neg_anchor = (data['preprocess_neg_anchor'].squeeze().numpy() * 255).astype(np.uint8)
            
            results['name'].append(name)
            results['pred'].append(probs)
            results['ground_truth'].append(ground_truth)
            results['pos_anchor'].append(pos_anchor)
            results['neg_anchor'].append(neg_anchor)

            if pos_prob_dir:

                pos_prob_sub_dir = os.path.join(pos_prob_dir, sub_dir)
                if not os.path.exists(pos_prob_sub_dir):
                    os.makedirs(pos_prob_sub_dir)

                cv2.imwrite(os.path.join(pos_prob_dir, name), 
                    add_border(probs, border_value=0).astype(np.uint8))

            if neg_prob_dir:

                neg_prob_sub_dir = os.path.join(neg_prob_dir, sub_dir)
                if not os.path.exists(neg_prob_sub_dir):
                    os.makedirs(neg_prob_sub_dir)

                cv2.imwrite(os.path.join(neg_prob_dir, name), 
                    add_border(255 - probs, border_value=255).astype(np.uint8))

        model.train()

    avg_iou = {
        'pred': [],
        'pos_anchor': [],
        'neg_anchor': [],
    }

    avg_dice = {
        'pred': [],
        'pos_anchor': [],
        'neg_anchor': [],
    }
        
    print('Saving Preds...')

    for i in trange(len(test_loader.dataset)):

        name = results['name'][i]
        sub_dir = name.split('/')[0]
        
        pred = get_mask(results['pred'][i], threshold=255*0.6)
        pos_anchor = get_mask(results['pos_anchor'][i], threshold=255*0.6)
        neg_anchor = get_mask(255 - results['neg_anchor'][i], threshold=255*0.6)
        
        pred = morphology.remove_small_objects(pred, 10000)
        pos_anchor = morphology.remove_small_objects(pos_anchor, 10000)
        neg_anchor = morphology.remove_small_objects(neg_anchor, 10000)
        
        if results['ground_truth'][i] is not None:
            
            ground_truth = get_mask(results['ground_truth'][i], threshold=0)
            
            avg_iou['pred'].append(get_iou(pred, ground_truth))
            avg_iou['pos_anchor'].append(get_iou(pos_anchor, ground_truth))
            avg_iou['neg_anchor'].append(get_iou(neg_anchor, ground_truth))

            avg_dice['pred'].append(get_dice(pred, ground_truth))
            avg_dice['pos_anchor'].append(get_dice(pos_anchor, ground_truth))
            avg_dice['neg_anchor'].append(get_dice(neg_anchor, ground_truth))

        pred = (pred * 255).astype(np.uint8)

        if pos_mask_dir:

            pos_mask_sub_dir = os.path.join(pos_mask_dir, sub_dir)
            if not os.path.exists(pos_mask_sub_dir):
                os.makedirs(pos_mask_sub_dir)

            cv2.imwrite(os.path.join(pos_mask_dir, name), 
                add_border(pred, border_value=0).astype(np.uint8))

        if neg_mask_dir:

            neg_mask_sub_dir = os.path.join(neg_mask_dir, sub_dir)
            if not os.path.exists(neg_mask_sub_dir):
                os.makedirs(neg_mask_sub_dir)

            cv2.imwrite(os.path.join(neg_mask_dir, name), 
                add_border(255 - pred, border_value=255).astype(np.uint8))

    avg_iou = {k: np.array(v).mean() for k, v in avg_iou.items()}
    avg_dice = {k: np.array(v).mean() for k, v in avg_dice.items()}

    return avg_iou, avg_dice


def train(train_train_loader, train_test_loader, test_test_loader, model, 
        log_dir, model_dir, pos_prob_dir, neg_prob_dir, pos_mask_dir, neg_mask_dir):
        
    logger = SummaryWriter(log_dir)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    feature_extractor = FeatureExtraction(
        feature_extraction_cnn=sim_feature_cnn, 
        normalization=True, last_layer=','.join(sim_feature_layers))

    feature_extractor.eval()

    steps = 0
    
    npy_log = []

    for e in range(epoch_num):

        print('Epoch: {}'.format(e))

        model.train()

        for _, data in enumerate(tqdm(train_train_loader)):

            img = data['model_img'].to(device)
            pos_anchor = data['model_pos_anchor'].to(device).permute(0, 1, 3, 4, 2)
            neg_anchor = data['model_neg_anchor'].to(device).permute(0, 1, 3, 4, 2)
            
            optimizer.zero_grad()
            
            if merge_batches:
            
                clip_size = img.shape[1]

                img = img.view(-1, *img.shape[2:])
                pos_anchor = pos_anchor.view(-1, *pos_anchor.shape[2:])
                neg_anchor = neg_anchor.view(-1, *neg_anchor.shape[2:])

                logits = model(img)

                anchor_loss, anchor_loss_details = get_anchor_loss(
                    logits, -logits, pos_anchor, neg_anchor, pos_ratio, neg_ratio
                )

                logits = logits.view(-1, clip_size, *logits.shape[1:])
                img = img.view(-1, clip_size, *img.shape[1:])

                logits_1, logits_2 = logits[:,0,:,:,:], logits[:,1,:,:,:]
                img_1, img_2 = img[:,0,:,:,:], img[:,1,:,:,:]
                
            else:

                img_1, img_2 = img[:,0,:,:,:], img[:,1,:,:,:]
                pos_anchor_1, pos_anchor_2 = pos_anchor[:,0,:,:,:], pos_anchor[:,1,:,:,:]
                neg_anchor_1, neg_anchor_2 = neg_anchor[:,0,:,:,:], neg_anchor[:,1,:,:,:]

                logits_1 = model(img_1)
                logits_2 = model(img_2)

                anchor_loss_1, anchor_loss_details_1 = get_anchor_loss(
                    logits_1, -logits_1, pos_anchor_1, neg_anchor_1, pos_ratio, neg_ratio
                )

                anchor_loss_2, anchor_loss_details_2 = get_anchor_loss(
                    logits_2, -logits_2, pos_anchor_2, neg_anchor_2, pos_ratio, neg_ratio
                )

                anchor_loss = anchor_loss_1 + anchor_loss_2

                anchor_loss_details = {k: anchor_loss_details_1[k] + anchor_loss_details_2[k]
                    for k in anchor_loss_details_1.keys()}

            # Semantic diffusion
            with torch.no_grad():
                feature_maps_1 = feature_extractor(img_1)
                feature_maps_2 = feature_extractor(img_2)

            diffusion_loss = {}

            diffusion_loss_details = {}

            for i, key in enumerate(sim_feature_layers):

                feature_maps_1[i] = F.interpolate(feature_maps_1[i], 
                    size=feature_map_size, mode='bicubic', align_corners=True)

                feature_maps_2[i] = F.interpolate(feature_maps_2[i], 
                    size=feature_map_size, mode='bicubic', align_corners=True)

                _diff_loss, _diff_details = get_diffusion_loss(
                    feature_maps_1[i], feature_maps_2[i], logits_1, logits_2,
                    fg_margin=sim_fg_margins[i], bg_margin=sim_bg_margins[i],
                    fg_ratio=fg_ratio, bg_ratio=bg_ratio,
                    naming='sim_{}'.format(key))

                diffusion_loss[key] = _diff_loss
                diffusion_loss_details[key] = _diff_details

            #######

            total_loss = anchor_loss

            for key in sim_feature_layers:
                total_loss += diffusion_loss[key]

            total_loss.backward()

            optimizer.step()

            for k in anchor_loss_details.keys():
                logger.add_scalar('Running-{}'.format(k), 
                    anchor_loss_details[k], steps)

            for v in diffusion_loss_details.values():
                for k in v.keys():
                    logger.add_scalar('Running-{}'.format(k), v[k], steps)

            steps += 1

        if e % 1 == 0:

            results = {}

            train_iou, train_dice = test(train_test_loader, model, 
                pos_prob_dir + '-{}'.format(e), neg_prob_dir + '-{}'.format(e), 
                pos_mask_dir + '-{}'.format(e), neg_mask_dir + '-{}'.format(e))

            results['train_iou'] = train_iou
            results['train_dice'] = train_dice
            
            if test_test_loader is not None:

                test_iou, test_dice = test(test_test_loader, model, 
                    pos_prob_dir + '-{}'.format(e), neg_prob_dir + '-{}'.format(e), 
                    pos_mask_dir + '-{}'.format(e), neg_mask_dir + '-{}'.format(e))

                results['test_iou'] = test_iou
                results['test_dice'] = test_dice
                
                npy_log.append([train_iou, train_dice, test_iou, test_dice])
            else:
                npy_log.append([train_iou, train_dice])

            for k1 in results.keys():
                for k2 in results[k1].keys():
                    logger.add_scalar('{}-{}'.format(k1, k2), 
                        results[k1][k2], steps)

            torch.save(model.state_dict(), 
                os.path.join(model_dir, 'model-{}'.format(steps)))

    logger.close()
    
    np.save(os.path.join(log_dir, 'npy_log.npy'), npy_log)


if __name__ == '__main__':  # This is for the Single Stage setting.

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=str)
    parser.add_argument('--load_model_path', type=str, default=None)

    args = parser.parse_args()
    print(args.config)

    all_params = load_config_file(args.config)
    locals().update(all_params)
    
    size_pack = [
        original_img_size, 
        preprocess_crop_size, 
        preprocess_crop_location, 
        model_input_size
    ]

    if test_dir_list is not None:
        test_datadict = get_datadict(
            img_dir=img_dir,
            pos_anchor_dir=pos_anchor_dir,
            neg_anchor_dir=neg_anchor_dir,
            ground_truth_dir=None,  # Ground truth is not available for EndoVis test set.
            sub_dir_list=test_dir_list
        )  
        # For EndoVis test set, predictions will be generated but IoU / Dice will not be computed.

        test_test_dataset = SegmentationDataset(
            test_datadict, *size_pack, data_aug=False, clip_size=1)

        test_test_loader = torch.utils.data.DataLoader(
            test_test_dataset, batch_size=1, shuffle=False, num_workers=8)
    else:
        test_test_loader = None

    model_dir = os.path.join(naming, 'models')
    log_dir = os.path.join(naming, 'logs')
    pos_prob_dir = os.path.join(naming, 'pos_prob')
    neg_prob_dir = os.path.join(naming, 'neg_prob')
    pos_mask_dir = os.path.join(naming, 'pos_mask')
    neg_mask_dir = os.path.join(naming, 'neg_mask')

    for d in [model_dir, log_dir, pos_prob_dir, neg_prob_dir, pos_mask_dir, neg_mask_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    model = MyUNet(transposed_conv, align_corners)

    # print('{} GPUs Used.'.format(torch.cuda.device_count()))

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    model.to(device)

    if args.load_model_path:
        print('Loading Model: {}'.format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    train_train_datadict = get_datadict(   # training data in the Single Stage setting
        img_dir=img_dir,
        pos_anchor_dir=pos_anchor_dir,
        neg_anchor_dir=neg_anchor_dir,
        ground_truth_dir=None,             # Unsupervised training
        sub_dir_list=train_dir_list
    )
    
    train_test_datadict = get_datadict(     # testing data in the Single Stage setting
        img_dir=img_dir,
        pos_anchor_dir=pos_anchor_dir,
        neg_anchor_dir=neg_anchor_dir,
        ground_truth_dir=ground_truth_dir,  # IoU / Dice on EndoVis train set are reported.
        sub_dir_list=train_dir_list         # In Single Stage setting, training data and testing data are the same.
    )
    
    train_train_dataset = SegmentationDataset(
        train_train_datadict, *size_pack, data_aug=True, clip_size=2)

    train_train_loader = torch.utils.data.DataLoader(
        train_train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    train_test_dataset = SegmentationDataset(
        train_test_datadict, *size_pack, data_aug=False, clip_size=1)

    train_test_loader = torch.utils.data.DataLoader(
        train_test_dataset, batch_size=1, shuffle=False, num_workers=8)

    train(train_train_loader, train_test_loader, test_test_loader, model, 
        log_dir, model_dir, pos_prob_dir, neg_prob_dir, pos_mask_dir, neg_mask_dir)
