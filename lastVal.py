#!/usr/bin/env python
""" COCO validation script

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import time
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.parallel
from contextlib import suppress

from effdet import create_model, create_evaluator, create_dataset, create_loader
from effdet.data import resolve_input_config
from timm.utils import AverageMeter, setup_default_logging
try:
    from timm.layers import set_layer_config
except ImportError:
    from timm.models.layers import set_layer_config

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True


def add_bool_arg(parser, name, default=False, help=''):  # FIXME move to utils
    dest_name = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=dest_name, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=dest_name, action='store_false', help=help)
    parser.set_defaults(**{dest_name: default})


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('root', metavar='DIR',
                    help='path to dataset root')
parser.add_argument('--dataset', default='coco', type=str, metavar='DATASET',
                    help='Name of dataset (default: "coco"')
parser.add_argument('--split', default='val',
                    help='validation split')
parser.add_argument('--model', '-m', metavar='MODEL', default='tf_efficientdet_d1',
                    help='model architecture (default: tf_efficientdet_d1)')
add_bool_arg(parser, 'redundant-bias', default=None,
                    help='override model config for redundant bias layers')
add_bool_arg(parser, 'soft-nms', default=None, help='override model config for soft-nms')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='Override num_classes in model config if set. For fine-tuning from pretrained.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='bilinear', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--fill-color', default=None, type=str, metavar='NAME',
                    help='Image augmentation fill (background) color ("mean" or int)')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                    help="Enable compilation w/ specified backend (default: inductor).")
parser.add_argument('--results', default='', type=str, metavar='FILENAME',
                    help='JSON filename for evaluation results')


def validate(args):
    setup_default_logging()

    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    args.pretrained = args.pretrained or not args.checkpoint  # might as well try to validate something
    args.prefetcher = not args.no_prefetcher

    # create model
    with set_layer_config(scriptable=args.torchscript):
        extra_args = {}
        if args.img_size is not None:
            extra_args = dict(image_size=(args.img_size, args.img_size))
        bench = create_model(
            args.model,
            bench_task='predict',
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            redundant_bias=args.redundant_bias,
            soft_nms=args.soft_nms,
            checkpoint_path=args.checkpoint,
            checkpoint_ema=args.use_ema,
            **extra_args,
        )
    model_config = bench.config

    param_count = sum([m.numel() for m in bench.parameters()])
    print('Model %s created, param count: %d' % (args.model, param_count))

    bench = bench.cuda()

    if args.torchscript:
        assert not args.apex_amp, \
            'Cannot use APEX AMP with torchscripted model, force native amp with `--native-amp` flag'
        bench = torch.jit.script(bench)
    elif args.torchcompile:
        bench = torch.compile(bench, backend=args.torchcompile)

    amp_autocast = suppress
    if args.apex_amp:
        bench = amp.initialize(bench, opt_level='O1')
        print('Using NVIDIA APEX AMP. Validating in mixed precision.')
    elif args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        print('Using native Torch AMP. Validating in mixed precision.')
    else:
        print('AMP not enabled. Validating in float32.')

    if args.num_gpu > 1:
        bench = torch.nn.DataParallel(bench, device_ids=list(range(args.num_gpu)))

    dataset = create_dataset(args.dataset, args.root, args.split)
    input_config = resolve_input_config(args, model_config)
    loader = create_loader(
        dataset,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=args.workers,
        pin_mem=args.pin_mem,
    )

    evaluator = create_evaluator(args.dataset, dataset, pred_yxyx=False)
    bench.eval()
    batch_time = AverageMeter()
    end = time.time()
    last_idx = len(loader) - 1

    # Prepare saving of up to 10 images into 'sambels' next to this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, 'sambels')
    os.makedirs(save_dir, exist_ok=True)
    saved_count = 0

    # Precompute mean/std tensors for denormalization
    mean = torch.tensor(input_config['mean']).view(1, -1, 1, 1)
    std = torch.tensor(input_config['std']).view(1, -1, 1, 1)

    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            with amp_autocast():
                output = bench(input, img_info=target)
            evaluator.add_predictions(output, target)

            # Save up to 10 images from this batch (denormalized) with boxes & scores
            if saved_count < 10:
                # Move mean/std to same device as input
                m = mean.to(input.device, dtype=input.dtype)
                s = std.to(input.device, dtype=input.dtype)
                # input: NCHW
                batch = input
                n = batch.shape[0]
                for bi in range(n):
                    if saved_count >= 10:
                        break
                    img_t = batch[bi:bi+1]  # 1CHW
                    img_denorm = (img_t * s + m).clamp(0, 1)
                    img_np = (img_denorm[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype('uint8')
                    try:
                        img_pil = Image.fromarray(img_np)
                        draw = ImageDraw.Draw(img_pil)

                        # Draw detections for this image
                        det = output[bi].detach().cpu()
                        # PIL size returns (W, H)
                        W, H = img_pil.size
                        score_thresh = 0.3
                        for d in det:
                            score = float(d[4])
                            if score < score_thresh:
                                continue
                            # boxes are [xmin, ymin, xmax, ymax] (xyxy)
                            # outputs are scaled to original size (generate_detections * img_scale)
                            # we draw on network input image, so divide by img_scale per image
                            try:
                                if isinstance(target, dict) and 'img_scale' in target:
                                    # target['img_scale'] shape: [B] or [B, 1]
                                    scale_t = target['img_scale'][bi]
                                    img_scale_val = float(scale_t.item())
                                else:
                                    img_scale_val = 1.0
                                if img_scale_val == 0:
                                    img_scale_val = 1.0
                            except Exception:
                                img_scale_val = 1.0

                            x1 = int(d[0].item() / img_scale_val)
                            y1 = int(d[1].item() / img_scale_val)
                            x2 = int(d[2].item() / img_scale_val)
                            y2 = int(d[3].item() / img_scale_val)
                            # clamp within image bounds
                            x1 = max(0, min(W - 1, x1))
                            y1 = max(0, min(H - 1, y1))
                            x2 = max(0, min(W - 1, x2))
                            y2 = max(0, min(H - 1, y2))
                            # skip empty boxes
                            if x2 <= x1 or y2 <= y1:
                                continue
                            draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)
                            label = f"{score:.2f}"
                            # background for text
                            tw, th = draw.textlength(label), 10
                            draw.rectangle([(x1, max(0, y1 - th - 2)), (x1 + int(tw) + 6, y1)], fill=(255, 0, 0))
                            draw.text((x1 + 3, max(0, y1 - th - 1)), label, fill=(255, 255, 255))

                        img_name = f'val_{i:04d}_{bi:02d}.jpg'
                        img_path = os.path.join(save_dir, img_name)
                        img_pil.save(img_path)
                        saved_count += 1
                        if saved_count == 10:
                            print(f'Saved 10 images to: {save_dir}')
                    except Exception as e:
                        print(f'Error saving image batch {i} idx {bi}: {e}')

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.log_freq == 0 or i == last_idx:
                print(
                    f'Test: [{i:>4d}/{len(loader)}]  '
                    f'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {input.size(0) / batch_time.avg:>7.2f}/s)  '
                )

    mean_ap = 0.
    if dataset.parser.has_labels:
        mean_ap = evaluator.evaluate(output_result_file=args.results)
    else:
        evaluator.save(args.results)


def main():
    args = parser.parse_args()
    validate(args)


if __name__ == '__main__':
    main()
