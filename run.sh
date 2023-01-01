nohup python tools/train.py /home/gauthierli/code/mmdetection/workdir/cityscapes/mask2former/res101_right/mask2former_r101_lsj_8x2_50e_coco.py \
  --work-dir /home/gauthierli/code/mmdetection/workdir/cityscapes/mask2former/res101_right \
  --resume-from /home/gauthierli/code/mmdetection/workdir/cityscapes/mask2former/res101_right/iter_5000.pth \
  > /home/gauthierli/code/mmdetection/workdir/cityscapes/mask2former/res101_right/nohup.log 2>&1 &