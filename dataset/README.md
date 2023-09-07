You can download these datasets: [COCO](http://cocodataset.org/#download), [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose#dataset), [MPII](http://human-pose.mpi-inf.mpg.de/), [AIC](https://arxiv.org/abs/1711.06475), [COCO-Stuff](https://github.com/nightrome/cocostuff), [RefCOCO](https://github.com/lichengunc/refer), [GrefCOCO](https://github.com/henghuiding/gRefCOCO), [GoPro](https://seungjunnah.github.io/Datasets/gopro), [REDS](https://seungjunnah.github.io/Datasets/reds.html), [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/), [CLWD](https://arxiv.org/abs/2012.07616), [IP2PDataset](https://github.com/timothybrooks/instruct-pix2pix), [GIER](https://sites.google.com/view/gierdataset), [GQAInpaint](https://github.com/abyildirim/inst-inpaint), [MagicBrush](https://osu-nlp-group.github.io/MagicBrush/). The resulting data directory should look like this:

    InstructDiffusion
    |-- data
    `-- |-- coco
        |   |-- annotations
        |   `-- images
        |-- mpii
        |   |-- annot
        |   `-- images
        |-- crowdpose
        |   |-- json
        |   `-- images
        |-- aic
        |   |-- annotations
        |   `-- ai_challenger_keypoint_train_20170902
        |
        |-- coco-stuff
        |   |-- annotations
        |   |-- labels.txt
        |   `-- images
        |-- coco_2014
        |   |-- grefcoco
        |   |   |-- grefs(unc).json
        |   |   `-- instances.json
        |   |-- refcoco
        |   |   |-- instances.json
        |   |   |-- refs(google).p
        |   |   `-- refs(unc).p
        |   `-- images
        |        
        |-- GoPro
        |   |-- train
        |   `-- test
        |-- REDS
        |   |-- train
        |   `-- val
        |-- SIDD
        |   |-- train
        |   `-- val
        |-- CLWD
        |   |-- train
        |   |-- test
        |   `-- watermark_logo
        |
        |-- clip-filtered-dataset
        |   |-- shard-00.zip
        |   |-- shard-01.zip
        |   `-- ... 
        |-- GIER_editing_data
        |   |-- images
        |   `-- GIER.json
        |-- gqa-inpaint
        |   |-- images
        |   |-- images_inpainted
        |   |-- masks
        |   |-- train_scenes.json
        |   `-- meta_info.json
        `-- MagicBrush
            |-- data
            |-- processed-train
            `-- magic_train.json
