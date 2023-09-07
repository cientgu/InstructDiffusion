# Example: Image Editing
python edit_cli.py --input figure/animals.png --edit "Transform it to van Gogh, starry night style." --resolution 512 --steps 100 --config configs/instruct_diffusion.yaml --ckpt checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt --cfg-text 5.0 --cfg-image 1.25 --outdir logs/ --seed 93151
python edit_cli.py --input figure/animals.png --edit "Help the elephant wear a crown and maintain the appearance of others." --resolution 512 --steps 100 --config configs/instruct_diffusion.yaml --ckpt checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt --cfg-text 5.0 --cfg-image 1.25 --outdir logs/ --seed 51557 

# Example: Segmentation   More prompts can be found in the dataset/prompts/prompt_seg.txt
python edit_cli.py --input figure/mirrorcat.jpg --edit "Mark the pixels of the cat in the mirror to blue and leave the rest unchanged." --resolution 512 --steps 100 --config configs/instruct_diffusion.yaml --ckpt checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt --cfg-text 7.5 --cfg-image 1.5 --outdir logs/ --seed 94746
