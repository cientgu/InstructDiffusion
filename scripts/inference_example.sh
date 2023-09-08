# Example: Image Editing
python edit_cli.py --input figure/animals.png --edit "Transform it to van Gogh, starry night style." --resolution 768 --steps 100 --config configs/instruct_diffusion.yaml --ckpt checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt --cfg-text 5.0 --cfg-image 1.25 --outdir logs/ --seed 93151
python edit_cli.py --input figure/animals.png --edit "Help the elephant wear a crown and maintain the appearance of others." --resolution 512 --steps 100 --config configs/instruct_diffusion.yaml --ckpt checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt --cfg-text 5.0 --cfg-image 1.25 --outdir logs/ --seed 51557 

# Example: Segmentation   More prompts can be found in the dataset/prompts/prompt_seg.txt
python edit_cli.py --input figure/mirrorcat.jpg --edit "Mark the pixels of the cat in the mirror to blue and leave the rest unchanged." --resolution 512 --steps 100 --config configs/instruct_diffusion.yaml --ckpt checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt --cfg-text 7.5 --cfg-image 1.5 --outdir logs/ --seed 94746

# Example: Keypoint Detection   More prompts can be found in the dataset/prompts/prompt_pose.txt
python edit_cli.py --input figure/people.jpg --edit "Use yellow to encircle the left knee of the people on the far left and draw a blue circle over the nose of the tallest people." --resolution 512 --steps 100 --config configs/instruct_diffusion.yaml --ckpt checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt --cfg-text 6.0 --cfg-image 0.5 --outdir logs/ --seed 27775

# Example: Watermark Removal   More prompts can be found in the dataset/prompts/prompt_dewatermark.txt
python edit_cli.py --input figure/watermark.png --edit "Remove watermark from this picture." --resolution 512 --steps 100 --config configs/instruct_diffusion.yaml --ckpt checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt --cfg-text 1.0 --cfg-image 1.0 --outdir logs/ --seed 54763