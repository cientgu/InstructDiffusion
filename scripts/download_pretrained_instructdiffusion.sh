mkdir checkpoints

# if checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt exists
if [ -f checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt ]; then
    echo "checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt exists"
else
    echo "Downloading checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt"
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task-humanalign_aa
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task-humanalign_ab
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task-humanalign_ac
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task-humanalign_ad
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task-humanalign_ae
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task-humanalign_af
    
    cat v1-5-pruned-emaonly-adaption-task-humanalign_* > checkpoints/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt
    rm v1-5-pruned-emaonly-adaption-task-humanalign_*
fi

if [ -f checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt ]; then
    echo "checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt exists"
else
    echo "Downloading checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt"
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task_aa
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task_ab
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task_ac
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task_ad
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task_ae
    wget https://github.com/TiankaiHang/storage-2023/releases/download/0924/v1-5-pruned-emaonly-adaption-task_af

    cat v1-5-pruned-emaonly-adaption-task_* > checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt
    rm v1-5-pruned-emaonly-adaption-task_*
fi
