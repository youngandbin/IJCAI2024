for period in 1; do
    for bs in 1024; do 
        for memory_dim in 64; do
            for drop_out in 0; do
                for num_neg_train in 1 2 4 5; do
                    current_time=$(TZ="America/New_York" date -d '-5 hours' "+%y%m%d_%H%M%S")
                    python main_.py \
                        --prefix $current_time \
                        --period $period \
                        --bs $bs \
                        --memory_dim $memory_dim \
                        --drop_out $drop_out \
                        --num_neg_train $num_neg_train \
                        --model_name ours \
                        --wandb_name ours_ablation_ \
                        --gpu 0;
                done
            done
        done
    done
done

wait

for period in 2; do
    for bs in 2048; do 
        for memory_dim in 128; do
            for drop_out in 0.1; do
                for num_neg_train in 1 2 4 5; do
                    current_time=$(TZ="America/New_York" date -d '-5 hours' "+%y%m%d_%H%M%S")
                    python main_.py \
                        --prefix $current_time \
                        --period $period \
                        --bs $bs \
                        --memory_dim $memory_dim \
                        --drop_out $drop_out \
                        --num_neg_train $num_neg_train \
                        --model_name ours \
                        --wandb_name ours_ablation_ \
                        --gpu 0;
                done
            done
        done
    done
done

wait

for period in 3; do
    for bs in 1024; do 
        for memory_dim in 64; do
            for drop_out in 0; do
                for num_neg_train in 1 2 4 5; do
                    current_time=$(TZ="America/New_York" date -d '-5 hours' "+%y%m%d_%H%M%S")
                    python main_.py \
                        --prefix $current_time \
                        --period $period \
                        --bs $bs \
                        --memory_dim $memory_dim \
                        --drop_out $drop_out \
                        --num_neg_train $num_neg_train \
                        --model_name ours \
                        --wandb_name ours_ablation_ \
                        --gpu 0;
                done
            done
        done
    done
done

wait

for period in 4; do
    for bs in 1024; do 
        for memory_dim in 128; do
            for drop_out in 0; do
                for num_neg_train in 1 2 4 5; do
                    current_time=$(TZ="America/New_York" date -d '-5 hours' "+%y%m%d_%H%M%S")
                    python main_.py \
                        --prefix $current_time \
                        --period $period \
                        --bs $bs \
                        --memory_dim $memory_dim \
                        --drop_out $drop_out \
                        --num_neg_train $num_neg_train \
                        --model_name ours \
                        --wandb_name ours_ablation_ \
                        --gpu 0;
                done
            done
        done
    done
done

wait
