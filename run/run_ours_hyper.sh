for period in 1; do
    for bs in 1024 2048; do 
        for memory_dim in 32 64 128; do
            for drop_out in 0 0.1; do
                current_time=$(TZ="America/New_York" date -d '-5 hours' "+%y%m%d_%H%M%S")
                python main.py \
                    --prefix $current_time \
                    --period $period \
                    --bs $bs \
                    --memory_dim $memory_dim \
                    --drop_out $drop_out \
                    --model_name ours \
                    --gpu 0;
            done
        done
    done
done