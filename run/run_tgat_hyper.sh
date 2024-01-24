for period in 1; do
    for n_heads in 1 2 4; do 
        for memory_dim in 32 64 128; do
            current_time=$(TZ="America/New_York" date -d '-5 hours' "+%y%m%d_%H%M%S")
            python main.py \
                --prefix $current_time \
                --period $period \
                --n_heads $n_heads \
                --memory_dim $memory_dim \
                --model_name tgat \
                --gpu 0;
        done
    done
done